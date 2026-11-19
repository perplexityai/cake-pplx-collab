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

The non-Cake arms, the packs, the bytes model and the accuracy gate of
``bench_cake_gated_kv_compression_det_train.py``.

Every computing arm takes the operands the Cake autograd entries take: FP32 ``C`` and ``Z`` ``[T, 512]`` (the column
halves of one ``[T, 1024]`` tensor, the ``halves`` layout, or two contiguous tensors, the ``contiguous`` layout),
``cu_seqlens`` int32 ``[S + 1]`` on the device and the caller's pair count ``G`` (a Python int; no arm reads a device
value on the host), and returns ``out`` BF16 ``[G, 512]`` through autograd: the ``halves`` layout gets ``d[C | Z]`` as
one ``[T, 1024]`` gradient, the ``contiguous`` layout ``dC`` and ``dZ``.  Pair ``k`` of document ``s`` reads tokens
``t0 = cu_seqlens[s] + 2k`` and ``t1 = t0 + 1``; ``p0, p1 = softmax(Z[t0], Z[t1])`` per channel.

* ``compile_default`` / ``compile_max_autotune``: ``torch.compile`` (Inductor; ``mode="default"`` /
  ``"max-autotune-no-cudagraphs"``; ``fullgraph=True``; ``dynamic`` as the caller sets it) of the same pairwise math in
  a custom ``torch.autograd.Function``: a compiled forward (each output row finds its pair on the device from
  ``cu_seqlens`` and the prefix of the documents' pair counts; ``out = p0 C[t0] + p1 C[t1]`` in FP32, one BF16
  rounding) and a compiled token-major backward (one pass over the tokens: ``dC[t0] = p0 g``, ``dC[t1] = p1 g``,
  ``dZ[t0] = p0 p1 (C[t0] - C[t1]) g = -dZ[t1]`` with ``g = dout[r]``, tail rows +0.0; no scatter, no atomics).  The
  node saves the forward's pair prefix for the backward, as the Cake node saves its pair table, so neither backward
  scans the pair counts again.  Each arm built for a pack compiles its own copies of the
  functions (:func:`_fresh`), so no arm or pack reuses another's compiled code.
* ``eager``: the same two functions uncompiled: the reference, not a speed baseline.
* ``compile_autograd``: ``torch.compile`` of the reference forward written with ``torch.softmax``, differentiated by
  autograd.  The softmax backward forms ``dZ`` as ``p_i (g C_i - sum_j p_j g C_j)``, which loses the close-pair
  gradients the feature request names; the arm shows why the baselines spell the pairwise backward and is normally
  ineligible.

Accuracy gate (:func:`gate`): the feature request's acceptance criteria against a float64 oracle of its reference
(pairs from the host's list of document lengths, so the oracle shares no index arithmetic with any arm): every ``out``
element within one BF16 value of the float64 result rounded once to BF16 or, beyond that, with an absolute error of at
most one BF16 ulp of ``max(|C[t0, d]|, |C[t1, d]|)`` (the second case, for near-cancelling pairs; :func:`classify_out`),
``dC`` and ``dZ`` each within a relative L2 error of ``1e-6``, every output finite, the gradient rows of tail
tokens +0.0 and a second call bitwise equal to the first; on draws of the row's pack (plain, close pairs
``C[t1] = C[t0] (1 + 2^-12)`` and ``(1 + 2^-16)``, gate extremes) and on small edge packs (:data:`EDGE_PACKS`: empty
documents, single tokens, odd lengths, no pair at all).  An arm is eligible for the speed comparison
(:func:`eligibility`) when it meets these criteria on every draw, or when on every draw it is finite, zero on tail
rows, bitwise repeatable and no farther from the oracle than the Cake arm on the same draw (:func:`no_worse`: no more
``out`` elements outside both cases than Cake, ``dC`` and ``dZ`` errors within the bound or Cake's; the rule used is
recorded).
"""

from __future__ import annotations

import math
import random
import types
from collections.abc import Callable, Sequence
from typing import NamedTuple, Optional

import torch
from torch.autograd.function import once_differentiable

WIDTH = 512
CZ_WIDTH = 2 * WIDTH
LAYOUTS = ("halves", "contiguous")
# The feature request's acceptance bounds.
OUT_ULPS = 1
GRAD_REL_L2 = 1e-6
CLOSE_PAIRS = (12, 16)
# Small packs every computing arm is gated on once per layout.
EDGE_PACKS = ((0, 1, 3, 2, 7, 4), (2,), (3,), (5, 0, 0, 4), (127, 256, 1, 641), (1,), (0, 1, 0), (0,))

# Arm kinds: the Cake entry, a speed baseline, the reference, an informational arm, the bandwidth context.
CAKE, BASELINE, REFERENCE, INFORMATIONAL, CONTEXT = "cake", "baseline", "reference", "informational", "context"
COMPILE_MODES = {"compile_default": "default", "compile_max_autotune": "max-autotune-no-cudagraphs"}
BASELINE_ARMS = {
    "compile_default": BASELINE,
    "compile_max_autotune": BASELINE,
    "eager": REFERENCE,
    "compile_autograd": INFORMATIONAL,
}


class Unsupported(Exception):
    """An arm that cannot run on this device or with these libraries (the reason is reported)."""


# ----------------------------------------------------------------------------------------------------------------
# Packs


class Pack(NamedTuple):
    """One pack of documents: their lengths, in order (a length may be 0), and its row label."""

    lengths: tuple[int, ...]
    label: str

    @property
    def T(self) -> int:
        return sum(self.lengths)

    @property
    def S(self) -> int:
        return len(self.lengths)

    @property
    def G(self) -> int:
        return sum(n // 2 for n in self.lengths)

    @property
    def odd(self) -> int:
        return sum(n % 2 for n in self.lengths)


def lognormal_lengths(T: int, S: int, seed: int) -> tuple[int, ...]:
    """``S`` document lengths, each at least 1, summing to ``T``: weights from a seeded log-normal(0, 1) draw
    (Python's ``random``), the floor of each share, the remainder added to the longest document (the packs of the
    family's tests)."""
    rng = random.Random(seed)
    weights = [rng.lognormvariate(0.0, 1.0) for _ in range(S)]
    total = sum(weights)
    lengths = [max(1, math.floor(w / total * T)) for w in weights]
    lengths[lengths.index(max(lengths))] += T - sum(lengths)
    if min(lengths) < 1 or sum(lengths) != T:
        raise ValueError(f"cannot pack T = {T} into S = {S} documents of at least one token")
    return tuple(lengths)


def even_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    """``lengths`` made even (an odd length ``n`` becomes ``n - 1``, or 2 for ``n = 1``), the difference moved to the
    longest document, so the pack keeps ``T`` (even) and ``S`` and has no tail token."""
    out = [n + 1 if n == 1 else n - n % 2 for n in lengths]
    out[out.index(max(out))] += sum(lengths) - sum(out)
    if sum(out) != sum(lengths) or any(n % 2 or n < 2 for n in out):
        raise ValueError("these lengths cannot be made even with the same total")
    return tuple(out)


def tiny_lengths(T: int) -> tuple[int, ...]:
    """Lengths 1, 2, 3, 1, 2, 3, ... (the last one cut to fit) summing to ``T``: about ``T / 2`` documents, a third of
    them a single tail token."""
    out: list[int] = []
    total = 0
    while total < T:
        n = min(1 + len(out) % 3, T - total)
        out.append(n)
        total += n
    return tuple(out)


def bounds(lengths: Sequence[int]) -> list[int]:
    """``[0, b_1, ..., b_S]``: the prefix sums of ``lengths``."""
    out = [0]
    for n in lengths:
        out.append(out[-1] + int(n))
    return out


def pair_starts(lengths: Sequence[int]) -> list[int]:
    """``t0`` of every pair, in output-row order."""
    b = bounds(lengths)
    return [b[s] + 2 * k for s, n in enumerate(lengths) for k in range(n // 2)]


def tail_rows(lengths: Sequence[int]) -> list[int]:
    """The token row of every odd-length document's last token."""
    b = bounds(lengths)
    return [b[s + 1] - 1 for s, n in enumerate(lengths) if n % 2]


def phase_bytes(T: int, G: int) -> dict[str, int]:
    """The bytes of a dense pass of each phase (``cu_seqlens`` and the pair table neglected): ``fwd`` reads ``C`` and
    ``Z`` of every token (``4096 T``) and writes ``out`` (``1024 G``); ``bwd`` reads ``C``, ``Z`` and ``dout`` and
    writes ``dC`` and ``dZ`` (``8192 T + 1024 G``); ``step`` is both; ``ckpt`` two forwards and a backward.  No phase
    needs to read ``C`` or ``Z`` of a tail token, so on rows with many tail tokens (length-1 documents) these figures
    are more than a phase must move."""
    fwd = 2 * WIDTH * 4 * T + WIDTH * 2 * G
    bwd = 2 * (2 * WIDTH * 4 * T) + WIDTH * 2 * G
    return {"fwd": fwd, "bwd": bwd, "step": fwd + bwd, "ckpt": 2 * fwd + bwd}


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def draw(
    lengths: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    layout: str,
    close: Optional[int] = None,
    gates: bool = False,
) -> types.SimpleNamespace:
    """The operands of one call on ``lengths``: FP32 ``cz [T, 1024]`` and BF16 ``dout [G, 512]`` drawn with a CPU
    generator seeded with ``seed`` and moved to ``device`` (every GPU draws the same inputs, as in the family's
    tests); ``c`` and ``z`` its column halves (``halves``) or contiguous copies of them
    (``contiguous``).  ``close``: every pair has ``C[t1] = C[t0] (1 + 2^-close)`` in FP32.  ``gates``: fixed channel
    blocks of every pair get logit gaps of -100, +100, +2000 and -2000, exact ties and ties at 100 (where ``exp`` of
    the logit itself overflows FP32)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    T = sum(lengths)
    G = sum(n // 2 for n in lengths)
    generator = torch.Generator().manual_seed(seed)
    cz = torch.randn(T, CZ_WIDTH, generator=generator, dtype=torch.float32).to(device)
    dout = torch.randn(G, WIDTH, generator=generator, dtype=torch.float32).to(torch.bfloat16).to(device)
    t0 = torch.tensor(pair_starts(lengths), dtype=torch.long, device=device)
    if t0.numel():
        if close is not None:
            c = cz[:, :WIDTH]
            c[t0 + 1] = c[t0] * (1.0 + 2.0**-close)
        if gates:
            z = cz[:, WIDTH:]
            z0, z1 = z[t0], z[t0 + 1]
            z1[:, 0:64] = z0[:, 0:64] - 100.0
            z1[:, 64:128] = z0[:, 64:128] + 100.0
            z0[:, 128:192], z1[:, 128:192] = 1000.0, -1000.0
            z0[:, 192:256], z1[:, 192:256] = -1000.0, 1000.0
            z1[:, 256:320] = z0[:, 256:320]
            z0[:, 320:384], z1[:, 320:384] = 100.0, 100.0
            z[t0], z[t0 + 1] = z0, z1
    if layout == "halves":
        c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    else:
        c, z = cz[:, :WIDTH].contiguous(), cz[:, WIDTH:].contiguous()
    cu = torch.tensor(bounds(lengths), dtype=torch.int32, device=device)
    return types.SimpleNamespace(lengths=tuple(lengths), T=T, G=G, cz=cz, c=c, z=z, cu=cu, dout=dout, layout=layout)


def leaves_of(x: types.SimpleNamespace) -> list[torch.Tensor]:
    """The tensors an arm differentiates: ``[cz]`` (``halves``) or ``[c, z]`` (``contiguous``)."""
    return [x.cz] if x.layout == "halves" else [x.c, x.z]


# ----------------------------------------------------------------------------------------------------------------
# The pairwise math (FP32), forward and token-major backward


def _probabilities(z0: torch.Tensor, z1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(p0, p1) = softmax(z0, z1)`` per element, as the feature request's reference spells it."""
    m = torch.maximum(z0, z1)
    e0 = torch.exp(z0 - m)
    e1 = torch.exp(z1 - m)
    s = e0 + e1
    return e0 / s, e1 / s


def _row_pairs(
    cu: torch.Tensor, rows: int, tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(t0, t1, valid, ends)`` of output rows ``0 .. rows - 1``: row ``r`` is pair ``r - (ends[s] - pairs[s])`` of
    the document ``s`` whose pair range holds ``r`` (documents without a pair are skipped); rows past the pack's pair
    count are not ``valid`` and read token 0.  ``ends`` (int64 ``[S]``) is the inclusive prefix of the documents' pair
    counts."""
    cu = cu.to(torch.int64)
    starts = cu[:-1]
    pairs = (cu[1:] - starts) // 2
    ends = torch.cumsum(pairs, 0)
    r = torch.arange(rows, device=cu.device, dtype=torch.int64)
    doc = torch.searchsorted(ends, r, right=True)
    valid = doc < pairs.shape[0]
    doc = doc.clamp(max=pairs.shape[0] - 1)
    t0 = torch.where(valid, starts[doc] + 2 * (r - (ends[doc] - pairs[doc])), 0)
    t1 = (t0 + 1).clamp(max=tokens - 1)
    return t0, t1, valid, ends


def pair_forward(c: torch.Tensor, z: torch.Tensor, cu: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(out, ends)``: ``out [rows, 512]`` BF16, ``p0 C[t0] + p1 C[t1]`` in FP32, rounded once, zero past the pair
    count; ``ends``, the pair prefix the row search computed (:func:`_row_pairs`), for the backward."""
    t0, t1, valid, ends = _row_pairs(cu, rows, c.shape[0])
    p0, p1 = _probabilities(z[t0], z[t1])
    y = p0 * c[t0] + p1 * c[t1]
    return torch.where(valid[:, None], y, 0.0).to(torch.bfloat16), ends


def pair_forward_cz(cz: torch.Tensor, cu: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`pair_forward` on the column halves of ``cz [T, 1024]``."""
    return pair_forward(cz[:, :WIDTH], cz[:, WIDTH:], cu, rows)


def _token_pairs(cu: torch.Tensor, ends: torch.Tensor, rows: int, tokens: int):
    """Per token ``t``: ``(t0, t1, row, second, valid)``: the pair holding ``t``, its output row, whether ``t`` is
    the pair's second token, and whether ``t`` belongs to a pair at all (False for tail tokens and tokens outside
    every document).  ``ends`` is the forward's pair prefix (:func:`_row_pairs`): the pass does not scan the pair
    counts again."""
    cu = cu.to(torch.int64)
    starts, stops = cu[:-1], cu[1:]
    pairs = (stops - starts) // 2
    first = ends - pairs
    t = torch.arange(tokens, device=cu.device, dtype=torch.int64)
    doc = torch.searchsorted(stops, t, right=True).clamp(max=pairs.shape[0] - 1)
    pos = t - starts[doc]
    parity = pos % 2
    k = pos // 2
    valid = (pos >= 0) & (k < pairs[doc])
    t0 = (t - parity).clamp(min=0)
    t1 = (t0 + 1).clamp(max=tokens - 1)
    row = (first[doc] + k).clamp(min=0, max=rows - 1)
    return t0, t1, row, parity == 1, valid


def pair_backward(
    dout: torch.Tensor, c: torch.Tensor, z: torch.Tensor, cu: torch.Tensor, ends: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dC, dZ)`` FP32 ``[T, 512]`` in one pass over the tokens (no scatter): the first token of a pair gets
    ``p0 g`` and ``p0 p1 (C[t0] - C[t1]) g``, the second ``p1 g`` and the negation; tail tokens +0.0.  ``ends``: the
    forward's pair prefix."""
    t0, t1, row, second, valid = _token_pairs(cu, ends, rows, c.shape[0])
    p0, p1 = _probabilities(z[t0], z[t1])
    g = dout[row].float()
    flip = second[:, None]
    dc = torch.where(flip, p1, p0) * g
    dz0 = p0 * p1 * (c[t0] - c[t1]) * g
    dz = torch.where(flip, -dz0, dz0)
    keep = valid[:, None]
    return torch.where(keep, dc, 0.0), torch.where(keep, dz, 0.0)


def pair_backward_cz(
    dout: torch.Tensor, cz: torch.Tensor, cu: torch.Tensor, ends: torch.Tensor, rows: int
) -> torch.Tensor:
    """:func:`pair_backward` of the column halves, as one ``d[C | Z] [T, 1024]``."""
    dc, dz = pair_backward(dout, cz[:, :WIDTH], cz[:, WIDTH:], cu, ends, rows)
    return torch.cat([dc, dz], dim=1)


def release_forward(c: torch.Tensor, z: torch.Tensor, cu: torch.Tensor, rows: int) -> torch.Tensor:
    """The reference forward with ``torch.softmax`` over the stacked pair (differentiated by autograd)."""
    t0, t1, valid, _ = _row_pairs(cu, rows, c.shape[0])
    p = torch.softmax(torch.stack([z[t0], z[t1]], dim=1), dim=1)
    y = (p * torch.stack([c[t0], c[t1]], dim=1)).sum(dim=1)
    return torch.where(valid[:, None], y, 0.0).to(torch.bfloat16)


def release_forward_cz(cz: torch.Tensor, cu: torch.Tensor, rows: int) -> torch.Tensor:
    """:func:`release_forward` on the column halves of ``cz [T, 1024]``."""
    return release_forward(cz[:, :WIDTH], cz[:, WIDTH:], cu, rows)


# ----------------------------------------------------------------------------------------------------------------
# Compilation


def configure_dynamo() -> dict:
    """Raise the recompilation limit and make hitting it an error (Dynamo otherwise runs an over-limit frame
    uncompiled, which would time eager code under a compiled arm's name); returns the settings."""
    config = torch._dynamo.config
    found = {}
    for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(config, name):
            setattr(config, name, max(int(getattr(config, name)), 256))
            found[name] = getattr(config, name)
    for name in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit"):
        if hasattr(config, name):
            setattr(config, name, True)
            found[name] = True
    return found


def _fresh(fn: Callable, tag: str) -> Callable:
    """A copy of ``fn`` with its own code object: Dynamo keeps its compiled entries on the code object, so each copy
    compiles for itself (one arm's mode or pack never reuses another's entry)."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}__{tag}")
    copy = types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)
    copy.__qualname__ = code.co_name
    return copy


def _empty(tokens: int, cu: torch.Tensor, rows: int) -> bool:
    """No pair can be formed or no row is asked for (host metadata only)."""
    return tokens == 0 or rows == 0 or cu.numel() < 2


class PairMath:
    """The forward and the token-major backward of one arm, eager (``mode`` None) or compiled; the empty cases
    (``T == 0``, ``G == 0``, no document) are answered in Python, as the Cake host does.  The forward returns ``(out,
    ends)``, its pair prefix ``ends`` (None in an empty case) for the backward."""

    def __init__(self, tag: str, mode: Optional[str], dynamic: bool) -> None:
        fns = {
            "forward": pair_forward,
            "forward_cz": pair_forward_cz,
            "backward": pair_backward,
            "backward_cz": pair_backward_cz,
        }
        if mode is None:
            self.fns = fns
        else:
            self.fns = {
                key: torch.compile(_fresh(fn, f"{tag}_{key}"), mode=mode, dynamic=dynamic, fullgraph=True)
                for key, fn in fns.items()
            }

    def forward(self, c: torch.Tensor, z: torch.Tensor, cu: torch.Tensor, rows: int):
        if _empty(c.shape[0], cu, rows):
            return torch.zeros(rows, WIDTH, dtype=torch.bfloat16, device=c.device), None
        return self.fns["forward"](c, z, cu, rows)

    def forward_cz(self, cz: torch.Tensor, cu: torch.Tensor, rows: int):
        if _empty(cz.shape[0], cu, rows):
            return torch.zeros(rows, WIDTH, dtype=torch.bfloat16, device=cz.device), None
        return self.fns["forward_cz"](cz, cu, rows)

    def backward(self, dout, c, z, cu, ends, rows) -> tuple[torch.Tensor, torch.Tensor]:
        if _empty(c.shape[0], cu, rows):
            zeros = torch.zeros(c.shape[0], WIDTH, dtype=torch.float32, device=c.device)
            return zeros, zeros.clone()
        return self.fns["backward"](dout, c, z, cu, ends, rows)

    def backward_cz(self, dout, cz, cu, ends, rows) -> torch.Tensor:
        if _empty(cz.shape[0], cu, rows):
            return torch.zeros(cz.shape[0], CZ_WIDTH, dtype=torch.float32, device=cz.device)
        return self.fns["backward_cz"](dout, cz, cu, ends, rows)


class PairFunction(torch.autograd.Function):
    """``out = pair(C, Z)`` with the pairwise backward (``contiguous`` layout: ``dC`` and ``dZ``).  The node saves
    ``C``, ``Z``, ``cu_seqlens`` and the forward's pair prefix, which the backward reuses."""

    @staticmethod
    def forward(ctx, c, z, cu, rows, impl):  # type: ignore[override]
        ctx.impl, ctx.rows = impl, rows
        out, ends = impl.forward(c, z, cu, rows)
        ctx.save_for_backward(c, z, cu, ends)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):  # type: ignore[override]
        c, z, cu, ends = ctx.saved_tensors
        dc, dz = ctx.impl.backward(dout.contiguous(), c, z, cu, ends, ctx.rows)
        return dc, dz, None, None, None


class PairCzFunction(torch.autograd.Function):
    """``out = pair(cz[:, :512], cz[:, 512:])`` with the pairwise backward as one ``d[C | Z] [T, 1024]``.  The node
    saves ``cz``, ``cu_seqlens`` and the forward's pair prefix, which the backward reuses."""

    @staticmethod
    def forward(ctx, cz, cu, rows, impl):  # type: ignore[override]
        ctx.impl, ctx.rows = impl, rows
        out, ends = impl.forward_cz(cz, cu, rows)
        ctx.save_for_backward(cz, cu, ends)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):  # type: ignore[override]
        cz, cu, ends = ctx.saved_tensors
        return ctx.impl.backward_cz(dout.contiguous(), cz, cu, ends, ctx.rows), None, None, None


def _release_layer(layout: str, cu: torch.Tensor, rows: int, tag: str, dynamic: bool) -> Callable[..., torch.Tensor]:
    fn = release_forward_cz if layout == "halves" else release_forward
    compiled = torch.compile(_fresh(fn, tag), mode="default", dynamic=dynamic, fullgraph=True)

    def layer(*leaves: torch.Tensor) -> torch.Tensor:
        if _empty(leaves[0].shape[0], cu, rows):
            # Zero rows still connected to the leaves, so autograd gives them zero gradients.
            anchor = sum(leaf[:0].sum() for leaf in leaves)
            return (torch.zeros(rows, WIDTH, device=leaves[0].device) + anchor).to(torch.bfloat16)
        return compiled(*leaves, cu, rows)

    return layer


def make_layer(name: str, *, layout: str, cu: torch.Tensor, rows: int, dynamic: bool) -> Callable[..., torch.Tensor]:
    """``layer(*leaves) -> out`` of the non-Cake arm ``name`` (:data:`BASELINE_ARMS`) on the pack of ``cu`` with
    ``rows = G``; each call builds (and, when compiled, compiles on first use) its own functions."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if name == "compile_autograd":
        return _release_layer(layout, cu, rows, name, dynamic)
    if name == "eager":
        impl = PairMath(name, None, dynamic)
    elif name in COMPILE_MODES:
        impl = PairMath(name, COMPILE_MODES[name], dynamic)
    else:
        raise ValueError(f"unknown baseline arm {name!r}")
    if layout == "halves":
        return lambda cz: PairCzFunction.apply(cz, cu, rows, impl)
    return lambda c, z: PairFunction.apply(c, z, cu, rows, impl)


def describe(name: str, dynamic: bool) -> dict:
    """The implementation record of a non-Cake arm."""
    if name == "eager":
        return {"implementation": "PyTorch eager, pairwise forward and token-major backward (pair prefix saved)"}
    mode = "default" if name == "compile_autograd" else COMPILE_MODES[name]
    what = (
        "torch.compile of the reference forward (torch.softmax), autograd backward"
        if name == "compile_autograd"
        else "torch.compile of the pairwise forward and the token-major backward, custom autograd.Function saving "
        "the forward's pair prefix for the backward"
    )
    return {"implementation": what, "mode": mode, "dynamic": dynamic, "fullgraph": True}


# ----------------------------------------------------------------------------------------------------------------
# Accuracy gate


def oracle(x: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The float64 reference of ``x``: ``out [G, 512]`` unrounded, ``dc`` and ``dz [T, 512]`` (zero on tail rows),
    from the host's pair list."""
    t0 = torch.tensor(pair_starts(x.lengths), dtype=torch.long, device=x.cz.device)
    c, z = x.c.double(), x.z.double()
    out = torch.zeros(x.G, WIDTH, dtype=torch.float64, device=c.device)
    dc = torch.zeros(x.T, WIDTH, dtype=torch.float64, device=c.device)
    dz = torch.zeros_like(dc)
    if t0.numel():
        p0, p1 = _probabilities(z[t0], z[t0 + 1])
        out[:] = p0 * c[t0] + p1 * c[t0 + 1]
        g = x.dout.double()
        dz0 = p0 * p1 * (c[t0] - c[t0 + 1]) * g
        dc[t0], dc[t0 + 1] = p0 * g, p1 * g
        dz[t0], dz[t0 + 1] = dz0, -dz0
    return {"out": out, "dc": dc, "dz": dz}


def bf16_rounded_once(y: torch.Tensor) -> torch.Tensor:
    """FP64 ``y`` rounded once (to nearest, ties to even) to BF16: the quantum of ``y``'s binade at BF16's 8
    significant bits (the subnormal quantum ``2^-133`` below ``2^-126``), then an exact cast (a direct ``.to`` would
    round through FP32 first)."""
    _, exponent = torch.frexp(y)
    two = torch.tensor(2.0, dtype=torch.float64, device=y.device)
    quantum = torch.pow(two, (exponent.clamp(min=-125) - 8).to(torch.float64))
    return (torch.round(y / quantum) * quantum).to(torch.bfloat16)


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per element, the number of BF16 values from ``a`` to ``b``: 0 when equal (``+0.0`` and ``-0.0`` included)."""

    def ordered(t: torch.Tensor) -> torch.Tensor:
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


def bf16_ulp(m: torch.Tensor) -> torch.Tensor:
    """The BF16 ulp of FP64 ``m >= 0``: ``2^(clamp(e, min=-125) - 8)`` with ``(_, e) = frexp(m)``, the quantum rule of
    :func:`bf16_rounded_once` (``2^-133`` for every ``m`` below ``2^-125``), and ``2^-133`` for ``m == 0``."""
    _, exponent = torch.frexp(m)
    two = torch.tensor(2.0, dtype=torch.float64, device=m.device)
    quantum = torch.pow(two, (exponent.clamp(min=-125) - 8).to(torch.float64))
    return torch.where(m == 0, torch.pow(two, -133.0), quantum)


def classify_out(
    out: torch.Tensor, y: torch.Tensor, c0: torch.Tensor, c1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The feature request's ``out`` criterion per element (module docstring): ``(distance, second, outside)``.

    ``out`` is BF16, ``y`` the unrounded FP64 result, ``c0`` and ``c1`` the pair's stored ``C[t0, d]`` and
    ``C[t1, d]`` in FP64, all of one shape.  ``distance`` is the number of BF16 values from ``out`` to ``y`` rounded
    once to BF16 (the first case: at most :data:`OUT_ULPS`).  ``second`` marks the elements beyond that distance whose
    absolute error ``|out - y|`` is at most :func:`bf16_ulp` of ``max(|c0|, |c1|)``; ``outside`` the elements beyond
    that distance that exceed it, which fail the criterion.  The family's acceptance tests
    (``tests/experimental/test_cake_gated_kv_compression_det_train_acceptance.py``) keep an identical copy of this
    function and of :func:`bf16_ulp`: the tests do not import the benchmark."""
    distance = bf16_ulp_distance(out, bf16_rounded_once(y))
    within = (out.double() - y).abs() <= bf16_ulp(torch.maximum(c0.abs(), c1.abs()))
    beyond = distance > OUT_ULPS
    return distance, beyond & within, beyond & ~within


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    """``||got - want|| / ||want||`` in float64 (the denominator at least 1e-300)."""
    want = want.double()
    return float((got.double() - want).norm() / want.norm().clamp_min(1e-300))


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype]
    return t.contiguous().view(view)


def run(layer: Callable[..., torch.Tensor], x: types.SimpleNamespace) -> tuple[torch.Tensor, ...]:
    """``(out, dc, dz)`` of one forward and backward of ``layer`` on fresh leaves of ``x``'s storage."""
    leaves = [t.detach().requires_grad_(True) for t in leaves_of(x)]
    out = layer(*leaves)
    torch.autograd.backward([out], [x.dout])
    grads = [
        leaf.grad if leaf.grad is not None else torch.zeros(leaf.shape, dtype=leaf.dtype, device=leaf.device)
        for leaf in leaves
    ]
    if x.layout == "halves":
        dc, dz = grads[0][:, :WIDTH], grads[0][:, WIDTH:]
    else:
        dc, dz = grads
    return out.detach(), dc, dz


def measure(x: types.SimpleNamespace, got: Sequence[torch.Tensor], want: dict[str, torch.Tensor]) -> dict:
    """The acceptance measures of one call (module docstring).  On ``out`` (:func:`classify_out`): ``out_max_ulps``,
    the largest distance in BF16 values from the float64 result rounded once to BF16; ``out_elements_off``, the elements
    that differ from that rounded result (distance above 0); ``out_second_case``, the elements beyond one BF16 value
    within one BF16 ulp of ``max(|C[t0, d]|, |C[t1, d]|)``; ``out_outside``, the elements that meet neither case."""
    out, dc, dz = got
    found: dict = {
        "shapes": out.shape == (x.G, WIDTH)
        and out.dtype == torch.bfloat16
        and dc.shape == (x.T, WIDTH)
        and dz.shape == (x.T, WIDTH)
        and dc.dtype == torch.float32
        and dz.dtype == torch.float32
    }
    if not found["shapes"]:
        found["got"] = [[list(t.shape), str(t.dtype)] for t in got]
        return found
    c = x.c.double()
    t0 = torch.tensor(pair_starts(x.lengths), dtype=torch.long, device=c.device)
    ulps, second, outside = classify_out(out, want["out"], c[t0], c[t0 + 1])
    tails = torch.tensor(tail_rows(x.lengths), dtype=torch.long, device=dc.device)
    found.update(
        out_max_ulps=int(ulps.max()) if ulps.numel() else 0,
        out_elements_off=int((ulps > 0).sum()),
        out_second_case=int(second.sum()),
        out_outside=int(outside.sum()),
        dc_rel_l2=rel_l2(dc, want["dc"]),
        dz_rel_l2=rel_l2(dz, want["dz"]),
        finite=bool(torch.isfinite(out).all() and torch.isfinite(dc).all() and torch.isfinite(dz).all()),
        tail_rows=int(tails.numel()),
        tail_nonzero_words=int((_bits(dc[tails]) != 0).sum() + (_bits(dz[tails]) != 0).sum()),
    )
    return found


def passes(m: dict) -> bool:
    return bool(
        m.get("shapes")
        and m["out_outside"] == 0
        and m["dc_rel_l2"] <= GRAD_REL_L2
        and m["dz_rel_l2"] <= GRAD_REL_L2
        and m["finite"]
        and m["tail_nonzero_words"] == 0
        and m.get("repeat_bitwise")
    )


def variants(draws: int, seed: int, extremes: bool = True) -> list[tuple[str, dict]]:
    """The gate's draws: ``draws`` plain seeds, then (``extremes``) the close-pair and gate-extreme draws."""
    found = [(f"plain_seed{seed + i}", {"seed": seed + i}) for i in range(draws)]
    if extremes:
        found += [(f"close_2^-{k}", {"seed": seed, "close": k}) for k in CLOSE_PAIRS]
        found.append(("gate_extremes", {"seed": seed, "gates": True}))
    return found


def gate(
    layer: Callable[..., torch.Tensor],
    lengths: Sequence[int],
    *,
    layout: str,
    device: torch.device,
    draws: Sequence[tuple[str, dict]],
) -> dict:
    """The accuracy gate of ``layer`` on ``lengths`` (module docstring): ``{"pass": bool, "failed": [...],
    "draws": {label: measures}}``; an exception fails its draw and is recorded."""
    results: dict[str, dict] = {}
    for label, options in draws:
        try:
            x = draw(lengths, device=device, layout=layout, **options)
            want = oracle(x)
            first = run(layer, x)
            m = measure(x, first, want)
            if m["shapes"]:
                second = run(layer, x)
                m["repeat_bitwise"] = all(
                    bool(torch.equal(_bits(a), _bits(b))) for a, b in zip(first, second, strict=True)
                )
            m["pass"] = passes(m)
        except Exception as exc:  # noqa: BLE001 -- a failing arm is reported, the gate goes on
            m = {"pass": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        results[label] = m
        x = want = first = second = None  # noqa: F841 -- free the draw before the next one
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    failed = [label for label, m in results.items() if not m["pass"]]
    return {"pass": not failed, "failed": failed, "draws": results}


def no_worse(m: dict, reference: Optional[dict]) -> bool:
    """``m`` is finite, zero on tail rows, bitwise repeatable and at most as far from the oracle as ``reference``
    (the Cake arm's measures of the same draw): no more ``out`` elements outside both cases of the criterion than
    ``reference``, and ``dC`` and ``dZ`` within the bound or within ``reference``'s error where it misses the bound."""
    if reference is None or "error" in m or "error" in reference or not m.get("shapes") or not reference.get("shapes"):
        return False
    return bool(
        m["finite"]
        and m["tail_nonzero_words"] == 0
        and m.get("repeat_bitwise")
        and m["out_outside"] <= reference["out_outside"]
        and m["dc_rel_l2"] <= max(GRAD_REL_L2, reference["dc_rel_l2"])
        and m["dz_rel_l2"] <= max(GRAD_REL_L2, reference["dz_rel_l2"])
    )


def eligibility(found: dict, reference: Optional[dict]) -> tuple[bool, str]:
    """Whether a :func:`gate` result admits its arm to the speed comparison, and the rule that decided (module
    docstring); ``reference`` is the Cake arm's result on the same draws (None: the criteria alone)."""
    if found["pass"]:
        return True, "meets the feature request's criteria on every draw"
    if reference is None:
        return False, f"misses the feature request's criteria on {found['failed']}"
    if all(m["pass"] or no_worse(m, reference["draws"].get(label)) for label, m in found["draws"].items()):
        return True, f"misses the criteria on {found['failed']} but is no farther from the oracle than Cake there"
    return False, f"misses the criteria on {found['failed']} and is farther from the oracle than Cake"


def edge_eligibility(found: dict, reference: Optional[dict]) -> tuple[bool, str]:
    """:func:`eligibility` of an :func:`edge_gate` result, pack by pack."""
    if found.get("pass"):
        return True, "meets the feature request's criteria on every edge pack"
    if "packs" not in found:
        return False, "edge-pack gate not run"
    for label, pack in found["packs"].items():
        ref = None if reference is None or "packs" not in reference else reference["packs"].get(label)
        ok, rule = eligibility(pack, ref)
        if not ok:
            return False, f"edge pack {label}: {rule}"
    return True, "no farther from the oracle than Cake on the edge packs where it misses the criteria"


def edge_gate(
    make: Callable[[torch.Tensor, int], Callable[..., torch.Tensor]],
    *,
    layout: str,
    device: torch.device,
) -> dict:
    """:func:`gate` on every pack of :data:`EDGE_PACKS` (one plain draw and the close-pair draws), with a layer
    ``make(cu, G)`` built for each pack."""
    packs: dict[str, dict] = {}
    for lengths in EDGE_PACKS:
        label = "L" + "-".join(str(n) for n in lengths)
        try:
            cu = torch.tensor(bounds(lengths), dtype=torch.int32, device=device)
            layer = make(cu, sum(n // 2 for n in lengths))
            draws = variants(1, 0, extremes=False) + [(f"close_2^-{k}", {"seed": 0, "close": k}) for k in CLOSE_PAIRS]
            packs[label] = gate(layer, lengths, layout=layout, device=device, draws=draws)
        except Exception as exc:  # noqa: BLE001 -- recorded as a failed pack
            error = {"pass": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            packs[label] = {"pass": False, "failed": ["build"], "draws": {"build": error}}
    failed = [label for label, p in packs.items() if not p["pass"]]
    return {"pass": not failed, "failed": failed, "packs": packs}


# ----------------------------------------------------------------------------------------------------------------
# Retained state


def saved_bytes(
    layer: Callable[..., torch.Tensor], leaves: Sequence[torch.Tensor], inputs: Sequence[torch.Tensor]
) -> dict:
    """The tensors one forward of ``layer`` saves for its backward (``saved_tensors_hooks``): all of them, and those
    whose storage is not one of ``inputs``' (the state the backward retains beyond the operands)."""
    owned = {t.untyped_storage().data_ptr() for t in inputs}
    found: list[dict] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        found.append(
            {
                "bytes": t.numel() * t.element_size(),
                "input": t.untyped_storage().data_ptr() in owned,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
            }
        )
        return t

    fresh = [leaf.detach().requires_grad_(True) for leaf in leaves]
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = layer(*fresh)
    del out
    return {
        "saved_bytes": sum(f["bytes"] for f in found),
        "saved_beyond_inputs_bytes": sum(f["bytes"] for f in found if not f["input"]),
        "tensors": found,
    }
