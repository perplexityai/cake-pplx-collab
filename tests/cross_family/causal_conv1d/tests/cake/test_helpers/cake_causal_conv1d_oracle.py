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

The float64 oracle and the naive BF16 tier of the causal_conv1d training tests, and the loop that judges a kernel's
outputs against them over :data:`~tests.cake.test_helpers.cake_naive.DRAWS` independent inputs
(:mod:`tests.cake.test_helpers.cake_naive`).

The oracle (:func:`oracle`) is the function itself in float64, document-local over ``cu_seqlens``::

    pre[t, c]     = sum_k w[c, k] x[t - 3 + k, c] + b[c]   (x of another document, or before the first token, is 0)
    y[t, c]       = pre * sigmoid(pre)
    dpre[t, c]    = dy[t, c] * s (1 + pre (1 - s)),  s = sigmoid(pre)
    dx[t, c]      = sum_k w[c, k] dpre[t + 3 - k, c]   (dpre of another document, or past the last token, is 0)
    dweight[c, k] = sum_t dpre[t, c] x[t - 3 + k, c],  dbias[c] = sum_t dpre[t, c]

with no rounding: the kernels store ``y`` and ``dx`` in BF16 and the default SiLU' policy evaluates SiLU' at the
BF16-rounded preactivation, but the oracle is the function on every route and policy.  BF16 and FP32 inputs are
widened exactly.  It works in token blocks of :data:`BLOCK` rows with a three-token halo, so its float64 temporaries
stay bounded at ``S x C`` over 2^31.

The naive tier (:func:`naive_bf16`) is PyTorch in BF16 on the same inputs: every input cast to BF16 (FP32 taps and
bias rounded once), each document's rows behind three zero rows (no history), one ``F.conv1d`` over them
(``groups=C``, the taps as ``[C, 1, 4]``: tap ``k`` on row ``t - 3 + k``, the bias in the convolution), ``F.silu``
and ``torch.autograd.grad`` against ``dy``, all in BF16.  The channels are independent, so it runs on blocks of
channels of at most :data:`NAIVE_BLOCK` padded elements each.

:func:`judge_draws` makes each draw's inputs, launches the kernel and holds every output to its contract and to the
naive tier (``cake_naive.TierJudge``); it records the verdicts on the running test (``naive_tier`` in its user
properties, printed by the ``conftest.py`` terminal summary) and fails the test with every failure.  A deterministic
kernel's own bit checks (run to run, and the invariance contracts it documents) stay in the tests that call it.
"""

from __future__ import annotations

import random
import types
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from tests.cake.test_helpers.cake_naive import (
    DRAWS,
    TIERS,
    TierJudge,
    contract,
    draw_seed,
    seed_all,
)

WIDTH = 4
# The oracle's token block: each float64 temporary of a block is (BLOCK + 6) x C x 8 bytes.
BLOCK = 8192
# The most padded elements (rows x channels) of one block of channels of :func:`naive_bf16`.
NAIVE_BLOCK = 2**27
# The outputs of the backward (the oracle needs ``dy`` for them).
GRAD_OUTPUTS = ("dx", "dweight", "dbias")
# What a launch of :func:`judge_draws` returns: the outputs by name (a value, or the values of several launches) and
# the problem they are judged on, or a list of such pairs (one per tensor of a multi-tensor call).
Outputs = Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
Launched = tuple[Outputs, types.SimpleNamespace] | list[tuple[Outputs, types.SimpleNamespace]]

# The running pytest item, set by the ``conftest.py`` autouse fixture: the verdicts go to its user properties.
_NODE = None


def set_node(node) -> None:
    """Record verdicts on ``node`` (``None``: record nothing, as in a spawned rank process)."""

    global _NODE
    _NODE = node


def doc_ids(S: int, cu: torch.Tensor) -> torch.Tensor:
    """The document of each token as the kernels find it: the largest ``d`` with ``cu[d] <= t``."""

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
    b: torch.Tensor | None = None,
    dy: torch.Tensor | None = None,
    *,
    outputs: Sequence[str] = ("y", "dx", "dweight", "dbias"),
) -> dict[str, torch.Tensor]:
    """The float64 values of ``outputs`` for the rows ``x`` (``[S, C]``) of the documents ``cu``: ``y`` (``[S, C]``),
    and with ``dy`` ``dx`` (``[S, C]``), ``dweight`` (``[C, 4]``) and, with a bias ``b``, ``dbias`` (``[C]``)."""

    S, C = x.shape
    ids = doc_ids(S, cu)
    wd = w.double()
    bd = None if b is None else b.double()
    found: dict[str, torch.Tensor] = {}
    want_y = "y" in outputs
    want_grad = dy is not None and any(name in outputs for name in GRAD_OUTPUTS)
    if want_y:
        found["y"] = torch.empty(S, C, device=x.device, dtype=torch.float64)
    if want_grad:
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
        if want_y:
            found["y"][a:end] = (pre * s)[rows]
        if want_grad:
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
    return {name: found[name] for name in outputs if name in found}


def naive_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    b: torch.Tensor | None = None,
    dy: torch.Tensor | None = None,
    *,
    outputs: Sequence[str] = ("y", "dx", "dweight", "dbias"),
) -> dict[str, torch.Tensor]:
    """The naive BF16 tier of ``outputs`` (module docstring), as bfloat16 tensors of the oracle's shapes.  Reads its
    inputs and writes none."""

    S, C = x.shape
    ids = doc_ids(S, cu)
    # Row t of document d sits at t + 3 (d + 1): three zero rows before every document, an empty one's included.
    pos = torch.arange(S, device=x.device) + (WIDTH - 1) * (ids.to(x.device) + 1)
    rows = int(pos[-1]) + 1
    step = max(1, NAIVE_BLOCK // rows)
    grad = dy is not None and any(name in outputs for name in GRAD_OUTPUTS)
    parts = []
    for c in range(0, C, step):
        cols = slice(c, c + step)
        parts.append(
            _naive_channels(
                x[:, cols],
                w[cols],
                None if b is None else b[cols],
                pos,
                rows,
                dy[:, cols] if grad else None,
            )
        )
    found = {
        name: torch.cat([part[name] for part in parts], dim=0 if name in ("dweight", "dbias") else 1)
        for name in parts[0]
    }
    return {name: found[name] for name in outputs if name in found}


def _naive_channels(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    pos: torch.Tensor,
    rows: int,
    dy: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """:func:`naive_bf16` of one block of channels: ``x`` scattered to the padded rows ``pos`` of ``rows``."""

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


def problem(
    x: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    b: torch.Tensor | None = None,
    dy: torch.Tensor | None = None,
    *,
    rows: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    """The inputs a draw's outputs are judged on: the documents ``x`` (``[S, C]``), ``w``, ``cu``, ``b`` and ``dy``
    of the whole problem, and ``rows``, the global row of each row of the outputs ``y`` / ``dx`` (``None``: the
    outputs are in the problem's own row order, as ``[S, C]`` or ``[1, S, C]``)."""

    return types.SimpleNamespace(x=x, w=w, cu=cu, b=b, dy=dy, rows=rows)


def _shaped(name: str, value: torch.Tensor, p: types.SimpleNamespace, like: torch.Tensor) -> torch.Tensor:
    """An oracle or naive value of output ``name`` in the rows and the shape of the kernel's output ``like``."""

    if name in ("y", "dx") and p.rows is not None:
        value = value[p.rows]
    return value.reshape(like.shape)


def _output(name: str) -> str:
    """The output an output name stands for: ``dx`` of ``dx[1]`` (the second tensor of a multi-tensor call)."""

    return name.split("[", 1)[0]


def judge_draws(
    launch: Callable[[int, int], Launched],
    *,
    seed: int,
    label: str = "",
    draws: int = DRAWS,
) -> list[str]:
    """Judge one case over ``draws`` inputs and return every failure (empty when every output met the naive tier).

    For each draw ``j``, :func:`~tests.cake.test_helpers.cake_naive.seed_all` seeds the random state with
    :func:`~tests.cake.test_helpers.cake_naive.draw_seed` of ``seed`` and ``j``; ``launch(j, s)`` (``s`` that seed)
    makes the draw's inputs from ``s``, launches the kernel and returns ``(outputs, problem)``: each output by name
    (``y``, ``dx``, ``dweight``, ``dbias``), or a sequence of the values of several launches on the draw, and the
    :func:`problem` they are judged on -- or a list of such pairs, one per tensor of a multi-tensor call, each
    output name followed by the tensor's index (``y[0]``, ``dweight[2]``).  Every value is held to its contract
    against :func:`oracle` and judged against :func:`naive_bf16` (a second launch on a draw keeps the larger of the
    errors).  The random state is
    restored after the last draw, and the verdicts are recorded on the running test, each output named
    ``<output>@<label>`` when ``label`` is not empty."""

    judge = TierJudge(draws)
    failures: list[str] = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(draws):
            s = draw_seed(seed, j)
            seed_all(s)
            found = launch(j, s)
            for outputs, p in found if isinstance(found, list) else (found,):
                if not outputs:
                    continue  # nothing to judge (a rank without tokens returns no y / dx)
                names = tuple(dict.fromkeys(_output(name) for name in outputs))
                want = oracle(p.x, p.w, p.cu, p.b, p.dy, outputs=names)
                tiers = {tier: naive_bf16(p.x, p.w, p.cu, p.b, p.dy, outputs=names) for tier in TIERS}
                for name, values in outputs.items():
                    base = _output(name)
                    for actual in (values,) if isinstance(values, torch.Tensor) else tuple(values):
                        expected = _shaped(base, want[base], p, actual)
                        if message := contract(name, actual, expected):
                            failures.append(f"draw {j}: {message}")
                            continue
                        naive = {tier: _shaped(base, tiers[tier][base], p, actual) for tier in TIERS}
                        judge.add(name, j, actual, expected, naive)
                del outputs, p, want, tiers
            del found
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    failures += judge.failures()
    if _NODE is not None:
        for verdict in judge.verdicts().values():
            if label:
                verdict = {**verdict, "output": f"{verdict['output']}@{label}"}
            _NODE.user_properties.append(("naive_tier", verdict))
    return [f"{label}: {message}" if label else message for message in failures]


def assert_judged(
    launch: Callable[[int, int], Launched],
    *,
    seed: int,
    label: str = "",
    draws: int = DRAWS,
) -> None:
    """:func:`judge_draws`, failing the test with every failure."""

    failures = judge_draws(launch, seed=seed, label=label, draws=draws)
    assert not failures, "\n".join(failures)
