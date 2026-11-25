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

The naive tier of the KDA training accuracy tests (``test_cake_kda_*_naive.py``), for the judge of
``tests/cake_naive.py``: PyTorch BF16, and the judge of the family's accuracy checks (:func:`judge_arms`) with its
second reference, FLA (``cake_kda_fla.py``).

:func:`naive_with_grads` is the FP64 token recurrence of ``cake_kda_det_train_reference.py``
(:func:`kda_exact_with_grads`, :func:`kda_exact_raw_with_grads` and their in-kernel Q/K L2 norm forms in
``cake_kda_det_train_reference_l2n.py``) and its autograd on the same inputs, with every op in BF16:

- an input stored no wider than BF16 (``q``, ``k``, ``v``, ``beta``, ``dout``, the raw gate) is read as BF16; one
  stored wider (the FP32 activated gate ``g``, ``A_log``, ``dt_bias``) keeps its storage dtype, so an elementwise op
  of it runs in FP32 by torch's type promotion, as in BF16 user code that reads the FP32 tensor;
- the products ``k^T S`` and ``q^T S`` take BF16 operands (the state is cast to BF16 where a product takes it), and
  the outer product ``(beta k) (x) (v - k^T S)`` is a product of BF16 operands;
- with the in-kernel Q/K L2 norm the raw ``q`` / ``k`` rows are normalized in BF16 (``x / sqrt(sum x^2 + 1e-6)``);
- every result (the output and every gradient) is rounded to BF16.

Per document and head, with the state ``S`` ``[K, V]`` starting from zero::

    S = S * exp(g_t)[:, None];  S = S + (beta_t k_t) (x) (v_t - k_t^T S);  o_t = scale q_t^T S

In the raw gate mode the gate is activated in the same graph: ``g = lb * sigmoid(exp(A_log)[h] * (raw +
dt_bias[h]))``.  The recurrence is checkpointed in 64-token blocks, as the reference is, so the backward of a few
thousand tokens fits in memory.

:func:`judge_arms` is the judge of every accuracy check of the family's tests: a case's arms (routes, paths, forms,
repeated launches) on each draw's operands (:func:`draw_case`, or :func:`kcp_row_inputs` for a KCP harness row), each
output against :func:`oracle_with_grads` beside :func:`naive_of` and beside FLA on the same operands
(``cake_kda_fla.fla_of``), its verdicts recorded and settled against the owner-accepted gaps
(``cake_kda_known_gaps.py``).  This is the KDA rule: an output passes when it meets the naive BF16 tier or meets FLA
under the same rule (the same norms, :data:`tests.cake_naive.NEAR_TIE`, the max norm draw by draw, the zero rule), and
fails only when it misses both.  When FLA does not run a draw (the operands are not on a GPU, or FLA does not support
the case) the output is judged against the naive tier only, and its verdict says why; when FLA runs it and the pinned
FLA reference cannot (``cake_kda_fla.FlaReferenceError``), the test fails.  There is no tolerance: an output that
misses fails, unless it misses both references and is listed in the owner-accepted misses within its caps
(``tests/cake_naive_accepted.json``), or it is listed in the gaps within its gap's scope and caps.
:func:`assert_bitwise` is the deterministic package's run-to-run contract; the non-deterministic package has no bit
contract, and no output of one package is compared with the other's.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .cake_kda_det_train_reference import LOWER_BOUND
from .cake_kda_fla import FlaSteps, fla_of

D = 128
L2NORM_EPS = 1e-6  # the in-kernel Q/K L2 norm: x / sqrt(sum x^2 + 1e-6)
NAIVE_DTYPE = torch.bfloat16
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")
# the operands with a token axis: a batched draw holds them with a leading batch axis of 1
BATCHED = ("q", "k", "v", "g", "beta", "dout")


def stored(x: torch.Tensor, dtype: torch.dtype = NAIVE_DTYPE) -> torch.Tensor:
    """``x`` as the naive tier in ``dtype`` reads it: cast to ``dtype`` when ``x`` is stored no wider, else as is."""
    return x.to(dtype) if x.dtype.itemsize <= dtype.itemsize else x


def _block(S0, q, k, v, g, beta, scale, dtype):
    # S0 [H, K, V]; q, k, v [T, H, D] in dtype; g [T, H, D] (FP32, or dtype); beta [T, H] in dtype
    St = S0
    outs = []
    for t in range(q.shape[0]):
        St = St * torch.exp(g[t])[:, :, None]
        kt = k[t]
        pred = torch.einsum("hk,hkv->hv", kt, St.to(dtype))
        St = St + torch.einsum("hk,hv->hkv", beta[t][:, None] * kt, v[t] - pred)
        outs.append(scale * torch.einsum("hk,hkv->hv", q[t], St.to(dtype)))
    return torch.stack(outs, 0), St


def naive_forward(q, k, v, g, beta, cu_seqlens, scale, block=64, dtype=NAIVE_DTYPE):
    """``out [S, H, D]`` of the per-document recurrence on ``q, k, v, g [S, H, D]``, ``beta [S, H]`` (the tensors as
    the naive tier reads them, see :func:`stored`) with every product in ``dtype``; the output is in ``dtype``."""
    S, H, K = q.shape
    V = v.shape[-1]
    pieces = []
    cu = [int(x) for x in cu_seqlens.tolist()]
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        St = torch.zeros(H, K, V, dtype=dtype, device=q.device)
        for b0 in range(s0, s1, block):
            b1 = min(b0 + block, s1)
            ob, St = checkpoint(
                _block,
                St,
                q[b0:b1],
                k[b0:b1],
                v[b0:b1],
                g[b0:b1],
                beta[b0:b1],
                scale,
                dtype,
                use_reentrant=False,
            )
            pieces.append(ob)
    if not pieces:
        return torch.zeros(S, H, V, dtype=dtype, device=q.device)
    return torch.cat(pieces, 0)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + L2NORM_EPS)


def naive_with_grads(
    q,
    k,
    v,
    g,
    beta,
    dout,
    cu_seqlens,
    scale,
    *,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    l2n: bool = False,
    dtype=NAIVE_DTYPE,
) -> dict[str, torch.Tensor]:
    """The naive tier of the reference's ``out`` and gradients, each rounded to ``dtype``: ``out``, ``dq``, ``dk``,
    ``dv``, ``dg``, ``dbeta`` and with ``A_log`` (the raw gate mode: ``g`` is the raw BF16 gate, ``A_log`` FP32
    ``[H]``, ``dt_bias`` FP32 ``[H * 128]`` or ``[H, 128]``, ``lower_bound`` the gate's lower bound) also ``dA_log``
    and ``ddt_bias``.  ``l2n``: ``q`` / ``k`` are the raw rows, normalized inside the graph."""
    raw = A_log is not None
    srcs = (q, k, v, g, beta) + ((A_log, dt_bias) if raw else ())
    leaves = [stored(x.detach(), dtype).clone().requires_grad_(True) for x in srcs]
    qn, kn, vn, gn, bn = leaves[:5]
    if l2n:
        qn, kn = _normalize(qn), _normalize(kn)
    if raw:
        a_log, bias = leaves[5], leaves[6]
        H = gn.shape[-2]
        gn = lower_bound * torch.sigmoid(
            torch.exp(a_log).reshape(H, 1) * (gn + bias.reshape(H, D))
        )
    o = naive_forward(qn, kn, vn, gn, bn, cu_seqlens, scale, dtype=dtype)
    grads = torch.autograd.grad(o, leaves, stored(dout.detach(), dtype).to(o.dtype))
    names = RAW_GRAD_NAMES if raw else GRAD_NAMES
    out = {"out": o.detach()}
    out.update(dict(zip(names, (x.detach() for x in grads), strict=True)))
    return {name: t.to(dtype) for name, t in out.items()}


def draw_case(
    lengths,
    H: int,
    gate: str,
    l2n: bool,
    seed: int,
    device="cuda",
    *,
    dt_bias_mean: float = 0.0,
    dt_bias_2d: bool = False,
    batched: bool = False,
) -> dict[str, torch.Tensor]:
    """The operands of one draw of a judged case, every tensor a function of ``seed``: the reference helper's
    :func:`make_inputs` (``gate="g"``: the activated FP32 gate) or :func:`make_raw_inputs` (``"raw"``: the BF16 raw
    gate, FP32 ``A_log`` / ``dt_bias``, the draw of ``dt_bias`` shifted by ``dt_bias_mean`` and ``[H, 128]`` with
    ``dt_bias_2d``), and with ``l2n`` raw BF16 ``q`` / ``k`` rows of a few magnitudes per token (the caller's
    projections) in place of the normalized ones.  ``batched`` adds the leading batch axis of 1 to ``q, k, v, g, beta,
    dout`` (the judge's :func:`flat_inputs` takes it off again)."""
    from .cake_kda_det_train_reference import make_inputs, make_raw_inputs

    if gate == "g":
        if dt_bias_mean or dt_bias_2d:
            raise ValueError(
                "dt_bias_mean / dt_bias_2d are operands of the raw gate mode"
            )
        inputs = make_inputs(lengths, H, device, seed=seed)
    elif gate == "raw":
        inputs = make_raw_inputs(
            lengths,
            H,
            device,
            seed=seed,
            dt_bias_mean=dt_bias_mean,
            dt_bias_2d=dt_bias_2d,
        )
    else:
        raise ValueError(f"gate must be 'g' or 'raw', got {gate!r}")
    if l2n:
        S = int(sum(lengths))
        gen = torch.Generator().manual_seed(int(seed) + 7817)
        mag = torch.exp(torch.randn(S, H, 1, generator=gen) * 1.5)
        inputs["q"] = (
            (torch.randn(S, H, D, generator=gen) * mag).to(torch.bfloat16).to(device)
        )
        inputs["k"] = (
            (torch.randn(S, H, D, generator=gen) * mag.flip(0))
            .to(torch.bfloat16)
            .to(device)
        )
    if batched:
        for name in BATCHED:
            inputs[name] = inputs[name].unsqueeze(0)
    return inputs


def flat_inputs(inputs: dict) -> dict[str, torch.Tensor]:
    """``inputs`` without the leading batch axis of 1 of a batched draw (:func:`draw_case` ``batched``), as the
    reference and the naive tier read them."""
    if inputs["q"].dim() != 4:
        return inputs
    if inputs["q"].shape[0] != 1:
        raise ValueError(
            f"a batched draw has a batch axis of 1, got {tuple(inputs['q'].shape)}"
        )
    return {n: t.squeeze(0) if n in BATCHED else t for n, t in inputs.items()}


def kcp_row_draw(row, seed: int):
    """The KCP harness row ``row`` (``cake_kda_kcp_shapes.Row``) whose operands are those of draw seed ``seed``:
    ``cake_kda_kcp_inputs`` seeds a row's operands by its name, so the drawn row carries ``row.name`` and the seed in
    its name.  Partition with the original row (an explicit ragged table is keyed by the original name)."""
    import dataclasses

    return dataclasses.replace(row, name=f"{row.name}@{int(seed)}")


def kcp_row_inputs(row, seed: int, device) -> dict[str, torch.Tensor]:
    """The operands of draw seed ``seed`` of the KCP harness row ``row`` in :func:`draw_case`'s form (the whole pack
    in global order: ``q, k, v, dout`` BF16, ``g`` FP32 activated, ``beta`` BF16, ``cu_seqlens`` int32 on
    ``device``), for :func:`judge_arms`."""
    from .cake_kda_kcp_inputs import global_operands

    return global_operands(kcp_row_draw(row, seed), int(row.h), device)


def oracle_with_grads(
    inputs: dict, scale: float, lower_bound: float, l2n: bool
) -> dict[str, torch.Tensor]:
    """The FP64 reference of :func:`draw_case`'s operands (``out`` and every gradient by name), from the reference
    helpers unchanged."""
    from .cake_kda_det_train_reference import (
        kda_exact_raw_with_grads,
        kda_exact_with_grads,
    )
    from .cake_kda_det_train_reference_l2n import (
        kda_exact_l2n_with_grads,
        kda_exact_raw_l2n_with_grads,
    )

    a = tuple(inputs[n] for n in ("q", "k", "v", "g", "beta"))
    if "A_log" in inputs:
        fn = kda_exact_raw_l2n_with_grads if l2n else kda_exact_raw_with_grads
        o, grads = fn(
            *a,
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["dout"],
            inputs["cu_seqlens"],
            scale,
            lower_bound,
        )
    else:
        fn = kda_exact_l2n_with_grads if l2n else kda_exact_with_grads
        o, grads = fn(*a, inputs["dout"], inputs["cu_seqlens"], scale)
    return {"out": o, **grads}


def naive_of(
    inputs: dict, scale: float, lower_bound: float, l2n: bool
) -> dict[str, torch.Tensor]:
    """:func:`naive_with_grads` of :func:`draw_case`'s operands, keyed as :func:`oracle_with_grads`."""
    raw = {}
    if "A_log" in inputs:
        raw = {
            "A_log": inputs["A_log"],
            "dt_bias": inputs["dt_bias"],
            "lower_bound": lower_bound,
        }
    return naive_with_grads(
        *(inputs[n] for n in ("q", "k", "v", "g", "beta", "dout", "cu_seqlens")),
        scale,
        l2n=l2n,
        **raw,
    )


def cake_step(
    entry, inputs: dict, lower_bound: float, l2n: bool, *, ckpt: bool = False
) -> dict[str, torch.Tensor]:
    """``out`` and the gradients of every leaf (``q, k, v, g, beta`` and in the raw gate mode ``A_log``,
    ``dt_bias``) of one training step through the autograd entry ``entry`` (``cake_kda_det`` or
    ``cake_kda_nondet``), keyed as :func:`oracle_with_grads`; ``ckpt``: under ``torch.utils.checkpoint``."""
    raw = "A_log" in inputs
    names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw else ())
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
    cu = inputs["cu_seqlens"]

    def fn(*a):
        kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
        if raw:
            kw.update(
                use_gate_in_kernel=True,
                A_log=a[5],
                dt_bias=a[6],
                lower_bound=lower_bound,
            )
        return entry(*a[:5], cu, **kw)

    o = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
    grads = torch.autograd.grad(o, leaves, inputs["dout"])
    out = {"out": o.detach()}
    out.update(dict(zip(RAW_GRAD_NAMES if raw else GRAD_NAMES, grads, strict=True)))
    return out


def lengths_of(inputs: dict) -> tuple[int, ...]:
    """The document lengths of a draw's ``cu_seqlens``."""
    cu = [int(x) for x in inputs["cu_seqlens"].tolist()]
    return tuple(b - a for a, b in zip(cu, cu[1:]))


def _like(t: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    # the reference's / the tier's tensor in the shape of the kernel's result (a batched [1, S, H, 128] result, a
    # [H, 128] or [H * 128] ddt_bias): a reshape of the same elements, else unchanged (the contract reports the shape)
    if t.shape != actual.shape and t.numel() == actual.numel():
        return t.reshape(actual.shape)
    return t


def assert_bitwise(a: dict, b: dict, names=None, tag=None) -> None:
    """The deterministic package's run-to-run contract: ``a[name]`` and ``b[name]`` hold the same bytes for every
    ``name`` (default: every key of ``a``).  Never for the non-deterministic package, never across the packages."""
    names = tuple(a) if names is None else tuple(names)
    differ = [n for n in names if not torch.equal(a[n], b[n])]
    assert not differ, (tag, f"not bitwise equal: {differ}")


def judge_arms(
    record_property,
    draw,
    *,
    seed: int,
    scale: float,
    l2n: bool = False,
    lower_bound: float = LOWER_BOUND,
    draws: int | None = None,
):
    """Judge every arm of one case against the FP64 reference beside the naive BF16 tier and FLA over the judge's draws
    (``tests/cake_naive.py``; the KDA rule: an output passes when it meets the tier or FLA), record the verdicts and
    the FLA steps (``cake_kda_fla.FlaSteps``) and settle the verdicts against the owner-accepted gaps
    (``cake_kda_known_gaps.settle``).

    ``draw(j, s)`` makes draw ``j``'s operands from the seed ``s`` (:func:`draw_case` or :func:`kcp_row_inputs`:
    ``q, k, v, g, beta, dout, cu_seqlens`` and in the raw gate mode ``A_log, dt_bias``; batched or not), launches every
    arm of the case on them, runs the draw's own checks (the deterministic package's run-to-run bits on the first draw,
    launch records, plan fields) and returns ``(arms, inputs)``: ``arms`` maps an arm's name to its results, a dict
    ``{output: tensor}`` with the names of :func:`oracle_with_grads` (``out``, ``dq``, ``dk``, ``dv``, ``dg``,
    ``dbeta`` and in the raw gate mode ``dA_log``, ``ddt_bias``; any subset), or a list of such dicts (launches of the
    arm repeated on the draw: each is judged and the judge keeps the worst).  An output of the arm ``""`` is judged as
    ``name``, of another arm as ``"<arm>.<name>"``.  The reference (:func:`oracle_with_grads`), the tier
    (:func:`naive_of`) and FLA (``cake_kda_fla.fla_of``: one forward and backward, the reference of a checkpointed arm
    too) are formed once per draw from ``inputs`` and read in each result's shape.  ``l2n``: ``q`` /
    ``k`` are the raw rows the kernels normalize; ``scale`` the kernels' scale; ``lower_bound`` the raw gate's.

    Every output is held to its contract on every draw; the case's document lengths (of the first draw's
    ``cu_seqlens``) are the scope of the gaps.  Returns the judge."""
    import random

    from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed, seed_all

    from .cake_kda_known_gaps import settle

    judge = TierJudge(DRAWS if draws is None else draws)
    failures: list[str] = []
    lengths = None
    steps = FlaSteps()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(judge.draws):
            s = draw_seed(seed, j)
            seed_all(s)
            arms, inputs = draw(j, s)
            flat = flat_inputs(inputs)
            if lengths is None:
                lengths = lengths_of(flat)
            ref = oracle_with_grads(flat, scale, lower_bound, l2n)
            tier = naive_of(flat, scale, lower_bound, l2n)
            fla = steps.add(fla_of(flat, scale, lower_bound, l2n))
            for arm, results in arms.items():
                for res in (
                    results if isinstance(results, (list, tuple)) else (results,)
                ):
                    for name, actual in res.items():
                        key = f"{arm}.{name}" if arm else name
                        want = _like(ref[name], actual)
                        if message := contract(key, actual, want):
                            failures.append(f"draw {j}: {message}")
                            if actual.shape != want.shape:
                                continue
                        judge.add(
                            key,
                            j,
                            actual,
                            want,
                            {"bf16": _like(tier[name], actual)},
                            fla=fla.value(name, actual),
                        )
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    failures += judge.failures()
    judge.record(record_property)
    steps.record(record_property)
    settle(judge.verdicts(), failures, lengths=lengths)
    return judge


def judge_single_rank(
    record_property,
    entry,
    case: tuple,
    seed: int,
    device,
    scale: float,
    lower_bound: float,
    *,
    det: bool,
    dt_bias_2d: bool = False,
):
    """Judge one single-rank case ``(lengths, H, gate, l2n, ckpt)`` of the autograd entry ``entry`` with
    :func:`judge_arms`: every output of :func:`cake_step`, and with ``ckpt`` the same step under
    ``torch.utils.checkpoint`` (arm ``ckpt``: outputs ``ckpt.<name>``).  ``det``: on the first draw the step runs twice
    and every output must repeat bit for bit (the deterministic package's run-to-run contract).  ``dt_bias_2d``: the
    raw gate's ``dt_bias`` is drawn ``[H, 128]`` (:func:`draw_case`).  Returns the judge."""
    lengths, H, gate, l2n, ckpt = case

    def draw(j, s):
        inputs = draw_case(lengths, H, gate, l2n, s, device, dt_bias_2d=dt_bias_2d)
        got = cake_step(entry, inputs, lower_bound, l2n)
        if det and j == 0:
            assert_bitwise(
                got, cake_step(entry, inputs, lower_bound, l2n), tag="run to run"
            )
        arms = {"": got}
        if ckpt:
            arms["ckpt"] = cake_step(entry, inputs, lower_bound, l2n, ckpt=True)
        return arms, inputs

    return judge_arms(
        record_property, draw, seed=seed, scale=scale, l2n=l2n, lower_bound=lower_bound
    )


__all__ = [
    "BATCHED",
    "GRAD_NAMES",
    "L2NORM_EPS",
    "NAIVE_DTYPE",
    "RAW_GRAD_NAMES",
    "assert_bitwise",
    "cake_step",
    "draw_case",
    "fla_of",
    "flat_inputs",
    "judge_arms",
    "judge_single_rank",
    "kcp_row_draw",
    "kcp_row_inputs",
    "lengths_of",
    "naive_forward",
    "naive_of",
    "naive_with_grads",
    "oracle_with_grads",
    "stored",
]
