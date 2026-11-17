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

The CP = 1 local path of the NON-DETERMINISTIC Cake packed GDN training package
``cake_pplx._backends.cake_gdn_nondet_train_kcp`` (its public entries with a one-rank context).  Accuracy has no
tolerance: each floating output is judged by ``tests/cake_naive.py``, its error against an FP64 oracle beside the
error of a naive PyTorch BF16 implementation against the same oracle, over 32 random draws per case (``TIER`` lines in
the terminal summary).  The oracle is the exact FP64 reference of
``tests/experimental/test_helpers/cake_gdn_det_train_kcp_reference.py`` (imported read-only): ``reference_pack``
(its naive tier ``naive_pack``) and, with the in-kernel Q / K L2 norm, ``gdn_exact_l2n_with_grads`` (the raw rows
normalized in FP64; its naive tier ``gdn_naive_l2n_with_grads``); a region of three layers or with a later saver
chains them (``_form_oracle``).  This package promises no bit relation between two of its results, so every form of
a case (strided / contiguous, explicit / autograd, saved / replay, kept / replayed tapes, two caller streams, fused /
split walks, full / light, eager / graph steps, torch's deterministic algorithms on, ``"light_akk"`` / ``"light"``) is
judged against the oracle on its own (output ``"<arm>.<name>"``); an arm launched more than once on one draw is
judged by its worst launch.

Covered here: the six outputs ``out, dq, dk, dv, dbeta, dg`` on ragged packs of the normal gate regime (natural-log
gates down to the -42 class) and of the adversarial weak-erasure regimes, the six head pairs (16, 48), (8, 24),
(4, 12), (2, 6), (1, 3), (4, 4), the degenerate one-token pack and the zero-token pack (its contract only: its one
input does not change with the seed), the packed QKV strided view and the contiguous input, plain ``split`` views of a
packed QKV row, the packed views and contiguous operands with their hooks and ``retain_grad`` and the gradient layout
of each form (issue #8367; a CUDA-graph replay alternating the packed and plain views keeps each its own layout), the
explicit forward / backward entries and autograd (output order ``dq, dk, dv, dbeta, dg``, saved context and replay),
two caller streams with cached workspaces, the fused output walk and tape-and-output walk and the split forward on the
same build (test knobs ``fused_o`` / ``fused_o_tape`` 1 / 0), ``save_mode="full"`` without ``save_transfer`` at
``P = 2`` and the light form under NaN-filled allocations, ``save_mode="light_akk"`` (the light replay reads the
forward's ``Akk`` tape: ``fwd_prep_replay_l2n``) and the light form under NaN-filled allocations (with its launch
census and saved bytes, at the plan's ``P`` and ``P = 2``) and in ``graph="replay"`` steps, checkpointed
``graph="replay"`` steps, and the forward and backward under ``torch.use_deterministic_algorithms(True)`` (strict and
``warn_only``: no alert, no warning).  The multi-rank rows and the
runtime contracts live in ``test_cake_gdn_nondet_train_kcp_dist.py``; the CPU plan, table, byte-query and fake-backend tests in
``test_cake_gdn_nondet_train_kcp.py``.  Every test needs a CUDA device the package supports and the rendered ``build.json``
(skipped otherwise).  The suite imports the package through ``tests/cake/test_helpers/cake_package.py`` unconditionally (a
broken import fails, never skips) and nothing else from cake_pplx; the tests of the facade
``cake_pplx.cake_gdn_nondet_train_kcp`` live in ``test_cake_gdn_nondet_train_kcp_api.py``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pytest
import torch

from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed
from tests.experimental.test_helpers.cake_gdn_det_train_kcp_reference import (
    BLOCK,
    D,
    GRAD_NAMES,
    SEED,
    STORAGE_OF,
    TENSORS,
    gdn_exact_forward,
    gdn_exact_l2n_with_grads,
    gdn_naive_forward,
    gdn_naive_l2n_with_grads,
    head_params,
    make_inputs,
    naive_pack,
    reference_pack,
)


def _pkg(sub: str | None = None):
    """The package (or one of its modules) through the test-helper registry: a stale path fails here, never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_nondet_train_kcp", sub)


# the package root: the public entries and types (the names of the facade, which forwards to them)
gdn_pkg = _pkg()

OPERANDS = ("q", "k", "v", "g", "beta")
EXPLICIT_GRAD_ORDER = (
    "dq",
    "dk",
    "dv",
    "dbeta",
    "dg",
)  # the backward entries' output order
STORAGE_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}

# The pinned-decay GEN row: the normal regime's per-head decay rate e^A is raised to at least PINNED_DECAY_MIN, so every
# full 64-token chunk's gate sum |G_L| exceeds 100: the class where e^{G_L} overflows FP32 unless the kernel rescales
# within the chunk.
PINNED_DECAY_MIN = 4.0
PINNED_CHUNK_GATE_SUM_MIN = 100.0

# id, document lengths (none a multiple of 64 unless the case says so), (H_qk, HV), operand regime.  Each row is judged
# over 32 draws of the FP64 oracle and the naive tier, so a pack holds at most about 1000 tokens and 50k token x
# value-head rows; every row keeps documents of more than one 64-token chunk with a partial last chunk.
SHAPES = (
    ("c1-2x6-ragged", (100, 300, 77, 400, 129), (2, 6), "normal"),
    ("c1-2x6-gen-decay-pinned", (100, 300, 77, 400, 129), (2, 6), "normal+decay4"),
    ("c2-4x12-one-doc", (1000,), (4, 12), "normal"),
    ("c3-1x3-tiny-docs", (448, 5, 500, 1), (1, 3), "normal"),
    ("c4-8x24-chunk-interior", (500, 500), (8, 24), "normal"),
    ("c5-4x4-63-64-65", (63 + 64 + 65 + 500,), (4, 4), "normal"),
    ("c8s-16x48", (512 + 37, 400), (16, 48), "normal"),
    ("a1s-2x6-slow-decay", (1001,), (2, 6), "A1"),
    ("a2s-2x6-anisotropic", (1013,), (2, 6), "A2"),
    ("one-token", (1,), (2, 6), "normal"),
    ("zero-tokens", (0,), (2, 6), "normal"),
)


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(cu: list[int], heads: tuple[int, int], regime: str, seed: int) -> dict:
    """The row's operands of seed ``seed``; ``"normal+decay4"`` is the normal regime with every value head's decay
    rate e^A raised to at least PINNED_DECAY_MIN (g = -max(e^A, 4) * softplus(x), i.e. g scaled by max(1, 4 / e^A) per
    head)."""
    if regime != "normal+decay4":
        return make_inputs(cu, *heads, regime=regime, seed=seed)
    inputs = make_inputs(cu, *heads, regime="normal", seed=seed)
    rate = torch.exp(
        head_params(*heads, seed)["a_log"]
    )  # [HV] decay rates e^A of the normal regime
    inputs["g"] = (
        (inputs["g"] * torch.clamp(PINNED_DECAY_MIN / rate, min=1.0))
        .float()
        .contiguous()
    )
    for d in range(len(cu) - 1):
        for c0 in range(cu[d], cu[d + 1] - BLOCK + 1, BLOCK):
            gate_sum = inputs["g"][0, c0 : c0 + BLOCK].sum(
                0
            )  # [HV] per full in-document chunk
            assert bool((gate_sum.abs() > PINNED_CHUNK_GATE_SUM_MIN).all()), (
                d,
                c0,
                gate_sum,
            )
    return inputs


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (see ``test_cake_gdn_nondet_train_kcp.py``), so this module runs with at most 8
    threads and oneDNN off, as that suite does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _rendered() -> bool:
    """The package's generated ``build.json`` lists libraries (the export render delivered the programs)."""
    path = pathlib.Path(_pkg().__file__).resolve().parent / "build.json"
    if not path.is_file():
        return False
    return bool(json.loads(path.read_text()).get("libraries"))


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if not _rendered():
        pytest.skip("the package's build.json is not rendered (no kernels to run)")
    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    if _pkg("cake_backend").arch_for_capability(capability) is None:
        pytest.skip("the package does not support this device")
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _local_context(cu: list[int], device: torch.device):
    """The one-rank context of the local path (the public entry names; group None = single process)."""
    part = gdn_pkg.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, layout="contiguous"
    )
    return gdn_pkg.cake_gdn_nondet_train_kcp_build_context(part, None, 0, device)


def _run_fwd_bwd(inputs: dict, ctx, device: torch.device) -> dict:
    """Autograd forward + backward of the local path; the results keyed by tensor name (no batch axis)."""
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    out = gdn_pkg.cake_gdn_nondet_kcp(*leaves, ctx)
    grads = torch.autograd.grad(out, leaves, inputs["dout"].to(device))
    res = {"out": out.detach()[0]}
    res.update({n: gr.detach()[0] for n, gr in zip(GRAD_NAMES, grads, strict=True)})
    return res


def _check_dtypes(res: dict) -> None:
    """Each output in its storage dtype (``out`` / ``dq`` / ``dk`` / ``dv`` BF16, ``dg`` / ``dbeta`` FP32); its shape
    and finiteness are the judge's contract."""
    for n in TENSORS:
        assert res[n].dtype == STORAGE_DTYPE[STORAGE_OF[n]], (n, res[n].dtype)


def _pack(inputs: dict, device: torch.device, *, norm: bool, exact: bool) -> dict:
    """``out`` and the five gradients of one GDN layer on ``inputs`` (``q, k, v, g, beta, dout, cu_seqlens``, with or
    without the batch axis) on ``device``, without the batch axis: the FP64 oracle (``exact``) or its naive BF16 tier.
    Norm off: ``reference_pack`` / ``naive_pack``.  Norm on (``use_qk_l2norm_in_kernel=True``):
    ``gdn_exact_l2n_with_grads`` (the raw q / k rows normalized in FP64, no rounding of the normalized rows; dq / dk
    the gradients of the raw rows) / ``gdn_naive_l2n_with_grads`` (the norm in BF16)."""
    if not norm:
        res = reference_pack(inputs, device)[0] if exact else naive_pack(inputs, device)
    else:
        fn = gdn_exact_l2n_with_grads if exact else gdn_naive_l2n_with_grads
        ops = [inputs[n].to(device) for n in (*OPERANDS, "dout")]
        res = fn(*ops, inputs["cu_seqlens"])
    return {n: res[n].to(device) for n in TENSORS}


def _oracle(
    inputs: dict, device: torch.device, *, norm: bool = False
) -> tuple[dict, dict]:
    """The FP64 oracle and the naive BF16 tier of ``out`` and the five gradients on ``inputs`` (no batch axis), with
    the in-kernel Q / K L2 norm off or on (:func:`_pack`)."""
    return (
        _pack(inputs, device, norm=norm, exact=True),
        _pack(inputs, device, norm=norm, exact=False),
    )


def _form_oracle(
    inputs: dict,
    device: torch.device,
    *,
    layers: int = 1,
    w: torch.Tensor | None = None,
    norm: bool = False,
) -> tuple[dict, dict]:
    """The FP64 oracle and the naive BF16 tier of a region of ``layers`` GDN layers in sequence (layer i's output is
    layer i + 1's ``v``; ``q, k, g, beta`` feed every layer) and, with the later saver ``w`` (``[HV, 128]``), the
    region output ``x * w``: the last layer's cotangent is ``dout * w`` and ``dw`` is the sum over the tokens of
    ``dout * x``.  The oracle chains the FP64 layers on their FP64 outputs, the naive tier the BF16 layers on their
    BF16 outputs, each with its own cotangents; ``dq, dk, dg, dbeta`` sum over the layers (the naive tier in FP32,
    then rounded to BF16), ``dv`` is the first layer's.  Keys ``out``, the five gradients and (with ``w``) ``dw``,
    on ``device``, without the batch axis."""
    if norm and layers != 1:
        raise ValueError("the in-kernel norm forms here are one layer")
    cu = inputs["cu_seqlens"]
    tiers = []
    for exact in (True, False):
        xs = [inputs["v"]]
        for _ in range(layers - 1):
            fwd = gdn_exact_forward if exact else gdn_naive_forward
            ops = [inputs[n].to(device) for n in ("q", "k")]
            ops += [xs[-1].to(device)] + [inputs[n].to(device) for n in ("g", "beta")]
            xs.append(fwd(*ops, cu)[0])
        # the naive tier reads dout and w as stored (BF16 products, torch's promotion)
        dout = inputs["dout"][0].to(device)
        wt = None if w is None else w.detach()
        if exact:
            dout, wt = dout.double(), None if wt is None else wt.double()
        c = dout if wt is None else dout * wt
        acc: dict = {}
        top = None
        for i in reversed(range(layers)):
            r = _pack({**inputs, "v": xs[i], "dout": c}, device, norm=norm, exact=exact)
            if top is None:
                top = r["out"]
            for n in ("dq", "dk", "dg", "dbeta"):
                t = r[n] if exact else r[n].float()
                acc[n] = t if n not in acc else acc[n] + t
            c = r["dv"]
        res = {n: t if exact else t.to(torch.bfloat16) for n, t in acc.items()}
        res["dv"] = c
        res["out"] = top if wt is None else top * wt
        if wt is not None:
            res["dw"] = (dout * top).sum(0)
        tiers.append(res)
    return tiers[0], tiers[1]


def _named(res: dict, *arms: str, names=TENSORS) -> dict:
    """The outputs ``names`` of ``res`` under the judged names of ``arms``: ``"<arm>.<name>"`` (``name`` alone when
    no arm is given)."""
    return {f"{a}.{n}" if a else n: res[n] for a in (arms or ("",)) for n in names}


def _judge(draw, *, seed: int) -> tuple[TierJudge, list[str]]:
    """Judge a case over ``DRAWS`` draws with the naive BF16 tier (``tests/cake_naive.py``).  ``draw(j, s)`` makes
    draw ``j``'s inputs from the seed ``s`` (``make_inputs`` draws from explicit generators only, so the global random
    state is not touched), launches the arms and returns ``(runs, oracle, naive)``: ``runs`` a list of output dicts by
    judged name (see :func:`_named`), ``oracle`` and ``naive`` the FP64 oracle and the naive tier of every judged name.
    A name launched more than once on a draw is judged by its worst launch (``TierJudge.add``).  Every output is held
    to its contract first.  Returns the judge and every failure."""
    judge, failures = TierJudge(), []
    for j in range(DRAWS):
        runs, oracle, naive = draw(j, draw_seed(seed, j))
        for got in runs:
            for name, actual in got.items():
                if (m := contract(name, actual, oracle[name])) is not None:
                    failures.append(f"draw {j}: {m}")
                    continue
                judge.add(name, j, actual, oracle[name], {"bf16": naive[name]})
    return judge, failures + judge.failures()


@pytest.mark.parametrize(
    ("lengths", "heads", "regime"),
    [s[1:] for s in SHAPES],
    ids=[s[0] for s in SHAPES],
)
def test_local_path_meets_the_naive_tier(
    device, record_property, lengths, heads, regime
) -> None:
    """``out`` and the five gradients of the autograd path, judged against the FP64 oracle with the naive BF16 tier
    over 32 draws, in every regime.  The zero-token pack has one input whatever the seed: it is held to the contract
    only."""
    cu = _cu(lengths)
    ctx = _local_context(cu, device)
    if cu[-1] == 0:
        inputs = _inputs(cu, heads, regime, SEED)
        res = _run_fwd_bwd(inputs, ctx, device)
        _check_dtypes(res)
        ref, _ = reference_pack(inputs, device)
        failures = [
            m for n in TENSORS if (m := contract(n, res[n], ref[n])) is not None
        ]
        assert not failures, failures
        return

    def draw(j, s):
        inputs = _inputs(cu, heads, regime, s)
        res = _run_fwd_bwd(inputs, ctx, device)
        _check_dtypes(res)
        ref, naive = _oracle(inputs, device)
        return [_named(res)], _named(ref), _named(naive)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 700), (2, 6)), ((950, 37), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_packed_qkv_strided_view_and_the_contiguous_input_meet_the_naive_tier(
    device, record_property, lengths, heads
) -> None:
    """The packed-QKV strided views pass through (no copy); their results and the contiguous input's (arms
    ``strided`` / ``contiguous``) are each judged against the FP64 oracle with the naive BF16 tier."""
    H, HV = heads
    cu = _cu(lengths)
    T = cu[-1]
    ctx = _local_context(cu, device)

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        # the packed QKV row: q at head 0, k at head H, v at head 2H; token stride (2H + HV) * 128 elements
        packed = torch.empty(1, T, 2 * H + HV, D, dtype=torch.bfloat16, device=device)
        packed[:, :, :H] = inputs["q"].to(device)
        packed[:, :, H : 2 * H] = inputs["k"].to(device)
        packed[:, :, 2 * H :] = inputs["v"].to(device)
        views = {
            "q": packed[:, :, :H],
            "k": packed[:, :, H : 2 * H],
            "v": packed[:, :, 2 * H :],
        }
        for n, t in views.items():
            assert not t.is_contiguous(), n
            assert t.stride(1) == (2 * H + HV) * D, (n, t.stride())
            assert torch.equal(t.cpu(), inputs[n]), n
        contiguous = _run_fwd_bwd(inputs, ctx, device)
        strided = _run_fwd_bwd({**inputs, **views}, ctx, device)
        ref, naive = _oracle(inputs, device)
        arms = ("contiguous", "strided")
        return (
            [_named(contiguous, "contiguous"), _named(strided, "strided")],
            _named(ref, *arms),
            _named(naive, *arms),
        )

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


def _split_view_pass(form, row0, g0, b0, dout, ctx, heads, *, ckpt, **o) -> dict:
    """One autograd pass with q / k / v taken from the packed ``[1, T, (2 H_qk + HV) * 128]`` row ``row0``:
    ``"plain"`` = ``split`` + ``view`` head views, ``"packed"`` = ``cake_gdn_nondet_train_kcp_packed_qkv_views``, ``"contiguous"`` =
    separate contiguous leaves holding the same values.  A tensor hook (recording the gradient) and ``retain_grad``
    are set on every view (and on the split outputs) of the original forward.  Returns the results keyed by tensor
    name (no batch axis; ``dq`` / ``dk`` / ``dv`` cut from the row gradient), the row gradient, the hooks' records
    and the views."""
    pkg = _pkg()
    H, HV = heads
    T = row0.shape[1]
    sizes = (H * D, H * D, HV * D)
    row = row0.detach().clone().requires_grad_(True)
    g, beta = (x.detach().clone().requires_grad_(True) for x in (g0, b0))
    leaves = []
    if form == "contiguous":
        leaves = [
            p.detach()
            .clone(memory_format=torch.contiguous_format)
            .view(1, T, n, D)
            .requires_grad_(True)
            for p, n in zip(row0.split(sizes, dim=-1), (H, H, HV), strict=True)
        ]
    fired: dict = {}
    views: dict = {}

    def fn(row, g, beta, *leaves):
        parts = ()
        if form == "contiguous":
            q, k, v = leaves
        elif form == "plain":
            parts = row.split(sizes, dim=-1)
            q, k, v = (
                p.view(1, T, n, D) for p, n in zip(parts, (H, H, HV), strict=True)
            )
        else:
            q, k, v = pkg.cake_gdn_nondet_train_kcp_packed_qkv_views(row, H_qk=H, HV=HV)
        if not views:  # the original forward (a checkpoint recompute builds a graph no backward runs)
            views.update(q=q, k=k, v=v, **{f"part{i}": p for i, p in enumerate(parts)})
            for name, t in views.items():
                t.register_hook(
                    lambda gr, name=name: fired.setdefault(name, []).append(gr.clone())
                )
                t.retain_grad()
        return gdn_pkg.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, **o)

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, row, g, beta, *leaves, use_reentrant=False)
    else:
        out = fn(row, g, beta, *leaves)
    out.backward(dout)
    drow = (
        torch.cat([x.grad.reshape(1, T, -1) for x in leaves], -1)
        if leaves
        else row.grad
    )
    res = {"out": out.detach()[0], "dg": g.grad[0], "dbeta": beta.grad[0]}
    for n, (a, b), h in zip(
        ("dq", "dk", "dv"),
        ((0, H * D), (H * D, 2 * H * D), (2 * H * D, None)),
        (H, H, HV),
        strict=True,
    ):
        res[n] = drow[0, :, a:b].reshape(T, h, D)
    return {"res": res, "row": drow, "fired": fired, "views": views}


@pytest.mark.parametrize("ckpt", [False, True], ids=["direct", "checkpoint"])
@pytest.mark.parametrize("norm", [False, True], ids=["l2n-off", "l2n-on"])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 700), (2, 6)), ((512 + 37, 400), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_plain_split_views_keep_hooks_and_meet_the_naive_tier(
    device, record_property, lengths, heads, norm, ckpt
) -> None:
    """Issue #8367 on the device programs: plain ``split`` / ``view`` head views of one packed QKV row, the
    ``cake_gdn_nondet_train_kcp_packed_qkv_views`` views of it and contiguous operands (arms ``plain`` / ``packed`` /
    ``contiguous``) each give ``out``, ``dq | dk | dv`` (the row gradient), ``dg`` and ``dbeta``, judged against the
    FP64 oracle with the naive BF16 tier, with the in-kernel norm off and on (norm on: the oracle of the normalized
    rows), direct and checkpointed (bitwise equality between two results of this package is not part of its
    contract).  Exact and structural, on every draw: plain views keep torch's autograd path -- their tensor hooks
    and ``retain_grad`` (and those on the split outputs) fire once with the gradient the row receives in the same
    pass -- and only the packed views' split hands the packed dQKV row on without a concatenation."""
    kcp = _pkg("cake_kcp")
    H, HV = heads
    cu = _cu(lengths)
    T = cu[-1]
    ctx = _local_context(cu, device)
    o = {"use_qk_l2norm_in_kernel": norm}
    cols = {"q": (0, H * D), "k": (H * D, 2 * H * D), "v": (2 * H * D, None)}
    expect_split = {
        "contiguous": {"aliased": 0, "concatenated": 0},
        "plain": {"aliased": 0, "concatenated": 0},
        "packed": {"aliased": 1, "concatenated": 0},
    }

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        row0 = torch.cat(
            [inputs[n].to(device).reshape(1, T, -1) for n in ("q", "k", "v")], -1
        ).contiguous()
        g0, b0, dout = (inputs[n].to(device) for n in ("g", "beta", "dout"))
        runs = []
        for form, split in expect_split.items():
            before = dict(kcp.PACKED_SPLIT_STATS)
            got = _split_view_pass(form, row0, g0, b0, dout, ctx, heads, ckpt=ckpt, **o)
            delta = {n: kcp.PACKED_SPLIT_STATS[n] - before[n] for n in before}
            assert delta == split, (form, delta)
            # one packed dQKV row only for the split that hands it on; plain views get three contiguous gradients
            assert kcp.LAST_CALL["bwd"]["grad_layout"] == (
                "packed" if form == "packed" else "separate"
            ), form
            _check_dtypes(got["res"])
            runs.append(_named(got["res"], form))
            assert sorted(got["fired"]) == sorted(got["views"]), form
            for i, n in enumerate(("q", "k", "v")):
                a, b = cols[n]
                want = got["row"][..., a:b]
                assert len(got["fired"][n]) == 1, (form, n)
                assert torch.equal(got["fired"][n][0].reshape(want.shape), want), (
                    form,
                    n,
                )
                # contiguous: q / k / v are the leaves the row gradient is read from, so
                # their .grad is that gradient by construction (only the hook is checked)
                if form != "contiguous":
                    vg = got["views"][n].grad.reshape(want.shape)
                    assert torch.equal(vg, want), (form, n)
                if form == "plain":
                    part = got["fired"][f"part{i}"]
                    assert len(part) == 1 and torch.equal(part[0], want), (form, n)
                    assert torch.equal(got["views"][f"part{i}"].grad, want), (form, n)
        ref, naive = _oracle(inputs, device, norm=norm)
        return runs, _named(ref, *expect_split), _named(naive, *expect_split)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


def test_explicit_entries_and_autograd_meet_the_naive_tier_and_keep_the_output_order(
    device, record_property
) -> None:
    """The explicit forward / backward entries (output order ``dq, dk, dv, dbeta, dg``, storage dtypes), the replay
    backward (``saved=None``) and the autograd path (arms ``explicit_saved`` / ``explicit_replay`` (its five
    gradients; its ``out`` is the explicit forward's) / ``autograd``) are each judged against the FP64 oracle with the
    naive BF16 tier."""
    cu = _cu((100, 640, 77))
    ctx = _local_context(cu, device)

    def draw(j, s):
        inputs = make_inputs(cu, 2, 6, seed=s)
        ops = [inputs[n].to(device) for n in OPERANDS]
        dout = inputs["dout"].to(device)
        out, saved = gdn_pkg.cake_gdn_nondet_train_kcp_forward(
            *ops, ctx, save_for_backward=True
        )
        grads = gdn_pkg.cake_gdn_nondet_train_kcp_backward(*ops, ctx, dout, saved=saved)
        assert len(grads) == len(EXPLICIT_GRAD_ORDER)
        auto = _run_fwd_bwd(inputs, ctx, device)
        for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True):
            assert t.dtype == STORAGE_DTYPE[STORAGE_OF[n]], (n, t.dtype)
        replay = gdn_pkg.cake_gdn_nondet_train_kcp_backward(*ops, ctx, dout, saved=None)
        explicit = {
            "out": out[0],
            **{n: t[0] for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True)},
        }
        replayed = {n: t[0] for n, t in zip(EXPLICIT_GRAD_ORDER, replay, strict=True)}
        ref, naive = _oracle(inputs, device)
        arms = ("autograd", "explicit_saved", "explicit_replay")
        return (
            [
                _named(auto, "autograd"),
                _named(explicit, "explicit_saved"),
                _named(replayed, "explicit_replay", names=GRAD_NAMES),
            ],
            _named(ref, *arms),
            _named(naive, *arms),
        )

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


@pytest.mark.parametrize("warn_only", [False, True], ids=("strict", "warn_only"))
def test_torch_deterministic_algorithms_run_this_package_and_meet_the_naive_tier(
    device, record_property, warn_only
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``; torch then also fills the
    memory ``torch.empty`` returns) the explicit forward, the saved and replay backwards and the autograd node run
    this package with no alert, no warning and no fallback, and their results (arms ``explicit_saved`` /
    ``explicit_replay`` / ``autograd``) are judged against the FP64 oracle with the naive BF16 tier.  The same arms
    with the setting off, on the same pack and seeds, are judged by
    ``test_explicit_entries_and_autograd_meet_the_naive_tier_and_keep_the_output_order``."""
    cu = _cu((100, 640, 77))
    ctx = _local_context(cu, device)

    def run(inputs: dict) -> list[dict]:
        ops = [inputs[n].to(device) for n in OPERANDS]
        dout = inputs["dout"].to(device)
        out, saved = gdn_pkg.cake_gdn_nondet_train_kcp_forward(
            *ops, ctx, save_for_backward=True
        )
        grads = gdn_pkg.cake_gdn_nondet_train_kcp_backward(*ops, ctx, dout, saved=saved)
        replay = gdn_pkg.cake_gdn_nondet_train_kcp_backward(*ops, ctx, dout, saved=None)
        auto = _run_fwd_bwd(inputs, ctx, device)
        torch.cuda.synchronize(device)
        explicit = {
            "out": out[0],
            **{n: t[0] for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True)},
        }
        replayed = {n: t[0] for n, t in zip(EXPLICIT_GRAD_ORDER, replay, strict=True)}
        _check_dtypes(auto)
        _check_dtypes(explicit)
        return [
            _named(auto, "autograd"),
            _named(explicit, "explicit_saved"),
            _named(replayed, "explicit_replay", names=GRAD_NAMES),
        ]

    run(make_inputs(cu, 2, 6))  # the setting off: builds the programs

    def draw(j, s):
        inputs = make_inputs(cu, 2, 6, seed=s)
        enabled = torch.are_deterministic_algorithms_enabled()
        was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                runs = run(inputs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
        ref, naive = _oracle(inputs, device)
        arms = ("autograd", "explicit_saved", "explicit_replay")
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


# --------------------------------------------------------------------------- issue #7074 (1): sync-free context preparation (one GPU)
ISSUE_CU = [
    0,
    131071,
    218602,
    261987,
]  # the issue's layout: 261,987 tokens in 3 documents, (H_qk, HV) = (2, 6)


def test_context_preparation_does_not_synchronize(device) -> None:
    """``cake_gdn_nondet_train_kcp_partition`` + ``cake_gdn_nondet_train_kcp_build_context`` + ``prepare`` of a new layout make no synchronizing CUDA call (the
    tables travel in pinned, asynchronous copies), and every device table is bitwise the table built on the CPU."""
    plan = _pkg("cake_kcp_plan")
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        part = gdn_pkg.cake_gdn_nondet_train_kcp_partition(
            torch.tensor(ISSUE_CU, dtype=torch.int64),
            1,
            layout="contiguous",
        )
        ctx = gdn_pkg.cake_gdn_nondet_train_kcp_build_context(part, None, 0, device)
        prepared = ctx.prepare(2, 6)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)
    ref = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=ctx.sm_count
    )
    ref_prepared = ref.prepare(2, 6)
    pairs = (
        [(f"tables.{k}", ctx.tables[k], ref.tables[k]) for k in ctx.tables]
        + [
            (f"halo.{k}", ctx.halo_tables[k], ref.halo_tables[k])
            for k in ctx.halo_tables
        ]
        + [
            ("frag_table", ctx.frag_table, ref.frag_table),
            ("cu_seqlens_local", ctx.cu_seqlens_local, ref.cu_seqlens_local),
        ]
        + [
            (f"prepared.{k}", prepared.device[k], ref_prepared.device[k])
            for k in prepared.device
        ]
    )
    assert len(pairs) > 20
    for name, dev_t, cpu_t in pairs:
        assert dev_t.device.type == "cuda", name
        assert dev_t.dtype == cpu_t.dtype and tuple(dev_t.shape) == tuple(
            cpu_t.shape
        ), name
        assert dev_t.data_ptr() % plan.UPLOAD_ALIGN_BYTES == 0, name
        assert torch.equal(dev_t.cpu(), cpu_t), name
    assert prepared.P == ref_prepared.P


# --------------------------------------------------------------------------- issue #7074 (4): the workspace bytes the backward binds (one GPU)


class _PeakWorkspace:
    """A ``CakeGDNNondetKcpWorkspace`` recording the most bytes it binds (every allocation / release path)."""

    def __new__(cls, kcp):
        class Tracked(kcp.CakeGDNNondetKcpWorkspace):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.peak = 0

            def _upd(self):
                self.peak = max(self.peak, self.bytes)

            def tape(self, name):
                t = super().tape(name)
                self._upd()
                return t

            def pool(self, name, n=None):
                t = super().pool(name, n)
                self._upd()
                return t

            def ensure(self, *a, **k):
                tb = super().ensure(*a, **k)
                self._upd()
                return tb

            def adopt_tape(self, name, t):
                r = super().adopt_tape(name, t)
                self._upd()
                return r

        return Tracked()


@pytest.mark.parametrize(
    ("cu", "heads"),
    [(ISSUE_CU, (2, 6)), ([0, 11000, 23000, 32768], (16, 48))],
    ids=["issue-2x6", "32k-16x48"],
)
def test_workspace_bytes_match_what_the_backward_binds(device, cu, heads) -> None:
    """``cake_gdn_nondet_train_kcp_workspace_layout`` is what a caller-held workspace binds (tolerance 0): the forward's peak (the fused
    forward's #1; with test knob ``fused_o=0`` the split forward's tape walk, ``forward_split``), and the backward's
    phase maximum of each tape form -- the kept-tape form (the ten light tapes are the recompute's hand-off tensors,
    the tapes the forward leaves re-adopted after #8) and the replay form (the ten light tapes are workspace)."""
    import gc

    from torch.utils.checkpoint import checkpoint

    kcp = _pkg("cake_kcp")
    gb = _pkg("cake_bytes")
    H, HV = heads
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    ctx.prepare(H, HV)
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    for form in (True, False):
        layout = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H, HV=HV, reuse_recompute_tapes=form
        )
        ws = _PeakWorkspace(kcp)

        def fn(*a, _ws=ws, _form=form):
            return gdn_pkg.cake_gdn_nondet_kcp(
                *a, ctx, workspace=_ws, reuse_recompute_tapes=_form
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["forward"], (form, ws.peak, layout["forward"])
        ws.peak = 0
        out.backward(dout)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["backward"], (form, ws.peak, layout["backward"])
        assert ws.bytes == layout["bwd.phase.end"], (
            form,
            ws.bytes,
            layout["bwd.phase.end"],
        )
        assert gdn_pkg.cake_gdn_nondet_train_kcp_workspace_bytes(
            ctx, H_qk=H, HV=HV, reuse_recompute_tapes=form
        ) == max(layout["forward"], layout["backward"])
        for lf in leaves:
            lf.grad = None
        del out, ws
        gc.collect()
        torch.cuda.empty_cache()
        # the split forward (fused_o = 0) binds the h / v_new tapes in its tape walk: forward_split
        ws = _PeakWorkspace(kcp)

        def split(*a, _ws=ws, _form=form):
            return gdn_pkg.cake_gdn_nondet_kcp(
                *a,
                ctx,
                workspace=_ws,
                reuse_recompute_tapes=_form,
                test_knobs={"fused_o": 0},
            )

        out = checkpoint(split, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["forward_split"], (
            form,
            ws.peak,
            layout["forward_split"],
        )
        out.backward(dout)
        torch.cuda.synchronize(device)
        for lf in leaves:
            lf.grad = None
        del out, ws
        gc.collect()
        torch.cuda.empty_cache()
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- issue #7074 (2): reuse_recompute_tapes="auto" (one GPU)


@pytest.mark.parametrize(
    "later_saver", [False, True], ids=["gdn-closes-the-region", "later-saver"]
)
def test_reuse_recompute_tapes_auto_follows_the_region_shape(
    device, record_property, later_saver
) -> None:
    """``"auto"`` hands the recompute's tapes to the backward when the GDN node is the region's last saver (frame
    proof ``dead``, kept form) and takes the replay form when a later operation of the region saves a tensor
    (``holders_remain``); ``True`` always hands off (``always``), ``False`` never (``off``).  The direct autograd
    result and every checkpointed form (arms ``direct`` / ``checkpoint_True`` / ``checkpoint_False`` /
    ``checkpoint_auto``) are each judged against the FP64 oracle of the region with the naive BF16 tier, the registry
    is empty afterwards, and the ``"auto"`` backward -- the recompute forward, the frame proof over the autograd
    graph, the take -- makes no synchronizing CUDA call."""
    kcp = _pkg("cake_kcp")
    H, HV = 2, 6
    cu = [0, 300, 700, 1000]
    ctx = _local_context(cu, device)
    w = None
    if later_saver:
        # the later saver: a projection of the flattened heads (its weight is part of the region, the same on every
        # draw)
        gen = torch.Generator(device="cpu").manual_seed(7074)
        w = (
            (torch.randn(HV * D, HV * D, generator=gen) * 0.02)
            .to(torch.bfloat16)
            .to(device)
        )
    verified = kcp.torch_release_verified()
    forms = (
        (True, "always", "kept"),
        (False, "off", "replay"),
        (
            "auto",
            ("holders_remain" if later_saver else "dead") if verified else None,
            "replay" if (later_saver or not verified) else "kept",
        ),
    )
    names = ("out", *OPERANDS)
    arms = ("direct", *(f"checkpoint_{form}" for form, _r, _s in forms))
    grad_of = {"q": "dq", "k": "dk", "v": "dv", "g": "dg", "beta": "dbeta"}

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
        dout = inputs["dout"].to(device)
        if w is not None:
            dout = dout.reshape(1, -1, HV * D)

        def region(ckpt, form, sync_error=False):
            from torch.utils.checkpoint import checkpoint

            def fn(*a):
                out = gdn_pkg.cake_gdn_nondet_kcp(
                    *a,
                    ctx,
                    **({} if form is None else {"reuse_recompute_tapes": form}),
                )
                if w is None:
                    return out
                return torch.nn.functional.linear(
                    out.reshape(out.shape[0], out.shape[1], -1), w
                )

            for lf in leaves:
                lf.grad = None
            out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
            torch.cuda.synchronize(device)
            if sync_error:
                torch.cuda.set_sync_debug_mode("error")
            try:
                out.backward(dout)
            finally:
                if sync_error:
                    torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize(device)
            return {
                "out": out.detach(),
                **{
                    n: lf.grad.detach().clone()
                    for n, lf in zip(OPERANDS, leaves, strict=True)
                },
            }

        runs = [_named(region(False, None), "direct", names=names)]
        for form, reason, source in forms:
            res = region(True, form, sync_error=(form == "auto"))
            runs.append(_named(res, f"checkpoint_{form}", names=names))
            got_reason = kcp.LAST_CALL["fwd"]["hand_off_reason"]
            if reason is not None:
                assert got_reason == reason, (later_saver, form, got_reason)
            assert kcp.LAST_CALL["bwd"]["tapes_source"] == source, (
                later_saver,
                form,
                kcp.LAST_CALL["bwd"]["tapes_source"],
            )
            assert kcp.kept_registry_size() == 0, (later_saver, form)
        # the FP64 oracle of the region: the gradients of GDN are linear in its output's cotangent, so the projected
        # region's oracle is the pack's reference under the cotangent dout @ w, and its output the projected reference
        # output; the naive tier runs the same ops in BF16 (BF16 operands of both products, results rounded to BF16)
        if w is None:
            ref, naive = _oracle(inputs, device)
            ref_out, naive_out = ref["out"][None], naive["out"][None]
        else:
            d64, w64 = dout.double(), w.double()
            ref, _ = reference_pack(
                {**inputs, "dout": (d64 @ w64).reshape(inputs["v"].shape)}, device
            )
            naive = naive_pack(
                {**inputs, "dout": (dout @ w).reshape(inputs["v"].shape)}, device
            )
            ref_out = ref["out"].to(device).reshape(1, -1, HV * D) @ w64.T
            naive_out = naive["out"].to(device).reshape(1, -1, HV * D) @ w.T
        oracle = {"out": ref_out, **{n: ref[grad_of[n]][None] for n in OPERANDS}}
        tier = {"out": naive_out, **{n: naive[grad_of[n]][None] for n in OPERANDS}}
        return (
            runs,
            _named(oracle, *arms, names=names),
            _named(tier, *arms, names=names),
        )

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


# --------------------------------------------------------------------------- two caller streams (one GPU)


def test_two_caller_streams_with_cached_workspaces_meet_the_naive_tier(
    device, record_property
) -> None:
    """Two caller streams, each with its own cached caller-held workspace and its own pack, issue the explicit entries
    interleaved (A's forward, B's forward, A's backward, B's backward), twice: each stream's results (arms ``A`` /
    ``B``, the worse of its two steps) are judged against its pack's FP64 oracle with the naive BF16 tier."""
    kcp = _pkg("cake_kcp")
    packs = {"A": ([0, 300, 700, 1000], (2, 6)), "B": ([0, 900, 1000], (4, 12))}
    state = {
        key: {
            "ctx": _local_context(cu, device),
            "ws": kcp.CakeGDNNondetKcpWorkspace(),
            "stream": torch.cuda.Stream(device=device),
        }
        for key, (cu, _heads) in packs.items()
    }

    def draw(j, s):
        oracle, naive = {}, {}
        for key, (cu, heads) in packs.items():
            inputs = make_inputs(cu, *heads, seed=s)
            st = state[key]
            st["ops"] = [inputs[n].to(device) for n in OPERANDS]
            st["dout"] = inputs["dout"].to(device)
            st["results"] = []
            ref, nv = _oracle(inputs, device)
            oracle.update(_named(ref, key))
            naive.update(_named(nv, key))
        torch.cuda.synchronize(device)
        for _step in range(2):
            pending = {}
            for key, st in state.items():
                with torch.cuda.stream(st["stream"]):
                    pending[key] = gdn_pkg.cake_gdn_nondet_train_kcp_forward(
                        *st["ops"], st["ctx"], workspace=st["ws"]
                    )
            for key, st in state.items():
                with torch.cuda.stream(st["stream"]):
                    out, saved = pending[key]
                    grads = gdn_pkg.cake_gdn_nondet_train_kcp_backward(
                        *st["ops"],
                        st["ctx"],
                        st["dout"],
                        saved=saved,
                        workspace=st["ws"],
                    )
                    st["results"].append(
                        {
                            "out": out[0],
                            **{
                                n: t[0]
                                for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True)
                            },
                        }
                    )
            torch.cuda.synchronize(device)
        runs = [_named(r, key) for key, st in state.items() for r in st["results"]]
        return runs, oracle, naive

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


def test_per_call_workspace_ticket_slots_follow_the_caller_stream(
    device, record_property
) -> None:
    """Calls without ``workspace=`` on one context take the context's ticket slots of their caller stream: two
    streams interleaving the explicit entries (A's forward, B's forward, A's backward, B's backward), twice per draw,
    use two distinct slot sets, keyed by the two streams, every word zero afterwards; each stream's results (arms
    ``A`` / ``B``, the worse of its two steps) are judged against the FP64 oracle with the naive BF16 tier.  Under a
    CUDA stream capture a call gets no shared set (fresh slots)."""
    kcp = _pkg("cake_kcp")
    cu, heads = [0, 300, 700, 1000], (2, 6)
    ctx = _local_context(cu, device)
    streams = {key: torch.cuda.Stream(device=device) for key in ("A", "B")}

    def draw(j, s):
        inputs = make_inputs(cu, *heads, seed=s)
        ops = [inputs[n].to(device) for n in OPERANDS]
        dout = inputs["dout"].to(device)
        results = {key: [] for key in streams}
        torch.cuda.synchronize(device)
        for _step in range(2):
            pending = {}
            for key, stream in streams.items():
                with torch.cuda.stream(stream):
                    pending[key] = gdn_pkg.cake_gdn_nondet_train_kcp_forward(*ops, ctx)
            for key, stream in streams.items():
                with torch.cuda.stream(stream):
                    out, saved = pending[key]
                    grads = gdn_pkg.cake_gdn_nondet_train_kcp_backward(
                        *ops, ctx, dout, saved=saved
                    )
                    results[key].append(
                        {
                            "out": out[0],
                            **{
                                n: t[0]
                                for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True)
                            },
                        }
                    )
            torch.cuda.synchronize(device)
        ref, naive = _oracle(inputs, device)
        runs = [_named(r, key) for key, rs in results.items() for r in rs]
        return runs, _named(ref, *streams), _named(naive, *streams)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)

    def ticket_sets():
        return {
            k: v
            for k, v in ctx._caches.items()
            if isinstance(k, tuple) and k[:1] == ("kcp_call_tickets",)
        }

    sets = ticket_sets()
    assert {k[3] for k in sets} == {s.cuda_stream for s in streams.values()}, sets
    assert len({t.data_ptr() for t, _views in sets.values()}) == len(streams)
    for t, _views in sets.values():
        assert bool((t.view(torch.int32) == 0).all())
    graph, x = torch.cuda.CUDAGraph(), torch.zeros(1, device=device)
    with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=device)):
        x.add_(1)
        captured = kcp._call_tickets(ctx, device)
    assert captured is None
    assert ticket_sets().keys() == sets.keys()
    assert not failures, failures


# --------------------------------------------------------------------------- the launch scope (one GPU)


def test_launch_scope_enters_one_stream_context_per_entry_body(
    device, monkeypatch, record_property
) -> None:
    """The kernels backend's launch path on a caller stream: an autograd forward + backward, direct and under a
    non-reentrant checkpoint, binds the caller's stream for tvm-ffi (``tvm_ffi.use_raw_stream`` on the stream's
    device and raw handle) once per entry body (the launch scope) instead of once per launch, every binding is the
    caller stream, the launches are the same records on the same grids as with the scope off (one context per
    launch), and each arm's results (``scope`` / ``per-launch``, ``direct`` / ``checkpoint``) are judged against the
    FP64 oracle with the naive BF16 tier."""
    from torch.utils.checkpoint import checkpoint

    be = _pkg("cake_backend")
    kcp = _pkg("cake_kcp")
    real = be.tvm_ffi
    entered = [0]
    bound = []

    class _Counting:
        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def use_raw_stream(dev, raw):
            entered[0] += 1
            bound.append((dev.index, int(raw)))
            return real.use_raw_stream(dev, raw)

    monkeypatch.setattr(be, "tvm_ffi", _Counting())
    cu, heads = [0, 300, 700, 1000], (2, 6)
    ctx = _local_context(cu, device)
    stream = torch.cuda.Stream(device=device)

    def label(arm: str, ckpt: bool) -> str:
        return f"{arm}-{'checkpoint' if ckpt else 'direct'}"

    def draw(j, s):
        inputs = make_inputs(cu, *heads, seed=s)
        torch.cuda.synchronize(device)
        results, launches, contexts = {}, {}, {}
        for arm in ("scope", "per-launch"):
            with monkeypatch.context() as m:
                if arm == "per-launch":
                    m.setattr(
                        kcp.KernelBackend, "launch_scope", lambda self, call: None
                    )
                for ckpt in (False, True):
                    entered[0] = 0
                    with torch.cuda.stream(stream):
                        leaves = [
                            inputs[n].to(device).detach().requires_grad_(True)
                            for n in OPERANDS
                        ]

                        def layer(q, k, v, g, beta):
                            return gdn_pkg.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx)

                        out = (
                            checkpoint(layer, *leaves, use_reentrant=False)
                            if ckpt
                            else layer(*leaves)
                        )
                        grads = torch.autograd.grad(
                            out, leaves, inputs["dout"].to(device)
                        )
                    stream.synchronize()
                    results[label(arm, ckpt)] = {
                        "out": out.detach()[0],
                        **{
                            n: t.detach()[0]
                            for n, t in zip(GRAD_NAMES, grads, strict=True)
                        },
                    }
                    launches[(arm, ckpt)] = {
                        k: [(r, tuple(g)) for r, g in kcp.LAST_CALL[k]["records"]]
                        for k in ("fwd", "bwd")
                    }
                    contexts[(arm, ckpt)] = entered[0]
        for ckpt in (False, True):
            bodies = 3 if ckpt else 2  # forward, (the checkpoint's recompute,) backward
            n_launches = sum(len(v) for v in launches[("per-launch", ckpt)].values())
            assert launches[("scope", ckpt)] == launches[("per-launch", ckpt)], ckpt
            assert contexts[("scope", ckpt)] == bodies, (ckpt, contexts)
            assert contexts[("per-launch", ckpt)] >= n_launches > bodies, (
                ckpt,
                contexts,
            )
        ref, naive = _oracle(inputs, device)
        runs = [_named(res, arm) for arm, res in results.items()]
        return runs, _named(ref, *results), _named(naive, *results)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    caller = {(stream.device.index, stream.cuda_stream)}
    assert bound and set(bound) == caller, set(bound)
    assert not failures, failures


# --------------------------------------------------------------------------- the output walk kcp_fwdh_out (one GPU)
# A light, saved-boundary forward that launches #5 and hands no tape to a backward runs the output walk in place of the
# tape walk and #5 (test knob fused_o = 1, the shipped form); fused_o = 0 forces the split form.  Each form below runs
# both on the same build (arms ``fused`` / ``split``), each judged against the FP64 oracle of the form's region with
# the naive BF16 tier (:func:`_form_oracle`; a graph form by the worst of its calls).  The launch logs show which form
# each forward ran.  Each pack holds about 1000 tokens (the route does not depend on the size: the walk forms of both
# value widths are forced by the vh64 / vh128 forms, and P = 2 by the segments-2 forms).

FUSED_SHAPES = (
    ("2x6", (300, 400, 300), (2, 6)),
    ("16x48", (512 + 37, 400), (16, 48)),
)
FUSED_FORMS = (
    # id, layers, checkpoint, later saver, options (``explicit`` / ``handle`` select the entry form), test knobs
    ("direct", 1, False, False, {}, {}),
    ("explicit", 1, False, False, {"explicit": True}, {}),
    ("ckpt-closes", 1, True, False, {}, {}),
    ("ckpt-later-saver", 1, True, True, {}, {}),
    ("ckpt-replay-form", 1, True, False, {"reuse_recompute_tapes": False}, {}),
    ("ckpt-3-layers-later-saver", 3, True, True, {}, {}),
    ("ckpt-handle", 1, True, False, {"handle": True}, {}),
    ("segments-2", 1, True, True, {"segments": 2}, {}),
    ("vh64", 1, True, True, {}, {"fwdh_vh": 64}),
    ("vh128", 1, True, True, {}, {"fwdh_vh": 128}),
    ("l2n", 1, True, True, {"use_qk_l2norm_in_kernel": True}, {}),
    ("graph", 1, False, False, {"graph": "replay"}, {}),
)


def _fused_ctx(cu, form, device):
    """The one-rank context of a fused-walk form (``graph_replay`` for the graph form)."""
    part = gdn_pkg.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, layout="contiguous"
    )
    graph = form[4].get("graph") == "replay"
    return gdn_pkg.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, device, **({"graph_replay": True} if graph else {})
    )


def _later_saver_weight(HV: int, device) -> torch.Tensor:
    """The later saver's ``w`` (``[HV, 128]`` BF16 in [0.5, 1.5), one value for every draw): the region's output is
    ``x * w``, and w's gradient reads ``x``."""
    gen = torch.Generator(device="cpu").manual_seed(6098)
    w = (torch.rand(HV, D, generator=gen) + 0.5).to(torch.bfloat16).to(device)
    return w.requires_grad_(True)


def _fused_arm(
    device, ctx, inputs, form, fused_o: int, log: list, *, w=None, knob="fused_o"
) -> list[dict]:
    """The runs of ``form`` on ``inputs`` with test knob ``knob`` (``fused_o``; ``fused_o_tape`` for the
    ``save_mode="full"`` forms) set to ``fused_o``: one result (a graph form: its capture and two replays), each with
    ``out``, the five gradients and (with the later saver ``w``) ``dw``, without the batch axis; ``log`` receives
    ``(kind, sequence, fwd_o decision)`` of every recorded call."""
    from torch.utils.checkpoint import checkpoint

    kcp = _pkg("cake_kcp")
    _fid, layers, ckpt, _later, opts, knobs = form
    opts = dict(opts)
    explicit, handle = opts.pop("explicit", False), opts.pop("handle", False)
    graph = opts.get("graph") == "replay"
    kw = {"test_knobs": {knob: fused_o, **knobs}, **opts}
    if graph:
        kw["workspace"] = gdn_pkg.CakeGDNNondetKcpWorkspace()
    dout = inputs["dout"].to(device)
    del log[:]
    if explicit:
        q, k, v, g, beta = (inputs[n].to(device) for n in OPERANDS)
        out, saved = gdn_pkg.cake_gdn_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, save_for_backward=True, **kw
        )
        dq, dk, dv, dbeta, dg = gdn_pkg.cake_gdn_nondet_train_kcp_backward(
            q, k, v, g, beta, ctx, dout, saved=saved, **kw
        )
        torch.cuda.synchronize(device)
        res = {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}
        return [{n: t[0] for n, t in res.items()}]

    def fn(q, k, v, g, beta, *handles):
        x = v
        for i in range(layers):
            hk = {"recompute_handle": handles[i]} if handles else {}
            x = gdn_pkg.cake_gdn_nondet_kcp(q, k, x, g, beta, ctx, **hk, **kw)
        # a later saver of the region: w's gradient reads the output
        return x if w is None else x * w

    results = []
    if graph:
        kcp.graph_cache_clear()
    try:
        # graph: capture, then replays (a replay records no call)
        for _rep in range(3 if graph else 1):
            leaves = [
                inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS
            ]
            handles = (
                [gdn_pkg.CakeGDNNondetKcpRecomputeHandle() for _ in range(layers)]
                if handle
                else []
            )
            out = (
                checkpoint(fn, *leaves, *handles, use_reentrant=False)
                if ckpt
                else fn(*leaves, *handles)
            )
            grads = torch.autograd.grad(
                out, leaves + ([w] if w is not None else []), dout
            )
            torch.cuda.synchronize(device)
            res = {"out": out.detach()[0].clone()}
            res.update(
                {n: gr[0].clone() for n, gr in zip(GRAD_NAMES, grads, strict=False)}
            )
            if w is not None:
                res["dw"] = grads[-1].clone()
            results.append(res)
    finally:
        if graph:
            kcp.graph_cache_clear()
    return results


def _judge_fused_form(device, cu, heads, form, log, check, *, knob="fused_o"):
    """Judge ``form`` over 32 draws: per draw the split arm (``knob`` = 0), then the fused arm (``knob`` = 1), each
    run's outputs named ``"split.<name>"`` / ``"fused.<name>"`` against the FP64 oracle of the form's region with the
    naive BF16 tier; ``check(fused_log, split_log)`` asserts each draw's launch logs."""
    H, HV = heads
    ctx = _fused_ctx(cu, form, device)
    w = _later_saver_weight(HV, device) if form[3] else None
    names = TENSORS + (("dw",) if w is not None else ())
    arms = ("split", "fused")

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        split = _fused_arm(device, ctx, inputs, form, 0, log, w=w, knob=knob)
        split_log = list(log)
        fused = _fused_arm(device, ctx, inputs, form, 1, log, w=w, knob=knob)
        check(list(log), split_log)
        ref, naive = _form_oracle(
            inputs,
            device,
            layers=form[1],
            w=w,
            norm=bool(form[4].get("use_qk_l2norm_in_kernel")),
        )
        runs = [_named(r, "split", names=names) for r in split]
        runs += [_named(r, "fused", names=names) for r in fused]
        return runs, _named(ref, *arms, names=names), _named(naive, *arms, names=names)

    return _judge(draw, seed=SEED)


@pytest.mark.parametrize("form", FUSED_FORMS, ids=[f[0] for f in FUSED_FORMS])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [s[1:] for s in FUSED_SHAPES],
    ids=[s[0] for s in FUSED_SHAPES],
)
def test_fused_output_walk_and_the_split_forward_meet_the_naive_tier(
    device, monkeypatch, record_property, lengths, heads, form
) -> None:
    """The same-build matrix of the output walk: direct, explicit, checkpointed (the region closed by the node, a later
    saver, the replay form, three layers with a later saver, a recompute handle), two segments, both value widths,
    the in-kernel L2 norm and the graph replay, each with test knob ``fused_o`` 1 and 0 (arms ``fused`` / ``split``)
    judged against the FP64 oracle with the naive BF16 tier; on every draw the fused arm's forwards that write the
    output run the output walk and its backwards are the split form's."""
    kcp = _pkg("cake_kcp")
    backend = _pkg("cake_backend")
    if not kcp.fused_o_shipped(backend.device_arch(device.index)):
        pytest.skip("the registry ships no output walk for this architecture")
    if form[4].get("use_qk_l2norm_in_kernel") and not kcp.l2n_records_shipped():
        pytest.skip("the registry ships no in-kernel L2 norm records")
    log: list = []
    record = kcp._record

    def recording(kind, call):
        record(kind, call)
        entry = kcp.LAST_CALL[kind]
        log.append((kind, tuple(entry.get("sequence") or ()), entry.get("fwd_o")))

    monkeypatch.setattr(kcp, "_record", recording)

    def check(fused_log, split_log):
        # the same calls; the fused arm's forwards that write the output run the output walk (the first forward
        # always: it hands no tape over), the split arm never does
        assert [k for k, _s, _o in fused_log] == [k for k, _s, _o in split_log]
        fwd = [(s, o) for k, s, o in fused_log if k == "fwd"]
        assert fwd and fwd[0][1] == "fused", fwd[:1]
        for seq, decision in fwd:
            assert ("kcp_fwdh_out" in seq) == (decision == "fused"), (decision, seq)
            if decision == "fused":
                assert "kcp_fwdh_tape" not in seq and "fwd_o" not in seq, seq
        assert not any("kcp_fwdh_out" in s or o == "fused" for _k, s, o in split_log)
        assert [s for k, s, _o in fused_log if k == "bwd"] == [
            s for k, s, _o in split_log if k == "bwd"
        ]  # every backward is the split form's

    judge, failures = _judge_fused_form(device, _cu(lengths), heads, form, log, check)
    judge.record(record_property)
    assert not failures, failures
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- the tape-and-output walk (one GPU)
# A save_mode="full" forward that keeps its saved set and launches #5 runs the tape-and-output walk kcp_fwdh_out_tape in
# place of the tape walk and #5 (test knob fused_o_tape = 1, the shipped form); fused_o_tape = 0 forces the split form.
# Each form below runs both on the same build with the saved h / v_new tapes pre-filled with 0xFF bytes (NaN): a tape
# row the backward reads but no walk wrote shows as a non-finite gradient (the judge's contract).  Arms ``fused`` /
# ``split`` are judged as in the output-walk matrix above.

FULL_TAPE_FORMS = (
    # id, layers, checkpoint, later saver, options (``explicit`` / ``handle`` select the entry form), test knobs
    ("direct", 1, False, False, {}, {}),
    ("explicit", 1, False, False, {"explicit": True}, {}),
    ("explicit-segments-2", 1, False, False, {"explicit": True, "segments": 2}, {}),
    ("explicit-vh64", 1, False, False, {"explicit": True}, {"fwdh_vh": 64}),
    ("explicit-vh128", 1, False, False, {"explicit": True}, {"fwdh_vh": 128}),
    ("ckpt-closes", 1, True, False, {}, {}),
    ("ckpt-later-saver", 1, True, True, {}, {}),
    ("ckpt-always", 1, True, False, {"recompute_output": "always"}, {}),
    ("ckpt-3-layers-later-saver", 3, True, True, {}, {}),
    ("ckpt-handle", 1, True, False, {"handle": True}, {}),
    ("segments-2", 1, True, True, {"segments": 2}, {}),
    (
        "segments-2-no-save-transfer",
        1,
        True,
        True,
        {"segments": 2, "save_transfer": False},
        {},
    ),
    ("vh64", 1, True, True, {}, {"fwdh_vh": 64}),
    ("vh128", 1, True, True, {}, {"fwdh_vh": 128}),
    ("l2n", 1, True, True, {"use_qk_l2norm_in_kernel": True}, {}),
)


@pytest.mark.parametrize("form", FULL_TAPE_FORMS, ids=[f[0] for f in FULL_TAPE_FORMS])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [s[1:] for s in FUSED_SHAPES],
    ids=[s[0] for s in FUSED_SHAPES],
)
def test_fused_tape_walk_and_the_split_full_forward_meet_the_naive_tier(
    device, monkeypatch, record_property, lengths, heads, form
) -> None:
    """The same-build matrix of the tape-and-output walk under ``save_mode="full"``: direct, explicit (its backward
    reads the saved tapes; two segments, both value widths), checkpointed (the region closed by the node, a later
    saver, ``recompute_output="always"``, three layers with a later saver, a recompute handle), two segments with and
    without ``save_transfer``, both value widths and the in-kernel L2 norm, each with test knob ``fused_o_tape`` 1 and
    0 (arms ``fused`` / ``split``) judged against the FP64 oracle with the naive BF16 tier.  On every draw each full
    forward of the fused arm that writes the output runs ``kcp_fwdh_out_tape`` and neither the tape walk nor #5; the
    split arm never runs it; the backward launches are the split form's."""
    kcp = _pkg("cake_kcp")
    backend = _pkg("cake_backend")
    if not kcp.fused_o_tape_shipped(backend.device_arch(device.index)):
        pytest.skip("the registry ships no tape-and-output walk for this architecture")
    if form[4].get("use_qk_l2norm_in_kernel") and not kcp.l2n_records_shipped():
        pytest.skip("the registry ships no in-kernel L2 norm records")
    full_form = (*form[:4], {"save_mode": "full", **form[4]}, form[5])
    log: list = []
    record = kcp._record

    def recording(kind, call):
        record(kind, call)
        entry = kcp.LAST_CALL[kind]
        log.append((kind, tuple(entry.get("sequence") or ()), entry.get("fwd_o")))

    monkeypatch.setattr(kcp, "_record", recording)
    fresh_tapes = kcp.CakeGDNNondetKcpCall.fresh_tapes

    def filled_tapes(self, names):
        fresh_tapes(self, names)
        for n in ("h_tape", "v_new"):
            if n in names:
                self.tapes[n].view(torch.uint8).fill_(0xFF)

    monkeypatch.setattr(kcp.CakeGDNNondetKcpCall, "fresh_tapes", filled_tapes)

    def check(fused_log, split_log):
        # the same calls; the fused arm's forwards that write the output run the tape-and-output walk (the first
        # forward always: it is no recompute), a recompute that leaves #5 out keeps the tape walk, and the split arm
        # never runs it
        assert [k for k, _s, _o in fused_log] == [k for k, _s, _o in split_log]
        fwd = [(s, o) for k, s, o in fused_log if k == "fwd"]
        assert fwd and fwd[0][1] == "fused_tape", fwd[:1]
        for seq, decision in fwd:
            assert decision in ("fused_tape", "skipped"), (decision, seq)
            assert ("kcp_fwdh_out_tape" in seq) == (decision == "fused_tape"), (
                decision,
                seq,
            )
            if decision == "fused_tape":
                assert not {"kcp_fwdh_tape", "kcp_fwdh_out", "fwd_o"} & set(seq), seq
        assert not any(
            "kcp_fwdh_out_tape" in s or o == "fused_tape" for _k, s, o in split_log
        )
        assert [s for k, s, _o in fused_log if k == "bwd"] == [
            s for k, s, _o in split_log if k == "bwd"
        ]  # every backward is the split form's

    judge, failures = _judge_fused_form(
        device, _cu(lengths), heads, full_form, log, check, knob="fused_o_tape"
    )
    judge.record(record_property)
    assert not failures, failures
    assert kcp.kept_registry_size() == 0


def test_graph_replay_keeps_each_view_form_its_own_gradient_layout(
    device, record_property
) -> None:
    """``graph="replay"`` with a caller-held workspace, autograd through ``cake_gdn_nondet_kcp``: q / k / v as
    ``cake_gdn_nondet_train_kcp_packed_qkv_views`` views and as plain ``split`` / ``view`` views of the same packed row (the same pointers
    and strides) share the forward graph but are two backward graph keys.  Each form captures its own backward with
    its own gradient layout (``"packed"`` / ``"separate"``), and alternating forms replay their own graphs, never the
    other form's (on every draw, from an empty graph cache); every ``out`` and row gradient of each form (arms
    ``packed`` / ``plain``, the worst of the form's calls) is judged against the FP64 oracle with the naive BF16 tier."""
    kcp = _pkg("cake_kcp")
    H, HV = 2, 6
    cu = _cu((300, 700))
    T = cu[-1]
    ctx = _local_context(cu, device)
    sizes = (H * D, H * D, HV * D)
    # each form's backward runs eager on its first call, captures on its second and replays from then on (the forward
    # graph is the same for both forms)
    forms = ("packed",) * 4 + ("plain",) * 3 + ("packed", "plain")
    layout = {"packed": "packed", "plain": "separate"}
    want = [
        (f, e, layout[f])
        for f, e in zip(
            forms,
            ("eager", "capture", "replay", "replay")
            + ("eager", "capture", "replay")
            + ("replay", "replay"),
            strict=True,
        )
    ]
    names = ("out", "dq", "dk", "dv")

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        ws = kcp.CakeGDNNondetKcpWorkspace()
        row = torch.cat(
            [inputs[n].to(device).reshape(1, T, -1) for n in ("q", "k", "v")], -1
        ).contiguous()
        row.requires_grad_(True)
        g, beta, dout = (inputs[n].to(device) for n in ("g", "beta", "dout"))

        def views(form):
            if form == "packed":
                return kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(
                    row, H_qk=H, HV=HV
                )
            parts = row.split(sizes, dim=-1)
            return tuple(
                p.view(1, T, n, D) for p, n in zip(parts, (H, H, HV), strict=True)
            )

        events, runs = [], []
        kcp.graph_cache_clear()
        try:
            for form in forms:
                q, k, v = views(form)
                out = gdn_pkg.cake_gdn_nondet_kcp(
                    q, k, v, g, beta, ctx, workspace=ws, graph="replay"
                )
                (drow,) = torch.autograd.grad(out, (row,), dout)
                rec = kcp.LAST_CALL["bwd"]
                events.append((form, rec["graph_event"], rec["grad_layout"]))
                res = {"out": out.detach()[0].clone()}
                for n, (a, b), h in zip(
                    ("dq", "dk", "dv"),
                    ((0, H * D), (H * D, 2 * H * D), (2 * H * D, None)),
                    (H, H, HV),
                    strict=True,
                ):
                    res[n] = drow[0, :, a:b].reshape(T, h, D).clone()
                runs.append(_named(res, form, names=names))
                del q, k, v, out, drow
        finally:
            kcp.graph_cache_clear()
        assert events == want, events
        ref, naive = _oracle(inputs, device)
        return (
            runs,
            _named(ref, *layout, names=names),
            _named(naive, *layout, names=names),
        )

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


# ----------------------------------------------------------------------------------------------------------------
# save_mode="full" without save_transfer at P > 1


@pytest.mark.parametrize("norm", [False, True], ids=["l2n-off", "l2n-on"])
def test_full_backward_without_save_transfer_and_light_meet_the_naive_tier_on_poisoned_memory(
    device, monkeypatch, record_property, norm
) -> None:
    """``save_mode="full"`` with ``save_transfer=False`` (the only form of ``exchange="ring"``) at ``P = 2``: every
    tensor the package allocates is filled with NaN bytes, so a read of a plane no program of the call wrote (the
    ``M_seg^T`` staging plane the backward's merges read) shows up as a non-finite output (the judge's contract).  The
    saved set carries ``M_seg``.  Out and the five gradients of the light form (arm ``light``) and of the full form on
    a per-call, a fresh and a shared workspace (twice on the shared one: the worse call) and under a non-reentrant
    checkpoint (arms ``full_per_call`` / ``full_fresh`` / ``full_shared`` / ``full_checkpoint``) are judged against
    the FP64 oracle with the naive BF16 tier.  P = 2 is forced (``segments=2``, bounded by the documents' 5 / 7 / 5
    chunks)."""
    kcp = _pkg("cake_kcp")
    real_alloc = kcp._alloc

    def nan_alloc(shape, dtype, dev):
        t = real_alloc(shape, dtype, dev)
        if t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    monkeypatch.setattr(kcp, "_alloc", nan_alloc)
    cu = _cu((300, 400, 300))
    H, HV = 16, 48
    ctx = _local_context(cu, device)
    o = {"segments": 2, "save_transfer": False, "use_qk_l2norm_in_kernel": norm}

    def run(inputs, workspace, *, ckpt: bool = False, **kw) -> dict:
        leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]

        def fn(*a):
            return gdn_pkg.cake_gdn_nondet_kcp(*a, ctx, workspace=workspace, **o, **kw)

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
        else:
            out = fn(*leaves)
        out.backward(inputs["dout"].to(device))
        torch.cuda.synchronize(device)
        res = {"out": out.detach()[0]}
        res.update({n: lf.grad[0] for n, lf in zip(GRAD_NAMES, leaves, strict=True)})
        return res

    inputs = make_inputs(cu, H, HV)
    q, k, v, g, beta = (inputs[n].to(device)[0] for n in OPERANDS)
    _, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        q, k, v, g, beta, ctx, save_mode="full", **o
    )
    assert saved.P == 2 and saved.c1_recv is None and saved.M_seg is not None
    del saved, q, k, v, g, beta, inputs
    arms = (
        "light",
        "full_per_call",
        "full_fresh",
        "full_shared",
        "full_checkpoint",
    )

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        runs = [_named(run(inputs, None), "light")]
        assert kcp.LAST_CALL["fwd"]["P"] == 2
        shared = kcp.CakeGDNNondetKcpWorkspace()
        full_arms = (
            ("full_per_call", None, False),
            ("full_fresh", kcp.CakeGDNNondetKcpWorkspace(), False),
            ("full_shared", shared, False),
            ("full_shared", shared, False),
            ("full_checkpoint", None, True),
        )
        for arm, ws, ckpt in full_arms:
            full = run(inputs, ws, ckpt=ckpt, save_mode="full")
            assert kcp.LAST_CALL["bwd"]["tapes_source"] == "saved", arm
            runs.append(_named(full, arm))
        ref, naive = _oracle(inputs, device, norm=norm)
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


# ----------------------------------------------------------------------------------------------------------------
# save_mode="light_akk": the light set plus #1_l2n's BF16 Akk tape; the light replay runs fwd_prep_replay_l2n

AKK_FORMS = (
    # id, document lengths, (H_qk, HV), options (the forced P = 2 is bounded by the documents' chunk counts)
    ("2x6", (100, 300, 77, 400, 129), (2, 6), {}),
    (
        "2x6-no-save-transfer",
        (100, 300, 77, 400, 129),
        (2, 6),
        {"save_transfer": False},
    ),
    ("16x48-P2", (300, 400, 300), (16, 48), {"segments": 2}),
    (
        "16x48-P2-no-save-transfer",
        (300, 400, 300),
        (16, 48),
        {"segments": 2, "save_transfer": False},
    ),
)


def _akk_or_skip(kcp):
    """The replay's program name; skips while the registry does not ship the norm programs or the replay."""
    if not kcp.l2n_records_shipped():
        pytest.skip("the in-kernel L2 norm programs are not rendered")
    jit = _pkg("cake_jit")
    replay = kcp.AKK_REPLAY_OF["fwd_prep_l2n"]
    if not jit.records_of_program(replay):
        pytest.skip(f"{replay} is not rendered")
    return jit, replay


@pytest.mark.parametrize("form", AKK_FORMS, ids=[f[0] for f in AKK_FORMS])
def test_light_akk_and_light_meet_the_naive_tier_on_poisoned_memory(
    device, monkeypatch, record_property, form
) -> None:
    """``save_mode="light_akk"`` (``use_qk_l2norm_in_kernel=True``) and ``"light"`` (the same programs on the same
    states but #1_l2n's block inverse, whose ``Akk`` the replay reads from the saved set) on the device programs with
    every tensor the package allocates filled with NaN bytes: out and the five gradients of each arm -- the explicit
    entries of both modes, autograd of both modes (the saved set through ``save_for_backward``) and ``"light_akk"``
    under a non-reentrant checkpoint (where the mode saves as ``"light"``) -- are judged against the FP64 oracle of the
    normalized rows with the naive BF16 tier.  The backward launches ``fwd_prep_replay_l2n`` once per light replay of
    ``fwd_prep_l2n`` and no ``fwd_prep_l2n``, on the same grids; the forward launches light's programs; the saved set
    gains the ``Akk`` tape (``128 R HV`` bytes; its rows of the real chunks are finite, as are the full set's; the pad
    chunks' rows are written by no program)."""
    kcp = _pkg("cake_kcp")
    jit, replay = _akk_or_skip(kcp)
    real_alloc = kcp._alloc

    def nan_alloc(shape, dtype, dev):
        t = real_alloc(shape, dtype, dev)
        if t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    monkeypatch.setattr(kcp, "_alloc", nan_alloc)
    _name, lengths, (H, HV), opts = form
    cu = _cu(lengths)
    ctx = _local_context(cu, device)
    o = {"use_qk_l2norm_in_kernel": True, **opts}
    arms = (
        "explicit_light",
        "explicit_light_akk",
        "autograd_light",
        "autograd_light_akk",
        "checkpoint_light_akk",
    )

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        q, k, v, g, beta = (inputs[n].to(device)[0] for n in OPERANDS)
        dout = inputs["dout"].to(device)[0]

        def saved_pass(**kw) -> dict:
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                q, k, v, g, beta, ctx, **o, **kw
            )
            fwd = dict(kcp.LAST_CALL["fwd"])
            grads = kcp.cake_gdn_nondet_train_kcp_backward(
                q, k, v, g, beta, ctx, dout, saved=saved, **o, **kw
            )
            torch.cuda.synchronize(device)
            return {
                "out": out,
                **dict(zip(("dq", "dk", "dv", "dbeta", "dg"), grads, strict=True)),
                "saved": saved,
                "fwd": fwd,
                "bwd": dict(kcp.LAST_CALL["bwd"]),
            }

        def autograd(ckpt: bool, **kw) -> dict:
            leaves = [
                inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS
            ]

            def fn(*a):
                return gdn_pkg.cake_gdn_nondet_kcp(*a, ctx, **o, **kw)

            if ckpt:
                from torch.utils.checkpoint import checkpoint

                out = checkpoint(fn, *leaves, use_reentrant=False)
            else:
                out = fn(*leaves)
            out.backward(inputs["dout"].to(device))
            torch.cuda.synchronize(device)
            res = {"out": out.detach()[0]}
            res.update(
                {n: lf.grad[0] for n, lf in zip(GRAD_NAMES, leaves, strict=True)}
            )
            return res

        light = saved_pass(
            save_mode="light"
        )  # the default with the norm is "light_akk"
        if "segments" in opts:
            assert light["fwd"]["P"] == opts["segments"]
        got = saved_pass(save_mode="light_akk")
        lb, ab = light["bwd"], got["bwd"]
        assert ab["akk_source"] == "saved" and lb["akk_source"] is None
        assert (
            ab["save_mode"] == "light_akk" and ab["tapes_source"] == lb["tapes_source"]
        )
        assert got["fwd"]["sequence"] == light["fwd"]["sequence"]
        n_l2n = lb["sequence"].count("fwd_prep_l2n")
        assert n_l2n >= 1 and "fwd_prep_l2n" not in ab["sequence"]
        assert ab["sequence"] == [
            replay if p == "fwd_prep_l2n" else p for p in lb["sequence"]
        ]
        replay_records = set(jit.records_of_program(replay))
        l2n_records = set(jit.records_of_program("fwd_prep_l2n"))
        assert sum(r in replay_records for r, _ in ab["records"]) == n_l2n
        assert sum(r in l2n_records for r, _ in lb["records"]) == n_l2n
        assert not any(r in l2n_records for r, _ in ab["records"])
        assert [gr for _, gr in ab["records"]] == [gr for _, gr in lb["records"]]
        tape = int(ctx.R) * HV * 128
        akk = got["saved"].tape("akk")
        assert got["saved"].bytes - light["saved"].bytes == tape == akk.numel() * 2
        _, full_saved = kcp.cake_gdn_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, save_mode="full", **o
        )
        torch.cuda.synchronize(device)
        full_akk = full_saved.tape("akk")
        # pad-row rule: the 64 rows of each pad chunk (chunk_len 0; NC_pad - NC of them) are written and read by no
        # program (NaN bytes here, in both modes); the rows of the NC real chunks are finite in both tapes
        rows_per_chunk = akk.shape[0] // len(ctx.chunk_len)
        assert rows_per_chunk * len(ctx.chunk_len) == akk.shape[0] == int(ctx.R)
        real = torch.tensor([n > 0 for n in ctx.chunk_len], device=akk.device)
        real = real.repeat_interleave(rows_per_chunk)
        assert int(real.sum()) == rows_per_chunk * int(ctx.NC)
        for tag, t in (("light_akk", akk), ("full", full_akk)):
            assert bool(torch.isfinite(t[real].float()).all()), tag
        del full_saved, full_akk, akk, real
        a_light = autograd(False, save_mode="light")
        a_akk = autograd(False, save_mode="light_akk")
        assert kcp.LAST_CALL["bwd"]["akk_source"] == "saved"
        a_ckpt = autograd(True, save_mode="light_akk")
        assert kcp.LAST_CALL["bwd"]["save_mode"] == "light"
        assert replay not in kcp.LAST_CALL["bwd"]["sequence"]
        ref, naive = _oracle(inputs, device, norm=True)
        runs = [
            _named(r, a)
            for a, r in zip(arms, (light, got, a_light, a_akk, a_ckpt), strict=True)
        ]
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 700), (2, 6)), ((512 + 37, 400), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_graph_replay_light_akk_steps_and_the_eager_light_step_meet_the_naive_tier(
    device, record_property, lengths, heads
) -> None:
    """Forward + backward steps of ``save_mode="light_akk"`` under ``graph="replay"`` on one caller-held workspace
    whose arena still holds an eager ``"light"`` step's tapes, with pointer-stable operands (a new workspace and an
    empty graph cache per draw): the steady state replays both graphs, every backward launches the replay in place of
    ``fwd_prep_l2n`` (``akk_source="saved"``), and the eager ``"light"`` step and the graph steps (arms ``eager_light``
    / ``graph_light_akk``, the worst of the five steps) are judged against the FP64 oracle of the normalized rows with
    the naive BF16 tier."""
    kcp = _pkg("cake_kcp")
    _jit, replay = _akk_or_skip(kcp)
    H, HV = heads
    cu = _cu(lengths)
    ctx = _local_context(cu, device)
    arms = ("eager_light", "graph_light_akk")

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        base = [inputs[n].to(device) for n in OPERANDS]
        dout = inputs["dout"].to(device)
        ws = kcp.CakeGDNNondetKcpWorkspace()

        def step(mode: str, graph: str):
            leaves = [t.detach().requires_grad_(True) for t in base]
            out = gdn_pkg.cake_gdn_nondet_kcp(
                *leaves,
                ctx,
                workspace=ws,
                save_mode=mode,
                graph=graph,
                use_qk_l2norm_in_kernel=True,
            )
            fwd = dict(kcp.LAST_CALL["fwd"])
            res = {"out": out.detach()[0].clone()}
            out.backward(dout)
            res.update(
                {
                    n: lf.grad.detach()[0].clone()
                    for n, lf in zip(GRAD_NAMES, leaves, strict=True)
                }
            )
            return res, fwd, dict(kcp.LAST_CALL["bwd"])

        kcp.graph_cache_clear()
        try:
            eager, _eager_fwd, eager_bwd = step("light", "off")
            steps = [step("light_akk", "replay") for _ in range(5)]
            torch.cuda.synchronize(device)
        finally:
            kcp.graph_cache_clear()
        events = [(f["graph_event"], b["graph_event"]) for _, f, b in steps]
        assert events[-2:] == [("replay", "replay")] * 2, events
        swap = [replay if p == "fwd_prep_l2n" else p for p in eager_bwd["sequence"]]
        assert eager_bwd["sequence"].count("fwd_prep_l2n") >= 1
        for i, (_res, _f, b) in enumerate(steps):
            assert b["akk_source"] == "saved" and b["sequence"] == swap, (
                i,
                b["sequence"],
            )
        ref, naive = _oracle(inputs, device, norm=True)
        runs = [_named(eager, "eager_light")]
        runs += [_named(res, "graph_light_akk") for res, _f, _b in steps]
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


# --------------------------------------------------------------------------- checkpoint under graph="replay"
# torch.utils.checkpoint(use_reentrant=False) of one GDN layer with graph="replay" on a caller-held workspace: the
# recompute decides its tape hand-off and its output program before the CUDA-graph lookup (both are part of its graph
# key), so the captured step replays the eager checkpoint step's programs: no #1 and no tape walk in the backward.


class _LaterSaver(torch.autograd.Function):
    """A later saver of the checkpointed region: torch's recompute runs in its backward, which first runs ``hook()``
    (a call on the GDN layer's workspace between the recompute and the GDN backward)."""

    hook = None

    @staticmethod
    def forward(fctx, x):
        fctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(fctx, gy):
        (_,) = fctx.saved_tensors
        if _LaterSaver.hook is not None:
            _LaterSaver.hook()
        return gy


def _ckpt_steps(
    kcp, ctx, inputs, device, *, steps, ws=None, later=False, hooks=None, **kw
):
    """``steps`` checkpointed steps of one layer on one caller-held workspace (``ws``, default a new one) with
    pointer-stable operands: per step the out and the five gradients (cloned at once: a replay returns the graph's
    static tensors), the records of the original forward, torch's recompute and the backward, and
    ``GRAPH_STATS["reset"]`` after the step.  ``later``: a :class:`_LaterSaver` closes the region; ``hooks[i]``: its
    backward hook in step ``i``."""
    from torch.utils.checkpoint import checkpoint

    ws = kcp.CakeGDNNondetKcpWorkspace() if ws is None else ws
    base = [inputs[n].to(device) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    results, records, resets = [], [], []
    kcp.graph_cache_clear()
    try:
        for i in range(steps):
            leaves = [t.detach().requires_grad_(True) for t in base]

            def fn(*a):
                out = gdn_pkg.cake_gdn_nondet_kcp(*a, ctx, workspace=ws, **kw)
                return _LaterSaver.apply(out) if later else out

            _LaterSaver.hook = (hooks or {}).get(i)
            try:
                out = checkpoint(fn, *leaves, use_reentrant=False)
                fwd = dict(kcp.LAST_CALL["fwd"])
                res = {"out": out.detach()[0].clone()}
                out.backward(dout)
            finally:
                _LaterSaver.hook = None
            res.update(
                {
                    n: lf.grad.detach()[0].clone()
                    for n, lf in zip(GRAD_NAMES, leaves, strict=True)
                }
            )
            records.append(
                (fwd, dict(kcp.LAST_CALL["fwd"]), dict(kcp.LAST_CALL["bwd"]))
            )
            resets.append(kcp.GRAPH_STATS["reset"])
            results.append(res)
            del out, leaves
        torch.cuda.synchronize(device)
    finally:
        kcp.graph_cache_clear()
    return results, records, resets


def _launches(step_records) -> list:
    """The launch inventory of one step: (record, grid) of every launch of the original forward, the recompute and
    the backward, in issue order."""
    return [(r, tuple(g)) for rec in step_records for r, g in rec["records"]]


@pytest.mark.parametrize("l2n", [False, True], ids=["raw", "l2n"])
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 700), (2, 6)), ((512 + 37, 400), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_graph_replay_checkpoint_replays_the_eager_checkpoint_programs(
    device, record_property, lengths, heads, l2n, monkeypatch
) -> None:
    """Checkpointed steps under ``graph="replay"`` (default ``reuse_recompute_tapes`` / ``recompute_output``) reach a
    steady state in which the original forward, the recompute and the backward all replay captured graphs, no entry is
    reset and every step's launch inventory is the eager checkpoint step's -- on a verified torch #1 runs twice (not
    three times: the backward binds the recompute's tapes and launches no #1 or tape walk) and the registry sees one
    put / take / release per step and no stale entry.  ``reuse_recompute_tapes=False`` (the previous graph form)
    replays #1 in the backward.  Out and the five gradients of the eager steps, the graph steps and the previous
    form's steps (arms ``eager`` / ``graph`` / ``previous``, the worst of the six steps) are judged against the FP64
    oracle with the naive BF16 tier; the launch facts hold on every draw (an empty graph cache per form and draw).
    The size rule of ``"auto"`` is set to 0 rows (the recompute decides at these sizes)."""
    kcp = _pkg("cake_kcp")
    if l2n and not kcp.l2n_records_shipped():
        pytest.skip("the in-kernel L2 norm programs are not rendered")
    _size_rule_at(kcp, monkeypatch, 0)
    H, HV = heads
    cu = _cu(lengths)
    ctx = _local_context(cu, device)
    kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
    steps = 6
    prep = "fwd_prep_l2n" if l2n else "fwd_prep"

    def n_prep(rec) -> int:
        return sum(1 for r, _ in _launches(rec) if r.startswith(prep))

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        eager, eager_rec, _ = _ckpt_steps(kcp, ctx, inputs, device, steps=steps, **kw)
        kept_before, stale_before = dict(kcp.KEPT_STATS), dict(kcp.KEPT_STALE_STATS)
        graph, graph_rec, resets = _ckpt_steps(
            kcp, ctx, inputs, device, steps=steps, graph="replay", **kw
        )
        kept_delta = {k: kcp.KEPT_STATS[k] - kept_before[k] for k in kept_before}
        stale_delta = {
            k: kcp.KEPT_STALE_STATS[k] - stale_before[k] for k in stale_before
        }
        previous, previous_rec, _ = _ckpt_steps(
            kcp,
            ctx,
            inputs,
            device,
            steps=steps,
            graph="replay",
            reuse_recompute_tapes=False,
            **kw,
        )
        events = [tuple(r["graph_event"] for r in rec) for rec in graph_rec]
        assert events[-3:] == [("replay", "replay", "replay")] * 3, events
        assert resets[-1] == resets[-3], resets  # the steady state resets no entry
        assert [tuple(r["graph_event"] for r in rec) for rec in previous_rec][-1] == (
            "replay",
            "replay",
            "replay",
        )
        for i in range(steps):
            assert _launches(graph_rec[i]) == _launches(eager_rec[i]), i
        assert n_prep(previous_rec[-1]) == 3
        assert stale_delta == {k: 0 for k in stale_delta}, stale_delta
        assert kcp.kept_registry_size() == 0
        if kcp.torch_release_verified():
            assert n_prep(graph_rec[-1]) == 2, _launches(graph_rec[-1])
            assert graph_rec[-1][2]["tapes_source"] == "kept"
            assert graph_rec[-1][1]["hand_off"] is True
            assert graph_rec[-1][1]["fwd_o"] == "skipped"
            assert (
                kept_delta["put"]
                == kept_delta["take"]
                == kept_delta["released"]
                == steps
            )
            assert kept_delta["miss"] == 0, kept_delta
        else:
            assert n_prep(graph_rec[-1]) == 3
        ref, naive = _oracle(inputs, device, norm=l2n)
        arms = {"eager": eager, "graph": graph, "previous": previous}
        runs = [_named(r, arm) for arm, results in arms.items() for r in results]
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


def _size_rule_at(kcp, monkeypatch, rows: int) -> None:
    """The size rule of ``reuse_recompute_tapes="auto"`` under ``graph="replay"`` at ``rows`` on every architecture
    (``cake_kcp.GRAPH_RECOMPUTE_MIN_ROWS`` and its default), with an empty architecture cache."""
    table = {arch: rows for arch in kcp.GRAPH_RECOMPUTE_MIN_ROWS}
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS", table)
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT", rows)
    monkeypatch.setattr(kcp, "_GRAPH_RECOMPUTE_ARCH", {})


@pytest.mark.parametrize("l2n", [False, True], ids=["raw", "l2n"])
def test_graph_replay_auto_below_the_size_rule_replays_the_original_forward_graph(
    device, record_property, l2n, monkeypatch
) -> None:
    """``reuse_recompute_tapes="auto"`` under ``graph="replay"`` with the size rule above this rank's ``T_r * HV``
    (``GRAPH_RECOMPUTE_MIN_ROWS``): torch's recompute looks up the original forward's key (the same cache entry in
    every step; the hand-off form, the rule at 0 rows, has its own) and, once the step is steady, replays that graph
    (the same launches as the original forward, reasons ``"small"``); the backward replays #1 and the tape walk (three
    prep launches, no entry put).  Out and the five gradients of the eager steps, the steps below the rule, the
    ``reuse_recompute_tapes=False`` steps and the hand-off form's steps (arms ``eager`` / ``small`` / ``previous`` /
    ``handed``, the worst of the six steps) are judged against the FP64 oracle with the naive BF16 tier; the launch
    facts hold on every draw."""
    kcp = _pkg("cake_kcp")
    if l2n and not kcp.l2n_records_shipped():
        pytest.skip("the in-kernel L2 norm programs are not rendered")
    H, HV = 2, 6
    cu = _cu((300, 700))
    ctx = _local_context(cu, device)
    kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
    steps = 6
    entries: list = []
    original_entry = kcp._graph_entry

    def spy_entry(kind, *args):
        e = original_entry(kind, *args)
        if kind == "fwd":
            entries.append(e)
        return e

    monkeypatch.setattr(kcp, "_graph_entry", spy_entry)
    prep = "fwd_prep_l2n" if l2n else "fwd_prep"

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        _size_rule_at(kcp, monkeypatch, cu[-1] * HV + 1)
        eager, _, _ = _ckpt_steps(kcp, ctx, inputs, device, steps=steps, **kw)
        entries.clear()
        kept_before = dict(kcp.KEPT_STATS)
        small, small_rec, resets = _ckpt_steps(
            kcp, ctx, inputs, device, steps=steps, graph="replay", **kw
        )
        small_pairs = [entries[2 * i : 2 * i + 2] for i in range(steps)]
        kept_delta = {k: kcp.KEPT_STATS[k] - kept_before[k] for k in kept_before}
        previous, _, _ = _ckpt_steps(
            kcp,
            ctx,
            inputs,
            device,
            steps=steps,
            graph="replay",
            reuse_recompute_tapes=False,
            **kw,
        )
        _size_rule_at(kcp, monkeypatch, 0)
        entries.clear()
        handed, _, _ = _ckpt_steps(
            kcp, ctx, inputs, device, steps=steps, graph="replay", **kw
        )
        handed_pairs = [entries[2 * i : 2 * i + 2] for i in range(steps)]
        entries.clear()
        events = [tuple(r["graph_event"] for r in rec) for rec in small_rec]
        assert events[-3:] == [("replay", "replay", "replay")] * 3, events
        assert resets[-1] == resets[-3], resets
        assert all(len(p) == 2 and p[0] is p[1] for p in small_pairs), small_pairs
        if kcp.torch_release_verified():
            assert all(p[0] is not p[1] for p in handed_pairs[-3:])
        for original, recompute, backward in small_rec[-3:]:
            assert recompute["records"] == original["records"]
            assert recompute["hand_off"] is False, recompute
            assert recompute["hand_off_reason"] == recompute["fwd_o_reason"] == "small"
            assert backward["tapes_source"] == "replay"
        rec = small_rec[-1]
        assert sum(1 for r, _ in _launches(rec) if r.startswith(prep)) == 3
        assert kept_delta["put"] == kept_delta["take"] == 0, kept_delta
        assert kcp.kept_registry_size() == 0
        ref, naive = _oracle(inputs, device, norm=l2n)
        arms = {"eager": eager, "small": small, "previous": previous, "handed": handed}
        runs = [_named(r, arm) for arm, results in arms.items() for r in results]
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures


def test_graph_replay_kept_tapes_are_refused_after_another_call_on_the_workspace(
    device, record_property
) -> None:
    """``graph="replay"``, ``reuse_recompute_tapes=True`` and a later saver: once the step replays, three consecutive
    steps run an explicit forward of OTHER operands on the same workspace in the later saver's backward -- between the
    recompute (whose kept tapes are the arena's own) and the GDN backward.  Each of them finds the entry stale
    (``KEPT_STALE_STATS["calls"]``) and the GDN backward replays #1 and the tape walk instead of binding the rewritten
    tapes, while that backward's own graph goes eager -> capture -> replay (the last hooked step replays it), and the
    next step binds the kept tapes again (on every draw, with a new workspace).  Out and the five gradients of the
    hooked steps and of the other steps (arms ``hooked`` / ``unhooked``, the worst of their steps) are judged against
    the FP64 oracle with the naive BF16 tier."""
    kcp = _pkg("cake_kcp")
    H, HV = 2, 6
    cu = _cu((300, 700))
    seeded = make_inputs(cu, H, HV, seed=7)
    other = {n: seeded[n].to(device) for n in OPERANDS}
    ctx = _local_context(cu, device)
    steps, hooked = 8, (4, 5, 6)
    arms = ("unhooked", "hooked")

    def draw(j, s):
        inputs = make_inputs(cu, H, HV, seed=s)
        ws = kcp.CakeGDNNondetKcpWorkspace()
        calls: list[str] = []

        def hook():
            calls.append("explicit forward")
            kcp.cake_gdn_nondet_train_kcp_forward(
                *(other[n] for n in OPERANDS),
                ctx,
                workspace=ws,
                graph="replay",
            )

        stale_before = dict(kcp.KEPT_STALE_STATS)
        results, records, _ = _ckpt_steps(
            kcp,
            ctx,
            inputs,
            device,
            steps=steps,
            ws=ws,
            later=True,
            hooks=dict.fromkeys(hooked, hook),
            graph="replay",
            reuse_recompute_tapes=True,
        )
        sources = [rec[2]["tapes_source"] for rec in records]
        events = [rec[2]["graph_event"] for rec in records]
        assert calls == ["explicit forward"] * len(hooked)
        last_clean = records[hooked[0] - 1][2]
        assert last_clean["graph_event"] == "replay", last_clean
        assert [sources[i] for i in hooked] == ["replay"] * len(hooked), sources
        assert [events[i] for i in hooked] == ["eager", "capture", "replay"], events
        assert sources[hooked[0] - 1] == sources[hooked[-1] + 1] == "kept", sources
        assert kcp.KEPT_STALE_STATS["calls"] - stale_before["calls"] == len(hooked)
        assert kcp.kept_registry_size() == 0
        ref, naive = _oracle(inputs, device)
        runs = [
            _named(r, "hooked" if i in hooked else "unhooked")
            for i, r in enumerate(results)
        ]
        return runs, _named(ref, *arms), _named(naive, *arms)

    judge, failures = _judge(draw, seed=SEED)
    judge.record(record_property)
    assert not failures, failures
