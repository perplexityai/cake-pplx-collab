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

Host tests of the family's naive-tier judge glue (``tests/experimental/test_helpers/cake_lm_head_judge.py``): the seed
offsets keep every draw's generators apart, the naive BF16 tiers return the oracle's shapes in BF16, the judged inputs
change with the draw, and the judge passes a launch that is the tier itself and fails one that is not.  No kernel
runs here; the judge's own verdicts go to a local list, so these tests print no ``TIER`` line.
"""

import pytest
import torch

from tests.cake_naive import DRAW_SEED_STRIDE, DRAWS, draw_seed
from tests.experimental.test_helpers.cake_lm_head_judge import (
    JUDGE_H,
    JUDGE_V,
    SEED_OFFSETS,
    W_DRAW_OFFSET,
    contract_lm_head,
    contract_support,
    judge_inputs,
    judge_lm_head,
    judge_support,
    judge_support_inputs,
    seed_offset_collisions,
)
from tests.experimental.test_helpers.cake_lm_head_loss_reference import (
    TEMPERATURE_SEED_OFFSET,
    make_inputs,
    naive_bf16,
    naive_bf16_forced_side,
    reference_fp64,
    reference_fp64_forced_side,
)
from tests.experimental.test_helpers.cake_lm_head_support_reference import (
    _TAU_SEED_OFFSET,
    _TEACHER_SEED_OFFSET,
    make_support,
    make_support_inputs,
    naive_support_bf16,
    reference_support_fp64,
)

DEV = torch.device("cpu")
SEED = 20261117
H_HOST, V_HOST = 256, 512
BF = torch.bfloat16


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.double().reshape(-1), b.double().reshape(-1)]))[0, 1])


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_seed_offsets_keep_the_draws_apart():
    """No two generator offsets of a judged case meet within the draws; the inputs' default ``W`` offset would."""
    assert JUDGE_H == 6144 and JUDGE_V == 8192
    assert seed_offset_collisions(SEED_OFFSETS) == []
    assert SEED_OFFSETS["W"] == W_DRAW_OFFSET
    assert SEED_OFFSETS["temperature"] == TEMPERATURE_SEED_OFFSET == _TAU_SEED_OFFSET
    assert SEED_OFFSETS["teacher"] == _TEACHER_SEED_OFFSET
    # the default offset of make_inputs / make_support_inputs is one stride: draw j's W is draw j + 1's X generator
    assert seed_offset_collisions({"X": 0, "W": DRAW_SEED_STRIDE}) == [("W", "X", 1)]
    assert seed_offset_collisions({"X": 0, "W": 31 * DRAW_SEED_STRIDE}) == [("W", "X", 31)]
    assert seed_offset_collisions({"X": 0, "W": DRAWS * DRAW_SEED_STRIDE}) == []
    assert seed_offset_collisions({"a": 5, "b": 5}) == [("a", "b", 0)]


def test_seed_offsets_are_the_ones_the_inputs_use():
    """The support inputs draw their upstreams from ``s + 7`` and their ids from ``s + 17``, as the table says."""
    s = SEED
    inp = make_support_inputs(29, seed=s, k_cap=8, labels=True, teacher=True)
    ug = torch.Generator(device="cpu")
    ug.manual_seed(s + SEED_OFFSETS["support upstreams"])
    assert torch.equal(inp.g, torch.randn(inp.M, generator=ug) * 0.05)
    ids, offsets = make_support(29, inp.V, 8, seed=s + SEED_OFFSETS["support ids"])
    assert torch.equal(ids, inp.support_ids) and torch.equal(offsets, inp.support_offsets)


def test_judge_inputs_change_with_the_draw():
    """Every draw of :func:`judge_inputs` has its own ``X``, ``W`` and labels, and draw ``j``'s ``W`` is unrelated
    to draw ``j + 1``'s ``X`` (with the inputs' default ``W`` it is the same normal draw)."""
    T = 37
    seeds = [draw_seed(SEED + T, j) for j in range(DRAWS)]
    draws = [
        judge_inputs(T, s, H=H_HOST, V=V_HOST, device=DEV, objective="policy") for s in seeds
    ]
    for a, b in zip(draws, draws[1:]):
        assert not torch.equal(a.X, b.X)
        assert not torch.equal(a.W, b.W)
        assert not torch.equal(a.labels, b.labels)
        assert not torch.equal(a.infer_logp, b.infer_logp)
        assert abs(_corr(a.W[:T], b.X)) < 0.1
    again = judge_inputs(T, seeds[3], H=H_HOST, V=V_HOST, device=DEV, objective="policy")
    for name in ("X", "W", "labels", "infer_logp", "loss_weights", "dlogp"):
        assert torch.equal(getattr(again, name), getattr(draws[3], name)), name
    plain = [make_inputs(T, seed=s, H=H_HOST, V=V_HOST, device=DEV) for s in seeds[:2]]
    assert _corr(plain[0].W[:T], plain[1].X) > 0.99
    sup = [judge_support_inputs(T, s, k_cap=8, labels=True) for s in seeds[:4]]
    for a, b in zip(sup, sup[1:]):
        assert not torch.equal(a.X, b.X) and not torch.equal(a.W, b.W)
        assert not torch.equal(a.support_ids, b.support_ids) or not torch.equal(
            a.support_offsets, b.support_offsets
        )
        assert abs(_corr(a.W[:T], b.X)) < 0.1


# ---------------------------------------------------------------------------
# The naive tiers
# ---------------------------------------------------------------------------


ENTRIES = [
    ("ce", "loss", None),
    ("policy", "loss", None),
    ("ce", "logprob", None),
    ("ce", "logprob", "scalar"),
    ("ce", "logprob", "hetero"),
]


@pytest.mark.parametrize(
    "objective, entry, temperature", ENTRIES, ids=[f"{o}-{e}-{t}" for o, e, t in ENTRIES]
)
def test_naive_bf16_has_the_oracle_shapes_in_bf16(objective, entry, temperature):
    T = 65
    inp = make_inputs(
        T, objective=objective, seed=SEED, H=H_HOST, V=V_HOST, device=DEV, ignore_frac=0.1,
        temperature=temperature,
    )
    naive = naive_bf16(inp, entry=entry)
    oracle = reference_fp64(inp, entry=entry)
    for name in ("logp", "lse", "dX", "dW"):
        assert naive[name].dtype == BF and naive[name].shape == oracle[name].shape, name
        assert bool(torch.isfinite(naive[name]).all()), name
    if entry == "loss":
        assert naive["loss"].dtype == BF and naive["loss"].shape == ()
    else:
        assert naive["loss"] is None
    assert not bool(naive["logp"][~inp.valid].any()) and not bool(naive["dX"][~inp.valid].any())
    # the tier rounds: it is not the oracle
    assert float((naive["logp"].double() - oracle["logp"])[inp.valid].abs().max()) > 0
    none = naive_bf16(inp, entry=entry, need_dx=False, need_dw=False)
    assert none["dX"] is None and none["dW"] is None
    assert torch.equal(none["logp"], naive["logp"])
    # a power-of-two upstream scale is exact in both
    twice, twice64 = naive_bf16(inp, entry=entry, scale=2.0), reference_fp64(inp, entry=entry, scale=2.0)
    for name in ("dX", "dW"):
        assert torch.equal(twice[name], 2 * naive[name]), name
        assert torch.equal(twice64[name], 2 * oracle[name]), name
    assert torch.equal(twice["logp"], naive["logp"]) and torch.equal(twice64["logp"], oracle["logp"])


def test_temperature_needs_the_logprob_entry():
    inp = make_inputs(5, seed=SEED, H=H_HOST, V=V_HOST, device=DEV, temperature="scalar")
    with pytest.raises(ValueError, match="temperature belongs"):
        naive_bf16(inp)


def test_forced_side_oracle_and_naive():
    """The rows of ``over`` are saturated (no gradient, loss term ``2 w``), the other valid rows unclipped."""
    T = 64
    inp = make_inputs(T, objective="policy", seed=SEED, H=H_HOST, V=V_HOST, device=DEV, ignore_frac=0.1)
    rows = inp.valid.nonzero().squeeze(1)
    over = torch.zeros(T, dtype=torch.bool)
    over[rows[1::2]] = True
    under = inp.valid & ~over
    oracle = reference_fp64_forced_side(inp, over)
    naive = naive_bf16_forced_side(inp, over)
    for name in ("loss", "logp", "lse", "dX", "dW"):
        assert naive[name].dtype == BF and naive[name].shape == oracle[name].shape, name
    assert not bool(oracle["dX"][over].any()) and not bool(naive["dX"][over].any())
    assert bool((oracle["dX"][under].abs().sum(-1) > 0).all())
    assert bool((naive["dX"][under].float().abs().sum(-1) > 0).all())
    ratio = torch.exp(oracle["logp"] - inp.infer_logp.double())
    w = inp.loss_weights.double()
    expected = -(w * torch.where(over, torch.full_like(ratio, 2.0), ratio))[inp.valid].sum()
    assert torch.equal(oracle["loss"], expected)  # the same float64 operations on the same values
    plain = reference_fp64(inp)
    assert torch.equal(oracle["logp"], plain["logp"])
    with pytest.raises(ValueError, match="policy loss"):
        reference_fp64_forced_side(make_inputs(4, seed=SEED, H=H_HOST, V=V_HOST, device=DEV), over[:4])
    with pytest.raises(ValueError, match="bool"):
        naive_bf16_forced_side(inp, over[:5])


SUPPORT_CASES = {
    "packed": (("packed",), dict()),
    "selected": (("selected",), dict(labels=True)),
    "kl": (("kl",), dict(teacher=True)),
    "packed and kl, hetero tau": (("packed", "kl"), dict(teacher=True, tau_form="hetero")),
}


@pytest.mark.parametrize("case", sorted(SUPPORT_CASES))
def test_naive_support_bf16_has_the_oracle_shapes_in_bf16(case):
    outputs, kw = SUPPORT_CASES[case]
    inp = make_support_inputs(41, seed=SEED, k_cap=16, **kw)
    naive = naive_support_bf16(inp, outputs=outputs)
    oracle = reference_support_fp64(inp, outputs=outputs)
    for name in (*outputs, "lse", "dX", "dW"):
        assert naive[name].dtype == BF and naive[name].shape == oracle[name].shape, name
        assert bool(torch.isfinite(naive[name]).all()), name
    for name in {"packed", "selected", "kl"} - set(outputs):
        assert naive[name] is None
    none = naive_support_bf16(inp, outputs=outputs, need_dx=False, need_dw=False)
    assert none["dX"] is None and none["dW"] is None


# ---------------------------------------------------------------------------
# The judge glue
# ---------------------------------------------------------------------------


def _as_kernel(inp, naive):
    """A stand-in kernel result: the naive tier at the kernel's dtypes, ``+0`` on the ignored rows."""
    valid = inp.valid
    out = dict(
        logp=torch.where(valid, naive["logp"].float(), torch.zeros_like(naive["logp"].float())),
        dX=torch.where(valid[:, None], naive["dX"], torch.zeros_like(naive["dX"])),
        dW=naive["dW"],
    )
    if naive["loss"] is not None:
        out["loss"] = naive["loss"].float()
    return out


def _make(T, **kw):
    return lambda s: judge_inputs(T, s, H=H_HOST, V=V_HOST, device=DEV, ignore_frac=0.1, **kw)


def test_judge_passes_the_tier_itself_and_restores_the_random_state():
    verdicts = []
    state = torch.get_rng_state()
    judge = judge_lm_head(
        lambda key, value: verdicts.append((key, value)),
        make=_make(37, objective="policy"),
        launch=lambda inp: {f: _as_kernel(inp, naive_bf16(inp)) for f in ("a", "b")},
        seed=SEED + 37,
        forms=("a", "b"),
    )
    assert torch.equal(torch.get_rng_state(), state)
    names = [v["output"] for _, v in verdicts]
    assert names == [f"{n}@{f}" for f in ("a", "b") for n in ("loss", "logp", "dX", "dW")]
    for key, verdict in verdicts:
        assert key == "naive_tier" and verdict["tier"] == "bf16" and verdict["draws"] == DRAWS
    assert judge.failures() == []


def test_judge_views_and_the_logprob_entry():
    verdicts = []

    def launch(inp):
        naive = naive_bf16(inp, entry="logprob")
        return dict(_as_kernel(inp, naive), lse=naive["lse"][inp.valid].float())

    judge_lm_head(
        lambda key, value: verdicts.append(value),
        make=_make(29, temperature="hetero"),
        launch=launch,
        entry="logprob",
        seed=SEED + 29,
        names=("logp", "dX", "dW", "lse"),
        views={"lse": ("lse", lambda inp, v: v[inp.valid])},
    )
    assert [v["output"] for v in verdicts] == ["logp", "dX", "dW", "lse"]
    assert all(v["tier"] == "bf16" for v in verdicts)
    with pytest.raises(ValueError, match="made by the test"):
        judge_lm_head(None, make=None, launch=None, entry="logprob", seed=0, names=("loss",))


def test_judge_fails_an_output_that_misses_the_tier():
    verdicts = []

    def launch(inp):
        out = _as_kernel(inp, naive_bf16(inp))
        out["dX"] = (out["dX"].float() * 1.25).bfloat16()
        return out

    with pytest.raises(AssertionError, match=r"dX: misses the naive tier over 32 inputs"):
        judge_lm_head(lambda key, value: verdicts.append(value), make=_make(37), launch=launch, seed=SEED)
    assert {v["output"]: v["tier"] for v in verdicts} == dict(loss="bf16", logp="bf16", dX=None, dW="bf16")


def test_judge_keeps_the_worst_of_repeated_launches():
    calls = []

    def launch(inp):
        out = _as_kernel(inp, naive_bf16(inp))
        calls.append(1)
        if len(calls) % 3 == 0:  # the third launch of each draw
            out["dW"] = (out["dW"].float() * 1.25).bfloat16()
        return out

    verdicts = []
    with pytest.raises(AssertionError, match=r"dW: misses the naive tier"):
        judge_lm_head(
            lambda key, value: verdicts.append(value), make=_make(37), launch=launch, seed=SEED, repeats=3
        )
    assert len(calls) == 3 * DRAWS
    assert {v["output"]: v["tier"] for v in verdicts}["dW"] is None
    good = []
    judge_lm_head(
        lambda key, value: good.append(value),
        make=_make(37),
        launch=lambda inp: _as_kernel(inp, naive_bf16(inp)),
        seed=SEED,
        repeats=2,
    )
    assert all(v["tier"] == "bf16" for v in good)


def test_judge_holds_every_draw_to_its_contract():
    def launch(inp):
        out = _as_kernel(inp, naive_bf16(inp))
        out["dX"][~inp.valid] = -0.0
        return out

    with pytest.raises(AssertionError, match=r"draw 0: dX: an ignored row holds a value other than \+0"):
        judge_lm_head(lambda key, value: None, make=_make(37), launch=launch, seed=SEED)

    def route(inp, result):
        assert result["logp"].shape[0] == 0, "not the expected route"

    with pytest.raises(AssertionError, match="draw 0: not the expected route"):
        judge_lm_head(
            lambda key, value: None,
            make=_make(37),
            launch=lambda inp: _as_kernel(inp, naive_bf16(inp)),
            seed=SEED,
            route=route,
        )


def test_contracts_name_every_break():
    inp = make_inputs(9, seed=SEED, H=H_HOST, V=V_HOST, device=DEV, ignore_frac=0.3)
    good = _as_kernel(inp, naive_bf16(inp))
    contract_lm_head(good, inp)
    bad = dict(good, loss=good["loss"].double(), dW=good["dW"].float())
    bad["logp"] = good["logp"].clone()
    bad["logp"][~inp.valid] = -0.0
    with pytest.raises(AssertionError) as info:
        contract_lm_head(bad, inp)
    text = str(info.value)
    assert "loss: torch.float64" in text and "dW: torch.float32" in text and "logp: an ignored row" in text
    contract_lm_head(dict(good, dW=good["dW"].float()), inp, grad_weight_dtype=torch.float32)
    sup = make_support_inputs(17, seed=SEED, k_cap=8, labels=True)
    res = naive_support_bf16(sup, outputs=("selected",))
    with pytest.raises(AssertionError, match="selected: torch.bfloat16"):
        contract_support(res, sup, outputs=("selected",))
    contract_support(dict(res, selected=res["selected"].float()), sup, outputs=("selected",))


def test_judge_support_passes_the_tier_itself():
    verdicts = []
    outputs = ("packed", "selected")

    def launch(inp):
        naive = naive_support_bf16(inp, outputs=outputs)
        return dict(packed=naive["packed"].float(), selected=naive["selected"].float(), dX=naive["dX"], dW=naive["dW"])

    judge_support(
        lambda key, value: verdicts.append(value),
        make=lambda s: judge_support_inputs(33, s, k_cap=8, labels=True),
        launch=launch,
        outputs=outputs,
        seed=SEED + 33,
    )
    assert [v["output"] for v in verdicts] == ["packed", "selected", "dX", "dW"]
    assert all(v["tier"] == "bf16" for v in verdicts)
