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

The LM-head family's glue on the shared naive-tier judge (``tests/cake_naive.py``).

A judged case runs :data:`DRAWS` draws.  Draw ``j`` of a case of seed ``seed`` makes its inputs from ``s =
draw_seed(seed, j)`` alone (:func:`judge_inputs`, :func:`judge_support_inputs`), launches the kernel once per form
(or ``repeats`` times), computes one float64 oracle and one naive BF16 value per draw and judges every form's
outputs against them as separately named outputs (``dX@compact``, ``dX@plain``, ...).  Each draw is also held to
its contract: the outputs' dtypes, shapes and finiteness (:func:`contract_lm_head`, :func:`contract_support`), the
exact outputs (an ignored row's ``logp`` / ``dX``, a single-candidate row's packed ``logp``, an ignored row's
selected ``logp``: 0) and the ``+0`` bits of the ignored rows.

Device cases judge at ``H = JUDGE_H``, ``V = JUDGE_V``: ``H`` keeps the production forms that depend on it (one pass,
column items, tiles), and every ``V >= 2048`` gives the production dX slice plans.

Seeds.  Every random input of a draw comes from a CPU generator seeded ``s`` plus a fixed offset
(:data:`SEED_OFFSETS`).  Draw ``j`` of an input with offset ``a`` and draw ``i`` of an input with offset ``b`` share a
generator when ``a - b = (i - j) * DRAW_SEED_STRIDE``; within a case no two offsets may differ by a multiple of the
stride up to ``DRAWS - 1`` strides (:func:`seed_offset_collisions`).  The weight's default offset of
``make_inputs`` / ``make_support_inputs`` (``seed + 1000003``) is one stride: draw ``j``'s ``W`` would come from the
generator of draw ``j + 1``'s ``X``.  The judged inputs therefore always draw ``W`` from ``s + W_DRAW_OFFSET``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Union

import torch

from tests.cake_naive import (
    DRAW_SEED_STRIDE,
    DRAWS,
    TIERS,
    TierJudge,
    contract,
    draw_seed,
    judge_case,
    seed_all,
)
from tests.experimental.test_helpers.cake_lm_head_loss_reference import (
    TEMPERATURE_SEED_OFFSET,
    Inputs,
    make_inputs,
    make_weight,
    naive_bf16,
    reference_fp64,
)
from tests.experimental.test_helpers.cake_lm_head_support_reference import (
    _TEACHER_SEED_OFFSET,
    IGNORE_INDEX,
    OUTPUTS as SUPPORT_OUTPUTS,
    SupportInputs,
    make_support_inputs,
    naive_support_bf16,
    reference_support_fp64,
)

#: The shape of the judged device cases.
JUDGE_H, JUDGE_V = 6144, 8192
#: The judged inputs' ``W`` generator: ``s + W_DRAW_OFFSET`` (see the module docstring).
W_DRAW_OFFSET = 500_009
#: Every generator seed offset a judged case uses, by input.
SEED_OFFSETS = {
    "X, labels and the objective operands": 0,
    "replay weight update": 5,
    "support upstreams": 7,
    "support ids": 17,
    "upstream gradients": 7637,
    "device upstream gradients": 7639,
    "zero sets": 7992,
    "W": W_DRAW_OFFSET,
    "temperature": TEMPERATURE_SEED_OFFSET,  # the support inputs' temperature uses the same offset
    "teacher": _TEACHER_SEED_OFFSET,
}

_View = tuple[str, Optional[Callable[[object, torch.Tensor], torch.Tensor]]]
_Kw = Union[None, Mapping, Callable[[object], Mapping]]


def seed_offset_collisions(
    offsets: Mapping[str, int], draws: int = DRAWS
) -> list[tuple[str, str, int]]:
    """Pairs of ``offsets`` whose generators meet within ``draws`` draws: ``(a, b, k)`` where offset ``a`` minus
    offset ``b`` is ``k`` strides, ``0 < |k| < draws`` (two inputs of one name always share their generator).  Two
    names with the same offset collide at ``k = 0``."""
    found = []
    names = sorted(offsets)
    for x, a in enumerate(names):
        for b in names[x + 1 :]:
            diff = offsets[a] - offsets[b]
            if diff % DRAW_SEED_STRIDE == 0 and abs(diff // DRAW_SEED_STRIDE) < draws:
                found.append((a, b, diff // DRAW_SEED_STRIDE))
    return found


def judge_inputs(
    T: int, s: int, *, H: int = JUDGE_H, V: int = JUDGE_V, device, **kw
) -> Inputs:
    """The inputs of the draw of seed ``s``: :func:`make_inputs` with ``seed = s`` and ``W`` from its own generator
    seeded ``s + W_DRAW_OFFSET``.  ``kw``: the other keywords of :func:`make_inputs` (``objective``, ``ignore_frac``,
    ``loss_div``, ``ld_pad``, ``temperature``, ``temperature_value``)."""
    W = make_weight(V, H, seed=s + W_DRAW_OFFSET, device=device)
    return make_inputs(T, seed=s, H=H, V=V, device=device, W=W, **kw)


def judge_support_inputs(
    T: int, s: int, *, H: int = 256, V: int = 512, device="cpu", **kw
) -> SupportInputs:
    """The candidate-support inputs of the draw of seed ``s``: :func:`make_support_inputs` with ``seed = s`` and ``W``
    from its own generator seeded ``s + W_DRAW_OFFSET``.  ``kw``: the other keywords of
    :func:`make_support_inputs`."""
    W = make_weight(V, H, seed=s + W_DRAW_OFFSET, device="cpu").to(device)
    return make_support_inputs(T, seed=s, H=H, V=V, device=device, W=W, **kw)


def _plus_zero(t: torch.Tensor) -> bool:
    """Every element of ``t`` is ``+0`` (bits)."""
    ints = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return not bool(t.contiguous().view(ints[t.element_size()]).any())


def _check(problems, name, value, dtype, shape) -> bool:
    if value.dtype != dtype or tuple(value.shape) != tuple(shape):
        problems.append(
            f"{name}: {value.dtype} {tuple(value.shape)}, expected {dtype} {tuple(shape)}"
        )
        return False
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        problems.append(f"{name}: not finite")
    return True


def contract_lm_head(
    result: Mapping,
    inp: Inputs,
    *,
    entry: str = "loss",
    grad_weight_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """The contract of one call's ``result`` (``loss`` / ``logp`` / ``dX`` / ``dW``; a missing or ``None`` entry is
    not checked), without an oracle: FP32 0-dim ``loss`` (loss entry), FP32 ``logp [T]``, BF16 ``dX [T, H]``,
    ``grad_weight_dtype`` ``dW [V, H]``, every one finite, and ``logp`` / ``dX`` ``+0`` (bits) on every ignored
    row.  Raises an ``AssertionError`` listing every break."""
    if entry not in ("loss", "logprob"):
        raise ValueError("entry must be 'loss' or 'logprob'")
    problems: list[str] = []
    ignored = ~inp.valid
    want = {
        "loss": (torch.float32, ()),
        "logp": (torch.float32, (inp.T,)),
        "dX": (torch.bfloat16, (inp.T, inp.H)),
        "dW": (grad_weight_dtype, (inp.V, inp.H)),
    }
    if entry != "loss":
        del want["loss"]  # the log-probability entry's loss is made by the test
    for name, (dtype, shape) in want.items():
        value = result.get(name)
        if value is None:
            continue
        value = value.detach()
        if _check(problems, name, value, dtype, shape) and name in ("logp", "dX"):
            if not _plus_zero(value[ignored.to(value.device)]):
                problems.append(f"{name}: an ignored row holds a value other than +0")
    assert not problems, "; ".join(problems)


def contract_support(
    result: Mapping, inp: SupportInputs, *, outputs: Sequence[str] = SUPPORT_OUTPUTS
) -> None:
    """The contract of one candidate-support call's ``result`` (a missing or ``None`` entry is not checked), without
    an oracle: FP32 ``packed [M]`` / ``selected [T]`` / ``kl [T]`` for the requested ``outputs``, BF16 ``dX [T, H]``
    and ``dW [V, H]``, every one finite.  Raises an ``AssertionError`` listing every break."""
    problems: list[str] = []
    want = {
        "packed": (torch.float32, (inp.M,)),
        "selected": (torch.float32, (inp.T,)),
        "kl": (torch.float32, (inp.T,)),
        "dX": (torch.bfloat16, (inp.T, inp.H)),
        "dW": (torch.bfloat16, (inp.V, inp.H)),
    }
    for name, (dtype, shape) in want.items():
        if name in SUPPORT_OUTPUTS and name not in outputs:
            continue
        value = result.get(name)
        if value is not None:
            _check(problems, name, value.detach(), dtype, shape)
    assert not problems, "; ".join(problems)


def _lm_head_exact(inp: Inputs, result: Mapping) -> dict:
    """The exact outputs of one call: ``logp`` and ``dX`` on the ignored rows, 0."""
    ignored = ~inp.valid
    found = {}
    for name in ("logp", "dX"):
        value = result.get(name)
        if value is not None:
            found[f"{name}_ignored"] = value.detach()[ignored.to(value.device)]
    return found


def _support_exact(inp: SupportInputs, result: Mapping) -> dict:
    """The exact outputs of one candidate-support call: the selected ``logp`` of an ignored row and the packed
    ``logp`` of a single-candidate row, 0."""
    found = {}
    selected, packed = result.get("selected"), result.get("packed")
    if selected is not None and inp.labels is not None:
        ignored = inp.labels.to(selected.device) == IGNORE_INDEX
        found["selected_ignored"] = selected.detach()[ignored]
    if packed is not None:
        counts = inp.counts.cpu()
        starts = inp.support_offsets.to(torch.int64).cpu()[:-1]
        found["packed_single"] = packed.detach()[starts[counts == 1].to(packed.device)]
    return found


def _kw(kw: _Kw, inp) -> dict:
    if kw is None:
        return {}
    return dict(kw(inp) if callable(kw) else kw)


def _judge(
    record_property,
    *,
    make,
    launch,
    seed,
    names,
    forms,
    views,
    route,
    contract_fn,
    exact_fn,
    exact_names,
    oracle_fn,
    naive_fn,
    repeats,
    draws,
) -> TierJudge:
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "the judge needs torch.backends.cuda.matmul.allow_tf32 False: the reference backend's FP32 GEMMs "
            "depend on it"
        )
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    views = dict(views or {})
    labels = [None] if forms is None else list(forms)
    if len(set(labels)) != len(labels):
        raise ValueError(f"forms must be distinct, got {labels}")

    def label(name, form):
        return name if form is None else f"{name}@{form}"

    exact = frozenset(label(n, f) for n in exact_names for f in labels)

    def select(inp, full, name):
        key, view = views.get(name, (name, None))
        value = full[key]
        if value is None:
            raise AssertionError(f"{name}: the reference holds no {key}")
        return value if view is None else view(inp, value)

    def launched(j, inp):
        results = launch(inp)
        per_form = {None: results} if forms is None else results
        if forms is not None and set(per_form) != set(labels):
            raise AssertionError(
                f"draw {j}: the launch returned forms {sorted(per_form)}, expected {sorted(labels)}"
            )
        try:
            if route is not None:
                route(inp, results)
            for form in labels:
                contract_fn(per_form[form], inp)
        except AssertionError as exc:
            raise AssertionError(f"draw {j}: {exc}") from exc
        outputs = {}
        for form in labels:
            result = per_form[form]
            for name in names:
                value = result.get(name)
                if value is None:
                    raise AssertionError(f"draw {j}: {label(name, form)}: the launch returned no value")
                outputs[label(name, form)] = value.detach()
            for name, value in exact_fn(inp, result).items():
                outputs[label(name, form)] = value
        return outputs

    def expected(inp, outputs):
        """The oracle's value of every output of one launch: the judged ones from one oracle call, the exact ones 0."""
        oracle = oracle_fn(inp)
        values = {}
        for form in labels:
            for name in names:
                values[label(name, form)] = select(inp, oracle, name)
        for name, actual in outputs.items():
            if name in exact:
                values[name] = torch.zeros(actual.shape, dtype=torch.float64, device=actual.device)
        return values

    def naive_values(inp):
        found = naive_fn(inp)
        values = {}
        for form in labels:
            for name in names:
                values[label(name, form)] = select(inp, found, name)
        return values

    if repeats == 1:
        inputs: dict[int, object] = {}

        def run(j, s):
            inp = make(s)
            outputs = launched(j, inp)
            values = expected(inp, outputs)
            inputs[j] = inp
            return outputs, values

        def naive(j, tier):
            return naive_values(inputs.pop(j))

        judge, failures = judge_case(run, naive, seed=seed, exact=exact, draws=draws)
    else:
        # judge_case's loop with every launch of a draw added: the judge keeps each output's largest error
        judge, failures = TierJudge(draws), []
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        state = (random.getstate(), torch.get_rng_state())
        try:
            for j in range(draws):
                s = draw_seed(seed, j)
                seed_all(s)
                inp = make(s)
                oracle, tiers = None, None
                for _ in range(repeats):
                    outputs = launched(j, inp)
                    if oracle is None:
                        oracle = expected(inp, outputs)
                        tiers = {tier: naive_values(inp) for tier in TIERS}
                    for name, actual in outputs.items():
                        want = oracle[name]
                        if message := contract(name, actual, want, name in exact):
                            failures.append(f"draw {j}: {message}")
                            if actual.shape != want.shape:
                                continue
                        if name in exact:
                            continue
                        judge.add(name, j, actual, want, {tier: tiers[tier][name] for tier in TIERS})
                del inp, oracle, tiers
        finally:
            random.setstate(state[0])
            torch.set_rng_state(state[1])
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
        failures = failures + judge.failures()
    judge.record(record_property)
    assert not failures, "\n".join(failures)
    return judge


def judge_lm_head(
    record_property,
    *,
    make: Callable[[int], Inputs],
    launch: Callable[[Inputs], Mapping],
    entry: str = "loss",
    seed: int,
    names: Optional[Sequence[str]] = None,
    forms: Optional[Sequence[str]] = None,
    route: Optional[Callable[[Inputs, Mapping], None]] = None,
    contract: Optional[Callable[[Mapping, Inputs], None]] = None,
    exact_rows: bool = True,
    oracle_kw: _Kw = None,
    naive_kw: _Kw = None,
    views: Optional[Mapping[str, _View]] = None,
    repeats: int = 1,
    draws: int = DRAWS,
) -> TierJudge:
    """Judge one LM-head case over ``draws`` draws against the naive BF16 tier, record every output's verdict with
    ``record_property`` and assert that every output met the tier and every draw kept its contract.

    * ``make(s)``: the draw's inputs from its seed alone (:func:`judge_inputs`).
    * ``launch(inp)``: the kernel's result (``loss`` / ``logp`` / ``dX`` / ``dW`` ...), or with ``forms`` one result
      per form name.  Called ``repeats`` times per draw; the judge keeps each output's largest error.
    * ``names``: the judged outputs of each result (default ``loss``, ``logp``, ``dX``, ``dW``; the
      log-probability entry: ``logp``, ``dX``, ``dW``, as the tests make that entry's loss themselves).  With
      ``forms`` the output ``n`` of form ``f`` is judged as ``n@f``.
    * ``views``: an output whose oracle and naive values are a view of another one: ``{name: (key, select)}``,
      ``select(inp, value)`` (or ``None``) picks the result's form from the full value of ``key`` (``lse`` of the
      compacted rows: ``("lse", lambda inp, v: v[inp.valid])``; an FP32 ``dX`` accumulator: ``("dX", None)``).
    * ``route(inp, results)`` asserts the launch's route on every draw; ``contract(result, inp)`` holds each result
      to its contract (default :func:`contract_lm_head` of ``entry``).
    * ``exact_rows``: judge ``logp`` and ``dX`` of the ignored rows exactly 0 (``logp_ignored`` / ``dX_ignored``).
    * ``oracle_kw`` / ``naive_kw``: more keywords of :func:`reference_fp64` / :func:`naive_bf16` (``scale``,
      ``over``), or a function of the inputs that returns them.

    One float64 oracle and one naive value per draw serve every form and launch.  Returns the judge."""
    if names is None:
        names = ("loss", "logp", "dX", "dW") if entry == "loss" else ("logp", "dX", "dW")
    keys = {dict(views or {}).get(n, (n, None))[0] for n in names}
    if entry != "loss" and "loss" in keys:
        raise ValueError("the log-probability entry's loss is made by the test and is not judged")
    need = dict(need_dx="dX" in keys, need_dw="dW" in keys)
    if contract is None:

        def contract(result, inp):
            contract_lm_head(result, inp, entry=entry)

    def oracle_fn(inp):
        return reference_fp64(inp, **{"entry": entry, **need, **_kw(oracle_kw, inp)})

    def naive_fn(inp):
        return naive_bf16(inp, **{"entry": entry, **need, **_kw(naive_kw, inp)})

    return _judge(
        record_property,
        make=make,
        launch=launch,
        seed=seed,
        names=tuple(names),
        forms=forms,
        views=views,
        route=route,
        contract_fn=contract,
        exact_fn=_lm_head_exact if exact_rows else (lambda inp, result: {}),
        exact_names=("logp_ignored", "dX_ignored") if exact_rows else (),
        oracle_fn=oracle_fn,
        naive_fn=naive_fn,
        repeats=repeats,
        draws=draws,
    )


def judge_support(
    record_property,
    *,
    make: Callable[[int], SupportInputs],
    launch: Callable[[SupportInputs], Mapping],
    outputs: Sequence[str],
    seed: int,
    names: Optional[Sequence[str]] = None,
    forms: Optional[Sequence[str]] = None,
    route: Optional[Callable[[SupportInputs, Mapping], None]] = None,
    contract: Optional[Callable[[Mapping, SupportInputs], None]] = None,
    exact_rows: bool = True,
    upstream: _Kw = None,
    views: Optional[Mapping[str, _View]] = None,
    repeats: int = 1,
    draws: int = DRAWS,
) -> TierJudge:
    """:func:`judge_lm_head` for the candidate-support entries: the oracle is :func:`reference_support_fp64` and the
    naive tier :func:`naive_support_bf16` of the requested ``outputs`` (``packed`` / ``selected`` / ``kl``).

    ``names`` defaults to ``outputs`` plus ``dX`` and ``dW``; ``upstream`` is the upstream gradients
    ``{name: tensor or None}`` the launch back-propagates (default the inputs' ``g`` / ``d`` / ``e``), or a function
    of the inputs that returns them; ``contract`` defaults to :func:`contract_support`.  ``exact_rows`` judges the
    selected ``logp`` of the ignored rows and the packed ``logp`` of the single-candidate rows exactly 0
    (``selected_ignored`` / ``packed_single``)."""
    for name in outputs:
        if name not in SUPPORT_OUTPUTS:
            raise ValueError(f"unknown output {name!r}")
    if names is None:
        names = (*outputs, "dX", "dW")
    keys = {dict(views or {}).get(n, (n, None))[0] for n in names}
    need = dict(need_dx="dX" in keys, need_dw="dW" in keys)
    if contract is None:

        def contract(result, inp):
            contract_support(result, inp, outputs=outputs)

    def up(inp):
        if upstream is None:
            return None
        return dict(upstream(inp) if callable(upstream) else upstream)

    def oracle_fn(inp):
        return reference_support_fp64(inp, outputs=tuple(outputs), upstream=up(inp), **need)

    def naive_fn(inp):
        return naive_support_bf16(inp, outputs=tuple(outputs), upstream=up(inp), **need)

    return _judge(
        record_property,
        make=make,
        launch=launch,
        seed=seed,
        names=tuple(names),
        forms=forms,
        views=views,
        route=route,
        contract_fn=contract,
        exact_fn=_support_exact if exact_rows else (lambda inp, result: {}),
        exact_names=("selected_ignored", "packed_single") if exact_rows else (),
        oracle_fn=oracle_fn,
        naive_fn=naive_fn,
        repeats=repeats,
        draws=draws,
    )
