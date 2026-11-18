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

Checkpoint hand-off suite of the Cake non-deterministic packed KDA chunked training package
(``cake_pplx._backends.cake_kda_nondet_train``: ``cake_kda_nondet(..., checkpoint_handoff="auto" | "on" |
"off")``).  On a GPU: under both checkpoint modes the hand-off's arm and ``"off"``'s arm (``out`` and the five
gradients) are each judged against the exact FP64 reference beside the naive BF16 tier (``tests/cake_naive.py``) over
the judge's draws, on every plan family of the device the packs below reach, with both kinds of record (a tape
record: the tapes; a row record: the recompute launches the intra twin, ``cake_jit.HANDOFF_TWINS``, and the record
also keeps the prep's rows), also with the tape WY twin (``cake_jit.TAPE_TWINS``) on the rows whose record keeps no
``w``; stacked layers are judged layer by layer on the operands each node read.  The plumbing stays exact: the record
is taken (the counters), its tape fields are the tape forward's, its pointers and bytes are the plan's, the launch
lists, stale or refused records fall back, the byte budget demotes and refuses, and ``"auto"`` engages exactly where
``HANDOFF_HEADS``, ``HANDOFF_VERIFIED``, the size bounds (``HANDOFF_MIN_HEAD_TOKENS``, ``HANDOFF_MIN_DOC_TOKENS``) and
``HANDOFF_ROWS_VERIFIED`` say.  A row above ``JUDGE_MAX_TOKENS`` tokens (P8 and a few others) keeps its exact checks
only, its results finite; each such test names the rows that judge its route.  The package has no bit contract: no
result is compared with another result.  Host only: the rules, the admission, the mode parsing and torch's recompute,
capture, compile and dispatch-mode state as the hand-off reads it.

Like the backend suite it names its package through the shared alias helper and imports nothing else from
``cake_pplx``.
"""

import contextlib
import dataclasses
import json
import os
import random
import re
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed, seed_all
from tests.experimental.test_helpers import cake_kda_known_gaps as known_gaps
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
)
from tests.experimental.test_helpers.cake_kda_naive import (
    draw_case,
    flat_inputs,
    judge_arms,
    lengths_of,
    naive_of,
    oracle_with_grads,
)

cake_backend = cake_package("cake.cake_kda_nondet_train", "cake_backend")
cake_build = cake_package("cake.cake_kda_nondet_train", "cake_build")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
CakeKDANondetWorkspace = cake_backend.CakeKDANondetWorkspace
D, SCALE = cake_backend.D, cake_backend.SCALE
ARCHES = tuple(cake_jit.ARCHES)
# the environment override the package no longer has: setting it changes nothing
ENV = "CAKE_KDA_CKPT_HANDOFF"

LEAVES = ("q", "k", "v", "g", "beta")
# the gradient of each leaf, in LEAVES order
GRAD_OF_LEAF = ("dq", "dk", "dv", "dg", "dbeta")
# the order the backward returns its results
RETURN_ORDER = ("dq", "dk", "dv", "dbeta", "dg")
# A judged draw runs the FP64 reference and the naive BF16 tier token by token: rows of up to this many tokens are
# judged at any head count, and the production-size rows below (one per route, up to about 16k tokens at up to four
# heads); a larger row keeps its exact checks only, its results finite.
JUDGE_MAX_TOKENS = 4096
PRODUCTION_ROWS = frozenset({((65,) * 257, 2)})
# the seed of every judged case and of the inputs of the exact-only rows (each draw's seed derives from it)
SEED = 7818
TAIL_STAGES = ("dqkg", "glm_intra", "gate_epilogue")

# A four-document pack of 261k tokens (P8) and its head counts.
P8 = (23496, 88510, 114974, 34413)
P8_HEADS = (1, 2, 4, 8, 16, 32)
# P8 at 64 heads: opt-in, one checkpoint mode per process (run each id in its own pytest process, for memory).
P8_H64_ENV = "CAKE_KDA_HANDOFF_TEST_P8_H64"
# Short documents, unaligned tails, zero-length documents and pad chunks on both routes: the overlap pair's rows
# (148 SMs), the separate dav program (12 heads), the 16-head rows whose light set carries w / u (sm_100a paired
# serial rows; sm_103a / sm_107a above 32 chains), the fused serial forward (32 heads: no forward w on sm_100a /
# sm_103a) and the 64-column fused recompute (64 heads); the rows where 148- / 152-SM sm_100a runs the ``wy_fwd``
# record ((65,) at 4 heads, 256 documents at 16 heads); more than 256 documents (the table / seg_table launches and
# the intra program past its 256-document form) on the segmented route (2 heads), at 16 heads (serial on sm_100a,
# segmented on sm_103a / sm_107a) and on the serial route (32 heads).  The family sweep below adds a row of every
# other plan family the device's plans reach.
SHORT_ROWS = [
    ((1,), 1),
    ((63,), 2),
    ((65,), 4),
    ((1000,), 1),
    ((63, 1, 65), 2),
    ((100, 0, 300), 2),
    ((100, 640, 77, 1000, 129), 2),
    ((63, 1, 65, 30), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
    ((300, 77), 12),
    ((129,), 16),
    ((129, 300, 70), 16),
    ((65, 200), 32),
    ((63, 1, 65), 64),
    (tuple(range(1, 257)), 16),
    ((65,) * 257, 2),
    ((65,) * 257, 16),
    ((65,) * 257, 32),
]
# The launch-sequence rows: both routes, the overlap pair, w from the light set, no forward w, the fused recompute
# and a pack that saves the segment states (light_states: the record keeps the set's transfers).
LAUNCH_ROWS = [
    ((100, 640, 77, 1000, 129), 2),
    ((63, 1, 65, 30), 2),
    ((129, 300, 70), 16),
    ((65, 200), 32),
    ((63, 1, 65), 64),
    (P8, 4),
]
LAYER_ROWS = [((100, 640, 77, 1000, 129), 2), ((129, 63), 16), ((65, 200), 32)]
# The rows whose record may keep no forward w (its tape backward launches wy): 32 and 64 heads, P8 at 32 heads among
# them (P8 at 64 heads: opt-in, below)
TAPE_WY_ROWS = [(P8, 32), ((65, 200), 32), ((63, 1, 65), 64), ((65,) * 257, 32)]
# The rows whose record may keep the forward's w, with both forward WY records (``wy_fwd``, ``wy_fwd_rows``)
WY_ROWS = [
    ((65,), 4),
    ((63, 1, 65, 30), 2),
    ((100, 640, 77, 1000, 129), 2),
    ((129, 300, 70), 16),
    (tuple(range(1, 257)), 16),
    ((65,) * 257, 2),
    (P8, 1),
    (P8, 4),
]
# The family sweep: candidate packs across the document-count, chain and segment thresholds, at one head and at
# every head count "auto" engages at (HANDOFF_HEADS); the cheapest candidate (tokens x heads) of each family runs,
# and every family must have a candidate up to P8 at 32 heads (the records read the document count, the heads, the
# SM count and the segments; the token count enters only through the segmented route's saved-states rule).
FAMILY_PACKS = [
    (65,),
    (1000,),
    (63, 1, 65),
    (63, 1, 65, 30),
    (100, 640, 77, 1000, 129),
    (20,) * 7,
    (129, 300, 70),
    (65,) * 33,
    (65,) * 64,
    (300,) * 100,
    tuple(range(1, 257)),
    (65,) * 257,
    (129,) * 300,
    (4000,) * 16,
    (2048,) * 64,
    P8,
]
FAMILY_HEADS = (1, *sorted(cake_backend.HANDOFF_HEADS))
FAMILY_CAP = sum(P8) * 32
# The intra twins (intra record -> twin) and the bytes per internal token-head a row record adds: gk (FP32) + kg
# (BF16) + beta_int (FP32), and kb (BF16) where the record keeps no w; the gk and beta_int arena slots its backward
# layout drops.
TWINS = dict(cake_jit.HANDOFF_TWINS)
ROW_BYTES = D * 4 + D * 2 + 4
ROW_KB_BYTES = D * 2
ROW_ARENA_SAVING = D * 4 + 4


def _shape_id(shape) -> str:
    lengths, heads = shape
    if tuple(lengths) == P8:
        return f"P8-h{heads}"
    if len(lengths) > 8:
        return f"n{len(lengths)}-S{sum(lengths)}-h{heads}"
    return "s" + "_".join(str(n) for n in lengths) + f"-h{heads}"


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {ARCHES}"
        )
    return torch.device("cuda", index)


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _manifest_or_skip()


@pytest.fixture(params=("tape", "rows"), ids=("kind_tape", "kind_rows"))
def record_kind(request, monkeypatch) -> str:
    """The kind of record the test's hand-offs publish, forced through ``_HANDOFF_ROWS`` (``"rows"`` skips the
    device, workspace and build checks of the rule; the byte budget still applies)."""
    monkeypatch.setattr(cake_backend, "_HANDOFF_ROWS", request.param == "rows")
    return request.param


def _with_rows(p, record_kind: str | None) -> bool:
    """Whether a record of plan ``p`` of the forced kind ``record_kind`` is a row record."""
    return record_kind == "rows" and p.handoff_twin is not None


@pytest.fixture(params=(False, True), ids=("tape_wy_record", "tape_wy_twin"))
def tape_wy(request, monkeypatch) -> bool:
    """Whether the tape backward launches the w-only twin of its WY record (``cake_jit.TAPE_TWINS``), forced
    through ``_TAPE_WY`` (the device, build and load checks of the rule skipped)."""
    monkeypatch.setattr(cake_backend, "_TAPE_WY", request.param)
    return request.param


def _tape_wy_runs(monkeypatch) -> list:
    """Record the ``record`` override of every ``wy`` launch on the tape path through ``CakeKDANondetWorkspace.run`` (``None``:
    the plan's WY record)."""
    runs: list = []
    real_run = CakeKDANondetWorkspace.run

    def run(self, stage, values, *, path, **kwargs):
        if stage == "wy" and path == "tape":
            runs.append(kwargs.get("record"))
        return real_run(self, stage, values, path=path, **kwargs)

    monkeypatch.setattr(CakeKDANondetWorkspace, "run", run)
    return runs


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    """Every test starts and ends with an empty hand-off registry, fresh counters and no environment override."""
    monkeypatch.delenv(ENV, raising=False)
    cake_backend._handoff_reset()
    yield
    cake_backend._handoff_reset()


# the package's autograd entry (it takes no mode argument)
cake_kda_nondet = cake_backend.cake_kda_nondet


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality: ``torch.equal`` on the integer views, so signed zeros and NaN payloads count."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        a.element_size()
    ]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _fill_nan(t: torch.Tensor) -> torch.Tensor:
    """NaN in every element of a floating-point tensor (one bit pattern per dtype): an element no launch writes
    stays recognizable, where a zero fill would equal a written zero."""
    if t.is_floating_point():
        t.fill_(float("nan"))
    return t


def _nan_alloc(shape_, dtype, dev) -> torch.Tensor:
    """``_alloc`` with floating-point tensors NaN-filled; integer tensors (the tables, offsets) zero-filled as
    before, so a stray index read cannot fault."""
    if dtype.is_floating_point:
        return torch.full(shape_, float("nan"), dtype=dtype, device=dev)
    return torch.zeros(shape_, dtype=dtype, device=dev)


@contextlib.contextmanager
def _poisoned():
    """The record's per-call tensors NaN-filled before the recompute's launches (``_HANDOFF_POISON``)."""
    old = cake_backend._HANDOFF_POISON
    cake_backend._HANDOFF_POISON = True
    try:
        yield
    finally:
        cake_backend._HANDOFF_POISON = old


def _stats(device: torch.device) -> dict:
    return cake_backend._handoff_stats(device.index)


def _plan(lengths, heads, device):
    return cake_backend.plan(
        sum(lengths),
        len(lengths),
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
    )


def _smallest_admitted(heads: int) -> int:
    """The tokens of the smallest one-document problem the size bounds admit at ``heads`` heads."""
    return max(
        cake_backend.HANDOFF_MIN_DOC_TOKENS,
        -(-cake_backend.HANDOFF_MIN_HEAD_TOKENS // heads),
    )


def _family(p) -> tuple:
    """The plan family of a row: the route, the light path, the head count and the records of the saving forward,
    the tape backward and the light backward (the SM count enters through the records it selects)."""
    records = tuple(
        (path, stage, name)
        for path in ("fwd", "tape", p.light_path)
        for stage, name in sorted(p.records[path].items())
        if name is not None
    )
    return (p.recur, p.light_path, p.H, records)


def _leaves(inputs: dict) -> list:
    return [inputs[n].detach().clone().requires_grad_(True) for n in LEAVES]


def _grads(leaves) -> dict:
    return {name: leaf.grad for name, leaf in zip(GRAD_OF_LEAF, leaves, strict=True)}


def _checkpointed_step(inputs: dict, *, reentrant: bool, handoff, workspace=None):
    """One forward of ``cake_kda_nondet`` under ``torch.utils.checkpoint`` and its backward on fresh leaves: ``out``
    (detached) and the leaves' gradients by name.  ``handoff=None`` passes no ``checkpoint_handoff`` (the
    default)."""
    leaves = _leaves(inputs)
    cu = inputs["cu_seqlens"]
    kw = {} if handoff is None else {"checkpoint_handoff": handoff}

    def layer(q, k, v, g, beta):
        return cake_kda_nondet(q, k, v, g, beta, cu, workspace=workspace, **kw)

    out = checkpoint(layer, *leaves, use_reentrant=reentrant)
    out.backward(inputs["dout"])
    return out.detach(), _grads(leaves)


def _results(out: torch.Tensor, grads: dict) -> dict:
    """One arm's results for the judge: ``out`` and the gradients by name."""
    return {"out": out, **grads}


def _inputs(lengths, heads: int, device, seed: int = SEED) -> dict:
    """The operands of draw seed ``seed`` (:func:`draw_case`: the activated FP32 gate)."""
    return draw_case(lengths, heads, "g", False, seed, device)


def _judged(lengths, heads: int) -> bool:
    """Whether a row is judged: up to :data:`JUDGE_MAX_TOKENS` tokens, or a production-size row of
    :data:`PRODUCTION_ROWS`; a larger row keeps its exact checks only."""
    return (
        sum(lengths) <= JUDGE_MAX_TOKENS or (tuple(lengths), heads) in PRODUCTION_ROWS
    )


def _finite(t: torch.Tensor) -> bool:
    """Whether every element of ``t`` is finite, in blocks of rows (bounded temporaries on the large rows)."""
    rows = t.reshape(t.shape[0], -1) if t.dim() and t.numel() else t.reshape(-1, 1)
    step = max(1, (1 << 24) // max(1, rows.shape[1]))
    return all(
        bool(torch.isfinite(rows[i : i + step]).all())
        for i in range(0, rows.shape[0], step)
    )


def _check_finite(results: dict, tag) -> None:
    """Every result of an arm too large to judge is finite: its contract (the oracle of finite inputs is finite)."""
    assert results, tag
    for name, t in results.items():
        assert _finite(t), (tag, name)


def _judge(record_property, draw) -> None:
    """Judge a case's arms against the FP64 reference beside the naive BF16 tier over the judge's draws."""
    judge_arms(record_property, draw, seed=SEED, scale=SCALE)


def _run_exact(draw) -> None:
    """A row too large to judge: its draw once, with its exact checks, and every result it returns finite."""
    arms, _ = draw(0, SEED)
    for arm, results in arms.items():
        _check_finite(results, arm)


def _handoff_arms(
    inputs,
    device,
    lengths,
    heads,
    *,
    reentrant: bool,
    record_kind: str | None = None,
    keep: bool = True,
    prefix: str = "",
) -> dict:
    """``"off"`` then ``"on"`` (the record's per-call tensors and the backward's ``dv2`` NaN-filled before the
    launches that write them) under one checkpoint mode: the off arm publishes nothing, the on arm publishes one
    record (of ``record_kind`` where the test forces one) and its backward takes it, and the registry is empty after
    each backward.  Returns the arms ``<prefix><mode>.off`` / ``<prefix><mode>.on`` (``mode`` the checkpoint mode:
    ``out`` and the five gradients) for the judge; with ``keep=False`` (a row too large to judge) each arm's results
    are checked finite as soon as they exist and dropped, and nothing is returned."""
    mode = "reentrant" if reentrant else "nonreentrant"
    tag = (_shape_id((lengths, heads)), mode, f"kind={record_kind}")
    cake_backend._handoff_reset()
    off = _results(*_checkpointed_step(inputs, reentrant=reentrant, handoff="off"))
    s = _stats(device)
    assert s["published"] == 0 and s["live_records"] == 0, (tag, "off", s)
    if not keep:
        _check_finite(off, (*tag, "off"))
        off = None
    with _poisoned():
        on = _results(*_checkpointed_step(inputs, reentrant=reentrant, handoff="on"))
    s = _stats(device)
    assert s["published"] == 1 and s["hits"] == 1 and s["misses"] == 0, (tag, s)
    assert s["live_records"] == 0 and s["live_bytes"] == 0, (tag, s)
    if record_kind is not None:
        with_rows = _with_rows(_plan(lengths, heads, device), record_kind)
        assert s["published_rows"] == int(with_rows) and s["demoted"] == 0, (tag, s)
    if not keep:
        _check_finite(on, (*tag, "on"))
        return {}
    return {f"{prefix}{mode}.off": off, f"{prefix}{mode}.on": on}


def _like(t: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    # the reference's / the tier's tensor in the result's shape (a reshape of the same elements), else unchanged
    if t.shape != actual.shape and t.numel() == actual.numel():
        return t.reshape(actual.shape)
    return t


# the operands the FP64 reference and the naive tier read
ORACLE_OPERANDS = ("q", "k", "v", "g", "beta", "dout", "cu_seqlens")


def _judge_layers(record_property, draw) -> None:
    """:func:`judge_arms` for stacks of KDA layers, whose layers read different operands: ``draw(j, s)`` returns
    ``(arms, inputs)`` pairs, each ``inputs`` the operands of one layer's call (its ``v`` the previous layer's output,
    its ``dout`` the gradient that reached its output: :func:`_tapped`) and each output name of each arm in one pair
    only; every pair is judged against the FP64 reference of its own operands beside the naive BF16 tier
    (formed once per draw for each distinct set of operands).  The judge, its records and the settling against the
    owner-accepted gaps are :func:`judge_arms`'s."""
    judge = TierJudge(DRAWS)
    failures: list[str] = []
    lengths = None
    cuda = torch.cuda.get_rng_state_all()
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(judge.draws):
            s = draw_seed(SEED, j)
            seed_all(s)
            formed: list[tuple[dict, dict, dict]] = []
            for arms, inputs in draw(j, s):
                flat = flat_inputs(inputs)
                if lengths is None:
                    lengths = lengths_of(flat)
                # operand identity (not a result): the reference and the tier of the same operands are formed once
                found = next(
                    (
                        (ref, tier)
                        for seen, ref, tier in formed
                        if all(torch.equal(seen[n], flat[n]) for n in ORACLE_OPERANDS)
                    ),
                    None,
                )
                if found is None:
                    found = (
                        oracle_with_grads(flat, SCALE, LOWER_BOUND, False),
                        naive_of(flat, SCALE, LOWER_BOUND, False),
                    )
                    formed.append((flat, *found))
                ref, tier = found
                for arm, results in arms.items():
                    for name, actual in results.items():
                        key = f"{arm}.{name}"
                        want = _like(ref[name], actual)
                        if message := contract(key, actual, want):
                            failures.append(f"draw {j}: {message}")
                            if actual.shape != want.shape:
                                continue
                        judge.add(
                            key, j, actual, want, {"bf16": _like(tier[name], actual)}
                        )
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        torch.cuda.set_rng_state_all(cuda)
    failures += judge.failures()
    judge.record(record_property)
    known_gaps.settle(judge.verdicts(), failures, lengths=lengths)


class _Tap(torch.autograd.Function):
    """A copy that records the gradient reaching it in ``sink[key]``: on a layer's ``v`` operand the ``dv`` its node
    returns, on its output the ``dout`` its node receives."""

    @staticmethod
    def forward(ctx, x, sink, key):
        ctx.sink, ctx.key = sink, key
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.sink[ctx.key] = grad.detach().clone()
        return grad, None, None


def _tapped(i: int, sink: dict, kda, v_in: torch.Tensor) -> torch.Tensor:
    """``kda(v)`` of layer ``i`` with its ``v`` operand and its output tapped (:class:`_Tap`).  ``sink`` keeps the ``v``
    of the last call that reached the node (a checkpoint's recompute, whose saved ``v`` the backward reads; a
    recompute may stop early, once the node's tensors are saved), the ``out`` of the last call that returned with
    that call's ``v``, and the gradients that reach the tapped ``v`` and output (the node's ``dv`` and its
    ``dout``)."""
    v = _Tap.apply(v_in, sink, (i, "dv"))
    seen = v.detach().clone()
    sink[(i, "v")] = seen
    o = kda(v)
    sink[(i, "out")] = (seen, o.detach().clone())
    return _Tap.apply(o, sink, (i, "dout"))


def _layer_results(sets, leaves, sink: dict, layers: int, prefix: str) -> list:
    """Per layer, two ``({"<prefix>layer<i>": results}, operands)`` pairs for :func:`_judge_layers`: ``out`` with the
    operands of the call that returned it, and the five gradients (``dv`` the gradient the node returned for its
    ``v``) with the operands the node's backward read."""
    pairs = []
    for i in range(layers):
        q, k, _v, g, beta = leaves[i]
        grads_inputs = {**sets[i], "v": sink[(i, "v")], "dout": sink[(i, "dout")]}
        out_v, out = sink[(i, "out")]
        arm = f"{prefix}layer{i}"
        pairs.append(({arm: {"out": out}}, {**grads_inputs, "v": out_v}))
        grads = {
            "dq": q.grad,
            "dk": k.grad,
            "dv": sink[(i, "dv")],
            "dg": g.grad,
            "dbeta": beta.grad,
        }
        pairs.append(({arm: grads}, grads_inputs))
    return pairs


# --------------------------------------------------------------------------- the hand-off judged beside off


@pytest.mark.parametrize("heads", P8_HEADS, ids=lambda h: f"P8-h{h}")
def test_handoff_takes_its_record_on_the_p8_pack(
    device, manifest, heads, record_kind
) -> None:
    """The P8 pack (261k tokens) with both kinds of record, under non-reentrant and reentrant checkpointing: the off
    arm publishes nothing, the on arm (its record's per-call tensors and the backward's ``dv2`` NaN-filled)
    publishes one record of the kind and its backward takes it, the registry is empty after each backward, and every
    result is finite.  The exact checks only: a judged draw runs the FP64 and the BF16 token recurrences, so the
    hand-off of both routes at these head counts is judged on the short packs
    (:func:`test_handoff_gradients_are_judged_on_short_packs`)."""
    inputs = _inputs(P8, heads, device)
    for reentrant in (False, True):
        _handoff_arms(
            inputs,
            device,
            P8,
            heads,
            reentrant=reentrant,
            record_kind=record_kind,
            keep=False,
        )


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_handoff_takes_its_record_on_the_p8_pack_at_64_heads(
    device, manifest, reentrant, record_kind
) -> None:
    """P8 at 64 heads: the same exact checks, one (checkpoint mode, record kind) per process (each id in its own
    pytest process, for memory).  Opt-in by ``CAKE_KDA_HANDOFF_TEST_P8_H64=1``.  The 64-head hand-off is judged on
    the short pack ``(63, 1, 65)`` at 64 heads."""
    if os.environ.get(P8_H64_ENV) != "1":
        pytest.skip(f"opt-in: {P8_H64_ENV}=1, one id per process")
    inputs = _inputs(P8, 64, device)
    _handoff_arms(
        inputs,
        device,
        P8,
        64,
        reentrant=reentrant,
        record_kind=record_kind,
        keep=False,
    )


@pytest.mark.parametrize("shape", SHORT_ROWS, ids=_shape_id)
def test_handoff_gradients_are_judged_on_short_packs(
    device, manifest, shape, record_kind, record_property
) -> None:
    """Short documents, unaligned tails, zero-length documents, pad chunks and more than 256 documents on both routes
    (forward w kept or not, w from the light set, the transfers per call, the table launches, and with row records
    both intra twins and the ``dv2`` pad-row fill kernel): under non-reentrant and reentrant checkpointing the off arm
    and the hand-off's on arm (``<mode>.off`` / ``<mode>.on``: ``out`` and the five gradients) are judged, with the
    record's exact checks on every draw.  The 257 documents at two heads (about 16k tokens) are the one
    production-size row judged here.  The 256 documents at 16 heads and the 257 documents at 16 and 32 heads keep the
    exact checks only (their results finite): the one document of 65 tokens at 4 heads judges the ``wy_fwd`` record
    of 148- / 152-SM ``sm_100a``, the rows at 16 and 32 heads judge those routes, and the 257 documents at two heads
    judge the more-than-256-documents form."""
    lengths, heads = shape
    keep = _judged(lengths, heads)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        arms = {}
        for reentrant in (False, True):
            arms.update(
                _handoff_arms(
                    inputs,
                    device,
                    lengths,
                    heads,
                    reentrant=reentrant,
                    record_kind=record_kind,
                    keep=keep,
                )
            )
        return arms, inputs

    if keep:
        _judge(record_property, draw)
    else:
        _run_exact(draw)


def test_handoff_is_judged_on_every_plan_family_of_the_device(
    device, manifest, record_kind, record_property
) -> None:
    """The plan families (:func:`_family`) the device's plans reach over ``FAMILY_PACKS`` x ``FAMILY_HEADS``: each
    has a row up to P8 at 32 heads, and the cheapest row of each family runs the off arm and the hand-off's on arm
    under both checkpoint modes with the test's kind of record, with the record's exact checks.  A row of up to
    ``JUDGE_MAX_TOKENS`` tokens is judged (``<row>.<mode>.off`` / ``<row>.<mode>.on``); a larger one keeps the exact
    checks only (its results finite; the hand-off of its route is judged on the short packs).  Every family is
    checked and every failure reported together."""
    sms, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    families: dict[tuple, tuple[int, tuple[int, ...], int]] = {}
    for lengths in FAMILY_PACKS:
        for heads in FAMILY_HEADS:
            try:
                p = cake_backend.plan(sum(lengths), len(lengths), heads, sms, arch)
            except ValueError:
                continue
            cost = sum(lengths) * heads
            key = _family(p)
            if key not in families or cost < families[key][0]:
                families[key] = (cost, lengths, heads)
    assert families
    too_big = [_shape_id(row[1:]) for row in families.values() if row[0] > FAMILY_CAP]
    assert not too_big, ("families without a row up to the cap", too_big)
    failures = []
    judged = 0
    for _cost, lengths, heads in sorted(families.values()):
        sid = _shape_id((lengths, heads))
        keep = sum(lengths) <= JUDGE_MAX_TOKENS

        def draw(j, s, lengths=lengths, heads=heads, sid=sid, keep=keep):
            inputs = _inputs(lengths, heads, device, s)
            arms = {}
            for reentrant in (False, True):
                arms.update(
                    _handoff_arms(
                        inputs,
                        device,
                        lengths,
                        heads,
                        reentrant=reentrant,
                        record_kind=record_kind,
                        keep=keep,
                        prefix=f"{sid}.",
                    )
                )
            return arms, inputs

        try:
            if keep:
                _judge(record_property, draw)
                judged += 1
            else:
                _run_exact(draw)
        except AssertionError as exc:
            failures.append((sid, str(exc)[:1000]))
        torch.cuda.empty_cache()
    assert not failures, (
        f"{len(failures)} of {len(families)} family checks failed",
        failures,
    )
    assert judged > 0


@pytest.mark.parametrize("shape", TAPE_WY_ROWS, ids=_shape_id)
def test_the_tape_wy_twin_hand_off_is_judged(
    device, manifest, monkeypatch, shape, record_kind, record_property
) -> None:
    """The rows whose record may keep no forward ``w`` with the tape WY twin forced (``_TAPE_WY = True``), with both
    kinds of record: under both checkpoint modes the off arm and the hand-off's on arm are judged, with the record's
    exact checks, and the backward from the record launches the twin in place of its WY record exactly where the
    record keeps no ``w`` (one launch per backward, as the record's).  P8 at 32 heads and the 257 documents at 32 heads
    keep the exact checks only (their results finite): the twin's hand-off on the serial route is judged on
    ``(65, 200)`` at 32 heads and ``(63, 1, 65)`` at 64 heads."""
    lengths, heads = shape
    p = _plan(lengths, heads, device)
    launches = p.dispatched("wy", "tape") and not p.handoff_keeps_w
    assert not launches or p.tape_wy_twin is not None, (p.arch, p.record("wy", "tape"))
    monkeypatch.setattr(cake_backend, "_TAPE_WY", True)
    runs = _tape_wy_runs(monkeypatch)
    keep = _judged(lengths, heads)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        arms = {}
        for reentrant in (False, True):
            runs.clear()
            arms.update(
                _handoff_arms(
                    inputs,
                    device,
                    lengths,
                    heads,
                    reentrant=reentrant,
                    record_kind=record_kind,
                    keep=keep,
                )
            )
            assert runs == ([p.tape_wy_twin] if launches else []), (reentrant, runs)
        return arms, inputs

    if keep:
        _judge(record_property, draw)
    else:
        _run_exact(draw)


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_the_tape_wy_twin_hands_off_on_p8_h64(
    device, manifest, monkeypatch, reentrant, record_kind
) -> None:
    """P8 at 64 heads with the tape WY twin forced: the exact checks (results finite), one (checkpoint mode, record
    kind) per process (each id in its own pytest process, for memory).  Opt-in by ``CAKE_KDA_HANDOFF_TEST_P8_H64=1``.
    The twin's hand-off at 64 heads is judged on ``(63, 1, 65)``."""
    if os.environ.get(P8_H64_ENV) != "1":
        pytest.skip(f"opt-in: {P8_H64_ENV}=1, one id per process")
    p = _plan(P8, 64, device)
    launches = p.dispatched("wy", "tape") and not p.handoff_keeps_w
    monkeypatch.setattr(cake_backend, "_TAPE_WY", True)
    runs = _tape_wy_runs(monkeypatch)
    inputs = _inputs(P8, 64, device)
    _handoff_arms(
        inputs,
        device,
        P8,
        64,
        reentrant=reentrant,
        record_kind=record_kind,
        keep=False,
    )
    assert runs == ([p.tape_wy_twin] if launches else []), runs


# --------------------------------------------------------------------------- the recompute's writes


def _check_layout(x, y, tag) -> None:
    """``x`` a tensor of ``y``'s dtype and shape; an integer tensor (a table) bitwise ``y``."""
    assert isinstance(x, torch.Tensor), tag
    assert (x.dtype, x.shape) == (y.dtype, y.shape), tag
    if not y.is_floating_point():
        assert _bits_equal(x, y), tag


def _check_set_layout(a, b, tag) -> None:
    """Two saved sets field for field: the same metadata, the same present tensors of the same dtypes and shapes, and
    the integer tensors (the tables) bitwise.  A floating tensor's values are judged through the backward that reads
    the set."""
    for field in dataclasses.fields(b):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(y, torch.Tensor):
            _check_layout(x, y, (tag, field.name))
        elif isinstance(y, tuple):
            assert isinstance(x, tuple) and len(x) == len(y), (tag, field.name)
            for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
                _check_layout(xi, yi, (tag, field.name, i))
        else:
            assert x == y, (tag, field.name)


@pytest.mark.parametrize("shape", LAUNCH_ROWS, ids=_shape_id)
def test_the_recompute_writes_are_the_tape_fields_and_out_is_judged(
    device, manifest, shape, record_kind, record_property
) -> None:
    """The forward with a hand-off request (``write_tapes = 1`` on the ``fwd`` records, the extras in per-call
    tensors; for a row record the intra twin), every floating-point per-call tensor NaN-filled at allocation
    (``_alloc``): its saved set has the light forward's metadata, present tensors, dtypes, shapes and tables; its
    record's ``h`` / ``v_new`` (and transfers) are bitwise the tape forward's (the record's tape fields); it keeps
    ``w`` exactly where the plan says; a row record keeps ``gk`` / ``kg`` / ``beta_int`` of the plan's shapes (``gk``
    fully written: no NaN of the fill left) and ``kb`` exactly where it keeps no ``w`` (a tape record none of them);
    its bytes are ``Plan.handoff_bytes(rows)`` and ``memory_report``'s.  ``out`` of the light forward and of the
    forward with the request and the gradients of the light backward from each forward's saved set (``light`` /
    ``request``) are judged.  P8 at 4 heads keeps the exact checks only (its results finite)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    p = _plan(lengths, heads, device)
    with_rows = _with_rows(p, record_kind)
    common = dict(
        scale=SCALE,
        save_for_backward=True,
        workspace=None,
        arena=None,
        doc_lengths=None,
    )

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        ops = [inputs[n] for n in (*LEAVES, "cu_seqlens")]
        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_backend, "_alloc", _nan_alloc)
            out, saved = cake_backend._fwd_call(*ops, save_mode=None, **common)
            request = {"task": 0, "index": device.index, "mode": "on"}
            out_h, saved_h = cake_backend._fwd_call(
                *ops, save_mode=None, handoff=request, **common
            )
            assert "tensors" in request, (lengths, heads)
            _check_set_layout(saved_h, saved, "request set")
            tensors = request["tensors"]
            _, tape = cake_backend._fwd_call(*ops, save_mode="tape", **common)
        # the record's tape fields: the tape forward's tapes bit for bit, the light set's own transfers and w
        assert _bits_equal(tensors["h"], tape.h)
        assert _bits_equal(tensors["v_new"], tape.v_new)
        if p.recur == "seg":
            assert _bits_equal(tensors["seg_m"], tape.seg_m)
            if p.seg_save:  # the record keeps the light set's own transfers
                assert tensors["seg_m"].data_ptr() == saved_h.seg_state[1].data_ptr()
        else:
            assert tensors["seg_m"] is None
        assert (tensors["w"] is not None) is p.handoff_keeps_w
        if p.light_saves_wu:  # the light set's own w
            assert tensors["w"].data_ptr() == saved_h.w.data_ptr()
        rows = (p.R, heads, D)
        row_names = ("gk", "kg", "beta_int", "kb")
        if with_rows:
            for name, shape_, dtype in (
                ("gk", rows, torch.float32),
                ("kg", rows, torch.bfloat16),
                ("beta_int", (p.R, heads), torch.float32),
                ("kb", rows, torch.bfloat16),
            ):
                if name in tensors:
                    t = tensors[name]
                    assert tuple(t.shape) == shape_ and t.dtype == dtype, name
            assert {"gk", "kg", "beta_int"} <= set(tensors)
            assert ("kb" in tensors) is not p.handoff_keeps_w
            # the twin writes every gk row, the pad chunks' too (zeros): no NaN of the fill is left
            assert not torch.isnan(tensors["gk"]).any()
            extra = sum(
                tensors[n].numel() * tensors[n].element_size()
                for n in row_names
                if n in tensors
            )
            assert extra == p.handoff_bytes(rows=True) - p.handoff_bytes()
        else:
            assert not set(row_names) & set(tensors), sorted(tensors)
        report = cake_backend.memory_report(
            S, N, heads, sm_count=p.sm_count, arch=p.arch
        )
        assert request["nbytes"] == p.handoff_bytes(rows=with_rows)
        assert (
            request["nbytes"]
            == report["handoff_bytes_rows" if with_rows else "handoff_bytes"]
        )
        assert request["key"] == cake_backend._handoff_key(p, SCALE)
        assert _stats(device)["published"] == 0  # filling a request publishes nothing
        del request, tensors, tape
        arms = {}
        for label, o, source in (("light", out, saved), ("request", out_h, saved_h)):
            grads = cake_backend._bwd_call(
                *ops,
                inputs["dout"],
                saved=source,
                scale=SCALE,
                workspace=None,
                arena=None,
                doc_lengths=None,
            )
            arms[label] = _results(o, dict(zip(RETURN_ORDER, grads, strict=True)))
        return arms, inputs

    if _judged(lengths, heads):
        _judge(record_property, draw)
    else:
        _run_exact(draw)


@pytest.mark.parametrize("shape", WY_ROWS, ids=_shape_id)
def test_the_kept_forward_w_is_the_tape_wy_w(
    device, manifest, monkeypatch, shape, tape_wy
) -> None:
    """Where the record keeps the forward's ``w`` (its backward launches no ``wy``), that ``w`` is bit for bit the
    ``w`` the tape backward's ``wy`` program writes (its WY record, or the record's w-only twin), pad rows included.
    Both start NaN-filled (the forward's through ``_alloc``, the backward's arena slot right before its ``wy``
    launch), so a row one program writes and the other does not differs."""
    lengths, heads = shape
    p = _plan(lengths, heads, device)
    if not p.handoff_keeps_w or not p.dispatched("wy", "tape"):
        pytest.skip(
            f"{p.arch} at {p.sm_count} SMs: the record keeps no forward w on this row (or the tape path runs no wy)"
        )
    inputs = make_inputs(lengths, heads, device)
    monkeypatch.setattr(cake_backend, "_alloc", _nan_alloc)
    ops = [inputs[n] for n in (*LEAVES, "cu_seqlens")]
    common = dict(scale=SCALE, save_for_backward=True, arena=None, doc_lengths=None)
    request = {"task": 0, "index": device.index, "mode": "on"}
    cake_backend._fwd_call(
        *ops, save_mode=None, workspace=None, handoff=request, **common
    )
    w_fwd = request["tensors"]["w"]
    captured: dict = {}

    def launcher(name, values, *, grid, device_index):
        wy = cake_jit.MODULES[name]["stage"] == "wy"
        if wy:
            _fill_nan(values["w"])
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)
        if wy:
            captured["w"], captured["record"] = values["w"].clone(), name

    # the tape set from a per-call forward; its backward on a tape-mode workspace whose launcher captures wy's w
    _, tape = cake_backend._fwd_call(*ops, save_mode="tape", workspace=None, **common)
    ws = CakeKDANondetWorkspace(
        p,
        device,
        backward=True,
        save_mode="tape",
        launcher=launcher,
    )
    cake_backend._bwd_call(
        *ops,
        inputs["dout"],
        saved=tape,
        scale=SCALE,
        workspace=ws,
        arena=None,
        doc_lengths=None,
    )
    assert "w" in captured, "the tape backward launched no wy"
    want_record = (p.tape_wy_twin if tape_wy else None) or p.record("wy", "tape")
    assert captured["record"] == want_record, (captured["record"], want_record)
    w_bwd = captured["w"]
    wy_fwd = p.record("wy_fwd", "fwd") if p.dispatched("wy_fwd", "fwd") else None
    info = (wy_fwd, captured["record"], tuple(w_fwd.shape), tuple(w_bwd.shape))
    assert w_bwd.numel() == w_fwd.numel(), info
    a = w_fwd.reshape(p.R, heads, D).contiguous().view(torch.int16)
    b = w_bwd.reshape(p.R, heads, D).contiguous().view(torch.int16)
    rows = (a != b).any(dim=2).any(dim=1).nonzero().flatten()
    assert rows.numel() == 0, (
        *info,
        f"{rows.numel()} of {p.R} rows differ",
        rows[:8].tolist(),
    )


# --------------------------------------------------------------------------- launches


def _expected_fwd(p, intra: str | None = None) -> list[str]:
    """The saving forward's launches (the original forward and a recompute with a tape record run the same records;
    ``intra``: the intra record a recompute with a row record launches instead of the plan's, its twin)."""
    seg = p.recur == "seg"
    seq: list[str] = []
    if p.dispatched("table", "fwd"):
        seq.append(p.record("table", "fwd"))
        if seg and p.dispatched("seg_table", "fwd"):
            seq.append(p.record("seg_table", "fwd"))
    seq.append(p.record("fwd_intra", "fwd") if intra is None else intra)
    if p.dispatched("wy_fwd", "fwd"):
        seq.append(p.record("wy_fwd", "fwd"))
    seq.append(p.record("fwdh", "fwd"))
    if seg:
        seq += [p.record("seg_carry", "fwd"), p.record("fwdh_p3", "fwd")]
    return seq


def _expected_bwd(
    p,
    path: str,
    *,
    prep_path=None,
    skip_wy=False,
    prep: bool = True,
    tape_wy: bool = False,
) -> list[str]:
    """A saved-set backward's launches on ``path`` (the set carries its tables; one tail pass); ``prep_path`` and
    ``skip_wy`` are the hand-off's light prep and its ``wy`` skip, ``prep=False`` a row record's backward (no
    prep launch: the record carries its rows), ``tape_wy`` a tape backward that launches the w-only twin of its WY
    record."""
    seg = p.recur == "seg"
    seq = [p.record("bwd_prep", prep_path or path)] if prep else []
    if p.dispatched("wy", path) and not skip_wy:
        twin = p.tape_wy_twin if (tape_wy and path == "tape") else None
        seq.append(twin or p.record("wy", path))
    if p.dispatched("fwdh", path):
        seq.append(p.record("fwdh", path))
        if seg:
            seq.append(p.record("seg_carry", path))
    if p.dispatched("fwdh_p3", path):
        seq.append(p.record("fwdh_p3", path))
    if p.dispatched("dav", path):
        seq.append(p.record("dav", path))
    dhu = p.record("dhu", path)
    seq += [dhu, p.record("seg_carry", path), dhu] if seg else [dhu]
    seq += [p.record(s, path) for s in TAIL_STAGES if p.dispatched(s, path)]
    return seq


@pytest.mark.parametrize("shape", LAUNCH_ROWS, ids=_shape_id)
def test_the_handoff_backward_launches_the_tape_path_with_the_light_prep(
    device, manifest, shape, record_kind, tape_wy, record_property
) -> None:
    """On a caller workspace: the original forward launches the saving forward's records, a recompute with a tape
    record too and one with a row record the intra twin in place of the intra record (on the record's grid); the
    backward from a tape record launches the tape path's records with the light path's prep record (no qg; no kb on
    the rows whose light set carries w), from a row record the same without any prep, and no ``wy`` where the
    record carries the forward's ``w`` (elsewhere its WY record, or with ``tape_wy`` the record's w-only twin, one
    launch either way); ``"off"`` launches the light path.  The launches are checked on every draw and both arms
    (``off`` / ``on``: ``out`` and the five gradients) judged; P8 at 4 heads keeps the exact checks only (its results
    finite)."""
    lengths, heads = shape
    p = _plan(lengths, heads, device)
    with_rows = _with_rows(p, record_kind)
    launched: list[str] = []
    intra_grids: list[tuple] = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        if cake_jit.MODULES[name]["stage"] == "fwd_intra":
            intra_grids.append((name, tuple(grid)))
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    ws = CakeKDANondetWorkspace(p, device, backward=True, launcher=recording)
    fwd = _expected_fwd(p)
    recompute = _expected_fwd(p, intra=p.handoff_twin if with_rows else None)
    handoff_bwd = _expected_bwd(
        p,
        "tape",
        prep_path=p.light_path,
        skip_wy=p.handoff_keeps_w,
        prep=not with_rows,
        tape_wy=tape_wy,
    )
    preps = [n for n in handoff_bwd if cake_jit.MODULES[n]["stage"] == "bwd_prep"]
    assert preps == ([] if with_rows else [p.record("bwd_prep", p.light_path)])
    intra_grid = tuple(p.grid("fwd_intra", "fwd"))

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        cu = inputs["cu_seqlens"]
        arms = {}
        for mode, bwd in (
            ("off", _expected_bwd(p, p.light_path)),
            ("on", handoff_bwd),
        ):
            cake_backend._handoff_reset()
            leaves = _leaves(inputs)
            launched.clear()
            out = checkpoint(
                lambda *a, m=mode: cake_kda_nondet(
                    *a, cu, workspace=ws, checkpoint_handoff=m
                ),
                *leaves,
                use_reentrant=False,
            )
            assert launched == fwd, mode
            # the original forward publishes nothing
            assert _stats(device)["published"] == 0
            launched.clear()
            intra_grids.clear()
            out.backward(inputs["dout"])
            want = (recompute if mode == "on" else fwd) + bwd
            assert launched == want, (mode, launched)
            # the twin runs on the intra record's grid
            assert [g for _, g in intra_grids] == [intra_grid], (mode, intra_grids)
            st = _stats(device)
            assert st["hits"] == (mode == "on") and st["live_records"] == 0, (mode, st)
            assert st["published_rows"] == int(mode == "on" and with_rows), (mode, st)
            arms[mode] = _results(out.detach(), _grads(leaves))
        return arms, inputs

    if _judged(lengths, heads):
        _judge(record_property, draw)
    else:
        _run_exact(draw)


# --------------------------------------------------------------------------- stale records and misses


class _AfterRecompute(torch.autograd.Function):
    """Identity inside a checkpointed region whose backward first unpacks its saved tensor (non-reentrant
    checkpointing then re-runs the region: the KDA recompute publishes its record) and then calls ``action``, before
    the KDA node's backward runs."""

    @staticmethod
    def forward(ctx, o, action):
        ctx.save_for_backward(o)
        ctx.action = action
        return o.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.saved_tensors  # noqa: B018 - the unpack re-runs the region
        if ctx.action is not None:
            ctx.action()
        return grad, None


STALE_ROWS = [((100, 640, 77, 1000, 129), 2), ((129, 300, 70), 16), ((65, 200), 32)]


@pytest.mark.parametrize("shape", STALE_ROWS, ids=_shape_id)
def test_a_stale_record_falls_back_and_stays_correct(
    device, manifest, shape, record_kind, record_property
) -> None:
    """An operand changed in place between the recompute (which published the record, of the test's kind) and the
    KDA node's backward: ``cu_seqlens`` bumped to a new version with the same values, or the gate ``g`` halved.
    The record's stamp no longer matches, the backward drops it (one miss, an empty registry) and runs the light
    path.  Without a change the record is taken.  The off and on arms of the unchanged inputs and of the version bump
    (``unchanged.*`` / ``cu_version.*``) are judged; with the halved gate the backward reads the recompute's set with
    the changed gate, which has no FP64 reference, so those arms' ``out`` (the forward of the unchanged inputs) is
    judged and their gradients are checked finite.  (Non-reentrant only: under reentrant checkpointing torch's own
    saved-tensor version check raises first; the failed-backward test covers that mode.)"""
    lengths, heads = shape
    with_rows = _with_rows(_plan(lengths, heads, device), record_kind)

    def run(inputs, mode: str, change: str | None) -> dict:
        leaves = _leaves(inputs)
        cu = inputs["cu_seqlens"].clone()
        g_leaf = leaves[3]

        def action():
            with torch.no_grad():
                if change == "cu_version":
                    cu.add_(0)
                elif change == "g_values":
                    g_leaf.mul_(0.5)

        def layer(q, k, v, g, beta):
            o = cake_kda_nondet(q, k, v, g, beta, cu, checkpoint_handoff=mode)
            return _AfterRecompute.apply(o, action if change else None)

        out = checkpoint(layer, *leaves, use_reentrant=False)
        out.backward(inputs["dout"])
        return _results(out.detach(), _grads(leaves))

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        arms = {}
        for change in (None, "cu_version", "g_values"):
            cake_backend._handoff_reset()
            off = run(inputs, "off", change)
            assert _stats(device)["published"] == 0
            cake_backend._handoff_reset()
            on = run(inputs, "on", change)
            st = _stats(device)
            assert st["published"] == 1 and st["live_records"] == 0, (change, st)
            assert st["published_rows"] == int(with_rows), (change, st)
            if change is None:
                assert st["hits"] == 1 and st["misses"] == 0, st
            else:
                assert st["hits"] == 0 and st["misses"] == 1, (change, st)
            label = change or "unchanged"
            for mode, results in (("off", off), ("on", on)):
                if change == "g_values":
                    _check_finite(results, (lengths, heads, change, mode))
                    results = {"out": results["out"]}
                arms[f"{label}.{mode}"] = results
        return arms, inputs

    _judge(record_property, draw)


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_inputs_changed_in_place_before_the_checkpointed_call_still_hand_off(
    device, manifest, reentrant, record_property
) -> None:
    """Inputs whose version counters moved before the checkpointed call (``cu_seqlens`` written in place, ``q``
    scaled in place by one): the take compares versions through the record's own aliases of the recompute's inputs,
    not through the tensors checkpoint unpacks, so the record is taken; the off and on arms are judged."""
    lengths, heads = (100, 640, 77, 1000, 129), 2

    def run(inputs, mode: str):
        cake_backend._handoff_reset()
        leaves = _leaves(inputs)
        cu = inputs["cu_seqlens"].clone()
        with torch.no_grad():
            cu.add_(0)
            leaves[0].mul_(1)
        assert cu._version > 0 and leaves[0]._version > 0
        out = checkpoint(
            lambda *a: cake_kda_nondet(*a, cu, checkpoint_handoff=mode),
            *leaves,
            use_reentrant=reentrant,
        )
        out.backward(inputs["dout"])
        return _results(out.detach(), _grads(leaves)), _stats(device)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        off, _ = run(inputs, "off")
        on, st = run(inputs, "on")
        assert st["published"] == 1 and st["hits"] == 1 and st["misses"] == 0, st
        return {"off": off, "on": on}, inputs

    _judge(record_property, draw)


def test_a_record_the_backward_refuses_is_released_and_stays_correct(
    device, manifest, record_kind, record_property
) -> None:
    """A record of the test's kind the node takes but the backward refuses (here: a plan key that differs from the
    forward's): one miss, the record's tensors released before the light path allocates its own (``h`` / ``v_new``
    / ``w`` / ``Aqk`` references cleared, and a row record's ``gk`` / ``kg`` / ``beta_int`` / ``kb``), the
    backward on the light path's own workspace (no hand-off layout: the full backward arena) and an empty registry,
    on every draw; the off arm and the refused on arm are judged."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    p = _plan(lengths, heads, device)
    real_key, real_take = cake_backend._handoff_key, cake_backend._handoff_take
    real_acquire = cake_backend._entry_acquire

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        off = _results(*_checkpointed_step(inputs, reentrant=False, handoff="off"))
        calls: list[int] = []
        taken: list = []
        backward_arenas: list = []

        def acquire(*args, **kwargs):
            ws, key = real_acquire(*args, **kwargs)
            if kwargs.get("backward"):
                backward_arenas.append(
                    (kwargs.get("layout"), type(ws).__name__, ws.bytes)
                )
            return ws, key

        def key(p, scale):  # the recompute's key, then another one in the backward
            calls.append(1)
            k = real_key(p, scale)
            return k if len(calls) == 1 else (*k, "mismatch")

        def take(*args):
            rec = real_take(*args)
            taken.append(rec)
            return rec

        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_backend, "_handoff_key", key)
            m.setattr(cake_backend, "_handoff_take", take)
            m.setattr(cake_backend, "_entry_acquire", acquire)
            cake_backend._handoff_reset()
            on = _results(*_checkpointed_step(inputs, reentrant=False, handoff="on"))
        st = _stats(device)
        assert len(calls) == 2, calls
        assert st["published"] == 1 and st["hits"] == 0 and st["misses"] == 1, st
        assert st["published_rows"] == int(_with_rows(p, record_kind)), st
        assert st["live_records"] == 0, st
        # taken by the node, refused by the backward's acceptance before its workspace
        assert len(taken) == 1 and taken[0] is not None
        rec = taken[0]
        assert rec.h is None and rec.v_new is None and rec.w is None and rec.aqk is None
        assert rec.seg_m is None and rec.aliases == ()
        assert (
            rec.gk is None
            and rec.kg is None
            and rec.beta_int is None
            and rec.kb is None
        )
        assert backward_arenas == [(None, "CakeKDANondetWorkspace", p.arena_bytes())], (
            backward_arenas
        )
        return {"off": off, "on": on}, inputs

    _judge(record_property, draw)


class _HandoffTestFailure(Exception):
    pass


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_records_of_a_failed_backward_are_dropped_by_the_next_forward(
    device, manifest, reentrant
) -> None:
    """A backward that fails after the recompute published its record (an exception in a later node's backward):
    torch may skip the graph task's final callbacks, so the record may outlive the backward; the next forward
    outside any backward drops it (one miss in total), and the next checkpointed step hands off again."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    cu = inputs["cu_seqlens"]

    def fail():
        raise _HandoffTestFailure("the backward fails after the recompute")

    def layer(q, k, v, g, beta):
        o = cake_kda_nondet(q, k, v, g, beta, cu, checkpoint_handoff="on")
        return _AfterRecompute.apply(o, fail)

    leaves = _leaves(inputs)
    out = checkpoint(layer, *leaves, use_reentrant=reentrant)
    with pytest.raises(Exception) as info:
        out.backward(inputs["dout"])
    assert "the backward fails after the recompute" in str(info.value), info.value
    del out, info
    s = _stats(device)
    assert s["published"] == 1 and s["hits"] == 0, s
    assert s["live_records"] in (0, 1), s  # 1 when torch skipped the final callbacks
    with torch.no_grad():
        cake_kda_nondet(*_leaves(inputs), cu, checkpoint_handoff="on")
    s = _stats(device)
    assert s["live_records"] == 0 and s["live_bytes"] == 0 and s["misses"] == 1, s
    _checkpointed_step(inputs, reentrant=reentrant, handoff="on")
    s = _stats(device)
    assert s["hits"] == 1 and s["live_records"] == 0, s


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_an_untaken_record_is_dropped_and_repeated_misses_turn_the_handoff_off(
    device, manifest, monkeypatch, reentrant, record_property
) -> None:
    """A backward that re-runs a checkpointed region but never reaches its KDA node (the gradient of another
    output only) leaves the record untaken: it is dropped at the end of its graph task (``queue_callback``) as a
    miss.  ``_HANDOFF_MAX_MISSES`` misses in a row turn the hand-off off on the device: the next checkpointed step
    publishes nothing (it runs the light path), and after a reset the hand-off is back on.  On every draw; the
    disabled step (``disabled.on``), the off step (``off``) and the step after the reset (``reenabled.on``) are
    judged."""
    lengths, heads = (100, 640, 77), 2
    monkeypatch.setattr(cake_backend, "_HANDOFF_MAX_MISSES", 2)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        cu = inputs["cu_seqlens"]
        cake_backend._handoff_reset()

        def pruned_step():
            leaves = _leaves(inputs)
            x = torch.randn(64, device=device, requires_grad=True)

            def region(q, k, v, g, beta, x_):
                # the KDA output leaves the graph (a reentrant region would otherwise run its node with a zero
                # gradient): the recompute publishes, no backward takes the record
                o = cake_kda_nondet(q, k, v, g, beta, cu, checkpoint_handoff="on")
                return o.detach(), x_ * x_

            _o, y = checkpoint(region, *leaves, x, use_reentrant=reentrant)
            y.sum().backward()
            # torch's own gradient of x * x (not this package's)
            assert torch.equal(x.grad, 2 * x.detach())
            assert all(leaf.grad is None for leaf in leaves)

        pruned_step()
        st = _stats(device)
        assert st["published"] == 1 and st["hits"] == 0, st
        assert st["live_records"] == 0 and st["misses"] == 1 and not st["disabled"], st
        pruned_step()
        st = _stats(device)
        assert st["misses"] == 2 and st["disabled"], st
        # turned off: no record, the light path
        on = _results(*_checkpointed_step(inputs, reentrant=reentrant, handoff="on"))
        st = _stats(device)
        assert st["published"] == 2 and st["hits"] == 0, st
        assert st["refused_disabled"] == 1, st
        off = _results(*_checkpointed_step(inputs, reentrant=reentrant, handoff="off"))
        cake_backend._handoff_reset()  # back on
        back = _results(*_checkpointed_step(inputs, reentrant=reentrant, handoff="on"))
        assert _stats(device)["hits"] == 1
        return {"disabled.on": on, "off": off, "reenabled.on": back}, inputs

    _judge(record_property, draw)


# --------------------------------------------------------------------------- checkpoint variants


@pytest.mark.parametrize(
    "outer_reentrant", (False, True), ids=("outer-nonreentrant", "outer-reentrant")
)
def test_nested_checkpoint_regions_publish_only_the_recompute(
    device, manifest, outer_reentrant, record_property
) -> None:
    """Three blocks, each an outer checkpoint region (non-reentrant or reentrant) around an inner non-reentrant
    region with the KDA layer, followed by an op that saves the KDA output (so the outer recompute re-runs the
    inner region's original forward inside the backward).  That forward runs under the inner region's pack hook
    and publishes nothing (``refused_hooks``); each inner recompute publishes one record its node takes: three
    hits, no miss, no budget refusal, never more than one live record.  Each block's KDA layer is judged on the
    operands its node read (its ``v`` the previous block's output, its ``dout`` the gradient that reached its output:
    :func:`_judge_layers`), with the hand-off off and on (``<mode>.layer<i>``)."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    blocks = 3

    def run(first, second, mode: str):
        cake_backend._handoff_reset()
        cu = first["cu_seqlens"]
        sets = [first] + [second] * (blocks - 1)
        leaves = [_leaves(s) for s in sets]
        sink: dict = {}

        def kda(i, v_in, q, k, g, beta):
            return _tapped(
                i,
                sink,
                lambda v: cake_kda_nondet(
                    q, k, v, g, beta, cu, checkpoint_handoff=mode
                ),
                v_in,
            )

        def block(i, v_in, q, k, g, beta):
            o = checkpoint(
                lambda *a: kda(i, *a), v_in, q, k, g, beta, use_reentrant=False
            )
            return torch.nn.functional.silu(o)

        x = leaves[0][2]
        for i in range(blocks):
            q, k, _v, g, beta = leaves[i]
            x = checkpoint(
                lambda *a, i=i: block(i, *a),
                x,
                q,
                k,
                g,
                beta,
                use_reentrant=outer_reentrant,
            )
        x.backward(second["dout"])
        return _layer_results(sets, leaves, sink, blocks, f"{mode}."), _stats(device)

    def draw(j, s):
        first, second = _two_layer_inputs(lengths, heads, device, s)
        off, st_off = run(first, second, "off")
        assert st_off["published"] == 0, st_off
        on, st = run(first, second, "on")
        assert st["published"] == blocks and st["hits"] == blocks, st
        assert st["misses"] == 0, st
        assert st["refused_budget"] == 0 and st["max_live_records"] <= 1, st
        assert st["live_records"] == 0, st
        return off + on

    _judge_layers(record_property, draw)


def test_handoff_with_checkpoint_early_stop_off(
    device, manifest, record_property
) -> None:
    """``torch.utils.checkpoint.set_checkpoint_early_stop(False)``: the recompute runs the whole region; the record
    is taken (the record's exact checks on every draw), and the off and on arms are judged."""
    set_early_stop = getattr(torch.utils.checkpoint, "set_checkpoint_early_stop", None)
    if set_early_stop is None:
        pytest.skip("this torch has no set_checkpoint_early_stop")
    lengths, heads = (100, 640, 77, 1000, 129), 2

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        with set_early_stop(False):
            arms = _handoff_arms(inputs, device, lengths, heads, reentrant=False)
        return arms, inputs

    _judge(record_property, draw)


def test_a_second_backward_with_retain_graph_hands_off_again(
    device, manifest, record_property
) -> None:
    """Non-reentrant checkpointing with ``retain_graph=True``: each backward recomputes, publishes and takes its own
    record.  The leaves' gradients are cleared between the two backwards, and each backward's gradients are judged
    with the hand-off off and on (``<mode>.first`` with ``out``, ``<mode>.second``)."""
    lengths, heads = (100, 640, 77, 1000, 129), 2

    def run(inputs, mode: str):
        cake_backend._handoff_reset()
        leaves = _leaves(inputs)
        cu = inputs["cu_seqlens"]
        out = checkpoint(
            lambda *a: cake_kda_nondet(*a, cu, checkpoint_handoff=mode),
            *leaves,
            use_reentrant=False,
        )
        out.backward(inputs["dout"], retain_graph=True)
        first = _results(out.detach(), _grads(leaves))
        for leaf in leaves:
            leaf.grad = None
        out.backward(inputs["dout"])
        return first, _grads(leaves), _stats(device)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        arms = {}
        for mode in ("off", "on"):
            first, second, st = run(inputs, mode)
            if mode == "on":
                assert st["published"] == 2 and st["hits"] == 2, st
                assert st["misses"] == 0 and st["live_records"] == 0, st
            arms[f"{mode}.first"] = first
            arms[f"{mode}.second"] = second
        return arms, inputs

    _judge(record_property, draw)


# --------------------------------------------------------------------------- layers sharing a workspace


def _two_layer_inputs(lengths, heads, device, seed: int):
    """The operands of the first layer and of the later layers, both functions of the draw seed ``seed``."""
    first = _inputs(lengths, heads, device, seed)
    second = _inputs(lengths, heads, device, seed + 12345)
    return first, second


def _layer_step(
    first, second, *, ws, handoff, reentrant, regions, layers=2, prefix=""
) -> list:
    """``layers`` KDA layers on one caller workspace (the output of each is the next one's ``v``), checkpointed
    per layer (``regions="each"``) or as one region (``"one"``): per layer, its results and the operands its node read
    (:func:`_layer_results`)."""
    cu = first["cu_seqlens"]
    sets = [first] + [second] * (layers - 1)
    leaves = [_leaves(s) for s in sets]
    sink: dict = {}

    def layer(i, v_in, q, k, g, beta):
        return _tapped(
            i,
            sink,
            lambda v: cake_kda_nondet(
                q, k, v, g, beta, cu, workspace=ws, checkpoint_handoff=handoff
            ),
            v_in,
        )

    def args(i):
        q, k, _v, g, beta = leaves[i]
        return q, k, g, beta

    if regions == "each":
        x = leaves[0][2]
        for i in range(layers):
            x = checkpoint(
                lambda *a, i=i: layer(i, *a), x, *args(i), use_reentrant=reentrant
            )
    else:

        def stack(v0, *flat):
            x = v0
            for i in range(layers):
                q, k, g, beta = flat[4 * i : 4 * i + 4]
                x = layer(i, x, q, k, g, beta)
            return x

        flat = [t for i in range(layers) for t in args(i)]
        x = checkpoint(stack, leaves[0][2], *flat, use_reentrant=reentrant)
    x.backward(second["dout"])
    return _layer_results(sets, leaves, sink, layers, prefix)


@pytest.mark.parametrize("shape", LAYER_ROWS, ids=_shape_id)
@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_two_checkpointed_layers_share_one_workspace(
    device, manifest, shape, reentrant, record_kind, record_property
) -> None:
    """Two KDA layers on one caller-owned workspace, checkpointed per layer or as one region (the recompute then
    publishes both records before either backward runs; the records live outside the arena the second layer's
    backward overwrites): two records published and taken and an empty registry.  With row records, per layer: two
    row records; as one region: the first record is published as a row record, the second's admission demotes it
    (two row records exceed the budget) and is published as a tape record, so both backwards take tape records.
    Each layer is judged on the operands its node read (:func:`_judge_layers`), with the hand-off off and on (the
    record's per-call tensors NaN-filled): ``<regions>.<mode>.layer<i>``."""
    lengths, heads = shape
    with_rows = int(_with_rows(_plan(lengths, heads, device), record_kind))
    ws = CakeKDANondetWorkspace.for_problem(sum(lengths), len(lengths), heads, device)

    def draw(j, s):
        first, second = _two_layer_inputs(lengths, heads, device, s)
        pairs = []
        for regions in ("each", "one"):
            tag = (lengths, heads, reentrant, regions)
            cake_backend._handoff_reset()
            kw = dict(ws=ws, reentrant=reentrant, regions=regions)
            pairs += _layer_step(
                first, second, handoff="off", prefix=f"{regions}.off.", **kw
            )
            assert _stats(device)["published"] == 0, tag
            with _poisoned():
                pairs += _layer_step(
                    first, second, handoff="on", prefix=f"{regions}.on.", **kw
                )
            st = _stats(device)
            assert st["published"] == 2 and st["hits"] == 2, (tag, st)
            assert st["misses"] == 0, (tag, st)
            assert st["live_records"] == 0 and st["refused_budget"] == 0, (tag, st)
            if regions == "each":
                assert st["published_rows"] == 2 * with_rows, (tag, st)
                assert st["demoted"] == 0, (tag, st)
            else:
                assert st["published_rows"] == with_rows == st["demoted"], (tag, st)
        return pairs

    _judge_layers(record_property, draw)


def test_the_byte_budget_admits_two_records_of_the_shape(
    device, manifest, record_kind, record_property
) -> None:
    """Three layers of one shape in one checkpointed region: the recompute publishes the first two records and
    the budget (two tape records of the shape, or one row record where larger) refuses the third before its
    extra writes; that layer's backward runs the light path.  With row records the first record is published as a
    row record and demoted by the second's admission (both then tape records).  The registry is empty after the
    backward.  Each layer is judged on the operands its node read (:func:`_judge_layers`), under both checkpoint
    modes with the hand-off off and on: ``<mode>.<handoff>.layer<i>``."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    with_rows = int(_with_rows(_plan(lengths, heads, device), record_kind))
    ws = CakeKDANondetWorkspace.for_problem(sum(lengths), len(lengths), heads, device)

    def draw(j, s):
        first, second = _two_layer_inputs(lengths, heads, device, s)
        pairs = []
        for reentrant in (False, True):
            mode = "reentrant" if reentrant else "nonreentrant"
            cake_backend._handoff_reset()
            kw = dict(ws=ws, reentrant=reentrant, regions="one", layers=3)
            pairs += _layer_step(
                first, second, handoff="off", prefix=f"{mode}.off.", **kw
            )
            pairs += _layer_step(
                first, second, handoff="on", prefix=f"{mode}.on.", **kw
            )
            st = _stats(device)
            assert st["published"] == 2 and st["refused_budget"] == 1, (reentrant, st)
            assert st["hits"] == 2 and st["misses"] == 0, (reentrant, st)
            assert st["live_records"] == 0, (reentrant, st)
            assert st["published_rows"] == with_rows == st["demoted"], (reentrant, st)
        return pairs

    _judge_layers(record_property, draw)


# --------------------------------------------------------------------------- switches and the quiet paths


def test_auto_follows_the_head_counts_and_the_verified_devices(
    device, manifest, monkeypatch, record_property
) -> None:
    """The default (``"auto"``, no keyword) with the toolkit, the verified set and the size bounds patched, so the
    test runs on any device: it hands off at a ``HANDOFF_HEADS`` head count on a verified device with the problem at
    both size bounds (``H * S == HANDOFF_MIN_HEAD_TOKENS``, ``S // N == HANDOFF_MIN_DOC_TOKENS``), and stays off on
    another device (counting ``unverified``), at a head count outside ``HANDOFF_HEADS`` and with either bound raised by
    one (no request: nothing counted), on every draw.  Each case's off and default arms (``<case>.off`` /
    ``<case>.auto``) are judged, one judge per head count."""
    lengths = (100, 640, 77, 1000, 129)
    S, N = sum(lengths), len(lengths)
    arch, sms = cake_jit.device_arch(device.index), cake_jit.sm_count(device.index)
    monkeypatch.setattr(
        cake_backend, "HANDOFF_VERIFIED", frozenset({(arch, sms, "9.9")})
    )
    # no device listed for row records: every record is a tape record
    monkeypatch.setattr(cake_backend, "HANDOFF_ROWS_VERIFIED", frozenset())
    heads = min(cake_backend.HANDOFF_HEADS)
    # the largest bounds the problem meets at that head count, and each one past it
    at = (heads * S, S // N)
    assert at[1] * N <= S < (at[1] + 1) * N, (S, N)
    # (case, heads, toolkit, (min head-tokens, min mean document length), published and hits, unverified)
    cases = (
        ("verified", heads, "9.9", at, 1, 0),
        ("other_toolkit", heads, "9.8", at, 0, 1),
        ("one_head", 1, "9.9", at, 0, 0),
        ("head_token_bound_raised", heads, "9.9", (at[0] + 1, at[1]), 0, 0),
        ("doc_bound_raised", heads, "9.9", (at[0], at[1] + 1), 0, 0),
    )
    for H in sorted({case[1] for case in cases}):

        def draw(j, s, H=H):
            inputs = _inputs(lengths, H, device, s)
            arms = {}
            for label, h, toolkit, bounds, engaged, unverified in cases:
                if h != H:
                    continue
                min_head_tokens, min_doc = bounds
                monkeypatch.setattr(
                    cake_backend, "_handoff_toolkit", lambda a, t=toolkit: t
                )
                monkeypatch.setattr(
                    cake_backend, "HANDOFF_MIN_HEAD_TOKENS", min_head_tokens
                )
                monkeypatch.setattr(cake_backend, "HANDOFF_MIN_DOC_TOKENS", min_doc)
                off = _checkpointed_step(inputs, reentrant=False, handoff="off")
                cake_backend._handoff_reset()
                auto = _checkpointed_step(inputs, reentrant=False, handoff=None)
                st = _stats(device)
                tag = (H, toolkit, min_head_tokens, min_doc)
                assert st["published"] == st["hits"] == engaged, (tag, st)
                assert st["unverified"] == unverified, (tag, st)
                assert st["live_records"] == 0, (tag, st)
                twin = _plan(lengths, H, device).handoff_twin
                assert st["published_rows"] == 0, (tag, st)
                assert st["rows_unverified"] == engaged * (twin is not None), (tag, st)
                if not engaged and not unverified:
                    nonzero = [n for n in cake_backend._HANDOFF_COUNTERS if st[n]]
                    assert not nonzero, (tag, nonzero, st)
                arms[f"{label}.off"] = _results(*off)
                arms[f"{label}.auto"] = _results(*auto)
            assert arms
            return arms, inputs

        _judge(record_property, draw)


def test_the_shipped_rule_on_this_device(device, manifest) -> None:
    """The default as shipped, nothing patched: this build's toolkit reads as ``major.minor``, and at the smallest
    ``HANDOFF_HEADS`` head count ``H``, on the smallest one-document problem the size bounds admit there
    (``max(HANDOFF_MIN_DOC_TOKENS, ceil(HANDOFF_MIN_HEAD_TOKENS / H))`` tokens), the default hands off exactly when
    this device (architecture, SM count, toolkit) is in ``HANDOFF_VERIFIED``, else it counts ``unverified``.  The
    toolkit of the one build of the twin, its intra record and the light prep reads as ``major.minor.patch``; the
    record (the call's own workspace) is a row record exactly when the device with that toolkit is in
    ``HANDOFF_ROWS_VERIFIED`` (else ``rows_unverified``) and the twin loads (else ``rows_unready``).  With one token
    fewer, and with the same tokens in more documents than the mean-length bound allows, the default makes no request
    on any device: nothing is published or counted.  The problem (about 98k tokens at two heads with the shipped
    bounds) keeps the exact checks only (the results of the off and default steps finite): the hand-off at this head
    count is judged on ``(100, 640, 77, 1000, 129)``
    (:func:`test_a_row_record_falls_back_to_a_tape_record_where_the_rule_refuses_it`)."""
    heads = min(cake_backend.HANDOFF_HEADS)
    min_doc = cake_backend.HANDOFF_MIN_DOC_TOKENS
    S = _smallest_admitted(heads)
    lengths = (S,)
    p = _plan(lengths, heads, device)
    key = (p.arch, p.sm_count, cake_backend._handoff_toolkit(p.arch))
    assert re.fullmatch(r"\d+\.\d+", key[2] or ""), key
    engaged = int(key in cake_backend.HANDOFF_VERIFIED)
    twin = p.handoff_twin
    assert twin is not None, (p.arch, p.record("fwd_intra", "fwd"))
    toolkit = cake_backend._handoff_rows_toolkit(
        twin,
        p.record("fwd_intra", "fwd"),
        p.record("bwd_prep", p.light_path),
        p.arch,
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", toolkit or ""), (key, toolkit)
    assert toolkit.rsplit(".", 1)[0] == key[2], (key, toolkit)
    listed = (p.arch, p.sm_count, toolkit) in cake_backend.HANDOFF_ROWS_VERIFIED
    inputs = _inputs(lengths, heads, device)
    off = _results(*_checkpointed_step(inputs, reentrant=False, handoff="off"))
    _check_finite(off, (key, "off"))
    del off
    cake_backend._handoff_reset()
    auto = _results(*_checkpointed_step(inputs, reentrant=False, handoff=None))
    s = _stats(device)
    assert s["published"] == s["hits"] == engaged, (key, s)
    assert s["unverified"] == 1 - engaged, (key, s)
    # a listed device's twin loads here (the suite's launches load it too)
    assert s["published_rows"] == int(engaged and listed), (key, toolkit, s)
    assert s["rows_unverified"] == int(engaged and not listed), (key, toolkit, s)
    assert s["rows_unready"] == 0, (key, toolkit, s)
    _check_finite(auto, (key, "auto"))
    del auto
    # below the bounds: one token fewer (under the token-head bound, or the mean-length bound where that one binds),
    # and the S tokens in one document more than S // HANDOFF_MIN_DOC_TOKENS (under the mean-length bound)
    n_docs = S // min_doc + 1
    base = S // n_docs
    split = (base,) * (n_docs - 1) + (S - base * (n_docs - 1),)
    for small in ((S - 1,), split):
        assert not cake_backend._handoff_large(sum(small), len(small), heads), small
        cake_backend._handoff_reset()
        inputs = make_inputs(small, heads, device)
        _checkpointed_step(inputs, reentrant=False, handoff=None)
        s = _stats(device)
        nonzero = [name for name in cake_backend._HANDOFF_COUNTERS if s[name]]
        assert not nonzero and s["live_records"] == 0, (len(small), s)


class _Built:
    """A stand-in for ``cake_build.prebuilt_path(name)``: ``is_file()`` is whether the library counts as prebuilt."""

    def __init__(self, built: bool) -> None:
        self.built = built

    def is_file(self) -> bool:
        return self.built


@pytest.fixture
def _fresh_rows_toolkit():
    """``_handoff_rows_toolkit`` caches its verdict per process: cleared around a test that patches the build
    state."""
    cake_backend._handoff_rows_toolkit.cache_clear()
    yield
    cake_backend._handoff_rows_toolkit.cache_clear()


@pytest.mark.parametrize(
    "case",
    ("unverified", "mixed_build", "load_fails", "caller_workspace", "verified"),
)
def test_a_row_record_falls_back_to_a_tape_record_where_the_rule_refuses_it(
    device, manifest, monkeypatch, _fresh_rows_toolkit, case, record_property
) -> None:
    """The rule (``_HANDOFF_ROWS`` unset) with ``"on"`` at a ``HANDOFF_HEADS`` head count, this device listed in
    ``HANDOFF_ROWS_VERIFIED`` with this build's toolkit (another patch version in the ``unverified`` case): a row
    record only on the call's own workspace with the one build and a twin that loads (``verified``: the twin
    launched, no prep); otherwise a tape record (the prep launched) and the counter of the refusal: another
    toolkit listed (``rows_unverified``), one of the three libraries prebuilt and another not (``mixed_build``:
    ``rows_unready``), the twin's load failing (``load_fails``: ``rows_unready``), a caller workspace (no
    counter).  On every draw; the off and on arms are judged in every case."""
    lengths, heads = (100, 640, 77, 1000, 129), min(cake_backend.HANDOFF_HEADS)
    p = _plan(lengths, heads, device)
    twin = p.handoff_twin
    assert twin is not None, (p.arch, p.record("fwd_intra", "fwd"))
    parent, prep = p.record("fwd_intra", "fwd"), p.record("bwd_prep", p.light_path)
    toolkit = cake_backend._handoff_rows_toolkit(twin, parent, prep, p.arch)
    assert toolkit is not None, (
        "this build's twin, intra record and light prep are not one build"
    )
    cake_backend._handoff_rows_toolkit.cache_clear()
    major, minor, patch = toolkit.split(".")
    listed = toolkit if case != "unverified" else f"{major}.{minor}.{int(patch) + 1}"
    monkeypatch.setattr(
        cake_backend,
        "HANDOFF_ROWS_VERIFIED",
        frozenset({(p.arch, p.sm_count, listed)}),
    )
    if case == "mixed_build":
        # one of the three libraries flips its build state (the launches load every library through the real
        # paths): the twin's where nothing is prebuilt here, else the intra record's
        twin_library = cake_jit.library_name(twin, p.arch)
        parent_library = cake_jit.library_name(parent, p.arch)
        real_prebuilt = cake_build.prebuilt_path
        flipped = (
            parent_library if real_prebuilt(twin_library).is_file() else twin_library
        )

        def prebuilt(name):
            if name == flipped:
                return _Built(not real_prebuilt(name).is_file())
            return real_prebuilt(name)

        monkeypatch.setattr(cake_build, "prebuilt_path", prebuilt)
    elif case == "load_fails":
        real_load = cake_jit.load

        def load(name, arch):
            if name == twin:
                raise RuntimeError("the twin does not load")
            return real_load(name, arch)

        monkeypatch.setattr(cake_jit, "load", load)
    workspace = None
    if case == "caller_workspace":
        workspace = CakeKDANondetWorkspace(p, device, backward=True)
    runs: list[tuple[str, str | None]] = []
    real_run = CakeKDANondetWorkspace.run

    def run(self, stage, values, *, path, **kwargs):
        runs.append((stage, kwargs.get("record")))
        return real_run(self, stage, values, path=path, **kwargs)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        off = _results(*_checkpointed_step(inputs, reentrant=False, handoff="off"))
        cake_backend._handoff_reset()
        runs.clear()
        monkeypatch.setattr(CakeKDANondetWorkspace, "run", run)
        on = _results(
            *_checkpointed_step(
                inputs, reentrant=False, handoff="on", workspace=workspace
            )
        )
        monkeypatch.setattr(CakeKDANondetWorkspace, "run", real_run)
        st = _stats(device)
        with_rows = case == "verified"
        assert st["published"] == st["hits"] == 1 and st["live_records"] == 0, (
            case,
            st,
        )
        assert st["published_rows"] == int(with_rows), (case, st)
        assert st["rows_unverified"] == int(case == "unverified"), (case, st)
        assert st["rows_unready"] == int(case in ("mixed_build", "load_fails")), (
            case,
            st,
        )
        # the "on" step: the backward from a tape record runs the light prep, from a row record none; the recompute
        # launches the twin exactly for a row record
        preps = [stage for stage, _ in runs if stage == "bwd_prep"]
        assert len(preps) == int(not with_rows), runs
        assert (("fwd_intra", twin) in runs) is with_rows, runs
        return {"off": off, "on": on}, inputs

    _judge(record_property, draw)


def test_the_environment_changes_nothing_end_to_end(
    device, manifest, monkeypatch, record_property
) -> None:
    """Through ``cake_kda_nondet``, with ``"auto"`` engaging nowhere (an empty verified set): the keyword alone
    decides; ``CAKE_KDA_CKPT_HANDOFF=off`` leaves ``"on"`` on and ``=on`` leaves ``"auto"`` off, on every draw.  Both
    arms (``env_off.on`` / ``env_on.auto``) are judged.  The host tests cover the other combinations."""
    monkeypatch.setattr(cake_backend, "HANDOFF_VERIFIED", frozenset())

    def draw(j, s):
        inputs = _inputs((63, 1, 65), 2, device, s)
        arms = {}
        for env, keyword, published in (("off", "on", 1), ("on", "auto", 0)):
            cake_backend._handoff_reset()
            monkeypatch.setenv(ENV, env)
            arms[f"env_{env}.{keyword}"] = _results(
                *_checkpointed_step(inputs, reentrant=False, handoff=keyword)
            )
            st = _stats(device)
            assert st["published"] == st["hits"] == published, (env, keyword, st)
            assert st["live_records"] == 0, (env, keyword, st)
        monkeypatch.delenv(ENV, raising=False)
        return arms, inputs

    _judge(record_property, draw)


def test_plain_and_original_forwards_publish_nothing(
    device, manifest, record_property
) -> None:
    """No record outside a checkpoint recompute: a plain forward and backward, a forward under ``no_grad``, the
    original forward of a checkpoint (before its backward), an explicit saved-set call; a plain backward takes
    nothing and the checkpointed forward retains no record; on every draw.  The plain step (``plain``) and the
    checkpointed step (``checkpoint``) are judged."""
    lengths, heads = (100, 640, 77, 1000, 129), 2

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, s)
        cu = inputs["cu_seqlens"]
        cake_backend._handoff_reset()
        leaves = _leaves(inputs)
        out = cake_kda_nondet(*leaves, cu, checkpoint_handoff="on")
        assert _stats(device)["published"] == 0
        out.backward(inputs["dout"])
        st = _stats(device)
        assert st["published"] == 0 and st["hits"] == 0 and st["misses"] == 0, st
        assert st["live_records"] == 0
        plain = _results(out.detach(), _grads(leaves))
        with torch.no_grad():
            cake_kda_nondet(*_leaves(inputs), cu, checkpoint_handoff="on")
        assert _stats(device)["published"] == 0
        cake_backend.cake_kda_nondet_train_forward(
            *(inputs[n] for n in (*LEAVES, "cu_seqlens"))
        )
        assert _stats(device)["published"] == 0
        # the plain call's output (its storage held by ``plain``): never released inside the measured window
        del out
        leaves = _leaves(inputs)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        out = checkpoint(
            lambda *a: cake_kda_nondet(*a, cu, checkpoint_handoff="on"),
            *leaves,
            use_reentrant=False,
        )
        torch.cuda.synchronize(device)
        st = _stats(device)
        # the original forward runs outside any backward (graph task -1)
        assert st["published"] == 0 and st["live_records"] == 0, st
        assert st["live_bytes"] == 0, st
        # the checkpointed forward keeps its output and nothing more (no record, no tapes); the allocator may also
        # have released earlier per-call temporaries inside the window, so the bound is one-sided
        grown = torch.cuda.memory_allocated(device) - before
        assert grown <= out.untyped_storage().nbytes(), grown
        out.backward(inputs["dout"])
        st = _stats(device)
        assert st["published"] == 1 and st["hits"] == 1 and st["live_records"] == 0, st
        return {"plain": plain, "checkpoint": _results(out.detach(), _grads(leaves))}, (
            inputs
        )

    _judge(record_property, draw)


def test_capture_blocks_the_handoff(device) -> None:
    """Inside CUDA-graph capture ``_handoff_blocked()`` holds (the request and the take both refuse)."""
    x = torch.zeros(1, device=device)
    graph = torch.cuda.CUDAGraph()
    assert not cake_backend._handoff_blocked()
    with torch.cuda.graph(graph):
        x.add_(1)
        inside = cake_backend._handoff_blocked()
    assert inside
    assert not cake_backend._handoff_blocked()


# --------------------------------------------------------------------------- host only


def test_the_auto_rule(monkeypatch) -> None:
    """Host only: ``"on"`` engages on every plan and ``"off"`` on none; ``"auto"`` at a ``HANDOFF_HEADS`` head count
    on a ``HANDOFF_VERIFIED`` (architecture, SM count, toolkit) device, counting ``unverified`` at those head counts
    on another architecture, SM count or toolkit (also one that cannot be read), on a one-document problem both size
    bounds admit (:func:`test_the_auto_size_bounds` covers the bounds).  The node makes no ``"auto"`` request at
    other head counts, below either size bound or with a ``cu_seqlens`` that is not a tensor, before it reads any
    torch state.  A plan of the raw gate mode (``gate="raw"``) or of an in-kernel Q/K L2 norm mode (``l2n``
    ``"fused"`` / ``"fallback"``) engages in no mode and counts nothing.  The shipped devices name exported
    architectures and ``major.minor`` toolkits; the head counts start at two (one head is host-bound)."""
    for arch, sms, toolkit in cake_backend.HANDOFF_VERIFIED:
        assert arch in ARCHES and sms > 0, (arch, sms, toolkit)
        assert re.fullmatch(r"\d+\.\d+", toolkit), (arch, sms, toolkit)
    heads = min(cake_backend.HANDOFF_HEADS)
    assert heads == 2, cake_backend.HANDOFF_HEADS
    gap = next(h for h in range(heads, 1024) if h not in cake_backend.HANDOFF_HEADS)
    # a one-document problem both size bounds admit at every HANDOFF_HEADS head count
    S0 = _smallest_admitted(heads)
    arch, other = ARCHES[0], ARCHES[1]
    monkeypatch.setattr(
        cake_backend, "HANDOFF_VERIFIED", frozenset({(arch, 152, "9.9")})
    )
    # (mode, architecture, SM count, toolkit, heads, engages, counts unverified)
    cases = (
        ("on", arch, 148, None, 1, True, False),
        ("off", arch, 152, "9.9", heads, False, False),
        ("auto", arch, 152, "9.9", heads, True, False),
        ("auto", arch, 152, "9.9", max(cake_backend.HANDOFF_HEADS), True, False),
        ("auto", arch, 152, "9.9", 1, False, False),
        ("auto", arch, 152, "9.9", gap, False, False),
        ("auto", arch, 148, "9.9", heads, False, True),
        ("auto", arch, 152, "9.8", heads, False, True),
        ("auto", arch, 152, None, heads, False, True),
        ("auto", other, 152, "9.9", heads, False, True),
    )
    unverified = 0
    for mode, arch_, sms, toolkit, H, want, counted in cases:
        monkeypatch.setattr(cake_backend, "_handoff_toolkit", lambda a, t=toolkit: t)
        p = types.SimpleNamespace(
            arch=arch_, sm_count=sms, H=H, S=S0, N=1, gate="g", l2n="off"
        )
        assert cake_backend._handoff_engages(p, mode, 0) is want, (mode, p, toolkit)
        unverified += counted
        assert cake_backend._handoff_stats(0)["unverified"] == unverified, (mode, p)
    monkeypatch.setattr(cake_backend, "_handoff_toolkit", lambda a: "9.9")
    refused = 0
    off_modes = [("raw", "off")] + [
        (g, m) for g in ("g", "raw") for m in ("fused", "fallback")
    ]
    for mode in ("on", "auto", "off"):
        for sms in (152, 148):
            for gate, l2n in off_modes:
                p = types.SimpleNamespace(
                    arch=arch, sm_count=sms, H=heads, gate=gate, l2n=l2n
                )
                assert cake_backend._handoff_engages(p, mode, 0) is False, (
                    mode,
                    sms,
                    gate,
                    l2n,
                )
                refused += 1
    assert refused == 3 * 2 * 5
    assert cake_backend._handoff_stats(0)["unverified"] == unverified

    def read_torch_state():
        raise RuntimeError("the request read torch state")

    monkeypatch.setattr(cake_backend, "_handoff_task", read_torch_state)
    ctx = types.SimpleNamespace(needs_input_grad=(True,) * 5)

    def v(H, S=S0):  # an operand on a CUDA device, without one
        return types.SimpleNamespace(
            device=torch.device("cuda", 0), shape=(S, H, D), dim=lambda: 3
        )

    def cu(N=1):  # the offsets of N documents (only their count is read)
        return torch.zeros(N + 1, dtype=torch.int32)

    n_docs = S0 // cake_backend.HANDOFF_MIN_DOC_TOKENS + 1
    no_request = (
        ("auto", v(1), cu()),
        ("auto", v(gap), cu()),
        ("off", v(heads), cu()),
        ("auto", v(heads, S0 - 1), cu()),
        ("auto", v(heads), cu(n_docs)),
        ("auto", v(heads), None),
    )
    for mode, v_, cu_ in no_request:
        assert cake_backend._handoff_request(ctx, mode, v_, cu_) is None, mode
    for mode, v_ in (("auto", v(heads)), ("on", v(1)), ("on", v(heads, 64))):
        with pytest.raises(RuntimeError, match="torch state"):
            cake_backend._handoff_request(ctx, mode, v_, cu())


def test_the_auto_size_bounds(monkeypatch) -> None:
    """Host only: ``"auto"`` admits a problem of ``S`` tokens in ``N`` documents at ``H`` heads by its size exactly
    when ``H * S >= HANDOFF_MIN_HEAD_TOKENS`` and ``S >= HANDOFF_MIN_DOC_TOKENS * N`` (zero-length documents count).
    At every ``HANDOFF_HEADS`` head count the smallest admitted one-document problem passes and one token fewer does
    not; at the mean-length bound ``N`` documents pass and ``N + 1`` do not.  The shipped bounds admit the measured
    12-head rows where the hand-off made the checkpointed step faster than the deterministic package (32768 tokens
    in one or four documents, 16384 in one) and refuse those where it made it slower (16384 tokens in four
    documents, 8192 in one or four).  On the plan, ``"auto"`` engages at the bounds on a verified device; a refusal
    by size counts nothing, on a verified device or another one; ``"on"`` ignores the bounds."""
    min_head_tokens = cake_backend.HANDOFF_MIN_HEAD_TOKENS
    min_doc = cake_backend.HANDOFF_MIN_DOC_TOKENS
    large = cake_backend._handoff_large
    # the measured rows at 12 heads: (tokens, documents) -> admitted
    measured = {
        (32768, 1): True,
        (32768, 4): True,
        (16384, 1): True,
        (16384, 4): False,
        (8192, 1): False,
        (8192, 4): False,
    }
    for (S, N), admitted in measured.items():
        assert large(S, N, 12) is admitted, (S, N)
    for H in sorted(cake_backend.HANDOFF_HEADS):
        S = _smallest_admitted(H)
        assert min_head_tokens <= H * S and min_doc <= S, (H, S)
        assert large(S, 1, H) and not large(S - 1, 1, H), (H, S)
    heads = max(cake_backend.HANDOFF_HEADS)
    for N in (1, 2, 4, 13, 300):
        S = min_doc * N
        assert heads * (S - 1) >= min_head_tokens, (N, S)
        assert large(S, N, heads) and not large(S - 1, N, heads), (N, S)
        assert not large(S, N + 1, heads), (N, S)

    arch = ARCHES[0]
    monkeypatch.setattr(
        cake_backend, "HANDOFF_VERIFIED", frozenset({(arch, 152, "9.9")})
    )
    monkeypatch.setattr(cake_backend, "_handoff_toolkit", lambda a: "9.9")
    heads = min(cake_backend.HANDOFF_HEADS)
    S = _smallest_admitted(heads)
    n_docs = S // min_doc + 1
    # (mode, SM count: 152 verified, 148 not; tokens; documents; engages; counts unverified)
    cases = (
        ("auto", 152, S, 1, True, False),
        ("auto", 152, S - 1, 1, False, False),
        ("auto", 152, S, n_docs, False, False),
        ("auto", 148, S - 1, 1, False, False),
        ("auto", 148, S, n_docs, False, False),
        ("auto", 148, S, 1, False, True),
        ("on", 148, 1, 1, True, False),
    )
    unverified = 0
    for mode, sms, S_, N, want, counted in cases:
        p = types.SimpleNamespace(
            arch=arch, sm_count=sms, H=heads, S=S_, N=N, gate="g", l2n="off"
        )
        assert cake_backend._handoff_engages(p, mode, 0) is want, (mode, sms, S_, N)
        unverified += counted
        s = cake_backend._handoff_stats(0)
        assert s["unverified"] == unverified, (mode, sms, S_, N, s)
        others = [n for n in cake_backend._HANDOFF_COUNTERS if n != "unverified"]
        assert not any(s[n] for n in others), (mode, sms, S_, N, s)


def test_a_full_window_of_misses_and_hits_turns_the_handoff_off(monkeypatch) -> None:
    """Host only: alternating hits and misses never reach ``_HANDOFF_MAX_MISSES`` in a row, but a full window of
    outcomes with as many misses as hits turns the hand-off off."""
    monkeypatch.setattr(cake_backend, "_HANDOFF_WINDOW", 4)
    cake_backend._handoff_reset()
    for hit in (True, False, True):
        cake_backend._handoff_count(0, hit=hit)
    assert not cake_backend._handoff_stats(0)["disabled"]
    cake_backend._handoff_count(0, hit=False)
    s = cake_backend._handoff_stats(0)
    assert s["hits"] == 2 and s["misses"] == 2 and s["disabled"], s


def test_handoff_mode_reads_no_environment(monkeypatch) -> None:
    """Host only: the mode is the keyword under every value of the removed variable (no override, no refusal of
    the variable), and the keyword's refusals."""
    assert not hasattr(cake_backend, "HANDOFF_ENV")
    for env in (None, "", "auto", "off", " Off ", "on", "1"):
        if env is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, env)
        got = tuple(cake_backend._handoff_mode(m) for m in ("auto", "on", "off"))
        assert got == ("auto", "on", "off"), env
    for bad in ("ON", "force", None, True, 1):
        with pytest.raises(ValueError, match="checkpoint_handoff"):
            cake_backend._handoff_mode(bad)


def test_cake_kda_nondet_passes_the_mode_to_the_node(monkeypatch) -> None:
    """Host only: ``cake_kda_nondet`` hands ``CakeKDANondetFunction`` the keyword's mode (the removed variable changes
    nothing), and ``"off"`` under ``no_grad`` (no node is recorded: no backward would take a record)."""
    seen: list = []

    class Spy:
        @staticmethod
        def apply(*args):
            seen.append(args[11])

    monkeypatch.setattr(cake_backend, "CakeKDANondetFunction", Spy)
    ops = [torch.zeros(1) for _ in range(6)]
    monkeypatch.setenv(ENV, "on")
    cake_backend.cake_kda_nondet(*ops)
    with torch.no_grad():
        cake_backend.cake_kda_nondet(*ops, checkpoint_handoff="on")
    assert seen == ["auto", "off"], seen


def test_the_apply_slot_refusals() -> None:
    """Host only: ``CakeKDANondetFunction.apply`` written for the deterministic Function's signature (its
    ``use_gate_in_kernel`` ``bool`` right after ``doc_lengths`` lands in the slot) fails with a ``TypeError`` naming
    the slot, and an unknown mode with a ``ValueError``, before any operand check or launch."""
    ops = [torch.zeros(1) for _ in range(6)]
    for flag in (False, True):
        with pytest.raises(TypeError, match="checkpoint_handoff"):
            cake_backend.CakeKDANondetFunction.apply(
                *ops, SCALE, None, None, None, None, flag
            )
    with pytest.raises(ValueError, match="checkpoint_handoff"):
        cake_backend.CakeKDANondetFunction.apply(
            *ops, SCALE, None, None, None, None, "force"
        )


def test_the_recompute_is_told_apart_by_its_saved_tensor_hooks() -> None:
    """Host only (CPU tensors): torch's innermost saved-tensor hooks and graph task as ``_handoff_request`` reads
    them.  Non-reentrant: the original forward sees the region's pack hook outside any backward, the recompute
    the recompute hook inside it.  Reentrant: no hooks, the first pass outside any backward without grad, the
    recompute inside it.  A caller's hooks are ``"other"``.  Nested non-reentrant regions: the outer recompute
    re-runs the inner region's original forward inside the backward under the inner pack hook (refused), and the
    inner recompute runs last under its recompute hook."""
    seen: list = []

    def fn(x):
        seen.append(
            (
                cake_backend._handoff_hooks(),
                cake_backend._handoff_task() not in (None, -1),
                torch.is_grad_enabled(),
            )
        )
        return x.sin()

    assert cake_backend._handoff_hooks() == "none"
    assert cake_backend._handoff_task() == -1
    x = torch.randn(8, requires_grad=True)
    checkpoint(fn, x, use_reentrant=False).sum().backward()
    assert seen == [("original", False, True), ("recompute", True, True)], seen
    seen.clear()
    checkpoint(fn, x, use_reentrant=True).sum().backward()
    assert seen == [("none", False, False), ("none", True, True)], seen
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        assert cake_backend._handoff_hooks() == "other"
    seen.clear()

    def outer(x_):
        return checkpoint(fn, x_, use_reentrant=False).cos()

    checkpoint(outer, x, use_reentrant=False).sum().backward()
    assert seen[0] == ("original", False, True), seen
    assert ("original", True, True) in seen[1:-1], seen
    assert seen[-1] == ("recompute", True, True), seen


class _ProbeFunction(torch.autograd.Function):
    """Identity whose forward records what ``_handoff_request`` reads there (it runs inside
    ``CakeKDANondetFunction.forward``): the hooks, whether a graph task runs, grad mode and ``needs_input_grad``."""

    @staticmethod
    def forward(ctx, x, seen):
        seen.append(
            (
                cake_backend._handoff_hooks(),
                cake_backend._handoff_task() not in (None, -1),
                torch.is_grad_enabled(),
                bool(ctx.needs_input_grad[0]),
            )
        )
        ctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def test_a_function_forward_sees_the_recompute_hooks_without_grad_mode() -> None:
    """Host only (CPU tensors): torch runs an autograd Function's forward with grad mode off, also in the checkpoint
    recompute (so ``_handoff_request`` cannot use ``torch.is_grad_enabled()``; ``cake_kda_nondet`` passes ``"off"`` when
    grad mode is off at its call), while the saved-tensor hooks, the graph task and ``needs_input_grad`` there tell
    the recompute apart."""
    seen: list = []

    def fn(x_):
        return _ProbeFunction.apply(x_, seen).sin()

    x = torch.randn(8, requires_grad=True)
    checkpoint(fn, x, use_reentrant=False).sum().backward()
    modes = [s[:3] for s in seen]
    assert modes == [("original", False, False), ("recompute", True, False)], seen
    assert seen[-1][3], seen
    seen.clear()
    checkpoint(fn, x, use_reentrant=True).sum().backward()
    modes = [s[:3] for s in seen]
    assert modes == [("none", False, False), ("none", True, False)], seen
    assert seen[-1][3], seen


def test_compile_compiled_autograd_and_dispatch_modes_block(monkeypatch) -> None:
    """Host only: ``torch.compiler.is_compiling()``, compiled autograd's flags (on, missing or not ``bool``: fail
    closed) and an active Python dispatch mode block the hand-off; nothing blocks it outside them."""
    from torch.utils._python_dispatch import TorchDispatchMode

    monkeypatch.delitem(sys.modules, "torch._dynamo.compiled_autograd", raising=False)
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not cake_backend._handoff_blocked()
    assert not cake_backend._handoff_dispatch_modes()
    with monkeypatch.context() as m:
        m.setattr(torch.compiler, "is_compiling", lambda: True, raising=False)
        assert cake_backend._handoff_blocked()
    names = ("in_compiled_autograd_region", "compiled_autograd_enabled")
    cases = (
        ({names[0]: False, names[1]: False}, False),
        ({names[0]: True, names[1]: False}, True),
        ({names[0]: False, names[1]: True}, True),
        ({names[1]: False}, True),  # a flag this module cannot read
        ({names[0]: 0, names[1]: False}, True),  # not a bool
    )
    for flags, want in cases:
        fake = types.ModuleType("torch._dynamo.compiled_autograd")
        for name, value in flags.items():
            setattr(fake, name, value)
        monkeypatch.setitem(sys.modules, "torch._dynamo.compiled_autograd", fake)
        assert cake_backend._handoff_blocked() is want, flags

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with Passthrough():
        assert cake_backend._handoff_dispatch_modes()
    assert not cake_backend._handoff_dispatch_modes()


def test_the_real_compiled_autograd_flags_block_inside_its_context() -> None:
    """Host only: this torch's ``torch._dynamo.compiled_autograd`` has both flags as ``bool`` (else the hand-off
    fails closed in every process that imports it), off outside its enable context and blocking inside it."""
    ca = pytest.importorskip("torch._dynamo.compiled_autograd")
    for name in ("in_compiled_autograd_region", "compiled_autograd_enabled"):
        value = getattr(ca, name, None)
        assert isinstance(value, bool), (name, value)
    if not torch.cuda.is_available():
        pytest.skip("the capture check needs CUDA")
    assert not cake_backend._handoff_blocked()
    enable = getattr(ca, "_enable", None) or getattr(ca, "enable", None)
    if enable is None:
        pytest.skip("this torch exposes no compiled-autograd enable context")
    try:
        ctx = enable(lambda gm: gm)
        ctx.__enter__()
    except Exception as exc:
        pytest.skip(
            f"the compiled-autograd enable context cannot be entered here: {exc!r}"
        )
    try:
        assert cake_backend._handoff_blocked()
    finally:
        ctx.__exit__(None, None, None)
    assert not cake_backend._handoff_blocked()


def test_memory_report_covers_the_handoff() -> None:
    """Host only: ``memory_report(...)["handoff_bytes"]`` is one tape record of the plan (the tapes; the forward's
    w where the record allocates it; two state blocks on the segmented route when the light set saves none);
    ``handoff_bytes_rows`` one row record (772 B more per internal token-head, 256 B more for ``kb`` where the
    record keeps no ``w``; ``None`` where the twin is not delivered for the architecture); ``handoff_budget_bytes``
    two tape records (the budget of a recompute that asks for a tape record); the per-call arenas of a backward from
    a tape record (``bwd_tape``) and from a row record (``HANDOFF_LAYOUT``: 516 B less per internal token-head, no
    ``gk`` / ``beta_int`` slots)."""
    for lengths in (P8, (100, 640, 77, 1000, 129), (63, 1, 65)):
        S, N = sum(lengths), len(lengths)
        for arch in ARCHES:
            for sm in (148, 152):
                for H in (1, 2, 4, 8, 12, 16, 32, 64):
                    tag = (_shape_id((lengths, H)), arch, sm)
                    p = cake_backend.plan(S, N, H, sm, arch)
                    want = p.NC_pad * H * D * D * 2 + p.R * H * D * 2
                    if p.dispatched("wy_fwd", "fwd") and not p.light_saves_wu:
                        want += p.R * H * D * 2
                    if p.recur == "seg" and not p.seg_save:
                        want += 2 * p.NSEG * H * D * D * 4
                    assert p.handoff_bytes() == want, tag
                    assert p.handoff_twin == TWINS[p.record("fwd_intra", "fwd")], tag
                    want_rows = want + p.R * H * ROW_BYTES
                    if not p.handoff_keeps_w:
                        want_rows += p.R * H * ROW_KB_BYTES
                    assert p.handoff_bytes(rows=True) == want_rows, tag
                    tape = p.workspace_bytes("bwd_tape")
                    lean = p.workspace_bytes(cake_backend.HANDOFF_LAYOUT)
                    assert lean == tape - ROW_ARENA_SAVING * p.R * H, tag
                    for save_mode in ("light", "tape"):
                        report = cake_backend.memory_report(
                            S, N, H, save_mode, sm_count=sm, arch=arch
                        )
                        assert report["handoff_bytes"] == want, tag
                        assert report["handoff_bytes_rows"] == want_rows, tag
                        assert (
                            report["handoff_budget_bytes"]
                            == cake_backend._HANDOFF_BUDGET_RECORDS * want
                        ), tag
                        assert report["arena_bwd_tape"] == tape, tag
                        assert report["arena_bwd_handoff"] == lean, tag


def test_the_row_record_rule(monkeypatch) -> None:
    """Host only (a stand-in plan): an engaged recompute asks for a row record when the plan has an intra twin, the
    call runs on its own workspace, the head count is a ``HANDOFF_HEADS`` one, the device (architecture, SM count)
    has an entry in ``HANDOFF_ROWS_VERIFIED`` (else ``rows_unverified``, the build not read), the toolkit of the one
    build of the twin, its intra record and its light prep completes the key (another toolkit: ``rows_unverified``;
    a mixed build or an unreadable toolkit: ``rows_unready``) and the twin loads (else ``rows_unready``);
    ``_HANDOFF_ROWS`` overrides all of it but the twin.  The shipped entries name exported architectures and
    ``major.minor.patch`` toolkits, each device also in ``HANDOFF_VERIFIED`` with its ``major.minor`` (a row record
    is a hand-off record first)."""
    for key in cake_backend.HANDOFF_ROWS_VERIFIED:
        arch, sms, toolkit = key
        assert arch in ARCHES and sms > 0, key
        assert re.fullmatch(r"\d+\.\d+\.\d+", toolkit), key
        assert (arch, sms, toolkit.rsplit(".", 1)[0]) in cake_backend.HANDOFF_VERIFIED
    heads = min(cake_backend.HANDOFF_HEADS)
    gap = next(h for h in range(heads, 1024) if h not in cake_backend.HANDOFF_HEADS)
    arch = ARCHES[0]
    monkeypatch.setattr(
        cake_backend, "HANDOFF_ROWS_VERIFIED", frozenset({(arch, 152, "9.9.1")})
    )
    asked: list[tuple] = []
    state = {"toolkit": "9.9.1", "loads": True}

    def rows_toolkit(twin, parent, prep, arch_):
        asked.append(("toolkit", twin, parent, prep, arch_))
        return state["toolkit"]

    def rows_loads(twin, arch_):
        asked.append(("load", twin, arch_))
        return state["loads"]

    monkeypatch.setattr(cake_backend, "_handoff_rows_toolkit", rows_toolkit)
    monkeypatch.setattr(cake_backend, "_handoff_rows_loads", rows_loads)

    def plan_(twin="twin", sms=152, H=heads, arch_=arch):
        return types.SimpleNamespace(
            handoff_twin=twin,
            arch=arch_,
            sm_count=sms,
            H=H,
            light_path="light",
            record=lambda stage, path: f"{stage}@{path}",
        )

    other_arch = plan_(arch_=ARCHES[1])
    # (override, plan, own workspace, toolkit, loads, row record, counter, checks asked)
    cases = (
        (None, plan_(), True, "9.9.1", True, True, None, 2),
        (None, plan_(sms=148), True, "9.9.1", True, False, "rows_unverified", 0),
        (None, other_arch, True, "9.9.1", True, False, "rows_unverified", 0),
        (None, plan_(), True, "9.9.2", True, False, "rows_unverified", 1),
        (None, plan_(), True, None, True, False, "rows_unready", 1),
        (None, plan_(), True, "9.9.1", False, False, "rows_unready", 2),
        (None, plan_(), False, "9.9.1", True, False, None, 0),
        (None, plan_(H=gap), True, "9.9.1", True, False, None, 0),
        (None, plan_(H=1), True, "9.9.1", True, False, None, 0),
        (None, plan_(twin=None), True, "9.9.1", True, False, None, 0),
        (False, plan_(), True, "9.9.1", True, False, None, 0),
        (True, plan_(sms=148, H=gap), False, None, False, True, None, 0),
        (True, plan_(twin=None), True, "9.9.1", True, False, None, 0),
    )
    for override, p, own_ws, toolkit, loads, want, counter, checks in cases:
        tag = (override, p.arch, p.sm_count, p.H, p.handoff_twin, own_ws, toolkit)
        cake_backend._handoff_reset()
        asked.clear()
        monkeypatch.setattr(cake_backend, "_HANDOFF_ROWS", override)
        state.update(toolkit=toolkit, loads=loads)
        assert cake_backend._handoff_wants_rows(p, 0, own_ws) is want, tag
        s = cake_backend._handoff_stats(0)
        for name in ("rows_unverified", "rows_unready"):
            assert s[name] == int(name == counter), (tag, s)
        # the build is read last but one, on the twin, the plan's intra record and its light prep; the load last
        expected = [
            ("toolkit", "twin", "fwd_intra@fwd", "bwd_prep@light", p.arch),
            ("load", "twin", p.arch),
        ]
        assert asked == expected[:checks], (tag, asked)


def test_the_row_record_toolkit_reads_one_build(
    monkeypatch, _fresh_rows_toolkit
) -> None:
    """Host only (the build state patched): the toolkit of the twin, the plan's intra record and its light prep is
    the builder's sidecar ``cuda`` when all three are prebuilt and the JIT's ``nvcc`` version when none of them and
    no library of the package is, as ``major.minor.patch``; a mixed state, a version without its patch level, an
    architecture the twin is not delivered for or any other exception gives ``None``.  Cached per arguments."""
    state = {"prebuilt": set(), "sidecar": "13.4.92", "nvcc": "13.4.93"}
    reads: list[str] = []
    others = {"other_sm_100a": {}, "twin_sm_100a": {}, "parent_sm_100a": {}}

    def library_name(name, arch):
        if name == "undelivered":
            raise RuntimeError("not delivered for this architecture")
        return f"{name}_{arch}"

    def prebuilt_info(path, arch):
        reads.append("sidecar")
        return {"cuda": state["sidecar"]}

    def nvcc():
        reads.append("nvcc")
        if state["nvcc"] is None:
            raise FileNotFoundError("nvcc")
        return cake_backend._full_version(state["nvcc"])

    monkeypatch.setattr(cake_jit, "library_name", library_name)
    monkeypatch.setattr(
        cake_build, "prebuilt_path", lambda name: _Built(name in state["prebuilt"])
    )
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: others)
    monkeypatch.setattr(cake_build, "_prebuilt_info", prebuilt_info)
    monkeypatch.setattr(cake_backend, "_jit_nvcc_version", nvcc)
    three = {"twin_sm_100a", "parent_sm_100a", "prep_sm_100a"}
    # (prebuilt, sidecar cuda, nvcc version, twin, toolkit, the read)
    cases = (
        (three, "13.4.92", "13.4.93", "twin", "13.4.92", ["sidecar"]),
        (set(), "13.4.92", "13.4.93", "twin", "13.4.93", ["nvcc"]),
        (three, "13.4", "13.4.93", "twin", None, ["sidecar"]),
        (set(), "13.4.92", None, "twin", None, ["nvcc"]),
        ({"other_sm_100a"}, "13.4.92", "13.4.93", "twin", None, []),
        ({"twin_sm_100a"}, "13.4.92", "13.4.93", "twin", None, []),
        (three - {"prep_sm_100a"}, "13.4.92", "13.4.93", "twin", None, []),
        (three, "13.4.92", "13.4.93", "undelivered", None, []),
    )
    for prebuilt, sidecar, nvcc_version, twin, want, read in cases:
        tag = (sorted(prebuilt), sidecar, nvcc_version, twin)
        cake_backend._handoff_rows_toolkit.cache_clear()
        reads.clear()
        state.update(prebuilt=set(prebuilt), sidecar=sidecar, nvcc=nvcc_version)
        got = cake_backend._handoff_rows_toolkit(twin, "parent", "prep", "sm_100a")
        assert got == want, (tag, got)
        assert reads == read, (tag, reads)
        # cached: a second call reads nothing again
        state.update(prebuilt=set(), sidecar="0.0.0", nvcc="0.0.0")
        again = cake_backend._handoff_rows_toolkit(twin, "parent", "prep", "sm_100a")
        assert again == want, (tag, again)
        assert reads == read, (tag, reads)
    for text, want in (
        ("13.4.92", "13.4.92"),
        ("13.04.092", "13.4.92"),
        ("13.4.92.1", "13.4.92"),
        ("13.4", None),
        ("13.x.92", None),
    ):
        assert cake_backend._full_version(text) == want, text


def test_a_failed_twin_load_is_retried_a_bounded_number_of_times(monkeypatch) -> None:
    """Host only: a failed load of the twin is counted, not kept (the next recompute loads again) up to
    ``_HANDOFF_ROWS_LOAD_TRIES`` failures per twin and architecture; after those the process stops loading it
    until ``_handoff_reset``.  A load that succeeds is asked every time (``cake_jit.load`` keeps the module)."""
    loads: list[tuple] = []
    state = {"fail": True}

    def load(name, arch):
        loads.append((name, arch))
        if state["fail"]:
            raise RuntimeError("JIT lock timeout")
        return types.SimpleNamespace()

    monkeypatch.setattr(cake_jit, "load", load)
    tries = cake_backend._HANDOFF_ROWS_LOAD_TRIES
    assert tries >= 2
    for _ in range(tries + 2):
        assert cake_backend._handoff_rows_loads("twin", "sm_100a") is False
    assert loads == [("twin", "sm_100a")] * tries, loads
    # another architecture counts apart; a success is not counted
    state["fail"] = False
    assert cake_backend._handoff_rows_loads("twin", "sm_103a") is True
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is False
    cake_backend._handoff_reset()
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True
    assert loads[tries:] == [("twin", "sm_103a")] + [("twin", "sm_100a")] * 2, loads
    # a transient failure then a success: the success counts
    cake_backend._handoff_reset()
    state["fail"] = True
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is False
    state["fail"] = False
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True


def _host_record(*, rows: bool, kb: bool = False, task: int = 7):
    """A CPU hand-off record and its byte count: 1000 B of tapes and, for a row record, 640 B of rows (and 256 B
    of ``kb``)."""
    row_tensors = {}
    if rows:
        row_tensors = {
            "gk": torch.zeros(100, dtype=torch.float32),
            "kg": torch.zeros(100, dtype=torch.bfloat16),
            "beta_int": torch.zeros(10, dtype=torch.float32),
        }
        if kb:
            row_tensors["kb"] = torch.zeros(128, dtype=torch.bfloat16)
    nbytes = 1000 + sum(t.numel() * t.element_size() for t in row_tensors.values())
    tensors = {
        "h": torch.zeros(250, dtype=torch.float32),
        "v_new": torch.zeros(1, dtype=torch.bfloat16),
        "w": None,
        "seg_m": None,
        **row_tensors,
    }
    request = {
        "index": 0,
        "task": task,
        "key": ("key",),
        "stream": 0,
        "nbytes": nbytes,
        "tensors": tensors,
    }
    return request, nbytes


def test_the_budget_demotes_row_records_before_it_refuses(monkeypatch) -> None:
    """Host only (CPU records): a call that asks for a tape record is admitted within ``_HANDOFF_BUDGET_RECORDS``
    tape records of its shape, one that asks for a row record within the larger of that and one row record.  A row
    record is admitted when the live records plus it fit; otherwise, and for a call that asks for a tape record, a
    tape record is admitted when the live records plus it fit, every live row record of the device first becoming
    a tape record (its rows and their bytes dropped, ``demoted``) where they do not (``refused_budget`` when it
    still does not fit).  Publishing counts ``published_rows`` for a row record; the end-of-task drop releases the
    records."""
    tape = 1000
    req_a, rows = _host_record(rows=True)
    assert rows == 1640
    aqk_a, aqk_b = torch.zeros(4), torch.zeros(4)
    inputs = [torch.zeros(2) for _ in range(6)]

    def live(aqk):
        return cake_backend._handoff_devices[0].records[aqk.data_ptr()]

    def dropped(rec):
        return all(t is None for t in (rec.gk, rec.kg, rec.beta_int, rec.kb))

    assert cake_backend._handoff_admit(0, 7, tape, rows) == "rows"  # an empty device
    cake_backend._handoff_publish(req_a, aqk_a, inputs)
    s = cake_backend._handoff_stats(0)
    assert s["published"] == s["published_rows"] == 1 and s["live_bytes"] == rows, s
    rec_a = live(aqk_a)
    # a call that asks for a tape record (1640 + 1000 > 2000): the live row record is demoted, the tape record
    # admitted
    assert cake_backend._handoff_admit(0, 7, tape, None) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 1 and s["live_bytes"] == tape and rec_a.nbytes == tape, s
    assert dropped(rec_a) and rec_a.h is not None and rec_a.v_new is not None
    req_b, _ = _host_record(rows=False)
    cake_backend._handoff_publish(req_b, aqk_b, inputs)
    s = cake_backend._handoff_stats(0)
    assert s["published"] == 2 and s["published_rows"] == 1, s
    assert s["live_bytes"] == 2 * tape, s
    # two tape records live: neither kind fits, nothing left to demote
    assert cake_backend._handoff_admit(0, 7, tape, rows) is None
    assert cake_backend._handoff_admit(0, 7, tape, None) is None
    s = cake_backend._handoff_stats(0)
    assert s["refused_budget"] == 2 and s["demoted"] == 1, s
    cake_backend._handoff_task_end(0, 7)
    s = cake_backend._handoff_stats(0)
    assert s["live_records"] == 0 and s["live_bytes"] == 0 and s["misses"] == 2, s
    # a second row record does not fit (1640 + 1640 > max(2000, 1640)): the live one is demoted, a tape record
    # admitted (1000 + 1000 <= 2000)
    cake_backend._handoff_reset()
    assert cake_backend._handoff_admit(0, 7, tape, rows) == "rows"
    cake_backend._handoff_publish(_host_record(rows=True)[0], aqk_a, inputs)
    assert cake_backend._handoff_admit(0, 7, tape, rows) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 1 and s["live_bytes"] == tape and dropped(live(aqk_a)), s
    # a row record larger than two tape records: the budget of a call that asks for one is one row record
    cake_backend._handoff_reset()
    assert cake_backend._handoff_admit(0, 7, 500, rows) == "rows"
    req_c, _ = _host_record(rows=True, kb=True)
    cake_backend._handoff_publish(req_c, aqk_a, inputs)
    assert cake_backend._handoff_stats(0)["live_bytes"] == rows + 256
    # 1896 live + 1640 > max(1000, 1640); 1896 + 500 too: demoted to its 1000 B of tapes, then 1000 + 500 fits
    assert cake_backend._handoff_admit(0, 7, 500, rows) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 1 and s["live_bytes"] == 1000, s
    # a call that asks for a tape record of that shape: 1000 + 500 > 2 * 500, nothing left to demote
    assert cake_backend._handoff_admit(0, 7, 500, None) is None
    # a live row record that fits as it is stays one: 1640 + 2000 <= 2 * 2000
    cake_backend._handoff_reset()
    cake_backend._handoff_publish(_host_record(rows=True)[0], aqk_a, inputs)
    assert cake_backend._handoff_admit(0, 7, 2000, None) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 0 and not dropped(live(aqk_a)), s
    # a turned-off device admits nothing
    cake_backend._handoff_devices[0].disabled = True
    assert cake_backend._handoff_admit(0, 7, 1, 1) is None
