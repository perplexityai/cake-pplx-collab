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

Public-API suite of ``cake_pplx.cake_engram_gate_nondet_train``, the non-deterministic Engram context gate.

Without a device: the entries are stable APIs, importing the module imports no backend, the module reads no
environment variable, every keyword of an entry is the backend's keyword of the same name, the autograd entries and
the explicit forward take the keyword-only ``dot_sum="compensated"`` (the explicit backward and the workspace query do
not) and refuse another choice before they look at the operands, the entries document their behaviour under
``torch.use_deterministic_algorithms``, operands on the CPU are a ``ValueError``, and a device of capability (10, 7)
maps to ``sm_107a``.

On SM100 / SM103 / SM107, for both ``dot_sum`` choices unless a test names the default: each choice launches its own
forward record and the one backward record (also under checkpoint), and the gradients of both choices are the same
bits (the weight gradients of the non-deterministic package excepted); the autograd entry
(``key`` and ``value`` contiguous or views of a packed ``kv``; BF16, FP32 or mixed weights; with and without a mask)
and the packed entry meet the naive BF16 tier against an FP64 oracle (``tests/cake_naive.py``, 32 draws) for the
output and every gradient; the packed entry equals the split entry bit
for bit (and its ``kv`` gradient is the split gradients side by side); views and contiguous operands give the same
bits; the explicit backward writes caller views into one ``dkv``; a masked token gives ``out = x`` and
``dx = dout`` exactly, zero ``dkey`` and ``dvalue`` and no weight-gradient contribution; rows with
``|dot| <= 5e-7`` get no gradient through the gate; an all-zero ``x`` or ``key`` stream gives a finite ``out`` and no
gradient through the gate; ``T = 0`` and ``T = 1`` work; every documented ``ValueError``; the
``torch.use_deterministic_algorithms`` behaviour; with the default choice, the launches run on the current stream
without synchronizing, a call captured in a CUDA graph replays, ``torch.utils.checkpoint(use_reentrant=False)`` gives
the gradients of plain autograd and the retained state and the workspace are reported; a device outside the
architectures is a ``RuntimeError``.  The issue's accuracy bound judges both choices
(``test_cake_engram_gate_acceptance.py``).
The weight gradients promise no bits: they are compared bit for bit with nothing; ``out``, ``dx``, ``dkey`` and
``dvalue`` have one fixed order.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.utils.checkpoint

import cake_pplx.cake_engram_gate_nondet_train as api
from tests.cake.test_helpers.cake_engram_gate_naive import (
    HIDDEN,
    KV_WIDTH,
    STREAMS,
    gate_outputs,
    judge_draws,
    make_inputs,
)

BACKEND = "cake_pplx._backends.cake_engram_gate_nondet_train"
ENTRIES = (
    "cake_engram_gate_nondet",
    "cake_engram_gate_nondet_kv",
    "cake_engram_gate_nondet_train_backward",
    "cake_engram_gate_nondet_train_forward",
    "cake_engram_gate_nondet_train_workspace_bytes",
    "is_cake_engram_gate_nondet_train_supported",
)
GRADS = ("dx", "dkey", "dvalue", "dq", "dk")
# The ``dot_sum`` choices (the default first) and the forward record suffix each launches; every backward launches
# ``bwd_dot2``.
DOT_SUMS = {"compensated": "dot2", "fp32": "f32"}
# The entries that take ``dot_sum``: the forward's sum (the backward is the same for both).
SUM_ENTRIES = ("cake_engram_gate_nondet", "cake_engram_gate_nondet_kv", "cake_engram_gate_nondet_train_forward")
# The gradients with one fixed order (the weight gradients are summed with atomics).
EXACT_GRADS = ("dx", "dkey", "dvalue")
# Cycles of the sleep kernel queued before a call on a side stream (about 0.1 s): the call returns long before it ends.
SLEEP_CYCLES = 200_000_000
ALERT = "adds the weight gradients with float atomics"


def _backend():
    from cake_pplx._backends.cake_engram_gate_nondet_train import cake_backend

    return cake_backend


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake Engram gate ships for SM100 / SM103 / SM107")
    from cake_pplx._backends.cake_engram_gate_nondet_train import cake_jit

    if not (Path(cake_jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    _build_libraries(0)
    return torch.device("cuda:0")


@functools.cache
def _build_libraries(index: int) -> None:
    """Build the device's libraries once, before the first judged or stream-ordered call."""
    backend = _backend()
    for spec in backend.specs_for_arch(backend.device_arch(index)):
        spec.build_and_load()


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool):
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)


def _leaves(ns):
    """Fresh autograd leaves of the operands of ``ns``: ``(x, key, value, kv, q_weight, k_weight)`` (``key`` and
    ``value`` are views of the ``kv`` leaf when ``ns`` is packed)."""
    x = ns.x.detach().clone().requires_grad_(True)
    qw = ns.q_weight.detach().clone().requires_grad_(True)
    kw = ns.k_weight.detach().clone().requires_grad_(True)
    if ns.kv is not None:
        kv = ns.kv.detach().clone().requires_grad_(True)
        key = kv[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN))
        return x, key, kv[:, STREAMS * HIDDEN :], kv, qw, kw
    key = ns.key.detach().clone().requires_grad_(True)
    value = ns.value.detach().clone().requires_grad_(True)
    return x, key, value, None, qw, kw


def _autograd(ns, *, packed: bool = False, checkpoint: bool = False, dot_sum: Optional[str] = None) -> dict:
    """Output and gradients of the autograd entry (``packed``: the ``kv`` entry; ``checkpoint``: under
    ``torch.utils.checkpoint(use_reentrant=False)``; ``dot_sum``: passed when given, else the entry's default)."""
    x, key, value, kv, qw, kw = _leaves(ns)
    sums = {} if dot_sum is None else {"dot_sum": dot_sum}
    if packed:
        fn = functools.partial(api.cake_engram_gate_nondet_kv, x, kv, qw, kw, ns.token_mask, **sums)
    else:
        fn = functools.partial(api.cake_engram_gate_nondet, x, key, value, qw, kw, ns.token_mask, **sums)
    out = torch.utils.checkpoint.checkpoint(fn, use_reentrant=False) if checkpoint else fn()
    out.backward(ns.dout)
    if kv is not None:
        dkey = kv.grad[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN))
        dvalue = kv.grad[:, STREAMS * HIDDEN :]
    else:
        dkey, dvalue = key.grad, value.grad
    found = {"out": out.detach(), "dx": x.grad, "dkey": dkey, "dvalue": dvalue, "dq": qw.grad, "dk": kw.grad}
    if kv is not None:
        found["dkv"] = kv.grad
    return found


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bytes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _record_failures(judge, failures, record_property) -> None:
    judge.record(record_property)
    assert not failures, failures


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_the_entries_are_stable_apis() -> None:
    for name in ENTRIES:
        entry = getattr(api, name)
        assert getattr(entry, "is_experimental", False) is False, name
        assert entry.__name__ == name


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import cake_pplx.cake_engram_gate_nondet_train as api\n"
        "names = dir(api)\n"
        f"print(any(m.startswith({BACKEND!r}) for m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_module_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(api.__file__).read_text())
    hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("environ", "getenv")]
    assert not hits, hits


def test_keywords_match_the_backend() -> None:
    for name in ENTRIES:
        ours = inspect.signature(inspect.unwrap(getattr(api, name))).parameters
        theirs = inspect.signature(getattr(_backend(), name)).parameters
        assert list(ours) == list(theirs), name
        for key, param in ours.items():
            assert param.kind == theirs[key].kind, (name, key)
            assert param.default == theirs[key].default, (name, key)
    gate = inspect.signature(api.cake_engram_gate_nondet).parameters
    assert gate["eps"].kind is inspect.Parameter.KEYWORD_ONLY and gate["eps"].default == 1e-20
    assert gate["clamp_value"].kind is inspect.Parameter.KEYWORD_ONLY and gate["clamp_value"].default == 1e-6
    assert gate["token_mask"].default is None
    for name in SUM_ENTRIES:
        param = inspect.signature(inspect.unwrap(getattr(api, name))).parameters["dot_sum"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default == "compensated", name
    for name in sorted(set(ENTRIES) - set(SUM_ENTRIES)):
        assert "dot_sum" not in inspect.signature(inspect.unwrap(getattr(api, name))).parameters, name
    for cls in ("CakeEngramGateNondetFunction", "CakeEngramGateNondetKvFunction"):
        assert list(inspect.signature(getattr(_backend(), cls).forward).parameters)[-1] == "dot_sum", cls


def test_another_dot_sum_is_refused_before_the_operands() -> None:
    """Every entry that takes ``dot_sum`` (and each autograd ``Function``) raises ``ValueError`` for another ``str``
    and ``TypeError`` for a non-``str``, naming the choices, before it looks at the operands (CPU operands here)."""
    ns = make_inputs(2, seed=1, device=torch.device("cpu"))
    backend = _backend()
    calls = {
        "cake_engram_gate_nondet": lambda **kw: api.cake_engram_gate_nondet(
            ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, **kw
        ),
        "cake_engram_gate_nondet_kv": lambda **kw: api.cake_engram_gate_nondet_kv(
            ns.x, torch.zeros(2, KV_WIDTH, dtype=torch.bfloat16), ns.q_weight, ns.k_weight, **kw
        ),
        "cake_engram_gate_nondet_train_forward": lambda **kw: api.cake_engram_gate_nondet_train_forward(
            ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, **kw
        ),
        "CakeEngramGateNondetFunction": lambda dot_sum: backend.CakeEngramGateNondetFunction.apply(
            ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, None, 1e-20, 1e-6, dot_sum
        ),
        "CakeEngramGateNondetKvFunction": lambda dot_sum: backend.CakeEngramGateNondetKvFunction.apply(
            ns.x, torch.zeros(2, KV_WIDTH, dtype=torch.bfloat16), ns.q_weight, ns.k_weight, None, 1e-20, 1e-6, dot_sum
        ),
    }
    assert set(SUM_ENTRIES) <= set(calls)
    for label, call in calls.items():
        for bad in ("dot2", "f32", "FP32", ""):
            with pytest.raises(ValueError, match=r"dot_sum must be one of \['compensated', 'fp32'\]"):
                call(dot_sum=bad)
                pytest.fail(f"{label} accepted dot_sum={bad!r}")
        for bad in (None, 1):
            with pytest.raises(TypeError, match=r"dot_sum must be a str, one of \['compensated', 'fp32'\]"):
                call(dot_sum=bad)
                pytest.fail(f"{label} accepted dot_sum={bad!r}")
    # Both choices pass the check; the CPU operands are refused next.
    for choice in DOT_SUMS:
        with pytest.raises(ValueError, match="CUDA"):
            calls["cake_engram_gate_nondet"](dot_sum=choice)


def test_the_entries_document_their_determinism() -> None:
    """Each entry states its behaviour under ``torch.use_deterministic_algorithms``."""
    for name in ENTRIES:
        assert "use_deterministic_algorithms" in getattr(api, name).__doc__, name


def test_cpu_operands_are_refused() -> None:
    ns = make_inputs(2, seed=1, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA"):
        api.cake_engram_gate_nondet(ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight)
    with pytest.raises(ValueError, match="CUDA"):
        api.cake_engram_gate_nondet_train_backward(ns.dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight)
    assert api.is_cake_engram_gate_nondet_train_supported("cpu") is False


def test_a_device_of_capability_10_7_is_served(monkeypatch) -> None:
    """A device of capability (10, 7) is the ``sm_107a`` target, and ``build.json`` holds an ``sm_107a`` library for
    every record (nothing is built or launched here)."""
    from cake_pplx._backends.cake_engram_gate_nondet_train import cake_jit

    path = Path(cake_jit.__file__).parent / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    device_arch = _backend().device_arch
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    device_arch.cache_clear()
    try:
        assert device_arch(0) == "sm_107a"
    finally:
        monkeypatch.undo()
        device_arch.cache_clear()
    entries = [entry for entry in json.loads(path.read_text())["libraries"] if entry["arch"] == "sm_107a"]
    assert sorted(e["name"] for e in entries) == sorted(f"{r['library']}_sm_107a" for r in cake_jit.MODULES.values())


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: the records of each dot_sum choice


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize("packed", (False, True), ids=("split", "packed"))
def test_each_dot_sum_launches_its_forward_record_and_the_one_backward(dot_sum, packed, monkeypatch) -> None:
    """``dot_sum`` selects the forward record (``fwd_<acc>``); every backward launches ``bwd_dot2`` then
    ``dw_reduce``, also when ``torch.utils.checkpoint`` recomputes the forward with the forward's choice; the explicit
    forward launches the record of the choice it is given."""
    device = _device_or_skip()
    backend = _backend()
    acc = DOT_SUMS[dot_sum]
    launched: list[str] = []
    entry = backend._entry

    def recording(name, arch):
        run = entry(name, arch)

        def call(*args):
            launched.append(name)
            return run(*args)

        return call

    monkeypatch.setattr(backend, "_entry", recording)
    ns = make_inputs(129, seed=53, device=device, weights="bf16", mask="some", layout="kv")
    _autograd(ns, packed=packed, dot_sum=dot_sum)
    assert launched == [f"fwd_{acc}", "bwd_dot2", "dw_reduce"]
    launched.clear()
    _autograd(ns, packed=packed, checkpoint=True, dot_sum=dot_sum)
    assert launched == [f"fwd_{acc}", f"fwd_{acc}", "bwd_dot2", "dw_reduce"]
    launched.clear()
    args = (ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask)
    api.cake_engram_gate_nondet_train_forward(*args, dot_sum=dot_sum)
    api.cake_engram_gate_nondet_train_backward(ns.dout, *args)
    assert launched == [f"fwd_{acc}", "bwd_dot2", "dw_reduce"]
    if dot_sum == "compensated":
        launched.clear()
        _autograd(ns, packed=packed)
        assert launched == ["fwd_dot2", "bwd_dot2", "dw_reduce"], "the default is the compensated sum"


@pytest.mark.parametrize("packed", (False, True), ids=("split", "packed"))
def test_the_gradients_do_not_depend_on_dot_sum(packed) -> None:
    """The one backward recomputes ``P`` from the inputs: for one ``dout``, both ``dot_sum`` choices give the same
    gradient bits (the weight gradients of the non-deterministic package are compared with nothing)."""
    device = _device_or_skip()
    ns = make_inputs(1031, seed=59, device=device, weights="mixed", mask="some", layout="kv")
    compensated = _autograd(ns, packed=packed, dot_sum="compensated")
    fp32 = _autograd(ns, packed=packed, dot_sum="fp32")
    for name in EXACT_GRADS:
        assert _same_bits(compensated[name], fp32[name]), name


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: accuracy

# (T, weights, mask, layout)
CASES = (
    (1, "fp32", "none", "contiguous"),
    (7, "bf16", "some", "kv"),
    (129, "mixed", "some", "contiguous"),
    (129, "fp32", "none", "kv"),
    (1031, "bf16", "some", "kv"),
)
CASE_IDS = [f"T{T}-{w}-{m}-{lay}" for T, w, m, lay in CASES]


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize(("T", "weights", "mask", "layout"), CASES, ids=CASE_IDS)
def test_the_autograd_entry_matches_the_reference(dot_sum, T, weights, mask, layout, record_property) -> None:
    device = _device_or_skip()

    def run(ns):
        got = _autograd(ns, dot_sum=dot_sum)
        got.pop("dkv", None)
        return got

    def make(seed):
        return make_inputs(T, seed=seed, device=device, weights=weights, mask=mask, layout=layout)

    _record_failures(*judge_draws(run, make, seed=101 + T), record_property)


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize(("T", "weights", "mask"), ((1, "fp32", "none"), (129, "bf16", "some")), ids=("T1", "T129"))
def test_the_packed_entry_matches_the_reference(dot_sum, T, weights, mask, record_property) -> None:
    device = _device_or_skip()

    def run(ns):
        got = _autograd(ns, packed=True, dot_sum=dot_sum)
        got.pop("dkv")
        return got

    def make(seed):
        return make_inputs(T, seed=seed, device=device, weights=weights, mask=mask, layout="kv")

    _record_failures(*judge_draws(run, make, seed=211 + T), record_property)


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize("T", (1, 129, 4096))
def test_the_packed_entry_equals_the_split_entry(T, dot_sum) -> None:
    """The packed entry's output and gradients are the split entry's bits; its ``kv`` gradient is the key and value
    gradients side by side (one ``[T, 25600]`` tensor written by the backward)."""
    device = _device_or_skip()
    ns = make_inputs(T, seed=7, device=device, weights="bf16", mask="some", layout="kv")
    split, packed = _autograd(ns, dot_sum=dot_sum), _autograd(ns, packed=True, dot_sum=dot_sum)
    names = ("out", "dx", "dkey", "dvalue")
    for name in names:
        assert _same_bits(split[name], packed[name]), name
    assert tuple(packed["dkv"].shape) == (T, KV_WIDTH) and packed["dkv"].is_contiguous()
    assert _same_bits(packed["dkv"], torch.cat((split["dkey"].reshape(T, -1), split["dvalue"]), dim=1))


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize("T", (2, 129, 1031))
def test_views_and_contiguous_operands_give_the_same_bits(T, dot_sum) -> None:
    """``key`` / ``value`` as views of ``kv`` and as contiguous tensors, and ``dkey`` / ``dvalue`` as caller views into
    one ``dkv`` and as contiguous tensors, give the same bits."""
    device = _device_or_skip()
    ns = make_inputs(T, seed=13, device=device, weights="fp32", mask="some", layout="kv")
    key_c, value_c = ns.key.contiguous(), ns.value.contiguous()
    fwd = functools.partial(api.cake_engram_gate_nondet_train_forward, dot_sum=dot_sum)
    assert _same_bits(
        fwd(ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask),
        fwd(ns.x, key_c, value_c, ns.q_weight, ns.k_weight, ns.token_mask),
    )
    dkv = torch.full((T, KV_WIDTH), math.nan, dtype=torch.bfloat16, device=device)
    dkey_v = dkv[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN))
    dvalue_v = dkv[:, STREAMS * HIDDEN :]
    bwd = api.cake_engram_gate_nondet_train_backward
    views = bwd(ns.dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask, dkey=dkey_v, dvalue=dvalue_v)
    plain = bwd(ns.dout, ns.x, key_c, value_c, ns.q_weight, ns.k_weight, ns.token_mask)
    assert views[1] is dkey_v and views[2] is dvalue_v
    assert not torch.isnan(dkv.float()).any()
    for name, a, b in zip(GRADS, views, plain, strict=True):
        if name in EXACT_GRADS:
            assert _same_bits(a, b), name


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
def test_a_masked_token_is_exact(dot_sum) -> None:
    """``token_mask[t]`` False: ``out[t] == x[t]`` and ``dx[t] == dout[t]`` bit for bit, ``dkey[t]`` and ``dvalue[t]``
    zero; a call whose tokens are all masked gives zero weight gradients."""
    device = _device_or_skip()
    ns = make_inputs(129, seed=17, device=device, weights="bf16", mask="some", layout="kv")
    off = ~ns.token_mask
    got = _autograd(ns, dot_sum=dot_sum)
    assert _same_bits(got["out"][off], ns.x[off])
    assert _same_bits(got["dx"][off], ns.dout[off])
    assert not got["dkey"][off].float().abs().any() and not got["dvalue"][off].float().abs().any()
    ns.token_mask = torch.zeros_like(ns.token_mask)
    got = _autograd(ns, dot_sum=dot_sum)
    assert _same_bits(got["out"], ns.x) and _same_bits(got["dx"], ns.dout)
    for name in ("dkey", "dvalue", "dq", "dk"):
        assert not got[name].float().abs().any(), name


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
def test_clamped_rows_have_no_gate_gradient(dot_sum) -> None:
    """Streams with ``|dot| <= 5e-7`` (``4e-7`` and ``-4e-7``) get no gradient through the gate (``dx = dout``,
    ``dkey = 0``, no weight gradient); streams with ``|dot| = 3e-6`` have a weight gradient (the dot product is
    ``q_weight[s, 0]`` by construction: ``x = 1``, ``key = 4`` in column 0 and 0 elsewhere, ``k_weight = 1``; the
    near-cancelling draws of the acceptance suite judge rows of that size)."""
    device = _device_or_skip()
    T = 3
    targets = torch.tensor([4e-7, -4e-7, 3e-6, -3e-6])

    def make(seed):
        ns = make_inputs(T, seed=seed, device=device, weights="fp32", scaled=False)
        ns.x = torch.ones_like(ns.x)
        key = torch.zeros_like(ns.key)
        key[:, :, 0] = 4.0
        ns.key = key
        qw = torch.zeros(STREAMS, HIDDEN, device=device)
        qw[:, 0] = targets.to(device)
        ns.q_weight, ns.k_weight = qw, torch.ones_like(qw)
        return ns

    ns = make(5)
    got = _autograd(ns, dot_sum=dot_sum)
    for s in (0, 1):
        assert _same_bits(got["dx"][:, s].contiguous(), ns.dout[:, s].contiguous()), s
        assert not got["dkey"][:, s].float().abs().any(), s
        assert not got["dq"][s].abs().any() and not got["dk"][s].abs().any(), s
    want = gate_outputs(ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, None, ns.dout, torch.float64)
    for s in (2, 3):
        assert got["dq"][s].abs().any() and want["dq"][s].abs().any(), s
        assert torch.isfinite(got["dq"][s]).all() and torch.isfinite(got["dx"][:, s].float()).all(), s


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize("which", ("x", "key"))
def test_an_all_zero_stream_gives_a_finite_output_and_no_gate_gradient(which, dot_sum, record_property) -> None:
    """An all-zero ``x`` or ``key`` stream gives ``dot = +0`` at ``eps = 1e-20``: a finite ``out`` that meets the
    tier, ``dx = dout`` and ``dkey = 0`` on that stream, every output finite."""
    device = _device_or_skip()

    def make(seed):
        ns = make_inputs(16, seed=seed, device=device, weights="fp32")
        t = getattr(ns, which).clone()
        t[:, 1] = 0
        setattr(ns, which, t)
        return ns

    def run(ns):
        got = _autograd(ns, dot_sum=dot_sum)
        for name in ("out", *GRADS):
            assert torch.isfinite(got[name].float()).all(), name
        assert _same_bits(got["dx"][:, 1].contiguous(), ns.dout[:, 1].contiguous())
        assert not got["dkey"][:, 1].float().abs().any()
        return {"out": got["out"]}

    _record_failures(*judge_draws(run, make, seed=401), record_property)


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
def test_no_token(dot_sum) -> None:
    """``T = 0``: an empty output, empty gradients and zero weight gradients; nothing launches, nothing alerts."""
    device = _device_or_skip()
    ns = make_inputs(0, seed=3, device=device, weights="bf16", mask="some", layout="kv")
    with _torch_deterministic(warn_only=False):
        got = _autograd(ns, dot_sum=dot_sum)
        packed = _autograd(ns, packed=True, dot_sum=dot_sum)
        explicit = api.cake_engram_gate_nondet_train_backward(
            ns.dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask
        )
    assert tuple(got["out"].shape) == (0, STREAMS, HIDDEN)
    for result in (got, packed):
        assert result["dq"].dtype == torch.bfloat16 and not result["dq"].float().abs().any()
        assert result["dk"].dtype == torch.bfloat16 and not result["dk"].float().abs().any()
    assert tuple(packed["dkv"].shape) == (0, KV_WIDTH)
    assert explicit[3].dtype == torch.float32 and tuple(explicit[3].shape) == (STREAMS, HIDDEN)
    assert not explicit[3].abs().any() and not explicit[4].abs().any()


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: checks


def _valid(device):
    return make_inputs(4, seed=2, device=device, weights="fp32", mask="some", layout="kv")


def _call_forward(ns, **edit):
    args = dict(
        x=ns.x, key=ns.key, value=ns.value, q_weight=ns.q_weight, k_weight=ns.k_weight, token_mask=ns.token_mask
    )
    kwargs = {k: edit.pop(k) for k in ("eps", "clamp_value", "out") if k in edit}
    args.update(edit)
    return api.cake_engram_gate_nondet_train_forward(**args, **kwargs)


def _malformed(ns, device):
    """``(id, edit)``: one malformed operand each (the documented ``ValueError`` cases)."""
    bf16 = dict(dtype=torch.bfloat16, device=device)
    T = ns.T
    return [
        ("x-fp32", dict(x=ns.x.float())),
        ("x-rank", dict(x=ns.x[0])),
        ("x-streams", dict(x=torch.zeros(T, 3, HIDDEN, **bf16))),
        ("x-width", dict(x=torch.zeros(T, STREAMS, 4096, **bf16))),
        ("x-strided", dict(x=ns.x.transpose(0, 1).contiguous().transpose(0, 1))),
        ("key-tokens", dict(key=torch.zeros(T + 1, STREAMS, HIDDEN, **bf16))),
        ("key-stream-stride", dict(key=torch.zeros(T, HIDDEN, STREAMS, **bf16).transpose(1, 2))),
        ("key-token-stride", dict(key=torch.zeros(T, 20484, **bf16)[:, 4:].unflatten(1, (STREAMS, HIDDEN)))),
        ("value-fp32", dict(value=ns.value.float())),
        ("value-column-stride", dict(value=torch.zeros(HIDDEN, T, **bf16).t())),
        ("value-token-stride", dict(value=torch.zeros(T, 5124, **bf16)[:, 4:])),
        ("q-fp16", dict(q_weight=ns.q_weight.half())),
        ("k-shape", dict(k_weight=torch.ones(STREAMS, 4096, device=device))),
        ("mask-u8", dict(token_mask=ns.token_mask.to(torch.uint8))),
        ("mask-shape", dict(token_mask=torch.ones(T + 1, dtype=torch.bool, device=device))),
        ("x-cpu", dict(x=ns.x.cpu())),
        ("mask-cpu", dict(token_mask=ns.token_mask.cpu())),
        ("eps-nan", dict(eps=math.nan)),
        ("eps-negative", dict(eps=-1.0)),
        ("clamp-inf", dict(clamp_value=math.inf)),
        ("out-strided", dict(out=torch.zeros(STREAMS, T, HIDDEN, **bf16).transpose(0, 1))),
    ]


def test_malformed_operands_are_a_value_error() -> None:
    device = _device_or_skip()
    ns = _valid(device)
    for label, edit in _malformed(ns, device):
        with pytest.raises(ValueError):
            _call_forward(ns, **edit)
            pytest.fail(f"forward accepted {label}")
        if "out" in edit:
            continue
        with pytest.raises(ValueError):
            fields = dict(
                x=ns.x, key=ns.key, value=ns.value, q_weight=ns.q_weight, k_weight=ns.k_weight, token_mask=ns.token_mask
            )
            scalars = {k: edit[k] for k in ("eps", "clamp_value") if k in edit}
            fields.update({k: v for k, v in edit.items() if k not in scalars})
            api.cake_engram_gate_nondet_train_backward(ns.dout, **fields, **scalars)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    T = ns.T
    overlapping = torch.zeros(8 * T + STREAMS * HIDDEN, **bf16).as_strided((T, STREAMS, HIDDEN), (8, HIDDEN, 1))
    for label, dout, outputs in (
        ("dout-fp32", ns.dout.float(), {}),
        ("dout-strided", ns.dout.transpose(0, 1).contiguous().transpose(0, 1), {}),
        ("dx-strided", ns.dout, dict(dx=torch.zeros(T, STREAMS, HIDDEN + 8, **bf16)[..., :HIDDEN])),
        ("dkey-overlapping", ns.dout, dict(dkey=overlapping)),
        ("dvalue-fp32", ns.dout, dict(dvalue=torch.zeros(T, HIDDEN, device=device))),
    ):
        with pytest.raises(ValueError):
            api.cake_engram_gate_nondet_train_backward(
                dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask, **outputs
            )
            pytest.fail(f"backward accepted {label}")
    with pytest.raises(ValueError):
        api.cake_engram_gate_nondet_kv(ns.x, ns.kv[:, :-8], ns.q_weight, ns.k_weight)


@pytest.mark.parametrize("name", ("x", "key", "value"))
def test_a_base_off_a_16_byte_boundary_is_a_value_error(name) -> None:
    device = _device_or_skip()
    ns = _valid(device)
    t = getattr(ns, name)
    flat = torch.empty(t.numel() + 64, dtype=t.dtype, device=device)
    shifted = flat[1 : 1 + t.numel()].view(t.contiguous().shape)
    assert shifted.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte"):
        _call_forward(ns, **{name: shifted})


def test_the_token_stride_rule_holds_for_one_token() -> None:
    device = _device_or_skip()
    ns = make_inputs(1, seed=3, device=device, weights="fp32", mask="none")
    bf16 = dict(dtype=torch.bfloat16, device=device)
    key = torch.zeros(1, STREAMS * HIDDEN + 4, **bf16)[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN))
    value = torch.zeros(1, HIDDEN + 4, **bf16)[:, :HIDDEN]
    for edit in (dict(key=key), dict(value=value)):
        with pytest.raises(ValueError, match="token stride"):
            _call_forward(ns, **edit)


def test_a_device_outside_the_architectures_is_refused(monkeypatch) -> None:
    device = _device_or_skip()
    ns = _valid(device)
    device_arch = _backend().device_arch
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    device_arch.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            _call_forward(ns)
    finally:
        monkeypatch.undo()
        device_arch.cache_clear()


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: determinism, streams, graphs, checkpoint, memory


def test_deterministic_mode_raises_the_alert_in_the_backward_only() -> None:
    """Under ``torch.use_deterministic_algorithms(True)``: the forward runs; the backward of every entry raises
    PyTorch's ``RuntimeError`` naming the deterministic package's entry, before it launches."""
    device = _device_or_skip()
    ns = make_inputs(16, seed=23, device=device, weights="bf16", mask="some", layout="kv")
    x, key, value, kv, qw, kw = _leaves(ns)
    with _torch_deterministic(warn_only=False):
        out = api.cake_engram_gate_nondet(x, key, value, qw, kw, ns.token_mask)
        packed = api.cake_engram_gate_nondet_kv(x, kv, qw, kw, ns.token_mask)
        api.cake_engram_gate_nondet_train_forward(ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight)
        with pytest.raises(RuntimeError, match=rf"{ALERT}.*cake_engram_gate_det_train\.cake_engram_gate_det\b"):
            out.backward(ns.dout)
        with pytest.raises(RuntimeError, match=rf"{ALERT}.*cake_engram_gate_det_train\.cake_engram_gate_det_kv\b"):
            packed.backward(ns.dout)
        with pytest.raises(RuntimeError, match=rf"{ALERT}.*cake_engram_gate_det_train_backward\b"):
            api.cake_engram_gate_nondet_train_backward(ns.dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight)


def test_warn_only_warns_once_and_runs(record_property) -> None:
    """With ``warn_only=True`` the backward warns once per call and runs; the results meet the tier."""
    device = _device_or_skip()

    def run(ns):
        with _torch_deterministic(warn_only=True), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = _autograd(ns)
        alerts = [w for w in caught if ALERT in str(w.message)]
        assert len(alerts) == 1, [str(w.message) for w in caught]
        return got

    def make(seed):
        return make_inputs(16, seed=seed, device=device, weights="fp32", mask="some")

    _record_failures(*judge_draws(run, make, seed=29), record_property)


def test_runs_on_the_current_stream_without_synchronizing() -> None:
    """On a side stream, behind a long sleep kernel that precedes the write of ``dout`` (NaN until then): the forward
    and the backward return while the sleep still runs, and the gradients are finite."""
    device = _device_or_skip()
    ns = make_inputs(129, seed=31, device=device, weights="bf16", mask="some", layout="kv")
    dout = torch.full_like(ns.dout, math.nan)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    fwd, bwd = api.cake_engram_gate_nondet_train_forward, api.cake_engram_gate_nondet_train_backward
    with torch.cuda.stream(side):
        fwd(ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask)
        bwd(ns.dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask)
    side.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        dout.copy_(ns.dout)
        out = fwd(ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask)
        got = bwd(dout, ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask)
        pending = not side.query()
    side.synchronize()
    assert pending, "the call synchronized: the sleep kernel queued before it had finished"
    assert torch.isfinite(out.float()).all()
    for name, t in zip(GRADS, got, strict=True):
        assert torch.isfinite(t.float()).all(), name


@pytest.mark.parametrize("T", (1, 129, 4096))
def test_a_call_captured_in_a_cuda_graph_replays(T) -> None:
    """The forward and the backward captured in one CUDA graph: after the outputs are filled with NaN, the replay
    writes every element; the forward and the non-atomic gradients equal the eager call's bits."""
    device = _device_or_skip()
    ns = make_inputs(T, seed=37, device=device, weights="bf16", mask="some", layout="kv")
    fwd, bwd = api.cake_engram_gate_nondet_train_forward, api.cake_engram_gate_nondet_train_backward
    args = (ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight, ns.token_mask)
    eager_out = fwd(*args)
    eager = bwd(ns.dout, *args)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        fwd(*args)
        bwd(ns.dout, *args)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fwd(*args)
        got = bwd(ns.dout, *args)
    for t in (out, *got):
        t.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize(device)
    assert _same_bits(out, eager_out)
    for name, a, b in zip(GRADS, got, eager, strict=True):
        assert torch.isfinite(a.float()).all(), name
        if name in EXACT_GRADS:
            assert _same_bits(a, b), name


@pytest.mark.parametrize("packed", (False, True), ids=("split", "packed"))
def test_checkpoint_gives_the_plain_gradients(packed) -> None:
    """Under ``torch.utils.checkpoint(use_reentrant=False)`` the output and the gradients are plain autograd's bits
    (the weight gradients too in the deterministic package)."""
    device = _device_or_skip()
    ns = make_inputs(1031, seed=41, device=device, weights="mixed", mask="some", layout="kv")
    plain = _autograd(ns, packed=packed)
    ckpt = _autograd(ns, packed=packed, checkpoint=True)
    names = ["out", "dx", "dkey", "dvalue"]
    for name in names:
        assert _same_bits(plain[name], ckpt[name]), name


@pytest.mark.parametrize("T", (16384, 65536))
def test_retained_state_and_workspace_are_reported(T, record_property) -> None:
    """The memory one training step keeps and allocates beside its operands: after the forward the autograd graph
    holds the output and nothing else (the saved set is the inputs); the backward's peak beyond its gradients is at
    most :func:`cake_engram_gate_nondet_train_workspace_bytes` plus the two FP32 weight gradients autograd casts to
    the parameters' BF16."""
    device = _device_or_skip()
    ns = make_inputs(T, seed=43, device=device, weights="bf16", mask="some", layout="kv")
    x, key, value, kv, qw, kw = _leaves(ns)
    api.cake_engram_gate_nondet_kv(x, kv, qw, kw, ns.token_mask).backward(ns.dout)  # loads, warms the allocator
    x.grad = kv.grad = qw.grad = kw.grad = None
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    out = api.cake_engram_gate_nondet_kv(x, kv, qw, kw, ns.token_mask)
    torch.cuda.synchronize(device)
    retained = torch.cuda.memory_allocated(device) - base - out.untyped_storage().nbytes()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    out.backward(ns.dout)
    torch.cuda.synchronize(device)
    grads = sum(t.untyped_storage().nbytes() for t in (x.grad, kv.grad, qw.grad, kw.grad))
    peak = torch.cuda.max_memory_allocated(device) - before - grads
    workspace = api.cake_engram_gate_nondet_train_workspace_bytes(T, device)
    record_property("retained_bytes_beyond_out", retained)
    record_property("backward_peak_beyond_grads_bytes", peak)
    record_property("workspace_bytes", workspace)
    print(f"T={T} retained_beyond_out={retained} backward_peak_beyond_grads={peak} workspace_bytes={workspace}")
    assert retained <= 0, retained
    # Autograd also holds the two FP32 weight gradients before their cast to the BF16 parameters' dtype; the caching
    # allocator rounds each block up to 512 bytes.
    assert peak <= workspace + 2 * STREAMS * HIDDEN * 4 + 16 * 512, (peak, workspace)


def test_operands_on_another_device(record_property) -> None:
    """With two devices: operands on the device that is not current give results there that meet the tier;
    operands on two devices are a ``ValueError``."""
    _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("one CUDA device")
    other = torch.device("cuda:1")
    if torch.cuda.get_device_capability(1) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the second device is not a target")
    _build_libraries(1)

    def run(ns):
        got = _autograd(ns)
        for t in got.values():
            assert t.device == other
        return got

    def make(seed):
        return make_inputs(16, seed=seed, device=other, weights="fp32", mask="some", layout="kv")

    _record_failures(*judge_draws(run, make, seed=47), record_property)
    ns = make_inputs(4, seed=1, device=other)
    with pytest.raises(ValueError, match="device"):
        api.cake_engram_gate_nondet_train_forward(ns.x, ns.key.to("cuda:0"), ns.value, ns.q_weight, ns.k_weight)
