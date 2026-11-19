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

Public entry points of ``cake_pplx.cake_dsa_sparse_attention_det_train`` over the backend package
``cake_pplx._backends.cake_dsa_sparse_attention_det_train``: the environment variables the facade resolves
into the backend's keywords, the parity keywords, the README's per-output statements, the four entries' gradients
repeating bit for bit (with repeated destination rows and a nonzero ``dkv_acc``) next to a resident side kernel, and
the explicit step entries (``cake_dsa_sparse_attention_det_train_*_forward`` / ``_backward``): the bits of the autograd
entries, repeating bit for bit, and the naive tier of ``tests/cake_naive.py`` (each floating output's error against an
FP64 oracle of the same inputs at most that of the same math in PyTorch BF16 over 32 draws, and 0 wherever that error
is 0), with the packed forward's global rows and lengths compared exactly and the NoPE forms also on a one-head ``kv``.
"""

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import cake_pplx.cake_dsa_sparse_attention_det_train as public_api
from cake_pplx._backends.cake_dsa_sparse_attention_det_train import (
    cake_backend,
    cake_jit,
)
from tests.experimental.test_helpers import cake_dsa_sparse_attention_naive as nv
from tests.experimental.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (
    last_valid_plus_one,
    make_inputs,
)

PACKAGE_DIR = Path(cake_backend.__file__).resolve().parent
SEED = 20261108
REPEATS = 5


def _bits(t):
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(view)


def _same_bits(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _require_backward(program=cake_jit.PROGRAM):
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    ):
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not cake_backend.generated_program_available(
        torch.device("cuda"), backward=True, program=program
    ):
        pytest.skip("backward stages not registered for this device")


# ---------------------------------------------------------------------------
# Facade (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), ("0", False), ("1", True), ("x", True)],
)
def test_check_dst_map_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(public_api.CHECK_DST_MAP_ENV, raising=False)
    else:
        monkeypatch.setenv(public_api.CHECK_DST_MAP_ENV, value)
    assert public_api._check_dst_map_from_environment() is expected


@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_binding_cache_capacity_environment_rejects_bad_values(value):
    code = "import cake_pplx.cake_dsa_sparse_attention_det_train"
    env = {
        **__import__("os").environ,
        public_api.BINDING_CACHE_CAPACITY_ENV: value,
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert public_api.BINDING_CACHE_CAPACITY_ENV in result.stderr


def test_binding_cache_is_this_packages_own():
    nondet = importlib.import_module(
        "cake_pplx._backends.cake_dsa_sparse_attention_nondet_train.cake_backend"
    )
    assert cake_backend.BINDING_CACHE is not nondet.BINDING_CACHE


def test_readme_states_every_output():
    """The README's table names every output of the package (A7: the per-output statements)."""
    text = (PACKAGE_DIR / "README.md").read_text()
    for name in (
        "`out`, `lse`, `o_lo`",
        "derived `topk_length`",
        "`dq_latent`, `dq_rope`",
        "`dkv_latent`, `dk_rope`",
        "NoPE `dkv`",
        "`dkv_acc` columns `0:576` (NoPE `0:512`)",
        "varlen global rows and lengths",
    ):
        assert name in text, name
    assert "repeats bit for bit" in text
    assert "use_deterministic_algorithms(True)" in text


@pytest.mark.parametrize(
    "name",
    [
        "cake_dsa_sparse_attention_det_nope",
        "cake_dsa_sparse_attention_det_nope_varlen",
        "cake_dsa_sparse_attention_det_train_nope_forward",
        "cake_dsa_sparse_attention_det_train_nope_backward",
        "cake_dsa_sparse_attention_det_train_nope_varlen_forward",
        "cake_dsa_sparse_attention_det_train_nope_varlen_backward",
    ],
)
def test_nope_entries_forward_the_sink(name, monkeypatch):
    """The NoPE entries take the ``sink`` keyword of the non-deterministic entries and hand it to the backend as given
    (``None`` by default) without reading it."""
    seen = []

    class Backend:
        def __getattr__(self, attr):
            def call(*args, **kwargs):
                seen.append((attr, kwargs))
                return attr

            return call

    monkeypatch.setattr(public_api, "_backend", lambda backend: Backend())
    entry = getattr(public_api, name)
    parameters = inspect.signature(entry).parameters.values()
    positional = [
        p for p in parameters if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    # the operands stand as ``None`` (the packed backward reads the length of ``cu_seqlens_k``), and so do the
    # required keyword-only arguments (the packed backward's ``topk_length``)
    args = [
        torch.zeros(3, dtype=torch.int32) if p.name.startswith("cu_seqlens") else None
        for p in positional
    ]
    required = {
        p.name: None
        for p in parameters
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
    }
    assert inspect.signature(entry).parameters["sink"].default is None
    entry(*args, **required)
    sink = torch.zeros(64)
    entry(*args, **required, sink=sink)
    assert len(seen) == 2 and seen[0][0] == seen[1][0]
    assert seen[0][1]["sink"] is None and seen[1][1]["sink"] is sink
    assert {k: v for k, v in seen[0][1].items() if k != "sink"} == {
        k: v for k, v in seen[1][1].items() if k != "sink"
    }


def test_nope_entries_hand_num_segments_to_the_backend(monkeypatch):
    """The flat NoPE entry and the NoPE step backward hand ``num_segments`` to the backend as a Python ``int``: 1 by
    default, the caller's count when given, with every other keyword as before; the packed step backward hands the
    documents' count (``len(cu_seqlens_k) - 1``); a bad count raises ``ValueError`` naming it before the backend."""
    seen = {}

    def flat(q, kv, indices, **kwargs):
        seen["flat"] = kwargs
        return "flat"

    def backward_nope(*args, **kwargs):
        seen["backward"] = (len(args), kwargs)
        return "backward"

    monkeypatch.setattr(cake_backend, "cake_dsa_sparse_attention_det_nope", flat)
    monkeypatch.setattr(cake_backend, "backward_nope", backward_nope)
    monkeypatch.delenv(public_api.CHECK_DST_MAP_ENV, raising=False)
    q = torch.zeros(8, 64, 512, dtype=torch.bfloat16)
    kv = torch.zeros(16, 512, dtype=torch.bfloat16)
    idx = torch.zeros(8, 5, dtype=torch.int32)
    out = torch.zeros(8, 64, 512, dtype=torch.bfloat16)
    lse = torch.zeros(8, 64)
    flat_default = dict(
        topk_length=None,
        softmax_scale=None,
        return_lse=False,
        key_passes=None,
        dkv_acc=None,
        dkv_dst_map=None,
        num_segments=1,
        dkv_direct=None,
        check_dst_map=False,
        sink=None,
    )
    assert public_api.cake_dsa_sparse_attention_det_nope(q, kv, idx) == "flat"
    assert seen["flat"] == flat_default
    public_api.cake_dsa_sparse_attention_det_nope(q, kv, idx, num_segments=8, key_passes=2)
    assert seen["flat"] == dict(flat_default, num_segments=8, key_passes=2)
    assert type(seen["flat"]["num_segments"]) is int
    step_backward = public_api.cake_dsa_sparse_attention_det_train_nope_backward
    assert step_backward(q, kv, idx, out, out, lse, out) == "backward"
    default = seen["backward"]
    assert default[0] == 7 and default[1]["num_segments"] == 1
    step_backward(q, kv, idx, out, out, lse, out, num_segments=3)
    assert seen["backward"] == (7, dict(default[1], num_segments=3))
    assert type(seen["backward"][1]["num_segments"]) is int
    varlen_backward = public_api.cake_dsa_sparse_attention_det_train_nope_varlen_backward
    cu_k = torch.tensor([0, 6, 6, 16], dtype=torch.int32)
    varlen_backward(q, kv, idx, out, out, lse, out, cu_k, topk_length=idx[:, 0])
    assert seen["backward"][1]["num_segments"] == 3
    for bad in (0, -1, True, 2.0, "2", torch.tensor(2)):
        seen.clear()
        with pytest.raises(ValueError, match="num_segments"):
            public_api.cake_dsa_sparse_attention_det_nope(q, kv, idx, num_segments=bad)
        with pytest.raises(ValueError, match="num_segments"):
            step_backward(q, kv, idx, out, out, lse, out, num_segments=bad)
        assert seen == {}


def _rope_host_views():
    """Rope operands of 8 queries over 16 keys, top-k 5, on the host (nothing here launches)."""
    q = torch.zeros(8, 64, 576, dtype=torch.bfloat16)
    kv = torch.zeros(16, 576, dtype=torch.bfloat16)
    idx = torch.zeros(8, 5, dtype=torch.int32)
    return (q[..., :512], q[..., 512:], kv[:, :512], kv[:, 512:]), idx


def test_rope_entries_hand_num_segments_to_the_backend(monkeypatch):
    """The flat rope entry and the rope step backward hand ``num_segments`` to the backend as a Python ``int``: 1 by
    default, the caller's count when given, with every other keyword as before."""
    seen = {}

    def flat(q_latent, q_rope, kv_latent, k_rope, indices, **kwargs):
        seen["flat"] = kwargs
        return "flat"

    def backward(*args, **kwargs):
        seen["backward"] = (len(args), kwargs)
        return "backward"

    monkeypatch.setattr(cake_backend, "cake_dsa_sparse_attention_det", flat)
    monkeypatch.setattr(cake_backend, "backward", backward)
    monkeypatch.delenv(public_api.CHECK_DST_MAP_ENV, raising=False)
    rope, idx = _rope_host_views()
    flat_default = dict(
        topk_length=None,
        softmax_scale=None,
        return_lse=False,
        key_passes=None,
        dkv_acc=None,
        dkv_dst_map=None,
        num_segments=1,
        dkv_direct=None,
        check_dst_map=False,
    )
    assert public_api.cake_dsa_sparse_attention_det(*rope, idx) == "flat"
    assert seen["flat"] == flat_default
    public_api.cake_dsa_sparse_attention_det(*rope, idx, num_segments=8, key_passes=2)
    assert seen["flat"] == dict(flat_default, num_segments=8, key_passes=2)
    assert type(seen["flat"]["num_segments"]) is int
    out = torch.zeros(8, 64, 512, dtype=torch.bfloat16)
    lse = torch.zeros(8, 64)
    step_backward = public_api.cake_dsa_sparse_attention_det_train_backward
    assert step_backward(*rope, idx, out, out, lse, out) == "backward"
    step_default = dict(
        topk_length=None,
        softmax_scale=None,
        dkv_fp32=False,
        key_passes=None,
        dkv_acc=None,
        dkv_dst_map=None,
        num_segments=1,
        dkv_direct=None,
        check_dst_map=False,
    )
    assert seen["backward"] == (9, step_default)
    step_backward(*rope, idx, out, out, lse, out, num_segments=3)
    assert seen["backward"] == (9, dict(step_default, num_segments=3))


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "2", torch.tensor(2)], ids=repr)
def test_rope_entries_refuse_a_bad_num_segments(monkeypatch, bad):
    """``num_segments`` is a host count: ``bool``, a float, a string, a tensor and a count below 1 raise ``ValueError``
    naming it before the backend is reached."""

    def refuse(*args, **kwargs):
        raise AssertionError("a bad num_segments must not reach the backend")

    monkeypatch.setattr(public_api, "_backend", refuse)
    rope, idx = _rope_host_views()
    out = torch.zeros(8, 64, 512, dtype=torch.bfloat16)
    lse = torch.zeros(8, 64)
    with pytest.raises(ValueError, match="num_segments"):
        public_api.cake_dsa_sparse_attention_det(*rope, idx, num_segments=bad)
    with pytest.raises(ValueError, match="num_segments"):
        public_api.cake_dsa_sparse_attention_det_train_backward(
            *rope, idx, out, out, lse, out, num_segments=bad
        )


# ---------------------------------------------------------------------------
# Public entries (GPU)
# ---------------------------------------------------------------------------


def _entry_call(entry, inp, mode, dkv_map):
    """One autograd step of a public entry: the returned gradients, and ``dkv_acc`` when the mode accumulates."""
    nope = "nope" in entry
    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (
            (inp.q_latent, inp.kv_latent)
            if nope
            else (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
        )
    ]
    cols = 512 if nope else 576
    kw = {}
    acc = None
    if mode != "fresh":
        rows = int(dkv_map.shape[0]) // 2 + 1 if mode == "acc_map" else inp.total_k
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        acc = torch.randn(rows, cols, device="cuda", generator=gen)
        acc[0, :4] = -0.0
        kw["dkv_acc"] = acc
        if mode == "acc_map":
            kw["dkv_dst_map"] = dkv_map
    fn = getattr(public_api, entry)
    if entry.endswith("varlen"):
        out = fn(
            *leaves,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            **kw,
        )
    else:
        out = fn(*leaves, inp.idx_global, **kw)
    out.backward(inp.dout)
    return [out] + [leaf.grad for leaf in leaves] + [acc]


ENTRIES = (
    "cake_dsa_sparse_attention_det",
    "cake_dsa_sparse_attention_det_varlen",
    "cake_dsa_sparse_attention_det_nope",
    "cake_dsa_sparse_attention_det_nope_varlen",
)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("mode", ["fresh", "acc", "acc_map"])
@pytest.mark.parametrize("side", [False, True])
def test_public_entries_repeat_bit_for_bit(entry, mode, side):
    """Five autograd steps of each public entry give the same bits for ``out`` and every gradient -- fresh, added
    into a nonzero ``dkv_acc`` (a planted ``-0.0`` included), and through a map with repeated destination rows --
    with or without a kernel resident on a side stream while the step runs."""
    _require_backward(cake_jit.NOPE_PROGRAM if "nope" in entry else cake_jit.PROGRAM)
    inp = make_inputs([200, 120], [600, 360], seed=SEED, topk=256)
    S = inp.total_k
    dkv_map = (
        torch.arange(S, device="cuda", dtype=torch.int32) % (S // 2)
    ).contiguous()
    side_stream = torch.cuda.Stream()
    first = None
    for _ in range(REPEATS):
        if side:
            with torch.cuda.stream(side_stream):
                torch.cuda._sleep(20_000_000)
        got = _entry_call(entry, inp, mode, dkv_map)
        torch.cuda.synchronize()
        if first is None:
            first = got
            continue
        for a, b in zip(got, first, strict=True):
            assert _same_bits(a, b)


ROPE_GRADS = ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")


def _flat_rope_call(inp, **kw):
    """One autograd step of the flat rope entry: ``out``, ``lse`` and the four leaf gradients by name."""
    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    out, lse = public_api.cake_dsa_sparse_attention_det(
        *leaves, inp.idx_global, return_lse=True, **kw
    )
    out.backward(inp.dout)
    torch.cuda.synchronize()
    return dict(
        out=out.detach(),
        lse=lse.detach(),
        **{name: leaf.grad for name, leaf in zip(ROPE_GRADS, leaves, strict=True)},
    )


def _step_rope_call(inp, **kw):
    """The rope step pair: ``out``, ``lse`` and the four gradients by name (``o_lo`` left out)."""
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    out, lse, o_lo = public_api.cake_dsa_sparse_attention_det_train_forward(*args)
    grads = public_api.cake_dsa_sparse_attention_det_train_backward(
        *args, out, o_lo, lse, inp.dout, **kw
    )
    torch.cuda.synchronize()
    return dict(out=out, lse=lse, **dict(zip(ROPE_GRADS, grads, strict=True)))


def test_flat_entry_with_num_segments_plans_one_pass_and_repeats_bit_for_bit(
    monkeypatch, record_property
):
    """On the global rows of a packed two-segment row that the policy would split into whole-row passes as one segment
    (T = 2 x 128, S = 2 x 65,536 keys, top-k 128: one token chunk), the flat rope entry and the rope step backward with
    ``num_segments=2`` plan one key-range pass, the plan of the varlen entry on the same rows, and repeat bit for bit;
    the step pair gives the autograd entry's bits; dK / dV keep the bits of the default (multi-pass) plan; every call
    meets the naive tier over the draws."""
    _require_backward()
    device = torch.device("cuda")
    _, record = cake_backend.record_for(device)
    stages = cake_jit.registered_stages()
    if not all(s in stages for s in cake_jit.KEY_PASS_STAGES):
        pytest.skip("the registered program has no key-range-pass stages")
    arch = cake_backend.arch_for(device)
    T, S, topk = 256, 131072, 128
    plan = cake_backend.plan_key_passes
    default_passes = plan(record, stages, T, S, topk, arch=arch)
    # the row would take whole-row passes as one segment, on every architecture
    assert default_passes > 1
    assert plan(record, stages, T, S, topk, num_segments=2, arch=arch) == 1
    # a cache of this test's own, cleared before each call below so that the call's backward plan is its one backward
    # entry; the det backward key holds the input geometry and the options that shape the plan, key_passes and
    # whether num_segments > 1 among them (cake_backend.backward_binding_key)
    monkeypatch.setattr(cake_backend, "BINDING_CACHE", cake_backend._BindingCache())
    cache = cake_backend.BINDING_CACHE

    def planned(call):
        cache.clear()
        got = call()
        plans = [b.plan for b in cache._entries.values() if b.plan.backward]
        assert len(plans) == 1
        return got, plans[0]

    inp = make_inputs([128, 128], [65536, 65536], seed=SEED + 21, topk=topk)
    seg, seg_plan = planned(lambda: _flat_rope_call(inp, num_segments=2))
    step, step_plan = planned(lambda: _step_rope_call(inp, num_segments=2))

    def varlen_call():
        leaves = [
            t.detach().clone().requires_grad_(True)
            for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
        ]
        out = public_api.cake_dsa_sparse_attention_det_varlen(
            *leaves, inp.idx_local, inp.cu_seqlens_q, inp.cu_seqlens_k
        )
        out.backward(inp.dout)
        torch.cuda.synchronize()

    _, varlen_plan = planned(varlen_call)
    default, default_plan = planned(lambda: _flat_rope_call(inp))
    assert varlen_plan.key_passes == 1
    assert not any(isinstance(k, tuple) for k in varlen_plan.backward_order)
    for plan in (seg_plan, step_plan):
        assert plan.key_passes == 1
        assert plan.backward_order == varlen_plan.backward_order
    assert default_plan.key_passes == default_passes
    assert any(isinstance(k, tuple) for k in default_plan.backward_order)
    # the same call repeats bit for bit; the step pair gives the autograd entry's bits
    for _ in range(2):
        again = _flat_rope_call(inp, num_segments=2)
        for name, value in seg.items():
            assert _same_bits(again[name], value), name
    for name, value in seg.items():
        assert _same_bits(step[name], value), name
    # the forward and dK / dV do not depend on the plan (dQ is summed through the FP32 partial of the passes)
    for name in ("out", "lse", "dkv_latent", "dk_rope"):
        assert _same_bits(default[name], seg[name]), name

    def case(s, j):
        draw = make_inputs([128, 128], [65536, 65536], seed=s, topk=topk)
        return nv.judged(
            nv.rope_problem(draw),
            {
                "flat num_segments=2": _flat_rope_call(draw, num_segments=2),
                "step num_segments=2": _step_rope_call(draw, num_segments=2),
                "flat default": _flat_rope_call(draw),
            },
        )

    nv.judge_draws(record_property, case, seed=SEED + 22)


# ---------------------------------------------------------------------------
# Explicit step entries (GPU): the forward and the backward as two calls
# ---------------------------------------------------------------------------

STEP_FORMS = ("rope", "nope", "nope_varlen")
STEP_GRADS = {
    "rope": ("dq_latent", "dq_rope", "dkv_latent", "dk_rope"),
    "nope": ("dq", "dkv"),
    "nope_varlen": ("dq", "dkv"),
}
# (form, one_head): the NoPE forms also take ``kv`` in its one-head form ``[S, 1, 512]``.
STEP_CASES = [
    ("rope", False),
    ("nope", False),
    ("nope_varlen", False),
    ("nope", True),
    ("nope_varlen", True),
]
STEP_CASE_IDS = [
    form + ("-kv_one_head" if one_head else "") for form, one_head in STEP_CASES
]


def _step_entries(form):
    stem = "cake_dsa_sparse_attention_det_train" + (
        "" if form == "rope" else f"_{form}"
    )
    return getattr(public_api, f"{stem}_forward"), getattr(
        public_api, f"{stem}_backward"
    )


def _step_acc(form, inp):
    return torch.zeros(inp.total_k, 576 if form == "rope" else 512, device="cuda")


def _step_kv(inp, one_head):
    """``kv`` of a NoPE call: ``[S, 512]``, or its one-head form ``[S, 1, 512]``."""
    return inp.kv_latent.unsqueeze(1) if one_head else inp.kv_latent


def _explicit_step(form, inp, acc=None, *, one_head=False):
    """One forward and one backward through the step entries of ``form`` (the varlen backward on the rows and
    lengths its forward returns; ``one_head``: a NoPE ``kv`` of shape ``[S, 1, 512]``): the :class:`nv.Problem` the
    outputs answer and the outputs by the names :func:`nv.judged` takes.  The backward leaves ``o_lo`` unchanged."""
    forward, backward = _step_entries(form)
    kw = {} if acc is None else dict(dkv_acc=acc)
    kv = _step_kv(inp, one_head)
    if form == "nope_varlen":
        out, lse, o_lo, rows, lengths = forward(
            inp.q_latent,
            kv,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
        )
        # The returned rows and lengths, exactly, against the test's own references: the inputs' rows are valid
        # first, so the forward keeps every slot in place on either index route.
        assert torch.equal(rows, inp.idx_global)
        assert torch.equal(lengths, last_valid_plus_one(inp.idx_global, inp.total_k))
        o_lo_before = o_lo.clone()
        grads = backward(
            inp.q_latent, kv, rows, out, o_lo, lse, inp.dout, inp.cu_seqlens_k,
            topk_length=lengths, **kw,
        )  # fmt: skip
        problem = nv.nope_problem(inp)
    else:
        if form == "rope":
            args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
            problem = nv.rope_problem(inp)
        else:
            args = (inp.q_latent, kv, inp.idx_global)
            problem = nv.nope_problem(inp)
        out, lse, o_lo = forward(*args)
        o_lo_before = o_lo.clone()
        grads = backward(*args, out, o_lo, lse, inp.dout, **kw)
    torch.cuda.synchronize()
    assert _same_bits(o_lo, o_lo_before)
    got = dict(out=out, lse=lse, o_lo=o_lo)
    for name, grad in zip(STEP_GRADS[form], grads, strict=True):
        assert (grad is None) == (
            acc is not None and name in ("dkv_latent", "dk_rope", "dkv")
        ), name
        if grad is not None:
            got[name] = grad
    return problem, got


def _autograd_step(form, inp, acc=None, *, one_head=False):
    """The autograd entry of ``form`` on the same inputs: ``(out, lse, leaf gradients)``."""
    nope = form != "rope"
    sources = (
        (inp.q_latent, _step_kv(inp, one_head))
        if nope
        else (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    )
    leaves = [t.detach().clone().requires_grad_(True) for t in sources]
    kw = dict(return_lse=True) if acc is None else dict(return_lse=True, dkv_acc=acc)
    if form == "rope":
        out, lse = public_api.cake_dsa_sparse_attention_det(
            *leaves, inp.idx_global, **kw
        )
    elif form == "nope":
        out, lse = public_api.cake_dsa_sparse_attention_det_nope(
            *leaves, inp.idx_global, **kw
        )
    else:
        out, lse = public_api.cake_dsa_sparse_attention_det_nope_varlen(
            *leaves, inp.idx_local, inp.cu_seqlens_q, inp.cu_seqlens_k, **kw
        )
    out.backward(inp.dout)
    torch.cuda.synchronize()
    return out.detach(), lse.detach(), [leaf.grad for leaf in leaves]


@pytest.mark.parametrize("form, one_head", STEP_CASES, ids=STEP_CASE_IDS)
@pytest.mark.parametrize("acc", [False, True], ids=["fresh", "acc"])
def test_step_entries_give_the_autograd_bits_and_repeat(form, one_head, acc):
    """The explicit forward and backward give the bits of the autograd entry on the same inputs (``out``, ``lse``,
    dQ, and dK + dV: the fresh gradients as the autograd engine hands them to the BF16 key leaves, or the rows added
    into a zeroed ``dkv_acc``), and repeat bit for bit; the NoPE forms also with a one-head ``kv [S, 1, 512]``."""
    _require_backward(cake_jit.PROGRAM if form == "rope" else cake_jit.NOPE_PROGRAM)
    inp = make_inputs([200, 120], [600, 360], seed=SEED + 7, topk=256)
    first = None
    for _ in range(3):
        step_acc = _step_acc(form, inp) if acc else None
        _, got = _explicit_step(form, inp, step_acc, one_head=one_head)
        flat = [got[n] for n in ("out", "lse", "o_lo", *STEP_GRADS[form]) if n in got]
        flat += [] if step_acc is None else [step_acc]
        if first is None:
            first = flat
        else:
            for a, b in zip(flat, first, strict=True):
                assert _same_bits(a, b)
    auto_acc = _step_acc(form, inp) if acc else None
    out, lse, leaf_grads = _autograd_step(form, inp, auto_acc, one_head=one_head)
    assert _same_bits(got["out"], out) and _same_bits(got["lse"], lse)
    names = STEP_GRADS[form]
    num_dq = 2 if form == "rope" else 1
    for name, leaf_grad in zip(names[:num_dq], leaf_grads[:num_dq], strict=True):
        assert _same_bits(got[name], leaf_grad), name
    if acc:
        assert all(g is None for g in leaf_grads[num_dq:])
        assert _same_bits(step_acc, auto_acc)
    else:
        for name, leaf_grad in zip(names[num_dq:], leaf_grads[num_dq:], strict=True):
            # the step's grad has the [S, 512] layout; the one-head leaf's is [S, 1, 512]
            step_grad = got[name].to(leaf_grad.dtype).reshape(leaf_grad.shape)
            assert _same_bits(step_grad, leaf_grad), name


@pytest.mark.parametrize("form, one_head", STEP_CASES, ids=STEP_CASE_IDS)
def test_step_entries_meet_the_naive_tier(form, one_head, record_property):
    """The explicit forward and backward (fresh dK + dV, and added into a zeroed ``dkv_acc``) meet the naive tier
    over the draws; the NoPE forms also with a one-head ``kv [S, 1, 512]``."""
    _require_backward(cake_jit.PROGRAM if form == "rope" else cake_jit.NOPE_PROGRAM)

    def case(s, j):
        inp = make_inputs([200, 120], [600, 360], seed=s, topk=256)
        problem, fresh = _explicit_step(form, inp, one_head=one_head)
        acc = _step_acc(form, inp)
        _, added = _explicit_step(form, inp, acc, one_head=one_head)
        added["dkv_acc"] = nv.Acc(before=torch.zeros_like(acc), after=acc)
        return nv.judged(problem, {"fresh": fresh, "acc": added})

    nv.judge_draws(record_property, case, seed=SEED + 8 + STEP_FORMS.index(form))
