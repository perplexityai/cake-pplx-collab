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

Public-API suite of ``cake_pplx.cake_fmha_nondet_train``, the dense multi-head attention backward of the Cake
non-deterministic training package: ``cake_fmha_nondet_train_backward(q, k, v, out, lse, dout, *, softmax_scale=None,
causal=False) -> (dq, dk, dv)``.

Without a device: the entry is a stable API (no ``ExperimentalWarning``) and the module's only public name, importing
the module imports no backend, the module reads no environment variable, every keyword of the entry is the backend's
keyword of the same name, the entry documents its behaviour under ``torch.use_deterministic_algorithms``, a
``deterministic`` keyword is a ``TypeError`` before any work, operands on the CPU are a ``ValueError``, and a device
of capability (10, 7) maps to ``sm_107a``, for which ``build.json`` lists every record's library.

On SM100 / SM103 / SM107: the entry's ``dq``, ``dk`` and ``dv`` for every variant (head dimension 64 or 128, causal or
not) meet the naive BF16 tier against an FP64 oracle (``tests/cake_naive.py``, 32 independent draws), with the default
and a given ``softmax_scale``, and with operands off a 256-byte boundary (16-byte aligned; ``lse`` 4-byte aligned);
every documented ``ValueError`` (dtype, device, layout, rank, shapes, head dimension, lengths, head count, causal
lengths, grid and index limits, a non-finite scale, a base not 16-byte aligned); a call whose FP32 workspace does
not fit is PyTorch's ``torch.OutOfMemoryError``; under ``torch.use_deterministic_algorithms`` the alert (a
``RuntimeError``; with ``warn_only=True`` one ``UserWarning``, then the results meet the tier); an empty batch returns
empty gradients and alerts nothing; the launches run on the current stream without synchronizing; a call
captured in a CUDA graph replays to results that meet the tier; a device outside the exported architectures is a
``RuntimeError``; with two devices, operands on the device that is not current give results there that meet the tier,
and operands on two devices are a ``ValueError``.  The package promises no bits: no test compares two runs bit for
bit. """

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

import pytest
import torch

import cake_pplx.cake_fmha_nondet_train as api
from tests.cake.test_helpers.cake_fmha_naive import (
    Case,
    assert_backward_contract,
    judge_draws,
    judged_backward,
    make_inputs,
)

BACKEND = "cake_pplx._backends.cake_fmha_nondet_train"
ENTRY = "cake_fmha_nondet_train_backward"
OPERANDS = ("q", "k", "v", "out", "lse", "dout")
VARIANTS = ((128, False), (128, True), (64, False), (64, True))
VARIANT_IDS = ["d128", "d128_causal", "d64", "d64_causal"]
# The alert of this package (PyTorch's wording, naming the entry).
ALERT = f"{ENTRY} does not have a deterministic implementation"
# Cycles of the sleep kernel queued before a call on a side stream (about 0.1 s): the call returns long before it ends.
SLEEP_CYCLES = 200_000_000


def _backend():
    from cake_pplx._backends.cake_fmha_nondet_train import cake_backend

    return cake_backend


def _call(ops: dict, **kwargs):
    return api.cake_fmha_nondet_train_backward(*(ops[name] for name in OPERANDS), **kwargs)


def _operands(device, *, B: int = 1, H: int = 2, Sq: int = 128, Skv: int = 128, D: int = 128) -> dict:
    """Well-formed operands of zeros (a call that should fail never reads them)."""
    bf16 = dict(device=device, dtype=torch.bfloat16)
    return {
        "q": torch.zeros(B, Sq, H, D, **bf16),
        "k": torch.zeros(B, Skv, H, D, **bf16),
        "v": torch.zeros(B, Skv, H, D, **bf16),
        "out": torch.zeros(B, Sq, H, D, **bf16),
        "lse": torch.zeros(B, H, Sq, device=device, dtype=torch.float32),
        "dout": torch.zeros(B, Sq, H, D, **bf16),
    }


def _offset(t: torch.Tensor, elements: int) -> torch.Tensor:
    """A contiguous copy of ``t`` that starts ``elements`` elements past the start of its allocation (the caching
    allocator aligns an allocation to at least 512 bytes)."""
    flat = torch.empty(t.numel() + elements + 256, dtype=t.dtype, device=t.device)
    view = flat[elements : elements + t.numel()].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous() and view.data_ptr() % 512 == elements * t.element_size()
    return view


def _strided(t: torch.Tensor) -> torch.Tensor:
    """``t`` with its values and shape in a non-contiguous layout (its last two dimensions stored transposed)."""
    view = t.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not view.is_contiguous()
    return view


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake dense attention backward ships for SM100 / SM103 / SM107")
    from cake_pplx._backends.cake_fmha_nondet_train import cake_jit

    if not (Path(cake_jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
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


def _grads(dq, dk, dv) -> dict:
    return {"dq": dq, "dk": dk, "dv": dv}


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_the_entry_is_a_stable_api() -> None:
    assert list(api.__all__) == [ENTRY]
    entry = getattr(api, ENTRY)
    assert getattr(entry, "is_experimental", False) is False
    assert entry.__name__ == ENTRY


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import cake_pplx.cake_fmha_nondet_train as api\n"
        "names = dir(api)\n"
        f"print(any(m.startswith({BACKEND!r}) for m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_module_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(api.__file__).read_text())
    hits = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
    ]
    assert not hits, hits


def test_keywords_match_the_backend() -> None:
    ours = inspect.signature(inspect.unwrap(getattr(api, ENTRY))).parameters
    theirs = inspect.signature(getattr(_backend(), ENTRY)).parameters
    assert list(ours) == list(theirs) == [*OPERANDS, "softmax_scale", "causal"]
    for name, param in ours.items():
        assert param.kind == theirs[name].kind, name
        assert param.default == theirs[name].default, name
    assert ours["softmax_scale"].kind is inspect.Parameter.KEYWORD_ONLY and ours["softmax_scale"].default is None
    assert ours["causal"].kind is inspect.Parameter.KEYWORD_ONLY and ours["causal"].default is False


def test_the_entry_documents_its_determinism() -> None:
    """The entry states its behaviour under ``torch.use_deterministic_algorithms``."""
    assert "use_deterministic_algorithms" in getattr(api, ENTRY).__doc__


def test_a_deterministic_keyword_is_a_type_error() -> None:
    ops = _operands("cpu")
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            _call(ops, deterministic=value)


def test_cpu_operands_are_refused() -> None:
    with pytest.raises(ValueError):
        _call(_operands("cpu"))


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: judged results


@pytest.mark.parametrize(("D", "causal"), VARIANTS, ids=VARIANT_IDS)
def test_the_entry_matches_the_reference(D, causal, record_property) -> None:
    """Every variant through the public entry, ``softmax_scale`` defaulting to ``D^-0.5``, three heads, more keys
    than queries without the causal mask: the gradients hold the contract and meet the tier; no warning."""
    device = _device_or_skip()
    case = Case(2, 3, 384) if causal else Case(2, 3, 256, 384)

    def draw(seed: int):
        x = make_inputs(case, D, causal=causal, seed=seed, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout, causal=causal)
        return judged_backward(x, {"entry": _grads(*got)})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("causal", (False, True), ids=("plain", "causal"))
def test_a_given_softmax_scale_matches_the_reference(causal, record_property) -> None:
    """A ``softmax_scale`` other than ``D^-0.5`` (the forward's): the gradients meet the tier of that scale."""
    device = _device_or_skip()
    scale = 0.0625

    def draw(seed: int):
        x = make_inputs(Case(1, 4, 256), 128, causal=causal, seed=seed, device=device, softmax_scale=scale)
        got = api.cake_fmha_nondet_train_backward(
            x.q, x.k, x.v, x.out, x.lse, x.dout, softmax_scale=scale, causal=causal
        )
        return judged_backward(x, {"scaled": _grads(*got)})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("D", (128, 64), ids=("d128", "d64"))
def test_operands_off_a_256_byte_boundary_match_the_reference(D, record_property) -> None:
    """Operands that start 16 bytes past an allocation (``lse`` 4 bytes past): accepted, and the gradients meet the
    tier."""
    device = _device_or_skip()

    def draw(seed: int):
        x = make_inputs(Case(1, 2, 256, 384), D, causal=False, seed=seed, device=device)
        ops = {name: _offset(getattr(x, name), 1 if name == "lse" else 8) for name in OPERANDS}
        return judged_backward(x, {"offset": _grads(*_call(ops))})

    judge_draws(record_property, draw)


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: refused calls


def _set(name, make):
    def edit(device):
        ops = _operands(device)
        ops[name] = make(ops[name])
        return ops, {}

    return edit


def _shape(causal=False, **sizes):
    def edit(device):
        return _operands(device, **sizes), {"causal": causal}

    return edit


def _pair(edits: dict):
    def edit(device):
        ops = _operands(device)
        for name, make in edits.items():
            ops[name] = make(ops[name], device)
        return ops, {}

    return edit


def _resize(name, **sizes):
    """``name`` replaced by zeros of another shape (:func:`_zeros_like`)."""
    return _pair({name: lambda t, d: _zeros_like(t, d, **sizes)})


def _resize_kv(**sizes):
    """``k`` and ``v`` both replaced by zeros of another shape."""
    return _pair({name: lambda t, d: _zeros_like(t, d, **sizes) for name in ("k", "v")})


def _retyped(name):
    """``name`` in a dtype the entry does not take (FP16 for a BF16 operand, BF16 for ``lse``)."""
    wrong = torch.bfloat16 if name == "lse" else torch.float16
    return _set(name, lambda t: t.to(wrong))


def _scale(value):
    def edit(device):
        return _operands(device), {"softmax_scale": value}

    return edit


def _zeros_like(t, device, **sizes):
    """Zeros of ``t``'s dtype with ``t``'s shape, some dimensions changed: ``B``, ``S``, ``H``, ``D`` of a
    ``[B, S, H, D]`` operand, ``B``, ``H``, ``S`` of ``lse``."""
    if t.dim() == 3:
        B, H, S = t.shape
        shape = (sizes.get("B", B), sizes.get("H", H), sizes.get("S", S))
    else:
        B, S, H, D = t.shape
        shape = (sizes.get("B", B), sizes.get("S", S), sizes.get("H", H), sizes.get("D", D))
    return torch.zeros(shape, dtype=t.dtype, device=device)


VALUE_ERRORS = [
    *[pytest.param(_retyped(n), id=f"dtype-{n}") for n in OPERANDS],
    pytest.param(_set("q", lambda t: t.float()), id="dtype-q-fp32"),
    pytest.param(_set("lse", lambda t: t.double()), id="dtype-lse-fp64"),
    *[pytest.param(_set(n, lambda t: t.cpu()), id=f"device-{n}") for n in OPERANDS],
    *[pytest.param(_set(n, _strided), id=f"strided-{n}") for n in OPERANDS],
    pytest.param(_set("q", lambda t: t.flatten(2)), id="rank-q-3"),
    pytest.param(_set("k", lambda t: t.unsqueeze(0)), id="rank-k-5"),
    pytest.param(_set("lse", lambda t: t.flatten(1)), id="rank-lse-2"),
    pytest.param(_resize("out", S=256), id="shape-out"),
    pytest.param(_resize("dout", S=256), id="shape-dout"),
    pytest.param(_resize("k", S=256), id="shape-k-not-v"),
    pytest.param(_resize_kv(H=1), id="kv-heads"),
    pytest.param(_resize_kv(D=64), id="kv-dim"),
    pytest.param(_resize_kv(B=2), id="kv-batch"),
    pytest.param(_resize("lse", S=256), id="shape-lse-rows"),
    pytest.param(_resize("lse", H=3), id="shape-lse-heads"),
    pytest.param(_resize("lse", B=2), id="shape-lse-batch"),
    *[pytest.param(_shape(D=D), id=f"head-dim-{D}") for D in (32, 96, 256)],
    *[pytest.param(_shape(Sq=S), id=f"sq-{S}") for S in (0, 64, 192)],
    *[pytest.param(_shape(Skv=S), id=f"skv-{S}") for S in (0, 64, 200)],
    pytest.param(_shape(H=0), id="heads-0"),
    pytest.param(_shape(causal=True, Sq=128, Skv=256), id="causal-sq-below-skv"),
    pytest.param(_shape(causal=True, Sq=256, Skv=128), id="causal-sq-above-skv"),
    *[pytest.param(_scale(value), id=f"scale-{value}") for value in (math.nan, math.inf, -math.inf)],
]


@pytest.mark.parametrize("edit", VALUE_ERRORS)
def test_malformed_operands_are_a_value_error(edit) -> None:
    device = _device_or_skip()
    ops, kwargs = edit(device)
    with pytest.raises(ValueError):
        _call(ops, **kwargs)


@pytest.mark.parametrize("elements", (1, 4), ids=("2-bytes", "8-bytes"))
@pytest.mark.parametrize("name", ("q", "k", "v", "out", "dout"))
def test_a_base_off_a_16_byte_boundary_is_a_value_error(name, elements) -> None:
    device = _device_or_skip()
    ops = _operands(device)
    ops[name] = _offset(ops[name], elements)
    with pytest.raises(ValueError):
        _call(ops)


@pytest.mark.parametrize(
    "sizes",
    (
        dict(B=65536, H=1, D=64),  # B past the grid's 65535 batches (grid z)
        dict(H=65536, D=64),  # H past the grid's 65535 heads (grid y)
        dict(H=1, Sq=2**24),  # B * H * Sq * D = 2^31 elements
        dict(H=1, Skv=2**24),  # B * H * Skv * D = 2^31 elements
    ),
    ids=("grid-batches", "grid-heads", "query-elements", "key-elements"),
)
def test_sizes_past_the_kernels_limits_are_a_value_error(sizes) -> None:
    device = _device_or_skip()
    ops = _operands(device, **sizes)
    with pytest.raises(ValueError):
        _call(ops)
    del ops
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: torch.use_deterministic_algorithms, empty batches, streams and devices


def test_a_workspace_that_does_not_fit_is_torch_out_of_memory_error() -> None:
    """A call whose FP32 workspace does not fit raises ``torch.OutOfMemoryError`` (a ``RuntimeError``), the error of
    a PyTorch allocation: the process is held by ``torch.cuda.set_per_process_memory_fraction`` to its operands, the
    three gradients and 60 MiB, below ``dq_accum``'s 160 MiB."""
    device = _device_or_skip()
    _call(_operands(device))  # loads the libraries
    ops = _operands(device, H=16, Sq=20480, Skv=20480)  # each BF16 operand and gradient 80 MiB
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gradients = 3 * ops["q"].numel() * ops["q"].element_size()
    limit = torch.cuda.memory_reserved(device) + gradients + (60 << 20)
    torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with pytest.raises(torch.OutOfMemoryError):
            _call(ops)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        del ops
        torch.cuda.empty_cache()


def test_deterministic_mode_raises_the_alert() -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` the call raises PyTorch's ``RuntimeError``, naming the entry
    and no other package to fall back to; outside the mode the same call runs."""
    device = _device_or_skip()
    x = make_inputs(Case(1, 2, 256), 128, causal=False, seed=5, device=device)
    with _torch_deterministic(warn_only=False):
        with pytest.raises(RuntimeError, match=ALERT) as info:
            api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)
    assert "_det_train" not in str(info.value)
    got = api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)
    assert_backward_contract("outside the mode", x, _grads(*got))


@pytest.mark.parametrize("causal", (False, True), ids=("plain", "causal"))
def test_warn_only_warns_once_and_runs(causal, record_property) -> None:
    """With ``warn_only=True`` each call warns once (``UserWarning``, naming the entry and no other package) and runs:
    its gradients meet the tier."""
    device = _device_or_skip()

    def draw(seed: int):
        x = make_inputs(Case(1, 2, 256), 128, causal=causal, seed=seed, device=device)
        with _torch_deterministic(warn_only=True):
            with pytest.warns(UserWarning) as record:
                got = api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout, causal=causal)
        ours = [str(w.message) for w in record if ALERT in str(w.message)]
        assert len(ours) == 1 and ours[0].startswith(ALERT), ours
        assert "_det_train" not in ours[0]
        return judged_backward(x, {"warn_only": _grads(*got)})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("causal", (False, True), ids=("plain", "causal"))
def test_an_empty_batch_launches_nothing_and_alerts_nothing(causal) -> None:
    """``B == 0``: empty gradients of the operands' shapes, without an alert even under
    ``torch.use_deterministic_algorithms(True)`` and without a warning."""
    device = _device_or_skip()
    Skv = 256 if causal else 384
    ops = _operands(device, B=0, H=2, Sq=256, Skv=Skv)
    with _torch_deterministic(warn_only=False), warnings.catch_warnings():
        warnings.simplefilter("error")
        dq, dk, dv = _call(ops, causal=causal)
    shapes = {"dq": (0, 256, 2, 128), "dk": (0, Skv, 2, 128), "dv": (0, Skv, 2, 128)}
    for name, value in _grads(dq, dk, dv).items():
        assert (tuple(value.shape), value.dtype, value.device) == (shapes[name], torch.bfloat16, device), name
        assert value.is_contiguous(), name


def test_runs_on_the_current_stream_without_synchronizing() -> None:
    """On a side stream, behind a long sleep kernel that precedes the write of ``dout`` (NaN until then): the call
    returns while the sleep still runs (no synchronization), and its gradients are finite (every launch read ``dout``
    after the write, on the current stream)."""
    device = _device_or_skip()
    x = make_inputs(Case(2, 3, 256, 384), 128, causal=False, seed=21, device=device)
    dout = torch.full_like(x.dout, math.nan)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)  # loads the libraries
    side.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        dout.copy_(x.dout)
        got = api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, dout)
        pending = not side.query()
    side.synchronize()
    assert pending, "the call synchronized: the sleep kernel queued before it had finished"
    assert_backward_contract("on a side stream", x, _grads(*got))


def test_a_call_captured_in_a_cuda_graph_replays(record_property) -> None:
    """A call captured in a CUDA graph (its launches go to the capture stream): after its gradients are filled with
    NaN, the replay writes every element and the results meet the tier."""
    device = _device_or_skip()
    x = make_inputs(Case(2, 3, 256, 384), 128, causal=False, seed=31, device=device)
    api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)  # loads the libraries before a capture

    def draw(seed: int):
        x = make_inputs(Case(2, 3, 256, 384), 128, causal=False, seed=seed, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)
        for t in got:
            t.fill_(math.nan)
        graph.replay()
        torch.cuda.synchronize(device)
        assert_backward_contract("replayed", x, _grads(*got))
        return judged_backward(x, {"replayed": _grads(*got)})

    judge_draws(record_property, draw)


def test_a_device_outside_the_architectures_is_refused(monkeypatch) -> None:
    """A device of capability (11, 0) (no delivered build) is a ``RuntimeError`` naming the delivered architectures."""
    device = _device_or_skip()
    x = make_inputs(Case(1, 2, 128), 128, causal=False, seed=3, device=device)
    device_arch = _backend().device_arch
    clear = getattr(device_arch, "cache_clear", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)
    finally:
        monkeypatch.undo()
        clear()


def test_a_device_of_capability_10_7_is_served(monkeypatch) -> None:
    """A device of capability (10, 7) is the delivered ``sm_107a`` target: the backend maps it to ``sm_107a``, and
    ``build.json`` holds an ``sm_107a`` library for every record and every host sequence (nothing is built or launched
    here)."""
    from cake_pplx._backends.cake_fmha_nondet_train import cake_jit

    path = Path(cake_jit.__file__).parent / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    device_arch = _backend().device_arch
    clear = getattr(device_arch, "cache_clear", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    clear()
    try:
        assert device_arch(0) == "sm_107a"
    finally:
        monkeypatch.undo()
        clear()
    entries = [entry for entry in json.loads(path.read_text())["libraries"] if entry["arch"] == "sm_107a"]
    assert sorted(entry["name"] for entry in entries) == sorted(
        f"{rec['library']}_sm_107a" for rec in (*cake_jit.MODULES.values(), *cake_jit.SEQUENCES.values())
    )
    assert all(entry["gencode"] == "-gencode=arch=compute_107a,code=sm_107a" for entry in entries)


def test_operands_on_another_device(record_property) -> None:
    """With two devices: operands on the device that is not current give gradients on that device that meet the tier;
    operands on two devices are a ``ValueError``."""
    _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")
    other = torch.device("cuda:1")
    if torch.cuda.get_device_capability(other) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the second device is not an exported architecture")
    previous = torch.cuda.current_device()
    torch.cuda.set_device(0)
    try:

        def draw(seed: int):
            x = make_inputs(Case(2, 3, 256), 128, causal=False, seed=seed, device=other)
            assert torch.cuda.current_device() == 0
            got = api.cake_fmha_nondet_train_backward(x.q, x.k, x.v, x.out, x.lse, x.dout)
            return judged_backward(x, {"other": _grads(*got)})

        judge_draws(record_property, draw)
        ops = _operands(torch.device("cuda:0"))
        ops["k"] = ops["k"].to(other)
        with pytest.raises(ValueError):
            _call(ops)
    finally:
        torch.cuda.set_device(previous)
