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

Public-API suite of ``cake_pplx.cake_attn_res_det_train``, the deterministic attention residual.

Without a device: the entries are stable APIs, importing the module imports no backend, the module reads no
environment variable, every keyword of an entry is the backend's keyword of the same name, the entries document their
behaviour under ``torch.use_deterministic_algorithms``, ``values`` that is not a list or tuple of 1 to 9 tensors is a
``ValueError``, operands on the CPU are a ``ValueError``, and a device of capability (10, 7) maps to ``sm_107a``.

On SM100 / SM103 / SM107: CUPTI counts the GPU activities of a forward and a backward call; the autograd entry meets
the naive BF16 tier against an FP64 oracle (``tests/cake_naive.py``, 32 draws) for the output and every gradient; the
explicit forward and backward give the autograd entry's ``out`` and value gradients bit for bit and write caller
outputs in place; ``n`` and the token count change from call to call without a new build; ``T = 0`` and ``T = 1``
work; every documented ``ValueError``; a device outside the architectures is a ``RuntimeError``; the
``torch.use_deterministic_algorithms`` behaviour; the launches run on the current stream without synchronizing; a
forward and a backward captured in one CUDA graph replay; ``torch.utils.checkpoint(use_reentrant=False)`` gives the
plain output and gradients; the retained state and the workspace are reported; every output repeats bit for bit run
to run, ``dw`` included.  The issue's acceptance criteria are ``test_cake_attn_res_acceptance.py``.
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

import pytest
import torch
import torch.utils.checkpoint

import cake_pplx.cake_attn_res_det_train as api
from tests.cake.test_helpers.cake_attn_res_naive import HIDDEN, judge_draws, make_inputs

BACKEND = "cake_pplx._backends.cake_attn_res_det_train"
ENTRIES = (
    "cake_attn_res_det",
    "cake_attn_res_det_train_backward",
    "cake_attn_res_det_train_forward",
    "cake_attn_res_det_train_retained_bytes",
    "cake_attn_res_det_train_workspace_bytes",
    "is_cake_attn_res_det_train_supported",
)
# GPU activities of one call: the forward launches ``fwd``; the backward launches ``bwd`` and ``dw_reduce``.
LAUNCHES = {"forward": 1, "backward": 2}
# Cycles of the sleep kernel queued before a call on a side stream (about 0.1 s): the call returns long before it ends.
SLEEP_CYCLES = 200_000_000


def _backend():
    from cake_pplx._backends.cake_attn_res_det_train import cake_backend

    return cake_backend


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake AttnRes kernels ship for SM100 / SM103 / SM107")
    from cake_pplx._backends.cake_attn_res_det_train import cake_jit

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


def _autograd(ns, *, checkpoint: bool = False) -> dict:
    """Output and gradients of the autograd entry (``checkpoint``: under
    ``torch.utils.checkpoint(use_reentrant=False)``), keyed as the judge's outputs."""
    values = [v.detach().clone().requires_grad_(True) for v in ns.values]
    w = ns.w.detach().clone().requires_grad_(True)
    fn = functools.partial(api.cake_attn_res_det, values, w)
    out = torch.utils.checkpoint.checkpoint(fn, use_reentrant=False) if checkpoint else fn()
    out.backward(ns.dout)
    found = {"out": out.detach(), **{f"dv{j}": v.grad for j, v in enumerate(values)}, "dw": w.grad}
    return found


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bytes."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _record_failures(judge, failures, record_property) -> None:
    judge.record(record_property)
    assert not failures, failures


def _cpu_operands(T: int = 2, n: int = 3):
    values = [torch.zeros(T, HIDDEN, dtype=torch.bfloat16) for _ in range(n)]
    return values, torch.ones(HIDDEN)


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
        "import cake_pplx.cake_attn_res_det_train as api\n"
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
    entry = inspect.signature(api.cake_attn_res_det).parameters
    assert list(entry) == ["values", "w", "eps"]
    assert entry["eps"].kind is inspect.Parameter.KEYWORD_ONLY and entry["eps"].default == 1e-5
    assert list(inspect.signature(_backend().CakeAttnResDetFunction.forward).parameters) == [
        "ctx",
        "w",
        "eps",
        "values",
    ]


def test_the_constants_are_the_backend_values() -> None:
    backend = _backend()
    for name in ("ARCHES", "HIDDEN", "MAX_VALUES", "SUPPORTED_COMPUTE_CAPABILITIES"):
        assert getattr(api, name) == getattr(backend, name), name
    assert api.HIDDEN == 7168 and api.MAX_VALUES == 9
    assert api.SUPPORTED_COMPUTE_CAPABILITIES == ((10, 0), (10, 3), (10, 7))


def test_the_entries_document_their_determinism() -> None:
    """Each entry states its behaviour under ``torch.use_deterministic_algorithms``."""
    for name in ENTRIES:
        assert "use_deterministic_algorithms" in getattr(api, name).__doc__, name


def test_values_must_be_a_list_of_one_to_nine_tensors() -> None:
    values, w = _cpu_operands(n=10)
    for bad in ((), values, values[0], None, (values[0], "v")):
        with pytest.raises(ValueError, match="values"):
            api.cake_attn_res_det(bad, w)
        with pytest.raises(ValueError, match="values"):
            api.cake_attn_res_det_train_forward(bad, w)


def test_cpu_operands_are_refused() -> None:
    values, w = _cpu_operands()
    with pytest.raises(ValueError, match="CUDA"):
        api.cake_attn_res_det(values, w)
    with pytest.raises(ValueError, match="CUDA"):
        api.cake_attn_res_det_train_forward(values, w)
    p = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="CUDA"):
        api.cake_attn_res_det_train_backward(values[0], values, w, p, p, p)
    assert api.is_cake_attn_res_det_train_supported("cpu") is False


def test_a_device_of_capability_10_7_is_served(monkeypatch) -> None:
    """A device of capability (10, 7) is the ``sm_107a`` target, and ``build.json`` holds an ``sm_107a`` library for
    every record (nothing is built or launched here)."""
    from cake_pplx._backends.cake_attn_res_det_train import cake_jit

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
# On SM100 / SM103 / SM107: launches and accuracy


def _gpu_activities(call):
    """The names of the GPU activities (kernels, memsets, copies) ``call`` runs, as CUPTI reports them through
    ``torch.profiler``; None when the profiler sees no GPU activity at all (its CUPTI does not serve the device)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names = [event.name for event in prof.events() if event.device_type == DeviceType.CUDA]
    return names or None


@pytest.mark.parametrize("n", (2, 9))
def test_launches_per_call(n) -> None:
    """CUPTI's count of the GPU activities of one forward and one backward call (autograd and explicit): no copy of
    a value, no stack, no cast."""
    device = _device_or_skip()
    ns = make_inputs(129, n, seed=61, device=device)
    values = [v.detach().clone().requires_grad_(True) for v in ns.values]
    w = ns.w.detach().clone().requires_grad_(True)
    api.cake_attn_res_det(values, w).backward(ns.dout)  # loads the programs
    for leaf in (*values, w):
        leaf.grad = None
    holder = {}
    forward = _gpu_activities(lambda: holder.setdefault("out", api.cake_attn_res_det(values, w)))
    if forward is None:
        pytest.skip("torch.profiler's CUPTI sees no GPU activity on this device")
    backward = _gpu_activities(lambda: holder["out"].backward(ns.dout))
    assert (len(forward), len(backward)) == (LAUNCHES["forward"], LAUNCHES["backward"]), (forward, backward)
    state = api.cake_attn_res_det_train_forward(ns.values, ns.w)
    explicit_fwd = _gpu_activities(lambda: api.cake_attn_res_det_train_forward(ns.values, ns.w))
    assert len(explicit_fwd) == LAUNCHES["forward"], explicit_fwd
    explicit = _gpu_activities(lambda: api.cake_attn_res_det_train_backward(ns.dout, ns.values, ns.w, *state[1:]))
    assert len(explicit) == LAUNCHES["backward"], explicit


# (n, T, data, weights)
CASES = (
    (1, 129, "scaled", "small"),
    (2, 1, "scaled", "small"),
    (2, 1025, "peaked", "normproj"),
    (5, 129, "zero_rows", "small"),
    (9, 1025, "scaled", "normproj"),
    (9, 129, "peaked", "small"),
)
CASE_IDS = [f"n{n}-T{T}-{data}-{w}" for n, T, data, w in CASES]


@pytest.mark.parametrize(("n", "T", "data", "weights"), CASES, ids=CASE_IDS)
def test_the_autograd_entry_matches_the_reference(n, T, data, weights, record_property) -> None:
    device = _device_or_skip()

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data=data, weights=weights)

    _record_failures(*judge_draws(_autograd, make, seed=31 + T + 100 * n), record_property)


@pytest.mark.parametrize("n", (1, 4, 9))
def test_the_explicit_steps_give_the_autograd_bits(n) -> None:
    """The explicit forward and backward give the autograd entry's ``out`` and gradients bit for bit (``dw`` too) and
    write caller-owned ``out`` and ``dv`` in place."""
    device = _device_or_skip()
    ns = make_inputs(257, n, seed=67, device=device, data="peaked")
    got = _autograd(ns)
    out = torch.full((257, HIDDEN), math.nan, dtype=torch.bfloat16, device=device)
    res = api.cake_attn_res_det_train_forward(ns.values, ns.w, out=out)
    assert res[0] is out and _same_bits(out, got["out"])
    for t in res[1:]:
        assert t.dtype == torch.float32 and tuple(t.shape) == (257, n) and t.is_contiguous()
    dv = [torch.full_like(out, math.nan) for _ in range(n)]
    got_dv, dw = api.cake_attn_res_det_train_backward(ns.dout, ns.values, ns.w, *res[1:], dv=dv)
    assert all(a is b for a, b in zip(got_dv, dv, strict=True))
    for j in range(n):
        assert _same_bits(dv[j], got[f"dv{j}"]), j
    assert dw.dtype == torch.float32 and tuple(dw.shape) == (HIDDEN,) and _same_bits(dw, got["dw"])


def test_dynamic_token_counts_and_values_need_no_new_build() -> None:
    """Calls with changing ``n`` (1 to 9) and token counts (1 to 4097) load no library beyond the first call's:
    one program per stage serves every shape."""
    device = _device_or_skip()
    backend = _backend()
    ns = make_inputs(3, 2, seed=71, device=device)
    _autograd(ns)
    loads = backend.load.cache_info().currsize
    entries = backend._entry.cache_info().currsize
    for T, n in ((1, 1), (7, 9), (4097, 5), (129, 3), (2, 8), (1025, 9), (33, 2)):
        got = _autograd(make_inputs(T, n, seed=73 + T, device=device))
        assert tuple(got["out"].shape) == (T, HIDDEN)
        assert all(torch.isfinite(t.float()).all() for t in got.values())
    assert backend.load.cache_info().currsize == loads
    assert backend._entry.cache_info().currsize == entries


@pytest.mark.parametrize("n", (1, 5, 9))
def test_no_token(n) -> None:
    """``T = 0``: an empty output and state, empty value gradients and a zero ``dw``; nothing launches, nothing
    alerts."""
    device = _device_or_skip()
    ns = make_inputs(0, n, seed=3, device=device)
    with _torch_deterministic(warn_only=False):
        got = _autograd(ns)
        out, p, r, d = api.cake_attn_res_det_train_forward(ns.values, ns.w)
        dv, dw = api.cake_attn_res_det_train_backward(ns.dout, ns.values, ns.w, p, r, d)
    assert tuple(got["out"].shape) == (0, HIDDEN) and tuple(out.shape) == (0, HIDDEN)
    assert all(tuple(t.shape) == (0, n) for t in (p, r, d))
    assert len(dv) == n and all(tuple(t.shape) == (0, HIDDEN) for t in dv)
    for t in (got["dw"], dw):
        assert t.dtype == torch.float32 and tuple(t.shape) == (HIDDEN,) and not t.abs().any()


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: checks


def _malformed(ns, device):
    """``(id, values, w, eps, out)``: one malformed operand each (the documented ``ValueError`` cases)."""
    bf16 = dict(dtype=torch.bfloat16, device=device)
    T = ns.T
    v = ns.values
    return [
        ("value-fp32", [v[0].float(), *v[1:]], ns.w, 1e-5, None),
        ("value-rank", [v[0][0], *v[1:]], ns.w, 1e-5, None),
        ("value-width", [torch.zeros(T, 4096, **bf16), *v[1:]], ns.w, 1e-5, None),
        ("value-tokens", [v[0], torch.zeros(T + 1, HIDDEN, **bf16), *v[2:]], ns.w, 1e-5, None),
        ("value-strided", [v[0], torch.zeros(T, HIDDEN + 8, **bf16)[:, :HIDDEN], *v[2:]], ns.w, 1e-5, None),
        ("value-cpu", [v[0], v[1].cpu(), *v[2:]], ns.w, 1e-5, None),
        ("w-bf16", v, ns.w.bfloat16(), 1e-5, None),
        ("w-shape", v, torch.ones(HIDDEN + 1, device=device), 1e-5, None),
        ("w-cpu", v, ns.w.cpu(), 1e-5, None),
        ("eps-nan", v, ns.w, math.nan, None),
        ("eps-negative", v, ns.w, -1.0, None),
        ("eps-inf", v, ns.w, math.inf, None),
        ("eps-str", v, ns.w, "eps", None),
        ("out-strided", v, ns.w, 1e-5, torch.zeros(HIDDEN, T, **bf16).t()),
        ("out-fp32", v, ns.w, 1e-5, torch.zeros(T, HIDDEN, device=device)),
    ]


def test_malformed_operands_are_a_value_error() -> None:
    device = _device_or_skip()
    ns = make_inputs(4, 3, seed=2, device=device)
    for label, values, w, eps, out in _malformed(ns, device):
        with pytest.raises(ValueError):
            api.cake_attn_res_det_train_forward(values, w, eps=eps, out=out)
            pytest.fail(f"forward accepted {label}")
        if out is not None or label.startswith("eps"):
            continue
        with pytest.raises(ValueError):
            api.cake_attn_res_det(values, w)
            pytest.fail(f"autograd entry accepted {label}")
    out, p, r, d = api.cake_attn_res_det_train_forward(ns.values, ns.w)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    for label, dout, state, dv in (
        ("dout-fp32", ns.dout.float(), (p, r, d), None),
        ("dout-strided", torch.zeros(4, HIDDEN + 8, **bf16)[:, :HIDDEN], (p, r, d), None),
        ("p-shape", ns.dout, (p[:, :2], r, d), None),
        ("r-bf16", ns.dout, (p, r.bfloat16(), d), None),
        ("d-strided", ns.dout, (p, r, d.t().contiguous().t()), None),
        ("dv-count", ns.dout, (p, r, d), [torch.empty_like(ns.dout)] * 2),
        ("dv-fp32", ns.dout, (p, r, d), [torch.empty(4, HIDDEN, device=device)] * 3),
    ):
        with pytest.raises(ValueError):
            api.cake_attn_res_det_train_backward(dout, ns.values, ns.w, *state, dv=dv)
            pytest.fail(f"backward accepted {label}")


def test_a_base_off_a_16_byte_boundary_is_a_value_error() -> None:
    device = _device_or_skip()
    ns = make_inputs(4, 2, seed=2, device=device)
    flat = torch.empty(4 * HIDDEN + 64, dtype=torch.bfloat16, device=device)
    shifted = flat[1 : 1 + 4 * HIDDEN].view(4, HIDDEN)
    assert shifted.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte"):
        api.cake_attn_res_det_train_forward([ns.values[0], shifted], ns.w)


def test_a_device_outside_the_architectures_is_refused(monkeypatch) -> None:
    device = _device_or_skip()
    ns = make_inputs(4, 2, seed=2, device=device)
    device_arch = _backend().device_arch
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    device_arch.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            api.cake_attn_res_det_train_forward(ns.values, ns.w)
    finally:
        monkeypatch.undo()
        device_arch.cache_clear()


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: determinism, streams, graphs, checkpoint, memory


def test_deterministic_mode_changes_nothing() -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict or ``warn_only``: nothing alerts or warns, and the
    output and every gradient are the bits of a call outside the mode."""
    device = _device_or_skip()
    ns = make_inputs(1031, 5, seed=19, device=device, data="zero_rows")
    first = _autograd(ns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with _torch_deterministic(warn_only=False):
            second = _autograd(ns)
        with _torch_deterministic(warn_only=True):
            third = _autograd(ns)
    for name in ("out", *(f"dv{j}" for j in range(5)), "dw"):
        assert _same_bits(first[name], second[name]), name
        assert _same_bits(first[name], third[name]), name


def test_runs_on_the_current_stream_without_synchronizing() -> None:
    """On a side stream, behind a long sleep kernel that precedes the write of ``dout`` (NaN until then): the forward
    and the backward return while the sleep still runs, and the gradients are finite."""
    device = _device_or_skip()
    ns = make_inputs(129, 9, seed=31, device=device)
    dout = torch.full_like(ns.dout, math.nan)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    fwd, bwd = api.cake_attn_res_det_train_forward, api.cake_attn_res_det_train_backward
    with torch.cuda.stream(side):
        state = fwd(ns.values, ns.w)
        bwd(ns.dout, ns.values, ns.w, *state[1:])
    side.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        dout.copy_(ns.dout)
        state = fwd(ns.values, ns.w)
        dv, dw = bwd(dout, ns.values, ns.w, *state[1:])
        pending = not side.query()
    side.synchronize()
    assert pending, "the call synchronized: the sleep kernel queued before it had finished"
    for t in (*state, *dv, dw):
        assert torch.isfinite(t.float()).all()


@pytest.mark.parametrize("T", (1, 129, 4097))
def test_a_call_captured_in_a_cuda_graph_replays(T) -> None:
    """The forward and the backward captured in one CUDA graph: after the outputs are filled with NaN, the replay
    writes every element; ``out``, the state and every gradient equal the eager call's bits."""
    device = _device_or_skip()
    ns = make_inputs(T, 5, seed=37, device=device, data="peaked")
    fwd, bwd = api.cake_attn_res_det_train_forward, api.cake_attn_res_det_train_backward
    eager_state = fwd(ns.values, ns.w)
    eager_dv, eager_dw = bwd(ns.dout, ns.values, ns.w, *eager_state[1:])
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        state = fwd(ns.values, ns.w)
        bwd(ns.dout, ns.values, ns.w, *state[1:])
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state = fwd(ns.values, ns.w)
        dv, dw = bwd(ns.dout, ns.values, ns.w, *state[1:])
    for t in (*state, *dv, dw):
        t.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize(device)
    for a, b in zip(state, eager_state, strict=True):
        assert _same_bits(a, b)
    for a, b in zip(dv, eager_dv, strict=True):
        assert _same_bits(a, b)
    assert _same_bits(dw, eager_dw)


def test_checkpoint_gives_the_plain_output_and_gradients() -> None:
    """Under ``torch.utils.checkpoint(use_reentrant=False)`` the output and every gradient are plain autograd's
    bits."""
    device = _device_or_skip()
    ns = make_inputs(1031, 9, seed=41, device=device, data="zero_rows", weights="normproj")
    plain = _autograd(ns)
    ckpt = _autograd(ns, checkpoint=True)
    for name in ("out", *(f"dv{j}" for j in range(9)), "dw"):
        assert _same_bits(plain[name], ckpt[name]), name


@pytest.mark.parametrize("n", (2, 5, 9))
def test_retained_state_and_workspace_are_reported(n, record_property) -> None:
    """The memory one training step keeps and allocates beside its operands at ``T = 16380``.  The autograd node
    saves the inputs by reference and one new FP32 ``[3, T, n]`` buffer (``p``, ``r``, ``d``):
    :func:`cake_attn_res_det_train_retained_bytes` bytes.  The allocator's view agrees: after the forward the
    memory beyond ``out`` is that state, and the backward's peak beyond its gradients is at most the workspace
    (:func:`cake_attn_res_det_train_workspace_bytes`), both up to the caching allocator's rounding (blocks of at
    least 10 MiB end on 2 MiB boundaries)."""
    device = _device_or_skip()
    T = 16380
    ns = make_inputs(T, n, seed=43, device=device)
    values = [v.detach().clone().requires_grad_(True) for v in ns.values]
    w = ns.w.detach().clone().requires_grad_(True)
    api.cake_attn_res_det(values, w).backward(ns.dout)  # loads, warms the allocator
    for leaf in (*values, w):
        leaf.grad = None
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    out = api.cake_attn_res_det(values, w)
    torch.cuda.synchronize(device)
    retained = torch.cuda.memory_allocated(device) - base - out.untyped_storage().nbytes()
    inputs = {t.untyped_storage().data_ptr() for t in (*values, w)}
    saved = out.grad_fn.saved_tensors
    new = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in saved}
    new = {ptr: size for ptr, size in new.items() if ptr not in inputs}
    assert len(saved) == n + 4 and inputs <= {t.untyped_storage().data_ptr() for t in saved}
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    out.backward(ns.dout)
    torch.cuda.synchronize(device)
    grads = sum(t.grad.untyped_storage().nbytes() for t in (*values, w))
    peak = torch.cuda.max_memory_allocated(device) - before - grads
    want = api.cake_attn_res_det_train_retained_bytes(T, n)
    workspace = api.cake_attn_res_det_train_workspace_bytes(T, n, device)
    record_property("retained_bytes_saved", sum(new.values()))
    record_property("retained_bytes_allocator", retained)
    record_property("retained_bytes_reported", want)
    record_property("backward_peak_beyond_grads_bytes", peak)
    record_property("workspace_bytes", workspace)
    print(
        f"T={T} n={n} retained saved={sum(new.values())} allocator={retained} reported={want} "
        f"backward_peak_beyond_grads={peak} workspace={workspace}"
    )
    assert list(new.values()) == [want], new
    slack = 2 * 2**20
    assert want <= retained <= want + slack, (retained, want)
    assert peak <= workspace + (n + 2) * slack, (peak, workspace)


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
        return make_inputs(16, 3, seed=seed, device=other)

    _record_failures(*judge_draws(run, make, seed=47), record_property)
    ns = make_inputs(4, 2, seed=1, device=other)
    with pytest.raises(ValueError, match="device"):
        api.cake_attn_res_det_train_forward([ns.values[0], ns.values[1].to("cuda:0")], ns.w)


@pytest.mark.parametrize(("n", "T"), ((2, 1), (5, 4097), (9, 16380)))
def test_every_output_repeats_bit_for_bit(n, T) -> None:
    """Five autograd calls on the same inputs give the same bytes for ``out`` and every gradient, ``dw`` included."""
    device = _device_or_skip()
    ns = make_inputs(T, n, seed=53 + n, device=device, data="scaled", weights="normproj")
    first = _autograd(ns)
    for _ in range(4):
        again = _autograd(ns)
        for name, t in first.items():
            assert _same_bits(t, again[name]), name
