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

Public-API suite of ``cake_pplx.cake_gated_kv_compression_det_train``, the ratio-2 gated KV compression of the Cake
deterministic training package: ``cake_gated_kv_compression_det_train_forward(c, z, cu_seqlens, num_pairs=None, *,
out=None)``, ``cake_gated_kv_compression_det_train_backward(dout, c, z, cu_seqlens, *, dc=None, dz=None,
pair_offsets=None)``, the autograd entries ``cake_gated_kv_compression_det(c, z, cu_seqlens, num_pairs)`` and
``cake_gated_kv_compression_det_cz(cz, cu_seqlens, num_pairs)``, and the memory report.

Without a device: the module's public names, each entry a stable API (no ``ExperimentalWarning``), importing the module
imports no backend, the module reads no environment variable, every entry's parameters are the backend's, a
``deterministic`` keyword is a ``TypeError`` before any work, operands on the CPU are a ``ValueError``, and the support
query is false off CUDA.

On SM100 / SM103 / SM107: the entries' ``out``, ``dC`` and ``dZ`` meet the naive BF16 tier against an FP64 oracle
(``tests/cake_naive.py``, 32 independent draws) through both autograd entries and through the explicit steps with
caller outputs (the gradients in the halves of one buffer); every documented ``ValueError`` and ``TypeError``; the
empty problems launch nothing and return empty or zero-filled outputs; the launches run on the current stream without
synchronizing; a captured CUDA graph replays to the bits of the eager call; ``torch.use_deterministic_algorithms(True)``
changes nothing; a forward launches the pair table only past 256 documents (its kernel builds the table of fewer);
every backward path launches ``bwd_warp`` for more than ``bwd_warp_docs`` documents (16384) and for
``bwd_warp_min_docs`` to ``bwd_warp_max_docs`` documents (15 to 32 on SM103, 41 to 1024 on SM107), else ``bwd`` for
more than ``bwd_small_docs`` documents (26 on SM100 / SM103, 125 on SM107) and at most ``bwd_max_docs`` (no bound on
SM100 / SM103, 16384 on SM107), else ``bwd_small``, and every backward role gives the same bits;
the autograd nodes save their inputs and the pair table only, compute only the requested gradients, read an incoming
gradient of any layout, and replay bit for bit under ``torch.utils.checkpoint(..., use_reentrant=False)``; a device
outside the delivered architectures is a ``RuntimeError`` and a device of capability (10, 7) is served; with two
devices, operands on the device that is not current give results there.
"""

from __future__ import annotations

import ast
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

import cake_pplx.cake_gated_kv_compression_det_train as api
from tests.cake.test_helpers.cake_gated_kv_compression_naive import (
    PERF_PACKS,
    WIDTH,
    Pack,
    assert_backward_contract,
    assert_bitwise,
    assert_forward_contract,
    judge_draws,
    judged,
    lognormal_lengths,
    make_inputs,
    nan_halves,
    nan_like_rows,
    pair_table_of,
)

BACKEND = "cake_pplx._backends.cake_gated_kv_compression_det_train"
NAMES = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeGatedKvCompressionDetCzFunction",
    "CakeGatedKvCompressionDetFunction",
    "cake_gated_kv_compression_det",
    "cake_gated_kv_compression_det_cz",
    "cake_gated_kv_compression_det_train_backward",
    "cake_gated_kv_compression_det_train_forward",
    "cake_gated_kv_compression_det_train_memory_bytes",
    "is_cake_gated_kv_compression_det_train_supported",
]
ENTRIES = {
    "cake_gated_kv_compression_det_train_forward": ["c", "z", "cu_seqlens", "num_pairs", "out"],
    "cake_gated_kv_compression_det_train_backward": ["dout", "c", "z", "cu_seqlens", "dc", "dz", "pair_offsets"],
    "cake_gated_kv_compression_det": ["c", "z", "cu_seqlens", "num_pairs"],
    "cake_gated_kv_compression_det_cz": ["cz", "cu_seqlens", "num_pairs"],
    "cake_gated_kv_compression_det_train_memory_bytes": ["num_tokens", "num_docs"],
}
MIXED = Pack((127, 256, 1, 641))
JUDGED_PACKS = (Pack((0, 1, 3, 2, 7, 4)), MIXED, *PERF_PACKS)
FORMS = ("cz", "separate")
# Cycles of the sleep kernel queued before a call on a side stream (about 0.1 s): the call returns long before it ends.
SLEEP_CYCLES = 200_000_000


def _backend():
    from cake_pplx._backends.cake_gated_kv_compression_det_train import cake_backend

    return cake_backend


def _jit():
    from cake_pplx._backends.cake_gated_kv_compression_det_train import cake_jit

    return cake_jit


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake gated KV compression ships for SM100 / SM103 / SM107")
    if not (Path(_jit().__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    _build_libraries(0)
    return torch.device("cuda:0")


@functools.cache
def _build_libraries(index: int) -> None:
    """Build the device's libraries once, before the first judged or stream-ordered call."""
    jit = _jit()
    for spec in jit.specs_for_arch(jit.device_arch(index)):
        spec.build_and_load()


def _autograd(x, form: str, *, requires=(True, True)):
    """``(out, dc, dz)`` of one autograd forward and backward of ``x`` (fresh leaves; ``dout`` as the gradient)."""
    if form == "cz":
        cz = x.cz.detach().clone().requires_grad_(True)
        out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
        out.backward(x.dout)
        return out.detach(), cz.grad[:, :WIDTH], cz.grad[:, WIDTH:]
    c = x.c.detach().clone().requires_grad_(requires[0])
    z = x.z.detach().clone().requires_grad_(requires[1])
    out = api.cake_gated_kv_compression_det(c, z, x.cu, x.G)
    out.backward(x.dout)
    return out.detach(), c.grad, z.grad


def _explicit(x):
    """``(out, dc, dz)`` of the explicit steps with caller outputs: ``out`` NaN first, ``dc`` / ``dz`` the halves of
    one NaN ``[T, 1024]`` buffer."""
    out = nan_like_rows(x.G, torch.bfloat16, x.c.device)
    api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, out=out)
    _, dc, dz = nan_halves(x.T, x.c.device)
    api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    return out, dc, dz


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_the_module_lists_its_names() -> None:
    assert sorted(api.__all__) == sorted(NAMES)
    for name in NAMES:
        assert hasattr(api, name), name


def test_the_entries_are_stable_apis() -> None:
    for name in ENTRIES:
        entry = getattr(api, name)
        assert getattr(entry, "is_experimental", False) is False, name
        assert entry.__name__ == name


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import cake_pplx.cake_gated_kv_compression_det_train as api\n"
        "names = dir(api)\n"
        f"print(any(m.startswith({BACKEND!r}) for m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_module_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(api.__file__).read_text())
    hits = [
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
    ]
    assert not hits, hits


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_parameters_match_the_backend(name) -> None:
    ours = inspect.signature(inspect.unwrap(getattr(api, name))).parameters
    theirs = inspect.signature(getattr(_backend(), name)).parameters
    assert list(ours) == list(theirs) == ENTRIES[name]
    for key, param in ours.items():
        assert param.kind == theirs[key].kind, (name, key)
        assert param.default == theirs[key].default, (name, key)


def test_keyword_only_options() -> None:
    forward = inspect.signature(api.cake_gated_kv_compression_det_train_forward).parameters
    backward = inspect.signature(api.cake_gated_kv_compression_det_train_backward).parameters
    assert forward["num_pairs"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and forward["num_pairs"].default is None
    assert forward["out"].kind is inspect.Parameter.KEYWORD_ONLY and forward["out"].default is None
    for key in ("dc", "dz", "pair_offsets"):
        assert backward[key].kind is inspect.Parameter.KEYWORD_ONLY and backward[key].default is None, key


def _cpu_operands(T: int = 6, G: int = 3) -> dict:
    return {
        "c": torch.zeros(T, WIDTH),
        "z": torch.zeros(T, WIDTH),
        "cz": torch.zeros(T, 2 * WIDTH),
        "cu": torch.tensor([0, T], dtype=torch.int32),
        "dout": torch.zeros(G, WIDTH, dtype=torch.bfloat16),
    }


def test_a_deterministic_keyword_is_a_type_error() -> None:
    o = _cpu_operands()
    calls = (
        lambda **kw: api.cake_gated_kv_compression_det_train_forward(o["c"], o["z"], o["cu"], 3, **kw),
        lambda **kw: api.cake_gated_kv_compression_det_train_backward(o["dout"], o["c"], o["z"], o["cu"], **kw),
        lambda **kw: api.cake_gated_kv_compression_det(o["c"], o["z"], o["cu"], 3, **kw),
        lambda **kw: api.cake_gated_kv_compression_det_cz(o["cz"], o["cu"], 3, **kw),
    )
    for call in calls:
        for value in (True, False):
            with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
                call(deterministic=value)


def test_cpu_operands_are_refused() -> None:
    o = _cpu_operands()
    with pytest.raises(ValueError):
        api.cake_gated_kv_compression_det_train_forward(o["c"], o["z"], o["cu"], 3)
    with pytest.raises(ValueError):
        api.cake_gated_kv_compression_det_train_backward(o["dout"], o["c"], o["z"], o["cu"])
    with pytest.raises(ValueError):
        api.cake_gated_kv_compression_det(o["c"], o["z"], o["cu"], 3)
    with pytest.raises(ValueError):
        api.cake_gated_kv_compression_det_cz(o["cz"], o["cu"], 3)


def test_the_support_query_is_false_off_cuda() -> None:
    assert api.is_cake_gated_kv_compression_det_train_supported("cpu") is False
    assert api.is_cake_gated_kv_compression_det_train_supported(torch.device("meta")) is False
    assert tuple(api.ARCHES) == ("sm_100a", "sm_103a", "sm_107a")
    assert tuple(api.SUPPORTED_COMPUTE_CAPABILITIES) == ((10, 0), (10, 3), (10, 7))


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: judged results


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("pack", JUDGED_PACKS, ids=[p.id for p in JUDGED_PACKS])
def test_the_autograd_entries_match_the_reference(pack, form, record_property) -> None:
    """``out`` and the gradients of ``C`` and ``Z`` through the autograd entry (``cz``: one ``[T, 1024]`` input whose
    gradient holds ``dC | dZ``) meet the tier."""
    device = _device_or_skip()

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout="halves")
        out, dc, dz = _autograd(x, form)
        assert_forward_contract(form, x, out)
        assert_backward_contract(form, x, dc, dz)
        return judged(x, {form: {"out": out, "dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("layout", ("contiguous", "halves"))
@pytest.mark.parametrize("pack", JUDGED_PACKS, ids=[p.id for p in JUDGED_PACKS])
def test_the_explicit_steps_match_the_reference(pack, layout, record_property) -> None:
    """The explicit forward into a caller ``out`` and the explicit backward into the halves of one caller buffer meet
    the tier, every element written over NaN."""
    device = _device_or_skip()

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout=layout)
        out, dc, dz = _explicit(x)
        assert_forward_contract("explicit", x, out)
        assert_backward_contract("explicit", x, dc, dz)
        return judged(x, {"explicit": {"out": out, "dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("variant", ({"close": 12}, {"close": 16}, {"gates": True}), ids=("2^-12", "2^-16", "gates"))
def test_extreme_inputs_match_the_reference(variant, record_property) -> None:
    device = _device_or_skip()

    def draw(seed: int):
        x = make_inputs(MIXED, seed=seed, device=device, layout="halves", **variant)
        out, dc, dz = _autograd(x, "cz")
        assert_backward_contract("cz", x, dc, dz)
        return judged(x, {"cz": {"out": out, "dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


# ----------------------------------------------------------------------------------------------------------------
# Errors


def _operands(device: torch.device) -> dict:
    """Well-formed operands of zeros on a pack of 2 and 5 tokens (G = 3), a call that should fail never reads them."""
    T, G = 7, 3
    cz = torch.zeros(T, 2 * WIDTH, device=device)
    return {
        "cz": cz,
        "c": cz[:, :WIDTH],
        "z": cz[:, WIDTH:],
        "cu_seqlens": torch.tensor([0, 2, 7], dtype=torch.int32, device=device),
        "num_pairs": G,
        "out": torch.zeros(G, WIDTH, dtype=torch.bfloat16, device=device),
        "dout": torch.zeros(G, WIDTH, dtype=torch.bfloat16, device=device),
        "dc": torch.zeros(T, WIDTH, device=device),
        "dz": torch.zeros(T, WIDTH, device=device),
        "pair_offsets": torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
    }


def _forward(o):
    return api.cake_gated_kv_compression_det_train_forward(
        o["c"], o["z"], o["cu_seqlens"], o["num_pairs"], out=o["out"]
    )


def _backward(o):
    return api.cake_gated_kv_compression_det_train_backward(
        o["dout"], o["c"], o["z"], o["cu_seqlens"], dc=o["dc"], dz=o["dz"], pair_offsets=o["pair_offsets"]
    )


def _flat(o, dtype, shape, offset_bytes=0):
    """A contiguous tensor of ``shape`` whose base is ``offset_bytes`` past a fresh allocation."""
    size = torch.tensor([], dtype=dtype).element_size()
    storage = torch.zeros(math.prod(shape) + 64, dtype=dtype, device=o["c"].device)
    return storage[offset_bytes // size :][: math.prod(shape)].view(shape)


def _overlapping_grads(o) -> None:
    """``dc`` and ``dz``: rows 0-6 and 1-7 of one ``[8, 512]`` buffer."""
    buffer = torch.zeros(8, WIDTH, device=o["c"].device)
    o.update(dc=buffer[:7], dz=buffer[1:])


def _dz_over_dout(o) -> None:
    """``dout`` (BF16 ``[3, 512]``, contiguous) inside the storage of ``dz``."""
    buffer = torch.zeros(7, WIDTH, device=o["c"].device)
    o.update(dz=buffer, dout=buffer.view(-1).view(torch.bfloat16)[: 3 * WIDTH].view(3, WIDTH))


# (label, call, edit): the edit makes one operand malformed; both steps check every operand they take.
VALUE_ERRORS = [
    ("c dtype", _forward, lambda o: o.update(c=o["c"].to(torch.bfloat16))),
    ("z dtype", _forward, lambda o: o.update(z=o["z"].double())),
    ("cu_seqlens dtype", _forward, lambda o: o.update(cu_seqlens=o["cu_seqlens"].long())),
    ("out dtype", _forward, lambda o: o.update(out=o["out"].float())),
    ("dout dtype", _backward, lambda o: o.update(dout=o["dout"].float())),
    ("dc dtype", _backward, lambda o: o.update(dc=o["dc"].bfloat16())),
    ("pair_offsets dtype", _backward, lambda o: o.update(pair_offsets=o["pair_offsets"].long())),
    ("c width", _forward, lambda o: o.update(c=o["cz"][:, :256])),
    ("z rows", _forward, lambda o: o.update(z=o["cz"][:6, WIDTH:])),
    ("cu_seqlens rank", _forward, lambda o: o.update(cu_seqlens=o["cu_seqlens"][None])),
    ("cu_seqlens empty", _forward, lambda o: o.update(cu_seqlens=o["cu_seqlens"][:0])),
    (
        "cu_seqlens strided",
        _forward,
        lambda o: o.update(cu_seqlens=torch.zeros(6, dtype=torch.int32, device=o["c"].device)[::2]),
    ),
    ("out width", _forward, lambda o: o.update(out=o["out"][:, :256].contiguous())),
    ("out rows", _forward, lambda o: o.update(num_pairs=2)),
    (
        "out strided",
        _forward,
        lambda o: o.update(out=torch.zeros(WIDTH, 3, dtype=torch.bfloat16, device=o["c"].device).t()),
    ),
    ("out misaligned", _forward, lambda o: o.update(out=_flat(o, torch.bfloat16, (3, WIDTH), 2))),
    ("out over c", _forward, lambda o: o.update(out=o["cz"].view(-1).view(torch.bfloat16)[: 3 * WIDTH].view(3, WIDTH))),
    ("no pair count", _forward, lambda o: o.update(num_pairs=None, out=None)),
    ("negative pair count", _forward, lambda o: o.update(num_pairs=-1, out=None)),
    ("pair count past the limit", _forward, lambda o: o.update(num_pairs=(1 << 26) + 1, out=None)),
    ("c feature stride", _forward, lambda o: o.update(c=torch.zeros(WIDTH, 7, device=o["c"].device).t())),
    ("c row stride off 8", _forward, lambda o: o.update(c=torch.zeros(7, 1028, device=o["c"].device)[:, :WIDTH])),
    (
        "c rows overlap",
        _forward,
        lambda o: o.update(c=torch.zeros(7 * 256 + 256, device=o["c"].device).as_strided((7, WIDTH), (256, 1))),
    ),
    ("c misaligned", _forward, lambda o: o.update(c=o["cz"][:, 4 : 4 + WIDTH])),
    ("z misaligned", _forward, lambda o: o.update(z=o["cz"][:, WIDTH - 4 : 2 * WIDTH - 4])),
    ("dout width", _backward, lambda o: o.update(dout=torch.zeros(3, 511, dtype=torch.bfloat16, device=o["c"].device))),
    (
        "dout strided",
        _backward,
        lambda o: o.update(dout=torch.zeros(WIDTH, 3, dtype=torch.bfloat16, device=o["c"].device).t()),
    ),
    ("dout misaligned", _backward, lambda o: o.update(dout=_flat(o, torch.bfloat16, (3, WIDTH), 8))),
    ("dc rows", _backward, lambda o: o.update(dc=o["dc"][:6])),
    ("dz misaligned", _backward, lambda o: o.update(dz=_flat(o, torch.float32, (7, WIDTH), 16))),
    ("dc is dz", _backward, lambda o: o.update(dz=o["dc"])),
    ("dc over dz", _backward, _overlapping_grads),
    ("dc over c", _backward, lambda o: o.update(dc=o["c"])),
    ("dz over dout", _backward, _dz_over_dout),
    ("pair_offsets length", _backward, lambda o: o.update(pair_offsets=o["pair_offsets"][:2])),
    ("z on the CPU", _forward, lambda o: o.update(z=o["z"].cpu())),
    ("dout on the CPU", _backward, lambda o: o.update(dout=o["dout"].cpu())),
]


@pytest.mark.parametrize(("label", "call", "edit"), VALUE_ERRORS, ids=[label for label, _, _ in VALUE_ERRORS])
def test_malformed_operands_are_a_value_error(label, call, edit) -> None:
    device = _device_or_skip()
    o = _operands(device)
    call(o)  # well formed
    edit(o)
    with pytest.raises(ValueError):
        call(o)


CZ_ERRORS = [
    ("dtype", lambda o: o["cz"].to(torch.bfloat16)),
    ("width", lambda o: o["cz"][:, :1000]),
    ("row stride off 8", lambda o: torch.zeros(7, 1028, device=o["c"].device)[:, : 2 * WIDTH]),
    ("misaligned", lambda o: _flat(o, torch.float32, (7, 2 * WIDTH), 16)),
    ("feature stride", lambda o: torch.zeros(2 * WIDTH, 7, device=o["c"].device).t()),
    ("on the CPU", lambda o: o["cz"].cpu()),
]


@pytest.mark.parametrize(("label", "make"), CZ_ERRORS, ids=[label for label, _ in CZ_ERRORS])
def test_a_malformed_cz_is_a_value_error(label, make) -> None:
    device = _device_or_skip()
    o = _operands(device)
    api.cake_gated_kv_compression_det_cz(o["cz"], o["cu_seqlens"], 3)  # well formed
    with pytest.raises(ValueError):
        api.cake_gated_kv_compression_det_cz(make(o), o["cu_seqlens"], 3)


@pytest.mark.parametrize("num_pairs", (torch.tensor(3), True, 3.0, "3"), ids=("tensor", "bool", "float", "str"))
def test_a_pair_count_that_is_not_a_host_int_is_a_type_error(num_pairs) -> None:
    device = _device_or_skip()
    o = _operands(device)
    with pytest.raises(TypeError):
        api.cake_gated_kv_compression_det_train_forward(o["c"], o["z"], o["cu_seqlens"], num_pairs)
    with pytest.raises(TypeError):
        api.cake_gated_kv_compression_det_cz(o["cz"], o["cu_seqlens"], num_pairs)


def test_a_non_tensor_operand_is_a_type_error() -> None:
    device = _device_or_skip()
    o = _operands(device)
    with pytest.raises(TypeError):
        api.cake_gated_kv_compression_det_train_forward(o["c"], [0.0] * WIDTH, o["cu_seqlens"], 3)
    with pytest.raises(TypeError):
        api.cake_gated_kv_compression_det_train_backward(o["dout"], o["c"], o["z"], [0, 2, 7])


# ----------------------------------------------------------------------------------------------------------------
# Empty problems, streams, graphs, modes


@pytest.fixture
def launches(monkeypatch):
    """The roles launched while the test runs (a spy on the registry's launch)."""
    jit = _jit()
    found = []
    original = jit.run_module

    def spy(role, values, *, rows, device_index, **options):
        found.append(role)
        return original(role, values, rows=rows, device_index=device_index, **options)

    monkeypatch.setattr(jit, "run_module", spy)
    return found


@pytest.mark.parametrize("lengths", ((), (0,), (1, 0, 1), (1,), (0, 0)), ids=("none", "0", "1-0-1", "1", "0-0"))
@pytest.mark.parametrize("form", FORMS)
def test_packs_without_a_pair_launch_nothing(lengths, form, launches) -> None:
    """A pack with no complete pair: an empty ``out``, all-zero ``dC`` and ``dZ`` (every byte), and no kernel
    launched by any entry."""
    device = _device_or_skip()
    x = make_inputs(Pack(lengths), seed=3, device=device, layout="halves")
    out, dc, dz = _autograd(x, form)
    assert tuple(out.shape) == (0, WIDTH) and out.dtype == torch.bfloat16
    for name, value in (("dc", dc), ("dz", dz)):
        assert tuple(value.shape) == (x.T, WIDTH), name
        assert int((value.view(torch.int32) != 0).sum()) == 0, name
    explicit = api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, 0)
    assert tuple(explicit.shape) == (0, WIDTH)
    _, gc, gz = nan_halves(x.T, device)
    api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=gc, dz=gz)
    assert int((gc.view(torch.int32) != 0).sum()) == int((gz.view(torch.int32) != 0).sum()) == 0
    assert launches == []


def test_no_token_or_no_document_zero_fills_without_a_kernel(launches) -> None:
    """``T == 0`` or ``S == 0`` with a caller ``G`` above the pair count: ``out`` is zero-filled, no kernel launched."""
    device = _device_or_skip()
    c = torch.zeros(0, WIDTH, device=device)
    for cu in (torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(3, dtype=torch.int32, device=device)):
        out = api.cake_gated_kv_compression_det_train_forward(c, c, cu, 4)
        assert tuple(out.shape) == (4, WIDTH) and int((out.view(torch.int16) != 0).sum()) == 0
        dc, dz = api.cake_gated_kv_compression_det_train_backward(
            torch.zeros(0, WIDTH, dtype=torch.bfloat16, device=device), c, c, cu
        )
        assert tuple(dc.shape) == tuple(dz.shape) == (0, WIDTH)
    assert launches == []


def test_runs_on_the_current_stream_without_synchronizing() -> None:
    """On a side stream, behind a long sleep kernel that precedes the write of ``dout`` (NaN until then): the calls
    return while the sleep still runs (no synchronization), and the gradients are finite (every launch read ``dout``
    after the write, on the current stream)."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=21, device=device, layout="halves")
    dout = torch.full_like(x.dout, math.nan)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        _explicit(x)  # loads the libraries
    side.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        dout.copy_(x.dout)
        out = api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
        dc, dz = api.cake_gated_kv_compression_det_train_backward(dout, x.c, x.z, x.cu)
        pending = not side.query()
    side.synchronize()
    assert pending, "the call synchronized: the sleep kernel queued before it had finished"
    assert_forward_contract("side stream", x, out)
    assert_backward_contract("side stream", x, dc, dz)


@pytest.mark.parametrize("form", ("explicit", "cz"))
def test_a_call_captured_in_a_cuda_graph_replays_the_eager_bits(form) -> None:
    """The forward and backward captured in a CUDA graph (explicit steps, or the ``cz`` autograd entry and its
    backward): after the outputs are filled with NaN, the replay writes the bits of the eager call."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=41, device=device, layout="halves")
    want = _explicit(x)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    if form == "explicit":
        with torch.cuda.stream(side):
            _explicit(x)
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=side):
            got = _explicit(x)
    else:
        cz = x.cz.detach().clone().requires_grad_(True)
        with torch.cuda.stream(side):
            api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G).backward(x.dout)
            cz.grad = None
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=side):
            out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
            out.backward(x.dout)
        got = (out.detach(), cz.grad[:, :WIDTH], cz.grad[:, WIDTH:])
    for t in got:
        t.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize(device)
    for name, a, b in zip(("out", "dc", "dz"), got, want, strict=True):
        assert_bitwise(a, b, f"{form}: the replayed {name}")


def test_deterministic_algorithms_change_nothing() -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` every entry runs without an alert or a warning and gives the
    bits of the default mode."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=43, device=device, layout="halves")
    want = _explicit(x)
    want_autograd = _autograd(x, "cz")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _explicit(x)
            got_autograd = _autograd(x, "cz")
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    names = [f"{entry} {name}" for entry in ("explicit", "autograd") for name in ("out", "dc", "dz")]
    for name, a, b in zip(names, got + got_autograd, want + want_autograd, strict=True):
        assert_bitwise(a, b, f"deterministic mode: {name}")
    assert "use_deterministic_algorithms" in api.cake_gated_kv_compression_det_train_forward.__doc__


@pytest.mark.parametrize("form", FORMS)
def test_the_autograd_node_saves_its_inputs_and_the_pair_table_only(form) -> None:
    """The tensors the node saves: its inputs (``c`` and ``z``, or ``cz``) and ``cu_seqlens`` by reference, and the
    forward's int32 ``[S + 1]`` pair table (equal to the prefix of the pair counts) -- nothing else."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=47, device=device, layout="halves")
    saved = []
    cz = x.cz.detach().clone().requires_grad_(True)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        if form == "cz":
            out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
            inputs = [cz]
        else:
            out = api.cake_gated_kv_compression_det(c, z, x.cu, x.G)
            inputs = [c, z]
    tables = [t for t in saved if t.dtype == torch.int32 and t.data_ptr() != x.cu.data_ptr()]
    rest = [t for t in saved if t.dtype != torch.int32 or t.data_ptr() == x.cu.data_ptr()]
    assert len(tables) == 1 and torch.equal(tables[0], pair_table_of(MIXED.lengths, device))
    assert sorted((t.data_ptr(), tuple(t.shape)) for t in rest) == sorted(
        (t.data_ptr(), tuple(t.shape)) for t in [*inputs, x.cu]
    )
    retained = api.cake_gated_kv_compression_det_train_memory_bytes(x.T, x.S)["retained_for_backward"]
    assert retained == tables[0].numel() * tables[0].element_size()
    out.backward(x.dout)


# Packs at the forward's fused bound (256 documents), past it, and on either side of the small pair table's bound (512
# documents): one document, 256, 257, 512 and 513 documents of 0 to 6 tokens.
BOUND_PACKS = tuple(Pack((2, *(i * 5 % 7 for i in range(1, n))), f"S{n}") for n in (1, 256, 257, 512, 513))


@pytest.mark.parametrize("pack", BOUND_PACKS, ids=[p.id for p in BOUND_PACKS])
def test_the_forward_launches_the_pair_table_only_past_256_documents(pack, launches) -> None:
    """A forward of at most ``fuse_docs`` (256) documents launches the ``fwd`` kernel alone, which writes the pair
    table the autograd node saves; a forward of more launches a pair table first, ``pair_table_small`` for at most
    ``small_docs`` (512) documents and ``pair_table`` for more, then ``fwd_unfused``, which reads it.  The saved table
    is the prefix of the pair counts every way, and the explicit forward gives the bits of the autograd one."""
    device = _device_or_skip()
    jit = _jit()
    arch = jit.device_arch(device.index)
    assert jit.fuse_docs(arch) == 256 and jit.small_docs(arch) == 512
    x = make_inputs(pack, seed=53, device=device, layout="halves")
    saved = []
    c = x.c.detach().clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        out = api.cake_gated_kv_compression_det(c, x.z, x.cu, x.G)
    table = "pair_table_small" if pack.S <= 512 else "pair_table"
    assert launches == (["fwd"] if pack.S <= 256 else [table, "fwd_unfused"]), pack.id
    tables = [t for t in saved if t.dtype == torch.int32 and t.data_ptr() != x.cu.data_ptr()]
    assert len(tables) == 1
    assert torch.equal(tables[0], pair_table_of(pack.lengths, device)), f"{pack.id}: saved pair table"
    explicit = api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
    assert_bitwise(explicit, out.detach(), f"{pack.id}: out")


# The backward's bounds on each architecture: the most documents of ``bwd_small`` below ``bwd``, the most documents
# of ``bwd`` (``None``: no bound), above which ``bwd_small`` runs again, the most documents of either, above which
# ``bwd_warp`` runs, and the documents below that ``bwd_warp`` also runs for (``None``: none).
BWD_SMALL_DOCS = {"sm_100a": 26, "sm_103a": 26, "sm_107a": 125}
BWD_MAX_DOCS = {"sm_100a": None, "sm_103a": None, "sm_107a": 16384}
BWD_WARP_DOCS = {"sm_100a": 16384, "sm_103a": 16384, "sm_107a": 16384}
BWD_WARP_RANGES = {"sm_100a": None, "sm_103a": (15, 32), "sm_107a": (41, 1024)}
# Packs on either side of the documents bounds (14, 26, 32, 40, 125, 1024 and 16384) and of the forward's fused bound
# (256 documents): one document, 14, 15, 16, 26, 27, 32, 33, 40, 41, 124, 125, 126, 256, 257, 1024, 1025, 16384, 16385
# and 32769 documents of 0 to 6 tokens; and log-normal packs of 16383 to 65536 tokens on either side of the documents
# bounds (the role does not depend on the tokens).
BWD_BOUND_PACKS = (
    *(
        Pack((2, *(i * 5 % 7 for i in range(1, n))), f"S{n}")
        for n in (1, 14, 15, 16, 26, 27, 32, 33, 40, 41, 124, 125, 126, 256, 257, 1024, 1025, 16384, 16385, 32769)
    ),
    *(
        Pack(lognormal_lengths(T, S, T + S))
        for T, S in (
            (16383, 26),
            (16384, 27),
            (16384, 124),
            (16384, 125),
            (16384, 126),
            (65536, 125),
            (65536, 126),
            (32768, 16384),
            (32768, 16385),
            (65536, 16385),
        )
    ),
)


@pytest.mark.parametrize("pack", BWD_BOUND_PACKS, ids=[p.id for p in BWD_BOUND_PACKS])
def test_every_backward_runs_the_role_of_its_documents(pack, launches) -> None:
    """A backward of ``S`` documents launches ``bwd_warp`` for ``S > bwd_warp_docs`` (16384) and for
    ``bwd_warp_min_docs <= S <= bwd_warp_max_docs`` (15 to 32 on sm_103a, 41 to 1024 on sm_107a), else ``bwd`` for
    ``bwd_small_docs < S <= bwd_max_docs`` (26 on sm_100a / sm_103a and 125 on sm_107a; no bound on sm_100a / sm_103a
    and 16384 on sm_107a), else ``bwd_small``.  The rule holds on the three backward paths: the explicit backward
    without ``pair_offsets`` (after a pair table), with them (alone), and the autograd backward (alone, on its forward's
    table).  The three give the same ``dC`` and ``dZ`` bits, and so does a launch of every other backward role served
    on the architecture, on the same table."""
    device = _device_or_skip()
    jit = _jit()
    arch = jit.device_arch(device.index)
    assert jit.bwd_small_docs(arch) == BWD_SMALL_DOCS[arch]
    assert jit.bwd_max_docs(arch) == BWD_MAX_DOCS[arch]
    span = BWD_WARP_RANGES[arch]
    assert jit.bwd_warp_region(arch) == (*(span or (None, None)), BWD_WARP_DOCS[arch])
    high = BWD_MAX_DOCS[arch]
    role = "bwd" if BWD_SMALL_DOCS[arch] < pack.S and (high is None or pack.S <= high) else "bwd_small"
    if pack.S > BWD_WARP_DOCS[arch] or (span is not None and span[0] <= pack.S <= span[1]):
        role = "bwd_warp"
    table_role = "pair_table_small" if pack.S <= 512 else "pair_table"
    forward = ["fwd"] if pack.S <= 256 else [table_role, "fwd_unfused"]
    x = make_inputs(pack, seed=59, device=device, layout="halves")
    table = pair_table_of(pack.lengths, device)
    _, dc, dz = nan_halves(x.T, device)
    launches.clear()
    api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    assert launches == [table_role, role], pack.id
    _, given_dc, given_dz = nan_halves(x.T, device)
    launches.clear()
    api.cake_gated_kv_compression_det_train_backward(
        x.dout, x.c, x.z, x.cu, dc=given_dc, dz=given_dz, pair_offsets=table
    )
    assert launches == [role], pack.id
    launches.clear()
    _, auto_dc, auto_dz = _autograd(x, "cz")
    assert launches == [*forward, role], pack.id
    for name, got, want in (
        ("dc with pair_offsets", given_dc, dc),
        ("dz with pair_offsets", given_dz, dz),
        ("autograd dc", auto_dc, dc),
        ("autograd dz", auto_dz, dz),
    ):
        assert_bitwise(got, want, f"{pack.id}: {name}")
    served = [name for name in ("bwd", "bwd_small", "bwd_warp") if arch in jit.ROLE_ARCHES.get(name, jit.ARCHES)]
    assert served == ["bwd", "bwd_small", "bwd_warp"]
    for other in served:
        if other == role:
            continue
        _, other_dc, other_dz = nan_halves(x.T, device)
        launches.clear()
        _backend().launch_bwd(
            dout=x.dout,
            c=x.c,
            z=x.z,
            cu_seqlens=x.cu,
            cu_pairs=table,
            dc=other_dc,
            dz=other_dz,
            c_ld=x.c.stride(0),
            z_ld=x.z.stride(0),
            dc_ld=other_dc.stride(0),
            dz_ld=other_dz.stride(0),
            role=other,
        )
        assert launches == [other], pack.id
        assert_bitwise(other_dc, dc, f"{pack.id}: dc of {other}")
        assert_bitwise(other_dz, dz, f"{pack.id}: dz of {other}")


def test_only_the_requested_gradients_are_returned() -> None:
    """With only ``c`` requiring a gradient, ``z.grad`` stays None and ``c.grad`` has the bits of the explicit
    ``dc``."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=53, device=device)
    _, dc, dz = _autograd(x, "separate", requires=(True, False))
    assert dz is None
    want, _ = api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
    assert_bitwise(dc, want, "c.grad")


def test_an_incoming_gradient_of_any_layout_is_read() -> None:
    """An expanded (zero-stride) incoming gradient and one off a 16-byte boundary give the bits of the explicit backward
    of the same values."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=59, device=device)
    cz = x.cz.detach().clone().requires_grad_(True)
    api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G).sum().backward()
    ones = torch.ones(x.G, WIDTH, dtype=torch.bfloat16, device=device)
    dc, dz = api.cake_gated_kv_compression_det_train_backward(ones, x.c, x.z, x.cu)
    assert_bitwise(cz.grad[:, :WIDTH], dc, "dc of an expanded gradient")
    assert_bitwise(cz.grad[:, WIDTH:], dz, "dz of an expanded gradient")
    storage = torch.empty(x.G * WIDTH + 8, dtype=torch.bfloat16, device=device)
    shifted = storage[1 : 1 + x.G * WIDTH].view(x.G, WIDTH)
    shifted.copy_(x.dout)
    cz.grad = None
    out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
    torch.autograd.backward(out, shifted)
    dc, dz = api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
    assert_bitwise(cz.grad[:, :WIDTH], dc, "dc of a misaligned gradient")
    assert_bitwise(cz.grad[:, WIDTH:], dz, "dz of a misaligned gradient")


@pytest.mark.parametrize("pack", (MIXED, PERF_PACKS[0]), ids=lambda p: p.id)
def test_checkpoint_replay_is_bitwise(pack) -> None:
    """Under ``torch.utils.checkpoint(..., use_reentrant=False)`` around a small graph (``out(cz * 2) * w``), the
    output and the gradients of ``cz`` and ``w`` have the bits of the same graph without the checkpoint, and the
    recompute's output has the bits of the first forward."""
    device = _device_or_skip()
    x = make_inputs(pack, seed=61, device=device)
    w = torch.randn(WIDTH, device=device).to(torch.bfloat16)
    outs: list[torch.Tensor] = []

    def block(cz_: torch.Tensor, w_: torch.Tensor) -> torch.Tensor:
        y = api.cake_gated_kv_compression_det_cz(cz_ * 2, x.cu, x.G)
        outs.append(y.detach().clone())
        return y * w_

    def run(checkpointed: bool):
        outs.clear()
        cz = x.cz.detach().clone().requires_grad_(True)
        wl = w.detach().clone().requires_grad_(True)
        y = torch.utils.checkpoint.checkpoint(block, cz, wl, use_reentrant=False) if checkpointed else block(cz, wl)
        y.backward(x.dout)
        return y.detach(), cz.grad, wl.grad, list(outs)

    plain = run(False)
    replay = run(True)
    assert len(plain[3]) == 1 and len(replay[3]) == 2
    for y in replay[3]:
        assert_bitwise(y, plain[3][0], "the recomputed forward")
    for name, got, want in zip(("output", "dcz", "dw"), replay[:3], plain[:3], strict=True):
        assert_bitwise(got, want, f"the checkpointed {name}")


def test_a_device_outside_the_architectures_is_refused(monkeypatch) -> None:
    """A device of capability (11, 0) (no delivered build) is a ``RuntimeError`` naming the delivered architectures."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=3, device=device)
    device_arch = _jit().device_arch
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    device_arch.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
        assert api.is_cake_gated_kv_compression_det_train_supported(device) is False
    finally:
        monkeypatch.undo()
        device_arch.cache_clear()


def test_a_device_of_capability_10_7_is_served(monkeypatch) -> None:
    """A device of capability (10, 7) is the delivered ``sm_107a`` target: ``build.json`` holds an ``sm_107a``
    library for every record that serves it, one per role served there, the forwards' being ``fwd_sm_107a`` and
    ``fwd_unfused_sm_107a`` and the backwards' ``bwd_sm_107a`` (for 1025 to 16384 documents), ``bwd_warp_sm_107a``
    (for 41 to 1024 and for more than 16384) and ``bwd_small_sm_107a`` (for the others; nothing is built or launched
    here)."""
    jit = _jit()
    path = Path(jit.__file__).parent / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    jit.device_arch.cache_clear()
    try:
        assert jit.device_arch(0) == "sm_107a"
    finally:
        monkeypatch.undo()
        jit.device_arch.cache_clear()
    entries = [entry for entry in json.loads(path.read_text())["libraries"] if entry["arch"] == "sm_107a"]
    roles = [role for role in jit.ROLES if "sm_107a" in jit.ROLE_ARCHES.get(role, jit.ARCHES)]
    assert roles == ["pair_table", "pair_table_small", "fwd", "fwd_unfused", "bwd", "bwd_small", "bwd_warp"]
    assert sorted(entry["name"] for entry in entries) == sorted(
        f"{jit.MODULES[jit.role_record(role, 'sm_107a')]['library']}_sm_107a" for role in roles
    )
    assert jit.role_record("fwd", "sm_107a") == "fwd_sm_107a"
    assert jit.role_record("fwd_unfused", "sm_107a") == "fwd_unfused_sm_107a"
    assert jit.role_record("bwd", "sm_107a") == "bwd_sm_107a"
    assert jit.role_record("bwd_small", "sm_107a") == "bwd_small_sm_107a"
    assert jit.bwd_small_docs("sm_107a") == 125
    assert jit.bwd_max_docs("sm_107a") == 16384
    assert jit.role_record("bwd_warp", "sm_107a") == "bwd_warp_sm_107a"
    assert jit.bwd_warp_region("sm_107a") == (41, 1024, 16384)
    assert all(entry["gencode"] == "-gencode=arch=compute_107a,code=sm_107a" for entry in entries)


def test_operands_on_another_device(record_property) -> None:
    """With two devices: operands on the device that is not current give results there that meet the tier; operands
    on two devices are a ``ValueError``."""
    _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")
    other = torch.device("cuda:1")
    if torch.cuda.get_device_capability(other) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the second device is not a delivered architecture")
    _build_libraries(1)
    previous = torch.cuda.current_device()
    torch.cuda.set_device(0)
    try:

        def draw(seed: int):
            x = make_inputs(MIXED, seed=seed, device=other, layout="halves")
            assert torch.cuda.current_device() == 0
            out, dc, dz = _autograd(x, "cz")
            assert out.device == other and dc.device == other
            return judged(x, {"other": {"out": out, "dc": dc, "dz": dz}})

        judge_draws(record_property, draw)
        o = _operands(torch.device("cuda:0"))
        o["z"] = o["z"].to(other)
        with pytest.raises(ValueError):
            _forward(o)
    finally:
        torch.cuda.set_device(previous)
