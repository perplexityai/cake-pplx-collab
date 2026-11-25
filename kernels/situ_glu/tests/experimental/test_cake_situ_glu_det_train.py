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

Backend suite of the Cake deterministic SiTU-GLU training package (``cake_pplx._backends.cake_situ_glu_det_train``,
the backend of ``cake_pplx.cake_situ_glu_det_train``): ``h = bf16(beta tanh(g / beta) sigmoid(g) lb tanh(u / lb))`` and
its gradients ``d_gate`` and ``d_up``, at any width that is a multiple of 256, on SM100 and SM103.

Without a device: the registry (one program per stage, every record delivered for the two architectures and for no
other, the launch facts, the argument plans), the launch grid, the build manifest (every record's library, the
recorded source digests), the capability map, the ``beta`` and ``lb`` keywords and their defaults, the refusal of CPU
tensors and no environment read outside the JIT shim.

On SM100 / SM103, every training output (``h``, ``d_gate`` and ``d_up``) judged against the naive BF16 tier
(``tests/cake/test_helpers/cake_naive.py``: the error against an FP64 oracle of the same inputs over 32 independent
draws, in the L2 and the max norm, median and largest, beside the error of the same operation in BF16,
:mod:`tests.cake.test_helpers.situ_glu_reference`) over row counts 1 to 4097, widths 256 to 33,792 (one segment,
partial passes of the CTA's warps, both sides of a full pass, and the issue's widths), the merged, two-tensor, sliced
and copied layouts, saturated, deep negative, small and wide values, zero gates and ups, every BF16 gate and scalars
other than the model's.  The kernels have no row-count boundary: one program per stage, one CTA per block of rows.
Beside the judge, exact facts of the contract (no tolerance): ``g = 0`` or ``u = 0`` gives ``h = 0`` exactly (and
``d_up = 0`` at ``g = 0``, ``d_gate = 0`` at ``u = 0``); saturated inputs give finite outputs with the oracle's signs
outside the flushed range; ``R = 0`` returns empty outputs and launches nothing; one library per stage serves every row
count, width, scalar and layout; the same bits at every call, in every layout, at every row count (a row's outputs do
not depend on the other rows), at large row counts whose offsets pass 2**31 bytes, on a side stream with no host
synchronization, in CUDA graph replays and under ``torch.use_deterministic_algorithms``; the autograd function matches
the steps bit for bit, saves only its inputs, gives the same bits under non-reentrant checkpointing and has no double
backward; inputs the kernels can read are read in place, others copied once; unfit outputs, operands and scalars are
refused.

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.situ_glu_reference import (
    BETA,
    LB,
    Case,
    arrange,
    fp32,
    judge,
    make_inputs,
    oracle,
)
from tests.cake.test_helpers.situ_glu_reference import outputs as step_outputs

pkg = cake_package("cake_situ_glu_det_train")
cake_jit = cake_package("cake_situ_glu_det_train", "cake_jit")

forward = pkg.cake_situ_glu_det_train_forward
backward = pkg.cake_situ_glu_det_train_backward
situ_glu = pkg.cake_situ_glu_det

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
STAGES = ("fwd", "bwd")
# The kernel parameters of each stage, in launch order (the kernels' signatures).
PARAMETERS = {
    "fwd": ("gate", "up", "h", "rows", "width", "gate_stride", "up_stride", "h_stride", "beta", "lb"),
    "bwd": (
        "gate",
        "up",
        "dh",
        "d_gate",
        "d_up",
        "rows",
        "width",
        "gate_stride",
        "up_stride",
        "dh_stride",
        "d_gate_stride",
        "d_up_stride",
        "beta",
        "lb",
    ),
}
BUFFERS = {"gate", "up", "h", "dh", "d_gate", "d_up"}
# The threads of a CTA of each stage's program.
THREADS = 128
# The smallest number that FP32 and BF16 hold as a normal number.
MIN_NORMAL = 2.0**-126

# The backward's outputs, the only ones the tiny-input cases judge.
GRADS = ("d_gate", "d_up")
# The judged cases: row counts 1 to 4097, widths 256 to 33792, every layout, the data kinds and other scalars.  A pass
# of the forward's CTA covers 12 segments of 256 columns (3072 columns), one of the backward's 8 (2048 columns).
CASES = [
    Case(1),
    Case(2, 256),  # one segment: three warps of the CTA idle
    Case(5, 1280, "merged"),  # five segments: one pass of the CTA, partial
    Case(33, 1792),  # just below one pass of the backward's CTA
    Case(33, 2048, "merged"),  # one pass of the backward's CTA
    Case(33, 2304),  # just above it
    Case(33, 2816, "merged"),  # just below one pass of the forward's CTA
    Case(33, 3328),  # just above it
    Case(3, 33792, "sliced"),
    Case(129),
    Case(129, layout="merged"),
    Case(129, layout="sliced"),
    Case(129, layout="copied"),
    Case(4097),
    Case(4097, layout="merged"),
    Case(1025, 6144),
    Case(1025, 6144, "merged"),
    Case(257, 33792),
    Case(257, 33792, "merged"),
    Case(257, data="wide"),
    Case(257, data="saturated"),
    Case(257, layout="merged", data="saturated"),
    Case(257, 6144, data="saturated"),
    Case(257, data="deep_negative"),
    Case(257, data="small"),
    # Tiny gates or ups, one band a case, for the backward: near where its two forms of beta (1 - E) meet (8), where
    # 1 - E from E alone is off by percents (20), and deep in the FP32 range, where E rounds to 1 (60).  The forward
    # takes 1 - E from E (its bound is in the package's README), so these cases judge d_gate and d_up only.
    Case(65, data="tiny_gate", band=8, judged=GRADS),
    Case(65, data="tiny_gate", band=20, judged=GRADS),
    Case(65, layout="merged", data="tiny_gate", band=60, judged=GRADS),
    Case(65, data="tiny_up", band=20, judged=GRADS),
    Case(65, 6144, data="tiny_up", band=60, judged=GRADS),
    Case(65, data="tiny_gate", band=20, beta=2.5, lb=10.0, judged=GRADS),
    Case(257, data="zero_gate"),
    Case(257, layout="merged", data="zero_up"),
    Case(22, data="all_gates"),  # 22 x 3072 >= 65536 gate patterns
    Case(257, beta=2.5, lb=10.0),  # a beta other than 4: P takes its own exponential
    Case(257, data="saturated", beta=3.0, lb=40.0),
    Case(257, data="deep_negative", beta=3.0),
]


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not pkg.is_cake_situ_glu_det_train_supported(torch.device("cuda")):
        pytest.skip(f"the package ships {ARCHES}; this device is {torch.cuda.get_device_capability()}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in pkg.specs_for_arch(pkg.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list:
    """``(stage, rows, values)`` of every launch, in order."""
    launched: list = []
    run_module = pkg.run_module

    def spy(stage, values, *, rows, device_index):
        launched.append((stage, rows, dict(values)))
        return run_module(stage, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(pkg, "run_module", spy)
    return launched


def assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape, (what, got.dtype, want.dtype, got.shape, want.shape)
    view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.contiguous().view(view), want.contiguous().view(view)), what


def assert_same_outputs(got: dict, want: dict, what: str) -> None:
    assert set(got) == set(want), what
    for name in want:
        assert_same_bits(got[name], want[name], f"{what}: {name}")


def _relayout(t: types.SimpleNamespace, layout: str) -> types.SimpleNamespace:
    """The inputs ``t`` (same values) in ``layout``."""
    return arrange(t.gate, t.up, t.dh, layout, beta=t.beta, lb=t.lb)


def _rows(t: types.SimpleNamespace, rows: int, offset: int = 0) -> types.SimpleNamespace:
    """Rows ``[offset, offset + rows)`` of the inputs ``t`` (views)."""
    span = slice(offset, offset + rows)
    out = types.SimpleNamespace(R=rows, I=t.I, beta=t.beta, lb=t.lb, layout=t.layout)
    out.gate, out.up, out.dh = t.gate[span], t.up[span], t.dh[span]
    if t.second is None:
        out.first, out.second = t.first[span], None
    else:
        out.first, out.second = out.gate, out.up
    return out


def _leaves(t: types.SimpleNamespace) -> tuple:
    """Fresh leaves ``(first, second)`` of the inputs ``t`` that require grad (None stays None)."""
    return tuple(None if x is None else x.detach().clone().requires_grad_(True) for x in (t.first, t.second))


def _autograd_step(t: types.SimpleNamespace, *, checkpoint: bool = False) -> dict:
    """``h`` and the gradients of the leaves of one autograd step on ``t`` (the forward under non-reentrant
    checkpointing when asked)."""
    leaves = _leaves(t)
    if checkpoint:
        h = torch.utils.checkpoint.checkpoint(situ_glu, *leaves, beta=t.beta, lb=t.lb, use_reentrant=False)
    else:
        h = situ_glu(*leaves, beta=t.beta, lb=t.lb)
    h.backward(t.dh)
    found = {"h": h.detach(), "d_first": leaves[0].grad}
    if leaves[1] is not None:
        found["d_up"] = leaves[1].grad
    return found


def _functional(t: types.SimpleNamespace) -> dict:
    """The steps' ``h`` and gradients on ``t``, keyed as :func:`_autograd_step` keys them."""
    h = forward(t.first, t.second, beta=t.beta, lb=t.lb)
    d_first, d_up = backward(t.dh, t.first, t.second, beta=t.beta, lb=t.lb)
    found = {"h": h, "d_first": d_first}
    if d_up is not None:
        found["d_up"] = d_up
    return found


def _large_inputs(rows: int, width: int, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """A merged ``[rows, 2 width]`` input and its ``dh``, drawn directly in BF16 (no FP32 copy of a large tensor)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    gate_up = torch.randn(rows, 2 * width, device=device, dtype=torch.bfloat16, generator=gen).mul_(2.0)
    dh = torch.randn(rows, width, device=device, dtype=torch.bfloat16, generator=gen)
    t = types.SimpleNamespace(R=rows, I=width, beta=fp32(BETA), lb=fp32(LB), layout="merged", gate_up=gate_up, dh=dh)
    t.gate, t.up = gate_up[:, :width], gate_up[:, width:]
    t.first, t.second = gate_up, None
    return t


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert dict(cake_jit.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    assert cake_jit.SEGMENT == 256
    assert cake_jit.PACKAGE == "cake_pplx._backends.cake_situ_glu_det_train"
    # One record per stage, keyed by its program name (its library); the host reaches it through PROGRAM_OF.
    assert sorted(record["stage"] for record in cake_jit.MODULES.values()) == sorted(STAGES)
    assert dict(cake_jit.PROGRAM_OF) == {record["stage"]: name for name, record in cake_jit.MODULES.items()}
    prefix = cake_jit.LIBRARY_PREFIX + "_"
    for name, record in cake_jit.MODULES.items():
        assert set(record) == {"arches", "arg_plan", "ffi_entry", "kernel_symbol", "launch", "library", "stage"}, name
        assert record["library"] == name
        assert re.fullmatch(re.escape(prefix) + "[0-9a-f]{20}", name), name
        assert tuple(record["arches"]) == ARCHES, (name, record["arches"])
        assert record["ffi_entry"] == "run", name
        assert record["kernel_symbol"] == "kernel_" + record["library"], name
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["dynamic_smem_bytes"] == 0, name
        assert tuple(launch["block"]) == (THREADS, 1, 1), name
    # The BF16 stride and alignment rule is one 16-byte load of 8 values.
    assert cake_jit.VECTOR_ELEMENTS == 8 and cake_jit.ALIGNMENT_BYTES == 16


def test_argument_plans_are_the_kernel_parameters() -> None:
    """Each record takes its kernel's parameters in order (tensors as buffers, scalars as parameters), then the
    grid; the host names every one of them."""
    for name, record in cake_jit.MODULES.items():
        assert pkg.parameters(name) == PARAMETERS[record["stage"]], name
        for kind, key in record["arg_plan"]:
            if kind == "grid":
                continue
            assert kind == ("buffer" if key in BUFFERS else "parameter"), (name, kind, key)
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == ["grid_x", "grid_y", "grid_z"]


def test_grid_is_one_cta_per_block() -> None:
    """One CTA per block of rows on the x axis; at the largest row count a CTA's step over the rows stays in 32
    bits."""
    assert cake_jit.ROWS_PER_CTA == {"fwd": 1, "bwd": 2}
    for stage in STAGES:
        per = cake_jit.ROWS_PER_CTA[stage]
        for rows in (1, 2, 3, 129, 4097, 16380, 262144, 1048576, cake_jit.MAX_ROWS):
            assert tuple(cake_jit.grid(stage, rows)) == (-(-rows // per), 1, 1), (stage, rows)
        assert cake_jit.MAX_ROWS - 1 + per * cake_jit.grid(stage, cake_jit.MAX_ROWS)[0] <= 2**31 - 1
        for rows in (0, -1, cake_jit.MAX_ROWS + 1):
            with pytest.raises(ValueError):
                cake_jit.grid(stage, rows)
    with pytest.raises(ValueError):
        cake_jit.grid("ckpt", 1)


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name, record in cake_jit.MODULES.items():
        for arch in ARCHES:
            assert pkg.library_name(name, arch) == f"{record['library']}_{arch}"
        for arch in ("sm_90a", "sm_107a", "sm_110a", "sm_120a"):
            with pytest.raises(RuntimeError):
                pkg.library_name(name, arch)


def test_capabilities_map_to_the_two_architectures(monkeypatch) -> None:
    """(10, 0) and (10, 3) map to ``sm_100a`` and ``sm_103a``; any other capability (SM107 included) is no target, and
    a device of one is a ``RuntimeError`` naming the delivered architectures."""
    assert tuple(pkg.ARCHES) == ARCHES
    assert tuple(pkg.SUPPORTED_COMPUTE_CAPABILITIES) == ((10, 0), (10, 3))
    assert pkg.WIDTH_MULTIPLE == 256
    for capability, arch in ARCH_BY_CAPABILITY.items():
        assert pkg.arch_for_capability(capability) == arch
    for capability in ((9, 0), (10, 1), (10, 7), (11, 0), (12, 0), (12, 1)):
        assert pkg.arch_for_capability(capability) is None, capability
    for capability, want in (((10, 3), "sm_103a"), ((10, 7), None)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        pkg._arch_of_index.cache_clear()
        try:
            if want is None:
                with pytest.raises(RuntimeError, match="sm_100a.*sm_103a"):
                    pkg.device_arch(0)
            else:
                assert pkg.device_arch(0) == want
        finally:
            monkeypatch.undo()
            pkg._arch_of_index.cache_clear()


def test_scalars_are_keywords_with_the_model_defaults() -> None:
    """``beta`` and ``lb`` are keyword-only everywhere, with the model's values 4.0 and 25.0 as defaults."""
    for entry in (forward, backward, situ_glu):
        for name, default in (("beta", 4.0), ("lb", 25.0)):
            parameter = inspect.signature(entry).parameters[name]
            assert parameter.kind is parameter.KEYWORD_ONLY, (entry.__name__, name)
            assert parameter.default == default, (entry.__name__, name)
    x = torch.zeros(2, 2 * 3072, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        forward(x, None, 4.0)


def test_cpu_tensors_are_refused() -> None:
    x = torch.zeros(2, 2 * 3072, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        forward(x)
    with pytest.raises(ValueError, match="CUDA"):
        backward(x[:, :3072], x)
    with pytest.raises(ValueError, match="CUDA"):
        situ_glu(x)


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{record['library']}_{arch}" for record in cake_jit.MODULES.values() for arch in record["arches"]}
    assert set(entries) == wanted
    assert len(entries) == len(cake_jit.MODULES) * len(ARCHES)
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert name.endswith("_" + entry["arch"]), name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_the_loader_imports_standalone_by_path() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a fresh process reads the manifest."""
    _manifest_or_skip()
    loader = str(PACKAGE_DIR / "cake_build.py")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {loader!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    records = sum(len(record["arches"]) for record in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


# The variables the JIT shim (``_cake_pplx_jit.py``) reads: the toolchain, cache and JIT switches.
JIT_SHIM_ENVIRONMENT_VARIABLES = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CAKE_PPLX_CACHE_DIR",
        "CAKE_PPLX_NVCC",
        "CXX",
        "CAKE_PPLX_JIT_WORKERS",
        "CAKE_PPLX_DISABLE_JIT",
    }
)


def _environment_reads(tree: ast.AST) -> tuple[set[str], list[int]]:
    """``(names, unresolved)``: the variable names ``tree`` reads through ``os.environ.get(...)``,
    ``os.environ[...]``, ``... in os.environ`` or ``os.getenv(...)`` -- a string literal, or a loop variable over a
    tuple / list of string literals -- and the lines of every other ``environ`` / ``getenv`` use."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def keys(expr: ast.AST, at: ast.AST) -> set[str] | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.Name):
            node = parents.get(at)
            while node is not None:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == expr.id
                    and isinstance(node.iter, (ast.Tuple, ast.List))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)
                ):
                    return {e.value for e in node.iter.elts}
                node = parents.get(node)
        return None

    names: set[str] = set()
    unresolved: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")):
            continue
        parent = parents.get(node)
        found = None
        if node.attr == "getenv":
            if isinstance(parent, ast.Call) and parent.func is node and parent.args:
                found = keys(parent.args[0], parent)
        elif isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and call.args:
                found = keys(call.args[0], call)
        elif isinstance(parent, ast.Subscript) and parent.value is node:
            found = keys(parent.slice, parent)
        elif (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is node
        ):
            found = keys(parent.left, parent)
        if found is None:
            unresolved.append(node.lineno)
        else:
            names |= found
    return names, unresolved


def test_the_package_reads_no_environment_variable() -> None:
    """No module of the package reads an environment variable, except the JIT shim (``_cake_pplx_jit.py``), which
    reads exactly :data:`JIT_SHIM_ENVIRONMENT_VARIABLES`, each by a literal name."""
    offenders = []
    shim = None
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        names, unresolved = _environment_reads(ast.parse(path.read_text()))
        if path.name == "_cake_pplx_jit.py":
            shim = names
            offenders += [f"{path.name}:{line} (variable not resolved)" for line in unresolved]
        else:
            offenders += [f"{path.name}:{line}" for line in unresolved]
            offenders += [f"{path.name}: {name}" for name in sorted(names)]
    assert not offenders, offenders
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES, (
        sorted(shim - JIT_SHIM_ENVIRONMENT_VARIABLES) if shim is not None else "no _cake_pplx_jit.py",
        sorted(JIT_SHIM_ENVIRONMENT_VARIABLES - shim) if shim is not None else None,
    )


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103: accuracy, judged against the naive BF16 tier


def test_the_records_load(device) -> None:
    arch = pkg.device_arch(device.index)
    assert arch in ARCHES
    for name, record in cake_jit.MODULES.items():
        assert callable(getattr(pkg.load(name, arch), record["ffi_entry"])), name


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_outputs_meet_the_naive_tier(case, device, record_property) -> None:
    """``h``, ``d_gate`` and ``d_up`` hold their contract (the oracle's shape, finite where it is finite: every input
    here is finite) and meet the naive BF16 tier over 32 draws."""
    judge(record_property, case, forward, backward, device=device)


# ----------------------------------------------------------------------------------------------------------------
# Exact facts of the contract


@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize("width", [256, 3072, 33792])
def test_a_zero_gate_or_up_gives_exact_zeros(layout, width, device) -> None:
    """Where ``g = 0`` or ``u = 0`` (a random third of the gates and of the ups), ``h`` is exactly 0; ``d_up`` is
    exactly 0 where ``g = 0`` (``a = 0``) and ``d_gate`` where ``u = 0`` (``u' = 0``), also with other scalars.
    Elsewhere the outputs are not all 0."""
    for beta, lb, seed in ((BETA, LB, 1), (2.5, 10.0, 2)):
        gen = torch.Generator(device=device).manual_seed(seed)
        rows = 33
        g = 2.0 * torch.randn(rows, width, device=device, generator=gen)
        u = 2.0 * torch.randn(rows, width, device=device, generator=gen)
        g = torch.where(torch.rand(rows, width, device=device, generator=gen) < 1 / 3, 0.0, g).bfloat16()
        u = torch.where(torch.rand(rows, width, device=device, generator=gen) < 1 / 3, 0.0, u).bfloat16()
        dh = torch.randn(rows, width, device=device, generator=gen).bfloat16()
        t = arrange(g, u, dh, layout, beta=fp32(beta), lb=fp32(lb))
        found = step_outputs(forward, backward, t)
        zero_g, zero_u = g == 0, u == 0
        assert bool(zero_g.any() and zero_u.any())
        assert bool((found["h"][zero_g | zero_u] == 0).all()), "h where g = 0 or u = 0"
        assert bool((found["d_up"][zero_g] == 0).all()), "d_up where g = 0"
        assert bool((found["d_gate"][zero_u] == 0).all()), "d_gate where u = 0"
        live = ~(zero_g | zero_u)
        for name in ("h", "d_gate", "d_up"):
            assert bool((found[name][live] != 0).any()), name


@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize(("beta", "lb"), [(BETA, LB), (3.0, 40.0)], ids=["model", "other"])
def test_saturated_inputs_are_finite_with_the_oracle_signs(layout, beta, lb, device) -> None:
    """Gates uniform in ``[-200, 200]`` and ups in ``[-1000, 1000]``, with the probes ``g`` in
    ``{-200, -20, 0, 20, 200}`` and ``u`` in ``{-1000, -50, 0, 50, 1000}`` crossed in the first row: every output is
    finite, and has the sign of the float64 oracle wherever the oracle is at least the smallest normal number (for
    ``d_gate``, where ``|g| >= 20``: there the gate derivative has no zero, so its sign is the operation's, not a
    rounding's).  Past ``|g| = 43 beta`` (``d_gate``) and ``|u| = 43 lb`` (``d_up``) the kernels flush the
    exponential, so a gradient under ``2^-124 |dh| lb`` or ``2^-124 |dh| beta`` is stored as 0: those elements are
    only checked to be finite."""
    gen = torch.Generator(device=device).manual_seed(5)
    rows, width = 129, 3072
    g = -200.0 + 400.0 * torch.rand(rows, width, device=device, generator=gen)
    u = -1000.0 + 2000.0 * torch.rand(rows, width, device=device, generator=gen)
    probes_g = torch.tensor([-200.0, -20.0, 0.0, 20.0, 200.0], device=device)
    probes_u = torch.tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0], device=device)
    g[0, :25] = probes_g.repeat_interleave(5)
    u[0, :25] = probes_u.repeat(5)
    dh = torch.randn(rows, width, device=device, generator=gen).bfloat16()
    t = arrange(g.bfloat16(), u.bfloat16(), dh, layout, beta=fp32(beta), lb=fp32(lb))
    found = step_outputs(forward, backward, t)
    want = oracle(t)
    gate = t.gate.float()
    for name in ("h", "d_gate", "d_up"):
        value = found[name]
        assert bool(value.isfinite().all()), name
        signed = want[name].abs() >= MIN_NORMAL
        if name == "d_gate":
            signed &= (gate.abs() >= 20.0) & (gate.abs() <= 43.0 * beta)
        if name == "d_up":
            signed &= t.up.float().abs() <= 43.0 * lb
        assert bool(signed.any()), name
        assert torch.equal(torch.sign(value.double())[signed], torch.sign(want[name])[signed]), name


@pytest.mark.parametrize("layout", ["split", "merged", "sliced"])
def test_no_rows_return_empty_outputs_and_launch_nothing(layout, device, monkeypatch) -> None:
    """``R = 0``: the steps and the autograd function return empty outputs of the right shapes and dtypes, and
    nothing is launched or loaded."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call with no rows launched or loaded a kernel")

    monkeypatch.setattr(pkg, "run_module", refuse)
    monkeypatch.setattr(pkg, "load", refuse)
    monkeypatch.setattr(pkg, "_entry", refuse)
    t = make_inputs(Case(0, 3072, layout), seed=1, device=device)
    h = forward(t.first, t.second)
    assert h.shape == (0, 3072) and h.dtype == torch.bfloat16 and h.device == device
    d_first, d_up = backward(t.dh, t.first, t.second)
    if layout == "merged":
        assert d_first.shape == (0, 2 * 3072) and d_up is None
    else:
        assert d_first.shape == (0, 3072) and d_up.shape == (0, 3072)
    assert d_first.dtype == torch.bfloat16
    leaves = _leaves(t)
    out = situ_glu(*leaves)
    assert out.shape == (0, 3072)
    out.sum().backward()
    for leaf in leaves:
        if leaf is not None:
            assert leaf.grad.shape == leaf.shape


def test_one_library_per_stage_serves_every_call(device, monkeypatch) -> None:
    """Row counts 1 to 4097, widths 256 to 33792, every layout and other scalars load one library per stage, once: no
    recompilation (``rows``, ``width``, ``beta``, ``lb`` and the strides are launch arguments)."""
    cake_build = cake_package("cake_situ_glu_det_train", "cake_build")
    loaded: list[str] = []
    load_library = cake_build.load_library

    def spy(name):
        loaded.append(name)
        return load_library(name)

    monkeypatch.setattr(cake_build, "load_library", spy)
    pkg.load.cache_clear()
    pkg._entry.cache_clear()
    cases = (
        Case(1),
        Case(4097, 3072, "merged"),
        Case(3, 256, "sliced", beta=2.5, lb=10.0),
        Case(129, 6144, "copied"),
        Case(2, 33792, "merged", beta=3.0),
        Case(16380, 6144),
    )
    try:
        for case in cases:
            step_outputs(forward, backward, make_inputs(case, seed=5, device=device))
    finally:
        pkg.load.cache_clear()
        pkg._entry.cache_clear()
    arch = pkg.device_arch(device.index)
    assert sorted(loaded) == sorted(pkg.library_name(cake_jit.PROGRAM_OF[stage], arch) for stage in STAGES)


@pytest.mark.parametrize(
    "case",
    [Case(1), Case(129, 3072, "merged", "saturated"), Case(257, 33792, "sliced"), Case(4097, 6144)],
    ids=lambda case: case.id,
)
def test_outputs_are_the_same_bits_at_every_call(case, device) -> None:
    t = make_inputs(case, seed=21, device=device)
    want = step_outputs(forward, backward, t)
    for k in range(4):
        assert_same_outputs(step_outputs(forward, backward, t), want, f"call {k + 1}")


@pytest.mark.parametrize("width", [256, 3072, 33792])
def test_layouts_give_the_same_bits(width, device) -> None:
    """The same values as two contiguous tensors, one merged tensor, row slices of wider buffers (read in place) and
    rows the host copies give the same bits."""
    t = make_inputs(Case(65, width, "split", "wide"), seed=4, device=device)
    want = step_outputs(forward, backward, t)
    for layout in ("merged", "sliced", "copied"):
        assert_same_outputs(step_outputs(forward, backward, _relayout(t, layout)), want, layout)


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_rows_do_not_depend_on_the_row_count(layout, device) -> None:
    """A row's outputs are the same bits whatever the other rows and the row count (each row is computed alone)."""
    t = make_inputs(Case(4097, 3072, layout, "wide"), seed=9, device=device)
    full = step_outputs(forward, backward, t)
    for rows in (1, 7, 129, 1000):
        part = step_outputs(forward, backward, _rows(t, rows))
        for name, value in part.items():
            assert_same_bits(value, full[name][:rows], f"{name} of the first {rows} rows")


@pytest.mark.parametrize(
    ("rows", "width"),
    [(262144, 3072), (1048576, 3072), (65536, 33792)],
    ids=["R262144-I3072", "R1048576-I3072", "R65536-I33792"],
)
def test_large_row_counts(rows, width, device) -> None:
    """The issue's routed call (262,144 rows), 1,048,576 rows, and 65,536 rows of the dense width, merged: byte
    offsets past 2**31 in every case, element offsets past 2**32 in the last two.  Finite outputs, the same bits at a
    second call, and the first and the last 4097 rows the bits of a call on those 4097 rows (views of the inputs, the
    last ones at a large row offset)."""
    t = _large_inputs(rows, width, seed=31, device=device)
    want = step_outputs(forward, backward, t)
    for name, value in want.items():
        assert bool(value.isfinite().all()), name
    again = step_outputs(forward, backward, t)
    assert_same_outputs(again, want, "second call")
    del again
    for offset in (0, rows - 4097):
        part = step_outputs(forward, backward, _rows(t, 4097, offset=offset))
        for name, value in part.items():
            assert_same_bits(value, want[name][offset : offset + 4097], f"{name} of rows {offset} to {offset + 4096}")


def test_beta_and_lb_are_rounded_to_fp32_once(device) -> None:
    """An integer scalar gives the bits of the same float; a scalar between two FP32 values gives the bits of its FP32
    rounding."""
    t = make_inputs(Case(33, 3072, "merged"), seed=2, device=device)
    want = forward(t.first, None, beta=4.0, lb=25.0)
    assert_same_bits(forward(t.first, None, beta=4, lb=25), want, "integer scalars")
    near = 3.0 + 2.0**-30  # rounds to 3.0 in FP32
    assert_same_bits(forward(t.first, None, beta=near, lb=near), forward(t.first, None, beta=3.0, lb=3.0), "rounding")


# ----------------------------------------------------------------------------------------------------------------
# Autograd, streams, graphs and the determinism setting


@pytest.mark.parametrize("layout", ["merged", "split", "sliced"])
def test_autograd_matches_the_steps_and_checkpoint(layout, device) -> None:
    """The autograd function gives the steps' bits (a merged input's gradient is one ``[R, 2I]`` tensor), saves only
    its inputs (no activation, no FP32 intermediate), and gives the same bits under non-reentrant checkpointing."""
    t = make_inputs(Case(513, 3072, layout), seed=8, device=device)
    want = _functional(t)
    if layout == "merged":
        assert want["d_first"].shape == (t.R, 2 * t.I)
    assert_same_outputs(_autograd_step(t), want, "autograd")
    assert_same_outputs(_autograd_step(t, checkpoint=True), want, "checkpoint")

    saved = []

    def pack(x):
        saved.append((str(x.dtype), x.numel(), x.data_ptr()))
        return x

    leaves = _leaves(t)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        situ_glu(*leaves, beta=t.beta, lb=t.lb)
    expected = [(str(x.dtype), x.numel(), x.data_ptr()) for x in leaves if x is not None]
    assert sorted(saved) == sorted(expected)


def test_autograd_returns_the_gradients_that_are_needed(device) -> None:
    """Only the inputs that require grad get one; ``beta`` and ``lb`` get none; an expanded upstream gradient
    (``h.sum()``) gives the bits of the steps with a gradient of ones."""
    t = make_inputs(Case(129, 3072, "split"), seed=12, device=device)
    want = _functional(t)
    gate = t.gate.detach().clone().requires_grad_(True)
    up = t.up.detach().clone()
    situ_glu(gate, up).backward(t.dh)
    assert_same_bits(gate.grad, want["d_first"], "d_gate alone")
    assert up.grad is None

    m = make_inputs(Case(129, 6144, "merged"), seed=13, device=device)
    leaves = _leaves(m)
    situ_glu(*leaves).sum().backward()
    ones = torch.ones(m.R, m.I, dtype=torch.bfloat16, device=device)
    d_first, _ = backward(ones, m.first, None)
    assert_same_bits(leaves[0].grad, d_first, "merged gradient of h.sum()")


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_double_backward_raises(layout, device) -> None:
    """Gradients computed with ``create_graph=True`` and an upstream gradient that requires grad have the steps' bits,
    and differentiating them raises ``RuntimeError``."""
    t = make_inputs(Case(33, 3072, layout), seed=3, device=device)
    want = _functional(t)
    leaves = _leaves(t)
    dh = t.dh.detach().clone().requires_grad_(True)
    wants = [leaf for leaf in leaves if leaf is not None]
    grads = torch.autograd.grad(situ_glu(*leaves), wants, dh, create_graph=True)
    names = ["d_first"] if t.second is None else ["d_first", "d_up"]
    for name, grad in zip(names, grads, strict=True):
        assert_same_bits(grad.detach(), want[name], f"{name} with create_graph=True")
    with pytest.raises(RuntimeError):
        grads[0].float().sum().backward()


def test_side_stream_and_no_host_sync(device) -> None:
    """On a side stream under ``torch.cuda.set_sync_debug_mode("error")``: the steps and an autograd step run without
    synchronizing and give the bits of the default stream."""
    t = make_inputs(Case(2049, 3072, "merged"), seed=9, device=device)
    want = step_outputs(forward, backward, t)
    want_autograd = _autograd_step(t)
    leaves = _leaves(t)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            got = step_outputs(forward, backward, t)
            h = situ_glu(*leaves)
            h.backward(t.dh)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    assert_same_outputs(got, want, "side stream")
    assert_same_bits(h.detach(), want_autograd["h"], "autograd h on a side stream")
    assert_same_bits(leaves[0].grad, want_autograd["d_first"], "autograd gradient on a side stream")


def test_cuda_graph_capture_and_replay(device) -> None:
    """A merged step at the model's scalars and a two-tensor step at another width and other scalars, with
    preallocated outputs, captured once after a warm-up call and replayed three times on new inputs copied into the
    captured tensors: each replay gives the bits of eager calls on those inputs."""
    R = 300
    a_case = Case(R, 3072, "merged")
    b_case = Case(R, 6144, "split", beta=2.5, lb=10.0)
    a = make_inputs(a_case, seed=40, device=device)
    b = make_inputs(b_case, seed=41, device=device)
    gu, dh_a = a.gate_up.clone(), a.dh.clone()
    gate, up, dh_b = b.gate.clone(), b.up.clone(), b.dh.clone()
    h_a = torch.empty(R, 3072, dtype=torch.bfloat16, device=device)
    d_gu = torch.empty(R, 2 * 3072, dtype=torch.bfloat16, device=device)
    h_b = torch.empty(R, 6144, dtype=torch.bfloat16, device=device)
    d_gate, d_up = torch.empty_like(h_b), torch.empty_like(h_b)

    def step():
        forward(gu, None, out=h_a)
        backward(dh_a, gu, None, grad_gate_up_or_gate=d_gu)
        forward(gate, up, beta=b.beta, lb=b.lb, out=h_b)
        backward(dh_b, gate, up, beta=b.beta, lb=b.lb, grad_gate_up_or_gate=d_gate, grad_up=d_up)

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        step()  # loads the libraries before the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for k in range(3):
        na = make_inputs(a_case, seed=50 + k, device=device)
        nb = make_inputs(b_case, seed=60 + k, device=device)
        gu.copy_(na.gate_up)
        dh_a.copy_(na.dh)
        gate.copy_(nb.gate)
        up.copy_(nb.up)
        dh_b.copy_(nb.dh)
        for buffer in (h_a, d_gu, h_b, d_gate, d_up):
            buffer.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        want_a = _functional(na)
        want_b = _functional(nb)
        assert_same_bits(h_a, want_a["h"], f"replay {k}: h (merged)")
        assert_same_bits(d_gu, want_a["d_first"], f"replay {k}: d_gate_up")
        assert_same_bits(h_b, want_b["h"], f"replay {k}: h (two tensors)")
        assert_same_bits(d_gate, want_b["d_first"], f"replay {k}: d_gate")
        assert_same_bits(d_up, want_b["d_up"], f"replay {k}: d_up")


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_steps_neither_raise_nor_warn_under_deterministic_algorithms(warn_only, device) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) with warnings as errors, the
    steps, an autograd step and a non-reentrant checkpointed step raise nothing, warn nothing and give the bits of the
    same calls with the setting off."""
    t = make_inputs(Case(1031, 3072, "merged"), seed=35, device=device)
    u = make_inputs(Case(77, 6144, "split"), seed=36, device=device)

    def run():
        return [
            step_outputs(forward, backward, t),
            step_outputs(forward, backward, u),
            _autograd_step(t),
            _autograd_step(t, checkpoint=True),
        ]

    want = run()
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = run()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_outputs(a, b, f"warn_only={warn_only} call {i}")


# ----------------------------------------------------------------------------------------------------------------
# Layouts, buffers and refusals


def test_fit_inputs_are_read_in_place_and_others_copied_once(device, launches) -> None:
    """The halves of a merged input and row slices of wider buffers reach the kernels as views of the caller's memory
    (their row strides), with no copy; rows the kernels cannot read are copied once to contiguous rows; a single row's
    stride is passed as 0; the width and the FP32 scalars are launch arguments."""
    m = make_inputs(Case(33, 3072, "merged"), seed=2, device=device)
    forward(m.first, None)
    stage, rows, values = launches[-1]
    assert stage == "fwd" and rows == 33
    assert values["gate"].data_ptr() == m.gate_up.data_ptr()
    assert values["up"].data_ptr() == m.gate_up.data_ptr() + 2 * 3072
    assert values["gate_stride"] == values["up_stride"] == 2 * 3072
    assert values["width"] == 3072 and values["h_stride"] == 3072
    assert values["beta"] == 4.0 and values["lb"] == 25.0
    backward(m.dh, m.first, None)
    stage, _, values = launches[-1]
    assert stage == "bwd"
    assert values["gate"].data_ptr() == m.gate_up.data_ptr()
    assert values["dh"].data_ptr() == m.dh.data_ptr()
    assert values["d_up"].data_ptr() == values["d_gate"].data_ptr() + 2 * 3072
    assert values["d_gate_stride"] == values["d_up_stride"] == 2 * 3072

    s = make_inputs(Case(33, 6144, "sliced"), seed=2, device=device)
    backward(s.dh, s.gate, s.up, beta=2.5, lb=10.0)
    _, _, values = launches[-1]
    for name, tensor in (("gate", s.gate), ("up", s.up), ("dh", s.dh)):
        assert values[name].data_ptr() == tensor.data_ptr(), name
        assert values[f"{name}_stride"] == tensor.stride(0) != 6144, name
    assert values["width"] == 6144 and values["beta"] == 2.5 and values["lb"] == 10.0

    c = make_inputs(Case(33, 3072, "copied"), seed=2, device=device)
    backward(c.dh, c.gate, c.up, beta=7.3)
    _, _, values = launches[-1]
    for name, tensor in (("gate", c.gate), ("up", c.up), ("dh", c.dh)):
        assert values[name].data_ptr() != tensor.data_ptr(), name
        assert values[name].is_contiguous() and values[f"{name}_stride"] == 3072, name
    assert values["beta"] == fp32(7.3)

    one = make_inputs(Case(1, 3072, "merged"), seed=2, device=device)
    backward(one.dh, one.first, None)
    _, _, values = launches[-1]
    for name in ("gate", "up", "dh", "d_gate", "d_up"):
        assert values[f"{name}_stride"] == 0, name


def test_preallocated_outputs(device) -> None:
    """Preallocated outputs, contiguous or row slices of wider aligned buffers, receive the bits of new outputs and are
    returned; outputs the kernels cannot write in place are refused, never copied."""
    width = 3072
    t = make_inputs(Case(65, width, "merged"), seed=14, device=device)
    want = _functional(t)

    def wide(rows, cols, pad=8):
        return torch.empty(rows, cols + 2 * pad, dtype=torch.bfloat16, device=device)[:, pad : pad + cols]

    def new(rows, cols):
        return torch.empty(rows, cols, dtype=torch.bfloat16, device=device)

    for h, d_gu in ((new(t.R, width), new(t.R, 2 * width)), (wide(t.R, width), wide(t.R, 2 * width))):
        assert forward(t.first, None, out=h) is h
        result = backward(t.dh, t.first, None, grad_gate_up_or_gate=d_gu)
        assert result[0] is d_gu and result[1] is None
        assert_same_bits(h, want["h"], "out")
        assert_same_bits(d_gu, want["d_first"], "grad_gate_up_or_gate")

    s = make_inputs(Case(65, width, "split"), seed=15, device=device)
    want_s = _functional(s)
    d_gate, d_up = wide(s.R, width), wide(s.R, width)
    result = backward(s.dh, s.gate, s.up, grad_gate_up_or_gate=d_gate, grad_up=d_up)
    assert result[0] is d_gate and result[1] is d_up
    assert_same_bits(d_gate, want_s["d_first"], "grad_gate")
    assert_same_bits(d_up, want_s["d_up"], "grad_up")

    raw = torch.empty(t.R * 2 * width + 64, dtype=torch.bfloat16, device=device)
    unfit = {
        "an FP32 output": torch.empty(t.R, width, device=device),
        "another shape": torch.empty(t.R + 1, width, dtype=torch.bfloat16, device=device),
        "a base that is not 16-byte aligned": raw[1 : 1 + t.R * width].view(t.R, width),
        "a row stride that is not a multiple of 8": raw[: t.R * (width + 2)].view(t.R, width + 2)[:, :width],
        "overlapping rows": raw.as_strided((t.R, width), (8, 1)),
        "a non-unit last stride": raw[: t.R * width * 2].view(t.R, width, 2)[:, :, 0],
    }
    for what, out in unfit.items():
        with pytest.raises(ValueError):
            forward(t.first, None, out=out)
        with pytest.raises(ValueError):
            backward(s.dh, s.gate, s.up, grad_gate_up_or_gate=out)
        with pytest.raises(ValueError):
            backward(s.dh, s.gate, s.up, grad_up=out)
    with pytest.raises(ValueError, match="grad_up"):
        backward(t.dh, t.first, None, grad_up=torch.empty_like(t.dh))


def test_operand_and_scalar_errors(device, launches) -> None:
    """Wrong types, dtypes, shapes, widths and devices, and scalars that are not finite positive Python numbers or whose
    kernel constants are past FP32, are refused before any launch."""
    t = make_inputs(Case(8, 3072, "split"), seed=1, device=device)
    gate, up, dh = t.gate, t.up, t.dh
    merged = torch.cat([gate, up], dim=1)
    value_errors = {
        "an FP16 gate": lambda: forward(gate.half(), up),
        "an FP32 up": lambda: forward(gate, up.float()),
        "a width that is not a multiple of 256": lambda: forward(gate[:, :3000], up[:, :3000]),
        "a width below 256": lambda: forward(gate[:, :128], up[:, :128]),
        "a merged input of an odd number of columns": lambda: forward(merged[:, :6143]),
        "a merged input whose half is not a multiple of 256": lambda: forward(merged[:, :6000]),
        "an up of another width": lambda: forward(gate, up[:, :2816]),
        "another row count of up": lambda: forward(gate, up[:7]),
        "a 3-D gate": lambda: forward(gate[None], up[None]),
        "a 1-D merged input": lambda: forward(merged[0]),
        "up on the CPU": lambda: forward(gate, up.cpu()),
        "dh of another shape": lambda: backward(dh[:, :256], gate, up),
        "an FP32 dh": lambda: backward(dh.float(), gate, up),
        "dh on the CPU": lambda: backward(dh.cpu(), gate, up),
        "a NaN beta": lambda: forward(gate, up, beta=float("nan")),
        "an infinite lb": lambda: forward(gate, up, lb=float("inf")),
        "a zero beta": lambda: forward(gate, up, beta=0.0),
        "a negative lb": lambda: forward(gate, up, lb=-1.0),
        "a beta past FP32": lambda: forward(gate, up, beta=1e39),
        "an lb that rounds to 0 in FP32": lambda: forward(gate, up, lb=1e-50),
        "a beta whose 2.88539 / beta is past FP32": lambda: forward(gate, up, beta=1e-39),
        "an lb whose 2.88539 / lb is past FP32 in the backward": lambda: backward(dh, gate, up, lb=1e-39),
        "a beta lb product past FP32 in the autograd entry": lambda: situ_glu(gate, up, beta=1e20, lb=1e20),
        "a negative beta in the backward": lambda: backward(dh, gate, up, beta=-4.0),
        "a negative lb in the autograd entry": lambda: situ_glu(gate, up, lb=-25.0),
    }
    type_errors = {
        "a tensor beta": lambda: forward(gate, up, beta=torch.tensor(4.0, device=device)),
        "a boolean lb": lambda: forward(gate, up, lb=True),
        "a string beta": lambda: forward(gate, up, beta="4"),
        "a list gate": lambda: forward([1.0], up),
        "a tensor lb in the autograd entry": lambda: situ_glu(gate, up, lb=torch.tensor(25.0)),
        "a tensor beta in the backward": lambda: backward(dh, gate, up, beta=torch.tensor(4.0)),
    }
    misses = []
    for kind, calls in ((ValueError, value_errors), (TypeError, type_errors)):
        for what, call in calls.items():
            try:
                call()
            except kind:
                continue
            except Exception as e:
                misses.append(f"{what}: {type(e).__name__} ({e}), not {kind.__name__}")
            else:
                misses.append(f"{what}: no {kind.__name__}")
    assert not misses, misses
    assert launches == []


def test_unsupported_devices_raise(device, monkeypatch) -> None:
    """A device of another architecture is a ``RuntimeError`` naming the delivered architectures, also with no rows,
    and is reported unsupported."""
    t = make_inputs(Case(5, 3072, "merged"), seed=1, device=device)
    monkeypatch.setattr(pkg, "_arch_of_index", lambda index: None)
    assert not pkg.is_cake_situ_glu_det_train_supported(device)
    for rows in (0, 5):
        with pytest.raises(RuntimeError, match="sm_100a"):
            forward(t.first[:rows], None)
        with pytest.raises(RuntimeError, match="sm_100a"):
            backward(t.dh[:rows], t.first[:rows], None)


def test_operands_on_another_device(device) -> None:
    """Operands on a second GPU run there and give the bits of the first; mixed devices are refused."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", 1 if device.index != 1 else 0)
    if not pkg.is_cake_situ_glu_det_train_supported(other):
        pytest.skip(f"{other} is not a supported device")
    t = make_inputs(Case(129, 3072, "merged"), seed=6, device=device)
    want = step_outputs(forward, backward, t)
    o = make_inputs(Case(129, 3072, "merged"), seed=6, device=other)
    got = step_outputs(forward, backward, o)
    for name, value in got.items():
        assert value.device == other, name
        assert_same_bits(value.to(device), want[name], f"{name} on {other}")
    with pytest.raises(ValueError):
        backward(t.dh, o.first, None)
