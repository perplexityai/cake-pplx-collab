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

Backend suite of the Cake non-deterministic (``nondet``) dense multi-head attention backward package
(``cake_pplx._backends.cake_fmha_nondet_train``, the backend of ``cake_pplx.cake_fmha_nondet_train``): head dimension
64 or 128, any head count, query and key lengths multiples of 128, causal (top-left, ``Sq == Skv``) or not, on SM100,
SM103 and SM107.  The package promises no bits: the main stage adds the FP32 dQ tiles of every key block of a head in
arrival order, so no test compares two runs bit for bit.

Without a device: the registry (the three stages in launch order, one program per record, every record delivered for
the three architectures and for no other, the launch facts), the routing table ``RECORD_OF`` (the preprocess,
main-stage and postprocess record of each head dimension and causal flag, every record used), the launch grids, the
stage launchers' keywords, their positional launch of every routed record (the FFI arguments and grid of the
registry-ordered launch), the build manifest (every record's library, the recorded source digests), the standalone
loader, the prebuilt libraries served first, and no environment read outside the JIT shim.

On SM100 / SM103 / SM107, each stage judged on its own against the naive BF16 tier (``tests/cake_naive.py``: the error
against an FP64 oracle of the same inputs over 32 independent draws, in the L2 and the max norm, median and largest,
beside the error of the same math in BF16), every output buffer filled with NaN before the launch so every element
must be written: the preprocess (``dPsum``, ``LSE_log2``, the cleared ``dQ_accum``) at both head dimensions; the main
stage on its own inputs (the FP32 ``LSE_log2`` and ``dPsum``, a zeroed ``dQ_accum``) for every variant, judged on
``dQ_accum``, ``dK`` and ``dV``; the postprocess (``dq`` in the input layout) at both head dimensions.  The public
backward, which composes the three stages, judged for every variant at four heads (the source tests' cases), one head
and five heads; each variant launched through the three records ``RECORD_OF`` names for it, its inputs left unchanged.

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_fmha_naive import (
    Case,
    assert_backward_contract,
    judge_draws,
    judged_backward,
    judged_stage,
    main_buffers,
    main_outputs,
    make_inputs,
    post_inputs,
    post_outputs,
    pre_inputs,
    pre_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake.cake_fmha_nondet_train")
cake_backend = cake_package("cake.cake_fmha_nondet_train", "cake_backend")
cake_build = cake_package("cake.cake_fmha_nondet_train", "cake_build")
cake_jit = cake_package("cake.cake_fmha_nondet_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STAGES = ("bwd_pre", "bwd_main", "bwd_post")
HEAD_DIMS = (128, 64)
# (head dimension, causal) of every variant; each is delivered for the three architectures.
VARIANTS = ((128, False), (128, True), (64, False), (64, True))
VARIANT_IDS = ["d128", "d128_causal", "d64", "d64_causal"]
# Threads per CTA of each stage's kernel (8, 16 and 4 warps).
BLOCK_THREADS = {"bwd_pre": 256, "bwd_main": 512, "bwd_post": 128}
ENTRY = "cake_fmha_nondet_train_backward"
# The keywords of each stage launcher, after the record name.
LAUNCHER_KEYWORDS = {
    "launch_bwd_pre": ("out", "dout", "lse", "lse_log2", "dpsum", "dq_accum"),
    "launch_bwd_main": ("q", "k", "v", "dout", "lse_log2", "dpsum", "dq_accum", "dk", "dv", "softmax_scale"),
    "launch_bwd_post": ("dq_accum", "dq", "softmax_scale"),
}


def main_cases(causal: bool) -> list[Case]:
    """The main stage's and the backward's cases of one causal flag: the source tests' four cases at four heads (the
    uneven one has more keys than queries without the causal mask, a longer sequence with it), then one head and an
    odd head count."""
    uneven = Case(1, 4, 640) if causal else Case(1, 4, 384, 640)
    odd = Case(2, 5, 640) if causal else Case(2, 5, 384, 640)
    return [Case(1, 4, 128), Case(2, 4, 256), uneven, Case(3, 4, 1024), Case(3, 1, 256), odd]


MAIN_PARAMS = [
    pytest.param(D, causal, case, id=f"{variant}-{case.id}")
    for (D, causal), variant in zip(VARIANTS, VARIANT_IDS, strict=True)
    for case in main_cases(causal)
]
# The source tests' preprocess cases at four heads, then one head and an odd head count.
PRE_CASES = (Case(1, 4, 128), Case(3, 4, 384), Case(2, 4, 1024), Case(2, 1, 256), Case(3, 5, 384))
# The source tests' postprocess cases (one, five and four heads).
POST_CASES = (Case(1, 1, 128), Case(3, 5, 256), Case(2, 4, 1024))


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the package ships {ARCHES}; this device is {capability}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert sorted(cake_jit.HEAD_DIMS) == sorted(HEAD_DIMS)
    assert cake_jit.BLOCK == 128
    assert cake_jit.LIBRARY_PREFIX == "cake_fmha_train"
    prefix = cake_jit.LIBRARY_PREFIX + "_"
    libraries = set()
    for name, rec in cake_jit.MODULES.items():
        assert rec["stage"] in STAGES, name
        assert name == rec["stage"] or name.startswith(rec["stage"] + "_"), name
        # Every variant is delivered for the three architectures.
        assert tuple(rec["arches"]) == ARCHES, (name, rec["arches"])
        assert rec["ffi_entry"] == "run", name
        assert re.fullmatch(re.escape(prefix) + "[0-9a-f]{20}", rec["library"]), (name, rec["library"])
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert tuple(launch["block"]) == (BLOCK_THREADS[rec["stage"]], 1, 1), name
        libraries.add(rec["library"])
    assert len(libraries) == len(cake_jit.MODULES)


def test_routing_table_names_three_records_per_variant() -> None:
    """``RECORD_OF[(head_dim, causal)]`` is the preprocess, main-stage and postprocess record of each variant: the
    preprocess and postprocess depend on the head dimension only, each variant has its own main stage, and every
    record of the registry serves some variant."""
    table = cake_jit.RECORD_OF
    assert set(table) == set(VARIANTS)
    used = set()
    for variant, names in table.items():
        assert tuple(cake_jit.MODULES[name]["stage"] for name in names) == STAGES, (variant, names)
        used.update(names)
    for D in HEAD_DIMS:
        assert table[(D, False)][0] == table[(D, True)][0], D
        assert table[(D, False)][2] == table[(D, True)][2], D
    assert table[(64, False)][0] != table[(128, False)][0]
    assert table[(64, False)][2] != table[(128, False)][2]
    assert len({table[variant][1] for variant in VARIANTS}) == len(VARIANTS)
    assert used == set(cake_jit.MODULES)


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name, rec in cake_jit.MODULES.items():
        for arch in ARCHES:
            assert cake_backend.library_name(name, arch) == f"{rec['library']}_{arch}"
        for arch in ("sm_110a", "sm_90a", "sm_120a"):
            with pytest.raises(RuntimeError):
                cake_backend.library_name(name, arch)


def test_grids_cover_every_tile() -> None:
    """A CTA per 128-row query tile of each (head, batch) for the preprocess and the postprocess, per 128-key block of
    each (head, batch) for the main stage: the grid is ``(tiles, H, B)`` (the kernels read the head from
    ``blockIdx.y`` and the batch from ``blockIdx.z``)."""
    rng = random.Random(7091)
    for _ in range(200):
        B, H = rng.randint(1, 64), rng.randint(1, 64)
        Sq, Skv = 128 * rng.randint(1, 96), 128 * rng.randint(1, 96)
        dims = dict(batch=B, heads=H, seqlen_q=Sq, seqlen_kv=Skv)
        assert tuple(cake_jit.grid("bwd_pre", **dims)) == (Sq // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_main", **dims)) == (Skv // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_post", **dims)) == (Sq // 128, H, B), dims


def test_stage_launchers_take_named_buffers() -> None:
    """Each stage launcher takes the record name, then its buffers and scalars by keyword only."""
    for launcher, keywords in LAUNCHER_KEYWORDS.items():
        params = list(inspect.signature(getattr(cake_backend, launcher)).parameters.values())
        assert params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD), launcher
        rest = params[1:]
        assert all(p.kind is p.KEYWORD_ONLY for p in rest), launcher
        assert sorted(p.name for p in rest) == sorted(keywords), launcher


@pytest.mark.parametrize(("D", "causal"), VARIANTS, ids=VARIANT_IDS)
def test_stage_launchers_pass_the_registry_arguments_positionally(D, causal, monkeypatch) -> None:
    """Every record ``RECORD_OF`` routes to is launched positionally by its stage launcher, and its FFI entry gets the
    same arguments and grid, of the same types, as through the registry-ordered launch of ``_launch`` (no device: the
    FFI entry is a recorder)."""
    calls = []
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: ARCHES[0])
    monkeypatch.setattr(cake_backend, "_entry", lambda name, arch: lambda *args: calls.append((name, args)))
    B, H, Sq, Skv = 2, 3, 256, 384
    bf16, fp32 = {"dtype": torch.bfloat16}, {"dtype": torch.float32}
    q, out, dout, dq = (torch.empty(B, Sq, H, D, **bf16) for _ in range(4))
    k, v, dk, dv = (torch.empty(B, Skv, H, D, **bf16) for _ in range(4))
    lse = torch.empty(B, H, Sq, **fp32)
    lse_log2, dpsum = torch.empty(B * H, Sq, **fp32), torch.empty(B * H, Sq, **fp32)
    dq_accum = torch.empty(B * H * Sq * D, **fp32)
    pre, main, post = cake_jit.RECORD_OF[(D, causal)]

    def launch_all() -> list:
        calls.clear()
        cake_backend.launch_bwd_pre(pre, out=out, dout=dout, lse=lse, lse_log2=lse_log2, dpsum=dpsum, dq_accum=dq_accum)
        cake_backend.launch_bwd_main(
            main,
            q=q,
            k=k,
            v=v,
            dout=dout,
            lse_log2=lse_log2,
            dpsum=dpsum,
            dq_accum=dq_accum,
            dk=dk,
            dv=dv,
            softmax_scale=0.0883883,
        )
        cake_backend.launch_bwd_post(post, dq_accum=dq_accum, dq=dq, softmax_scale=0.0883883)
        return list(calls)

    for stage, name in zip(STAGES, (pre, main, post), strict=True):
        assert cake_backend._positional(name, stage), (stage, name)
    positional = launch_all()
    monkeypatch.setattr(cake_backend, "_positional", lambda name, stage: False)
    registry = launch_all()
    assert [name for name, _ in positional] == [name for name, _ in registry] == [pre, main, post]
    for (name, got), (_, want) in zip(positional, registry, strict=True):
        assert len(got) == len(want), name
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            assert type(a) is type(b), (name, i, type(a), type(b))
            assert a is b if isinstance(a, torch.Tensor) else a == b, (name, i, a, b)


def test_the_package_exports_the_backend_entry() -> None:
    assert pkg.cake_fmha_nondet_train_backward is cake_backend.cake_fmha_nondet_train_backward
    assert ENTRY in pkg.__all__ and ENTRY in cake_backend.__all__


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]}
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
    records = sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path; without one the JIT spec names the entry."""
    _manifest_or_skip()
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_backend.library_name(name, arch)
            assert cake_build.prebuilt_path(library) == PACKAGE_DIR / "aot" / f"{library}.so"
            assert cake_backend.is_prebuilt(name, arch) == cake_build.prebuilt_path(library).is_file()


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
# On SM100 / SM103 / SM107: every record loads


def test_the_records_of_every_variant_load(device) -> None:
    arch = cake_backend.device_arch(device.index)
    assert arch in ARCHES
    for variant, names in cake_jit.RECORD_OF.items():
        for name in names:
            assert callable(getattr(cake_backend.load(name, arch), cake_jit.MODULES[name]["ffi_entry"])), (
                variant,
                name,
            )


# ----------------------------------------------------------------------------------------------------------------
# Each stage on its own, judged against the naive BF16 tier


@pytest.mark.parametrize("case", PRE_CASES, ids=[case.id for case in PRE_CASES])
@pytest.mark.parametrize("D", HEAD_DIMS, ids=[f"d{D}" for D in HEAD_DIMS])
def test_the_preprocess_matches_the_reference(D, case, device, record_property) -> None:
    """``dPsum`` and ``LSE_log2`` (0 on the rows whose ``lse`` is ``-inf``) meet the tier, and ``dQ_accum`` is
    cleared to exactly 0 (the tier's error there is 0)."""
    name = cake_jit.RECORD_OF[(D, False)][0]

    def draw(seed: int):
        t = pre_inputs(case, D, seed=seed, device=device)
        cake_backend.launch_bwd_pre(
            name, out=t.out, dout=t.dout, lse=t.lse, lse_log2=t.lse_log2, dpsum=t.dpsum, dq_accum=t.dq_accum
        )
        return judged_stage(pre_outputs, t, {"dPsum": t.dpsum, "LSE_log2": t.lse_log2, "dQ_accum": t.dq_accum})

    judge_draws(record_property, draw)


@pytest.mark.parametrize(("D", "causal", "case"), MAIN_PARAMS)
def test_the_main_stage_matches_the_reference(D, causal, case, device, record_property) -> None:
    """The main stage alone on its own inputs (the FP32 ``LSE_log2`` and ``dPsum`` as stored, ``dQ_accum`` zeroed,
    ``dK`` and ``dV`` filled with NaN): ``dQ_accum`` (the unscaled ``dS K`` in the reduction layout), ``dK`` and
    ``dV`` meet the tier of the same formula on the same inputs."""
    name = cake_jit.RECORD_OF[(D, causal)][1]

    def draw(seed: int):
        x = make_inputs(case, D, causal=causal, seed=seed, device=device)
        m = main_buffers(x)
        cake_backend.launch_bwd_main(
            name,
            q=x.q,
            k=x.k,
            v=x.v,
            dout=x.dout,
            lse_log2=m.lse_log2,
            dpsum=m.dpsum,
            dq_accum=m.dq_accum,
            dk=m.dk,
            dv=m.dv,
            softmax_scale=x.softmax_scale,
        )
        return judged_stage(
            lambda _, dtype: main_outputs(x, m, dtype), x, {"dQ_accum": m.dq_accum, "dK": m.dk, "dV": m.dv}
        )

    judge_draws(record_property, draw)


@pytest.mark.parametrize("case", POST_CASES, ids=[case.id for case in POST_CASES])
@pytest.mark.parametrize("D", HEAD_DIMS, ids=[f"d{D}" for D in HEAD_DIMS])
def test_the_postprocess_matches_the_reference(D, case, device, record_property) -> None:
    """``dq = bf16(dQ_accum * softmax_scale)``, written in the input layout ``[B, Sq, H, D]`` over a NaN-filled
    buffer, meets the tier."""
    name = cake_jit.RECORD_OF[(D, False)][2]

    def draw(seed: int):
        t = post_inputs(case, D, seed=seed, device=device)
        cake_backend.launch_bwd_post(name, dq_accum=t.dq_accum, dq=t.dq, softmax_scale=t.softmax_scale)
        return judged_stage(post_outputs, t, {"dq": t.dq})

    judge_draws(record_property, draw)


# ----------------------------------------------------------------------------------------------------------------
# The backward: the three stages composed


@pytest.mark.parametrize(("D", "causal", "case"), MAIN_PARAMS)
def test_the_backward_matches_the_reference(D, causal, case, device, record_property) -> None:
    """``dq``, ``dk`` and ``dv`` of the backward hold the contract (BF16, contiguous, the inputs' layouts, finite) and
    meet the tier of the backward formula on the same inputs (``dPsum`` from ``out`` and ``dout``)."""

    def draw(seed: int):
        x = make_inputs(case, D, causal=causal, seed=seed, device=device)
        dq, dk, dv = cake_backend.cake_fmha_nondet_train_backward(
            x.q, x.k, x.v, x.out, x.lse, x.dout, softmax_scale=x.softmax_scale, causal=causal
        )
        return judged_backward(x, {"backward": {"dq": dq, "dk": dk, "dv": dv}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize(("D", "causal"), VARIANTS, ids=VARIANT_IDS)
def test_the_backward_launches_the_records_of_its_variant(D, causal, device, monkeypatch) -> None:
    """The backward of a variant launches, in order, the preprocess, main-stage and postprocess records that
    ``RECORD_OF`` names for it, through the stage launchers; it writes none of its inputs."""
    launched = []
    for stage in STAGES:
        launcher = f"launch_{stage}"

        def spy(name, *args, _launch=getattr(cake_backend, launcher), _stage=stage, **kwargs):
            launched.append((_stage, name))
            return _launch(name, *args, **kwargs)

        monkeypatch.setattr(cake_backend, launcher, spy)
    x = make_inputs(Case(2, 3, 256), D, causal=causal, seed=11, device=device)
    inputs = (x.q, x.k, x.v, x.out, x.lse, x.dout)
    before = [t.clone() for t in inputs]
    dq, dk, dv = cake_backend.cake_fmha_nondet_train_backward(*inputs, softmax_scale=x.softmax_scale, causal=causal)
    assert launched == list(zip(STAGES, cake_jit.RECORD_OF[(D, causal)], strict=True))
    assert_backward_contract("routed", x, {"dq": dq, "dk": dk, "dv": dv})
    for name, old, new in zip(("q", "k", "v", "out", "lse", "dout"), before, inputs, strict=True):
        assert torch.equal(old, new), f"the backward wrote its input {name}"
