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
the three architectures and for no other, the launch facts: a cluster of two CTAs for the 2-CTA main kernels), the
routing table ``RECORD_OF`` (the preprocess, main-stage and postprocess record of each head dimension, causal flag and
route, every record used, a 2-CTA main stage with the pair-row postprocess and no other) and the host sequence of each
variant (``SEQUENCES``: its three records), the route of a call (``variant``: the pair route for head dimension 128
and a key length that is a multiple of 256 on the architectures of ``PAIR_ROUTE``), the launch grids, the stage
launchers' keywords, their positional launch of every routed record (the FFI arguments and grid of the
registry-ordered launch), the two layouts of the dQ accumulator in the judge's helper, the build manifest (every
record's and host sequence's library, the recorded source digests), the standalone loader, the prebuilt libraries
served first, and no environment read outside the JIT shim.

On SM100 / SM103 / SM107, each stage judged on its own against the naive BF16 tier (``tests/cake_naive.py``: the error
against an FP64 oracle of the same inputs over 32 independent draws, in the L2 and the max norm, median and largest,
beside the error of the same math in BF16), every output buffer filled with NaN before the launch so every element
must be written: the preprocess (``dPsum``, ``LSE_log2``, the cleared ``dQ_accum``) at both head dimensions; the main
stage on its own inputs (the FP32 ``LSE_log2`` and ``dPsum``, a zeroed ``dQ_accum``) for every variant, judged on
``dQ_accum`` (in the pair-row layout for the 2-CTA main kernels), ``dK`` and ``dV``; the postprocess (``dq`` in the
input layout) at both head dimensions and in the pair-row build.  The public backward, which composes the three
stages, judged on the cases of every variant at four heads (the source tests' cases), one head and five heads; a call
launches the kernels of the three records ``RECORD_OF`` names for its variant (``variant``), in order and no other
(observed on the device, on both routes of head dimension 128), its inputs left unchanged; the same records, and
``dq``, ``dk`` and ``dv`` within the tier, when the host sequence's ``run`` returns ``False`` (the workspace from
Python) and without a host sequence (the stage launchers).  Where ``torch.profiler`` records no kernel of a plain torch
operation on the device (a CUPTI that does not support it), the tests that observe kernels run every other check, then
skip the observation.

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
    accumulator,
    assert_backward_contract,
    from_accumulator,
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
# (head dimension, causal, pair) of every variant; each is delivered for the three architectures.  ``pair``: the
# pair route of head dimension 128 (the 2-CTA main kernel and the pair-row postprocess).
VARIANTS = (
    (128, False, False),
    (128, True, False),
    (64, False, False),
    (64, True, False),
    (128, False, True),
    (128, True, True),
)
VARIANT_IDS = ["d128", "d128_causal", "d64", "d64_causal", "d128_pair", "d128_pair_causal"]
# The 2-CTA main-stage records (a cluster of two CTAs along x) and the postprocess that reads their pair-row order.
PAIR_MAINS = ("bwd_main_2cta", "bwd_main_2cta_causal")
PAIR_POST = "bwd_post_pair"
# Threads per CTA of each stage's kernel (8, 16 and 4 warps).
BLOCK_THREADS = {"bwd_pre": 256, "bwd_main": 512, "bwd_post": 128}
ENTRY = "cake_fmha_nondet_train_backward"
# The keywords of each stage launcher, after the record name.
LAUNCHER_KEYWORDS = {
    "launch_bwd_pre": ("out", "dout", "lse", "lse_log2", "dpsum", "dq_accum"),
    "launch_bwd_main": ("q", "k", "v", "dout", "lse_log2", "dpsum", "dq_accum", "dk", "dv", "softmax_scale"),
    "launch_bwd_post": ("dq_accum", "dq", "softmax_scale"),
}


def main_cases(causal: bool, pair: bool = False) -> list[Case]:
    """The main stage's and the backward's cases of one causal flag and route.  The 1-CTA route: the source tests'
    four cases at four heads (the uneven one has more keys than queries without the causal mask, a longer sequence
    with it), then one head and an odd head count.  The pair route (key lengths that are multiples of 256): the 2-CTA
    source tests' cases, at four heads (without the mask also one query tile, more queries than keys and fewer), then
    one head and an odd head count."""
    if pair:
        if causal:
            return [
                Case(1, 4, 256),
                Case(2, 4, 512),
                Case(1, 4, 768),
                Case(3, 4, 1024),
                Case(2, 1, 512),
                Case(2, 5, 512),
            ]
        return [
            Case(1, 4, 256),
            Case(2, 4, 128, 256),
            Case(2, 4, 512),
            Case(1, 4, 384, 512),
            Case(2, 4, 768, 256),
            Case(3, 4, 1024),
            Case(2, 1, 512),
            Case(2, 5, 512),
        ]
    uneven = Case(1, 4, 640) if causal else Case(1, 4, 384, 640)
    odd = Case(2, 5, 640) if causal else Case(2, 5, 384, 640)
    return [Case(1, 4, 128), Case(2, 4, 256), uneven, Case(3, 4, 1024), Case(3, 1, 256), odd]


MAIN_PARAMS = [
    pytest.param(D, causal, pair, case, id=f"{variant}-{case.id}")
    for (D, causal, pair), variant in zip(VARIANTS, VARIANT_IDS, strict=True)
    for case in main_cases(causal, pair)
]
# The postprocess records, by (head dimension, pair).
POST_VARIANTS = ((128, False), (64, False), (128, True))
POST_IDS = ["d128", "d64", "d128_pair"]
# The calls whose launched kernels are observed: two (head, batch) pairs of three heads, 256 queries and keys for every
# head dimension and mask (the pair route at head dimension 128 where ``PAIR_ROUTE`` takes it), and 384 keys (an odd
# key-block count: the 1-CTA route) at head dimension 128.
ROUTE_PARAMS = [
    pytest.param(128, False, Case(2, 3, 256), id="d128"),
    pytest.param(128, True, Case(2, 3, 256), id="d128_causal"),
    pytest.param(64, False, Case(2, 3, 256), id="d64"),
    pytest.param(64, True, Case(2, 3, 256), id="d64_causal"),
    pytest.param(128, False, Case(2, 3, 384), id="d128-S384"),
    pytest.param(128, True, Case(2, 3, 384), id="d128_causal-S384"),
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
        # The 2-CTA main kernels run in clusters of two CTAs along x; every other kernel in single-CTA clusters.
        assert tuple(launch["cluster"]) == ((2, 1, 1) if name in PAIR_MAINS else (1, 1, 1)), name
        assert tuple(launch["block"]) == (BLOCK_THREADS[rec["stage"]], 1, 1), name
        libraries.add(rec["library"])
    assert len(libraries) == len(cake_jit.MODULES)
    assert set(PAIR_MAINS) | {PAIR_POST} <= set(cake_jit.MODULES)


def test_routing_table_names_three_records_per_variant() -> None:
    """``RECORD_OF[(head_dim, causal, pair)]`` is the preprocess, main-stage and postprocess record of each variant:
    the preprocess depends on the head dimension only, the postprocess on the head dimension and the route, each
    variant has its own main stage, a 2-CTA main stage goes with the pair-row postprocess and no other (the pair
    route, head dimension 128 only), and every record of the registry serves some variant."""
    table = cake_jit.RECORD_OF
    assert set(table) == set(VARIANTS)
    used = set()
    for variant, names in table.items():
        assert tuple(cake_jit.MODULES[name]["stage"] for name in names) == STAGES, (variant, names)
        used.update(names)
    for D, causal, pair in VARIANTS:
        assert table[(D, causal, pair)][0] == table[(D, False, False)][0], (D, causal, pair)
        assert table[(D, causal, pair)][2] == table[(D, False, pair)][2], (D, causal, pair)
        _, main, post = table[(D, causal, pair)]
        assert (main in PAIR_MAINS) == (post == PAIR_POST) == pair, (D, causal, pair)
        assert D == 128 or not pair, (D, causal, pair)
    assert table[(64, False, False)][0] != table[(128, False, False)][0]
    assert len({table[variant][2] for variant in VARIANTS}) == 3
    assert len({table[variant][1] for variant in VARIANTS}) == len(VARIANTS)
    assert used == set(cake_jit.MODULES)
    # The host sequence of each variant runs its three records, in order, on the three architectures.
    assert set(cake_jit.SEQUENCES) == set(table)
    for variant, sequence in cake_jit.SEQUENCES.items():
        assert tuple(sequence["records"]) == tuple(table[variant]), variant
        assert tuple(sequence["arches"]) == ARCHES, variant
    assert len({sequence["library"] for sequence in cake_jit.SEQUENCES.values()}) == len(VARIANTS)


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name, rec in cake_jit.MODULES.items():
        for arch in ARCHES:
            assert cake_backend.library_name(name, arch) == f"{rec['library']}_{arch}"
        for arch in ("sm_110a", "sm_90a", "sm_120a"):
            with pytest.raises(RuntimeError):
                cake_backend.library_name(name, arch)


def test_the_route_of_a_call() -> None:
    """``variant(D, causal, Skv, arch)`` is the ``RECORD_OF`` key of a call: ``(D, causal, pair)`` with ``pair`` True
    for head dimension 128 and a key length that is a multiple of 256 (an even key-block count: the 2-CTA kernel's
    CTA pairs) where ``PAIR_ROUTE[arch]`` is True, False otherwise (every head-dimension 64 call, every odd key-block
    count)."""
    assert set(cake_jit.PAIR_ROUTE) == set(ARCHES)
    assert all(type(on) is bool for on in cake_jit.PAIR_ROUTE.values())
    for arch in ARCHES:
        for D in HEAD_DIMS:
            for causal in (False, True):
                for blocks in range(1, 70):
                    Skv = blocks * cake_jit.BLOCK
                    found = cake_jit.variant(D, causal, Skv, arch)
                    pair = D == 128 and blocks % 2 == 0 and cake_jit.PAIR_ROUTE[arch]
                    assert found == (D, causal, pair), (D, causal, Skv, arch, found)
                    assert type(found[1]) is bool and type(found[2]) is bool, found
                    assert found in cake_jit.RECORD_OF, found


def test_the_accumulator_layouts() -> None:
    """The judge's helper writes and reads the FP32 dQ accumulator in the layout of each main kernel: per 128-row tile
    of a head, element ``(r, d)`` at ``(d // 4) * 512 + r * 4 + d % 4`` (the 1-CTA kernels) or at ``(r // 64) * 8192 +
    (d // 4) * 256 + (r % 64) * 4 + d % 4`` (the 2-CTA kernel, head dimension 128); each layout reads back."""
    B, H, S, D = 2, 3, 384, 128
    values = torch.arange(B * H * S * D, dtype=torch.float64).view(B * H, S, D)
    for pair in (False, True):
        acc = accumulator(values, pair=pair)
        assert acc.shape == (B * H * S * D,)
        for bh, row, d in ((0, 0, 0), (1, 5, 7), (2, 63, 127), (3, 64, 1), (4, 130, 66), (5, 383, 125)):
            tile, r = divmod(row, 128)
            if pair:
                at = (r // 64) * 8192 + (d // 4) * 256 + (r % 64) * 4 + d % 4
            else:
                at = (d // 4) * 512 + r * 4 + d % 4
            assert acc[(bh * (S // 128) + tile) * 128 * D + at] == values[bh, row, d], (pair, bh, row, d)
        back = from_accumulator(acc, B, H, S, D, pair=pair)
        assert torch.equal(back, values.view(B, H, S, D)), pair
    assert not torch.equal(accumulator(values, pair=True), accumulator(values, pair=False))


def test_grids_cover_every_tile() -> None:
    """A CTA per 128-row query tile of each (head, batch) for the preprocess and the postprocess, per 128-key block of
    each (head, batch) for the main stage: the grid is ``(tiles, H, B)`` (the kernels read the head from
    ``blockIdx.y`` and the batch from ``blockIdx.z``).  A call on the pair route has an even key-block count, so the
    2-CTA main kernel's clusters of two CTAs along x tile its grid."""
    rng = random.Random(7091)
    for _ in range(200):
        B, H = rng.randint(1, 64), rng.randint(1, 64)
        Sq, Skv = 128 * rng.randint(1, 96), 128 * rng.randint(1, 96)
        dims = dict(batch=B, heads=H, seqlen_q=Sq, seqlen_kv=Skv)
        assert tuple(cake_jit.grid("bwd_pre", **dims)) == (Sq // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_main", **dims)) == (Skv // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_post", **dims)) == (Sq // 128, H, B), dims
        for arch in ARCHES:
            if cake_jit.variant(128, False, Skv, arch)[2]:
                assert cake_jit.grid("bwd_main", **dims)[0] % 2 == 0, (dims, arch)


def test_stage_launchers_take_named_buffers() -> None:
    """Each stage launcher takes the record name, then its buffers and scalars by keyword only."""
    for launcher, keywords in LAUNCHER_KEYWORDS.items():
        params = list(inspect.signature(getattr(cake_backend, launcher)).parameters.values())
        assert params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD), launcher
        rest = params[1:]
        assert all(p.kind is p.KEYWORD_ONLY for p in rest), launcher
        assert sorted(p.name for p in rest) == sorted(keywords), launcher


@pytest.mark.parametrize(("D", "causal", "pair"), VARIANTS, ids=VARIANT_IDS)
def test_stage_launchers_pass_the_registry_arguments_positionally(D, causal, pair, monkeypatch) -> None:
    """Every record ``RECORD_OF`` routes to is launched positionally by its stage launcher, and its FFI entry gets the
    same arguments and grid, of the same types, as through the registry-ordered launch of ``_launch`` (no device: the
    FFI entry is a recorder)."""
    calls = []
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: ARCHES[0])
    monkeypatch.setattr(cake_backend, "_entry", lambda name, arch: lambda *args: calls.append((name, args)))
    # Sizes and tile counts all distinct, so a swapped integer argument or grid axis changes the recorded call (an even
    # key-block count on the pair route).
    B, H, Sq, Skv = 2, 3, 5 * cake_jit.BLOCK, (8 if pair else 7) * cake_jit.BLOCK
    assert len({B, H, Sq, Skv, Sq // cake_jit.BLOCK, Skv // cake_jit.BLOCK}) == 6
    bf16, fp32 = {"dtype": torch.bfloat16}, {"dtype": torch.float32}
    q, out, dout, dq = (torch.empty(B, Sq, H, D, **bf16) for _ in range(4))
    k, v, dk, dv = (torch.empty(B, Skv, H, D, **bf16) for _ in range(4))
    lse = torch.empty(B, H, Sq, **fp32)
    lse_log2, dpsum = torch.empty(B * H, Sq, **fp32), torch.empty(B * H, Sq, **fp32)
    dq_accum = torch.empty(B * H * Sq * D, **fp32)
    pre, main, post = cake_jit.RECORD_OF[(D, causal, pair)]

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
    libraries = (*cake_jit.MODULES.values(), *cake_jit.SEQUENCES.values())
    wanted = {f"{rec['library']}_{arch}" for rec in libraries for arch in rec["arches"]}
    assert set(entries) == wanted
    assert len(entries) == len(libraries) * len(ARCHES)
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
    records = sum(len(rec["arches"]) for rec in (*cake_jit.MODULES.values(), *cake_jit.SEQUENCES.values()))
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
    name = cake_jit.RECORD_OF[(D, False, False)][0]

    def draw(seed: int):
        t = pre_inputs(case, D, seed=seed, device=device)
        cake_backend.launch_bwd_pre(
            name, out=t.out, dout=t.dout, lse=t.lse, lse_log2=t.lse_log2, dpsum=t.dpsum, dq_accum=t.dq_accum
        )
        return judged_stage(pre_outputs, t, {"dPsum": t.dpsum, "LSE_log2": t.lse_log2, "dQ_accum": t.dq_accum})

    judge_draws(record_property, draw)


@pytest.mark.parametrize(("D", "causal", "pair", "case"), MAIN_PARAMS)
def test_the_main_stage_matches_the_reference(D, causal, pair, case, device, record_property) -> None:
    """The main stage alone on its own inputs (the FP32 ``LSE_log2`` and ``dPsum`` as stored, ``dQ_accum`` zeroed,
    ``dK`` and ``dV`` filled with NaN): ``dQ_accum`` (the unscaled ``dS K`` in the reduction layout, or in the pair-row
    layout for a 2-CTA main kernel), ``dK`` and ``dV`` meet the tier of the same formula on the same inputs."""
    name = cake_jit.RECORD_OF[(D, causal, pair)][1]

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
            lambda _, dtype: main_outputs(x, m, dtype, pair=pair),
            x,
            {"dQ_accum": m.dq_accum, "dK": m.dk, "dV": m.dv},
        )

    judge_draws(record_property, draw)


@pytest.mark.parametrize("case", POST_CASES, ids=[case.id for case in POST_CASES])
@pytest.mark.parametrize(("D", "pair"), POST_VARIANTS, ids=POST_IDS)
def test_the_postprocess_matches_the_reference(D, pair, case, device, record_property) -> None:
    """``dq = bf16(dQ_accum * softmax_scale)``, written in the input layout ``[B, Sq, H, D]`` over a NaN-filled
    buffer, meets the tier; the pair-row build reads ``dQ_accum`` in the 2-CTA main kernel's layout."""
    name = cake_jit.RECORD_OF[(D, False, pair)][2]

    def draw(seed: int):
        t = post_inputs(case, D, seed=seed, device=device)
        cake_backend.launch_bwd_post(name, dq_accum=t.dq_accum, dq=t.dq, softmax_scale=t.softmax_scale)
        return judged_stage(lambda u, dtype: post_outputs(u, dtype, pair=pair), t, {"dq": t.dq})

    judge_draws(record_property, draw)


# ----------------------------------------------------------------------------------------------------------------
# The backward: the three stages composed


@pytest.mark.parametrize(("D", "causal", "pair", "case"), MAIN_PARAMS)
def test_the_backward_matches_the_reference(D, causal, pair, case, device, record_property) -> None:
    """``dq``, ``dk`` and ``dv`` of the backward hold the contract (BF16, contiguous, the inputs' layouts, finite) and
    meet the tier of the backward formula on the same inputs (``dPsum`` from ``out`` and ``dout``), on the cases of
    each variant (``pair``: the pair route's cases, whose key lengths are multiples of 256); the call takes the route
    ``variant`` picks for it."""

    def draw(seed: int):
        x = make_inputs(case, D, causal=causal, seed=seed, device=device)
        dq, dk, dv = cake_backend.cake_fmha_nondet_train_backward(
            x.q, x.k, x.v, x.out, x.lse, x.dout, softmax_scale=x.softmax_scale, causal=causal
        )
        return judged_backward(x, {"backward": {"dq": dq, "dk": dk, "dv": dv}})

    judge_draws(record_property, draw)


def _kernels(call, trace: Path) -> tuple[object, list[str]]:
    """``(result, names)``: ``call()``'s result and the names of the CUDA kernels it ran, in launch order, observed on
    the device (``torch.profiler``; the trace is written to ``trace``)."""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as prof:
        result = call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"])
    return result, [event["name"] for event in kernels]


def _launched(call, device: torch.device, trace: Path) -> tuple[object, list[tuple[str, str]] | None]:
    """``(result, launched)``: ``call()``'s result and the CUDA kernels it ran, in launch order, observed on the device
    (``torch.profiler``; the trace is written to ``trace``), each as ``(stage, record)`` of the registry record whose
    kernel it is, or ``("no record", name)``.

    Positive control first: a plain torch operation on ``device``, profiled the same way.  When the profiler records
    no kernel of it (a CUPTI that does not support the device), ``call()`` runs without the profiler and ``launched``
    is ``None``: the test runs its other checks, then :func:`_assert_launched` skips the observation."""
    _, control = _kernels(lambda: torch.ones(1024, device=device).mul_(2), trace.with_name("control-" + trace.name))
    if not control:
        return call(), None
    record_of = {rec["kernel_symbol"]: (rec["stage"], name) for name, rec in cake_jit.MODULES.items()}
    result, names = _kernels(call, trace)
    return result, [record_of.get(name, ("no record", name)) for name in names]


def _assert_launched(launched: list[tuple[str, str]] | None, variant: tuple[int, bool, bool]) -> None:
    """The kernels observed are those of the three records of ``variant`` (a ``RECORD_OF`` key), in order, and no
    other.  Called after every other check of the test: when the profiler observed nothing (``None``), only this
    observation skips."""
    if launched is None:
        pytest.skip("torch.profiler records no CUDA kernel on this device (CUPTI); install a CUPTI that supports it")
    assert launched == list(zip(STAGES, cake_jit.RECORD_OF[variant], strict=True))


@pytest.mark.parametrize(("D", "causal", "case"), ROUTE_PARAMS)
def test_the_backward_launches_the_records_of_its_variant(D, causal, case, device, tmp_path) -> None:
    """A backward call launches, in order, the preprocess, main-stage and postprocess records that ``RECORD_OF``
    names for its variant (``variant(D, causal, Skv, arch)``: at head dimension 128 the pair route for 256 keys where
    ``PAIR_ROUTE`` takes it, the 1-CTA route for 384), and no other kernel: the kernels one call runs, observed on the
    device, are those records' kernels; it writes none of its inputs."""
    variant = cake_jit.variant(D, causal, case.kv, cake_backend.device_arch(device.index))
    x = make_inputs(case, D, causal=causal, seed=11, device=device)
    inputs = (x.q, x.k, x.v, x.out, x.lse, x.dout)
    before = [t.clone() for t in inputs]

    def call():
        return cake_backend.cake_fmha_nondet_train_backward(*inputs, softmax_scale=x.softmax_scale, causal=causal)

    (dq, dk, dv), launched = _launched(call, device, tmp_path / "trace.json")
    assert_backward_contract("routed", x, {"dq": dq, "dk": dk, "dv": dv})
    for name, old, new in zip(("q", "k", "v", "out", "lse", "dout"), before, inputs, strict=True):
        assert torch.equal(old, new), f"the backward wrote its input {name}"
    _assert_launched(launched, variant)


# The backward's paths besides the host sequence's ``run``: ``run`` returns False (no tvm-ffi environment allocator, or
# a failed allocation: the workspace from Python, then ``run_with_workspace``), and no host sequence for the variant
# (the stage launchers).
FALLBACKS = ("run_with_workspace", "stage_launchers")


@pytest.mark.parametrize("path", FALLBACKS)
@pytest.mark.parametrize(("D", "causal", "case"), ROUTE_PARAMS)
def test_the_backward_without_run_launches_the_same_records(
    D, causal, case, path, device, monkeypatch, tmp_path, record_property
) -> None:
    """When ``run`` returns ``False`` and when the variant has no host sequence, the backward launches the variant's
    three records in order, and its ``dq``, ``dk`` and ``dv`` meet the tier (on both routes of head dimension 128)."""
    arch = cake_backend.device_arch(device.index)
    variant = cake_jit.variant(D, causal, case.kv, arch)
    found = cake_backend._sequence(variant, arch)
    assert found is not None, "every variant ships a host sequence"
    library, run, run_with_workspace = found
    x = make_inputs(case, D, causal=causal, seed=11, device=device)
    inputs = (x.q, x.k, x.v, x.out, x.lse, x.dout)
    calls = []

    def without_allocator(*args):
        calls.append(len(args))
        return False

    replacement = None if path == "stage_launchers" else (library, without_allocator, run_with_workspace)
    monkeypatch.setattr(cake_backend, "_sequence", lambda *variant: replacement)

    def call():
        return cake_backend.cake_fmha_nondet_train_backward(*inputs, softmax_scale=x.softmax_scale, causal=causal)

    (dq_path, dk_path, dv_path), launched = _launched(call, device, tmp_path / "trace.json")
    assert calls == ([] if replacement is None else [11])
    assert_backward_contract(path, x, {"dq": dq_path, "dk": dk_path, "dv": dv_path})

    def draw(seed: int):
        y = make_inputs(case, D, causal=causal, seed=seed, device=device)
        got = cake_backend.cake_fmha_nondet_train_backward(
            y.q, y.k, y.v, y.out, y.lse, y.dout, softmax_scale=y.softmax_scale, causal=causal
        )
        return judged_backward(y, {path: dict(zip(("dq", "dk", "dv"), got, strict=True))})

    judge_draws(record_property, draw)
    _assert_launched(launched, variant)
