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
"""

"""Benchmark the non-deterministic Cake DSA indexer top-k against the deterministic one (SM100 / SM103 / SM107).

Times the *complete operator* of both packages on the rows of
``benchmarks/bench_cake_dsa_indexer_det_train.py`` (loaded by path: its row
definitions, input builders, CUPTI timer and deterministic arm are reused
unchanged; ``--rows`` takes its row and group names, ``iid`` / ``sink`` /
``pool`` / ``all`` / ``fp8``) in the input form of ``--form``:

* ``det``: ``cake_pplx._backends.cake_dsa_indexer_det_train`` -- the
  scan, the split merge where the dispatch splits the key ranges, and the
  ordering pass that writes every row in ascending id order;
* ``nondet``: ``cake_pplx._backends.cake_dsa_indexer_nondet_train``
  -- the same scan programs and the plain split merge, without an ordering
  pass: every row holds its selection in no specified slot order.

With ``--form bf16`` (the default) every arm runs the BF16 form.  With
``--form fp8`` every arm runs the FP8 form of its package on the
deterministic benchmark's FP8 inputs (``build_fp8_inputs``: float8_e4m3fn
``q`` / ``k`` with per-key FP32 ``k_scale``, amax scaling per (query, head)
and per key, the query and softmax scales folded into ``w``) of the plain iid
rows; ``--rows fp8`` (the default of the mode) is its FP8 shape set:
``P1``, ``glm2k_16k`` (2048 queries over 16384 keys), ``s1_8k_on_64k`` (8192
over 73728), ``s2_4x2k_packed`` (four segments of 2048 queries over 18432,
43008, 67584 and 100352 keys) and ``s3_cold16k`` (16384 over 16384), top-k
2048, causal.

The candidate-block rows (``--rows candidates``: the request's shapes at
top-k 512) add the arms ``nondet_cb`` / ``nondet_cb_aa`` (this package's
``prepare_dsa_indexer_nondet_topk_with_candidates``: the same top-k plus the
``[T, 2048]`` candidate blocks) and ``det_cb`` (the deterministic package's),
timed next to ``nondet`` / ``det``, the plain call of the same ``top_k``;
``candidate_blocks`` records each candidate-block arm's block structure
(counts, newest block, containment).  ``nondet_cb_scan`` runs this package's
call on its ``scan_blocks`` program where every row keeps all its blocks (the
dispatch before ``iota``, which runs the plain programs and writes the rows
after them: ``plan`` records ``cb_iota``).  ``nondet_in`` / ``nondet_in_aa`` (this
package's ``prepare_dsa_indexer_nondet_topk_in_candidates``) and ``det_in``
(the deterministic package's) time the top-k restricted to the row's own
candidate blocks (computed once by the package's candidate-block call outside
the timed span) next to ``nondet`` / ``det``; ``in_candidates`` records that
every id lies in a consumed block and the number of rows whose set is the plain
call's of the same package (``rows_with_plain_set``, information only: with the
same ``q`` / ``w`` and ``top_k``, every row on inputs whose FP32 scores are
exact in any summation order).

Both arms are prepared runners over caller-owned outputs and workspace;
``det_aa`` / ``nondet_aa`` prepare the same arm a second time (the in-process
A/A band) and ``det_nosample`` / ``nondet_nosample`` run with the sampled
first threshold off (``sample_tiles_max=0``).  Every arm's output is checked
on its own, never against another arm's: the reference's structure checks on
its rows put in ascending id order (unique visible ids, ``min(top_k,
visible)`` valid slots per row, ``-1`` / ``-inf`` padding).  Every arm's plan
is recorded; ``same_scan_program`` says whether ``det`` and ``nondet``
dispatched the same scan key and split.  A row where an arm's output breaks
the structure checks is flagged ``FAILED``: it gets no ``det / nondet`` ratio,
and the script exits non-zero after writing its JSON.

Timing: the deterministic benchmark's ``gpu_times``, CUPTI activity tracing
through ``torch.profiler`` with a cold L2 between iterations (per-iteration GPU
span of the whole operator);
``--cupti-required`` turns the CUDA-event fallback into a failure.  Every row
is measured in three paired phases -- the arms in the given order (``AB``), in
reverse order (``BA``) and interleaved one call at a time (``IL``) -- and the
medians of each phase and of all samples are reported with the wall time, the
peak extra device memory and the workspace bytes.  Per row the script prints
``det`` ms, ``nondet`` ms and ``det / nondet`` (above 1: the non-deterministic
package is faster), and a summary table at the end.  ``--verify`` compares
every arm, its rows put in ascending id order first, with the FP64 selection of
``tests/experimental/test_helpers/cake_dsa_indexer_reference.py`` (small rows
only): the structure checks, and the number of rows whose ids equal the FP64
selection's, as information (on Gaussian inputs the FP32 rounding may reorder
keys whose FP64 scores are near-tied).

Usage::

    python benchmarks/bench_cake_dsa_indexer_nondet_train.py [--rows P1 S2 ...] [--arms det,nondet]
        [--reps 20] [--strided-k] [--verify] [--cupti-required] [--json out.json]
    python benchmarks/bench_cake_dsa_indexer_nondet_train.py --rows sink --arms det,nondet,det_aa,nondet_aa
        --cupti-required --json out.json
    python benchmarks/bench_cake_dsa_indexer_nondet_train.py --form fp8 --arms det,nondet,det_aa,nondet_aa
        --cupti-required --json out.json
"""

import argparse
import contextlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cake_pplx._backends.cake_dsa_indexer_nondet_train import (  # noqa: E402
    cake_backend as nondet_backend,
)
from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref  # noqa: E402


def _load_det_bench():
    """The deterministic package's benchmark module, loaded by path (it is a script, not a package module)."""
    path = Path(__file__).resolve().with_name("bench_cake_dsa_indexer_det_train.py")
    spec = importlib.util.spec_from_file_location(
        "bench_cake_dsa_indexer_det_train", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


det_bench = _load_det_bench()
det_backend = det_bench.cake_backend


# ---------------------------------------------------------------------------
# Rows in ascending id order
# ---------------------------------------------------------------------------


def canonical(out):
    """``out`` with every row in ascending id order (valid slots, then the padding): the reference's slot order."""
    ids, scores = out
    key = torch.where(ids >= 0, ids.to(torch.int64), 1 << 32)
    order = key.sort(dim=1, stable=True).indices
    return ids.gather(1, order), scores.gather(1, order)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def nondet_plan_record(inputs, knobs: dict) -> dict:
    """The host plan of the non-deterministic package for ``inputs`` under ``knobs``."""
    plan = nondet_backend.plan_dsa_indexer_nondet_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        device=inputs.device,
        **det_bench.row_extra(inputs),
        **det_bench.form_extra(inputs),
        **knobs,
    )
    return dict(
        scan_key=plan.scan_key,
        merge_key=plan.merge_key,
        kind=plan.kind,
        block_q=int(plan.block_q),
        n_split=int(plan.n_split),
        grid_ctas=int(plan.grid_ctas),
        cand_cap=int(plan.cand_cap),
        check_period=int(plan.check_period),
        sample_tiles_max=int(plan.sample_tiles_max),
        sample_shift_permille=int(plan.sample_shift_permille),
        far_sample=int(plan.far_sample),
        sample_fit=bool(plan.sample_fit),
    )


class NondetArm(det_bench.Arm):
    """A prepared runner of the non-deterministic package over caller-owned outputs and workspace."""

    name = "nondet"
    package = "nondet"

    def __init__(
        self,
        inputs,
        *,
        knobs=None,
        name=None,
        with_candidates=False,
        in_candidates=False,
        cb_scan=False,
    ):
        super().__init__(inputs)
        self.knobs = dict(knobs or {})
        if name is not None:
            self.name = name
        self.blocks = None
        self.source_blocks = None
        if with_candidates:
            # cb_scan: the call on its scan_blocks program where every row keeps all its blocks (before iota)
            with (
                det_bench.scan_blocks_only(nondet_backend)
                if cb_scan
                else contextlib.nullcontext()
            ):
                self._prepare_with_candidates(inputs)
            return
        if in_candidates:
            self._prepare_in_candidates(inputs)
            return
        dev = inputs.device
        self.workspace_bytes = (
            nondet_backend.cake_dsa_indexer_nondet_train_topk_workspace_size(
                inputs.num_queries,
                inputs.num_keys,
                inputs.num_segments,
                top_k=inputs.top_k,
                ratio=inputs.ratio,
                device=dev,
                **det_bench.form_extra(inputs),
            )
        )
        self.workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=dev
        )
        self.indices = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=dev
        )
        self.scores = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=dev
        )
        self.plan = nondet_plan_record(inputs, self.knobs)
        self.runner = nondet_backend.prepare_dsa_indexer_nondet_topk(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            **det_bench.row_extra(inputs),
            **self.knobs,
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
        )

    def _prepare_with_candidates(self, inputs):
        dev = inputs.device
        extra = det_bench.row_extra(inputs)
        self.workspace_bytes = nondet_backend.cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size(
            inputs.num_queries,
            inputs.num_keys,
            inputs.num_segments,
            top_k=inputs.top_k,
            ratio=inputs.ratio,
            max_seqlen_k=extra.get("max_seqlen_k"),
            device=dev,
        )
        self.workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=dev
        )
        self.indices = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=dev
        )
        self.scores = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=dev
        )
        self.blocks = torch.empty(
            (inputs.num_queries, 2048), dtype=torch.int32, device=dev
        )
        plan = nondet_backend.plan_dsa_indexer_nondet_topk_with_candidates(
            inputs.num_queries,
            inputs.num_keys,
            inputs.num_segments,
            top_k=inputs.top_k,
            ratio=inputs.ratio,
            device=dev,
            **extra,
            **self.knobs,
        )
        c = plan.choice
        self.plan = dict(
            scan_key=c.scan_key,
            merge_key=c.merge_key,
            kind=c.kind,
            block_q=int(c.block_q),
            n_split=int(c.n_split),
            grid_ctas=int(c.grid_ctas),
            cand_cap=int(c.cand_cap),
            check_period=int(c.check_period),
            sample_tiles_max=int(c.sample_tiles_max),
            sample_shift_permille=int(c.sample_shift_permille),
            far_sample=int(c.far_sample),
            sample_fit=bool(c.sample_fit),
            blk_cap=int(plan.blk_cap),
            blk_bytes=int(plan.blk_bytes),
            cb_iota=bool(plan.iota),
        )
        self.runner = nondet_backend.prepare_dsa_indexer_nondet_topk_with_candidates(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
            **extra,
            **self.knobs,
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
            candidate_blocks=self.blocks,
        )

    def _prepare_in_candidates(self, inputs):
        dev = inputs.device
        extra = det_bench.row_extra(inputs)
        args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
        # the consumed candidate blocks: the row's own candidate-block call of this package, once, outside the timed
        # span (the torch deterministic setting is left as the benchmark found it)
        self.source_blocks = (
            nondet_backend.cake_dsa_indexer_nondet_train_topk_with_candidates(
                *args, **inputs.kwargs(), **extra
            )[2].clone()
        )
        self.workspace_bytes = nondet_backend.cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size(
            inputs.num_queries,
            inputs.num_keys,
            inputs.num_segments,
            top_k=inputs.top_k,
            ratio=inputs.ratio,
            device=dev,
        )
        self.workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=dev
        )
        self.indices = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.int32, device=dev
        )
        self.scores = torch.empty(
            (inputs.num_queries, inputs.top_k), dtype=torch.float32, device=dev
        )
        plan = nondet_backend.plan_dsa_indexer_nondet_topk_in_candidates(
            inputs.num_queries,
            inputs.num_keys,
            inputs.num_segments,
            top_k=inputs.top_k,
            ratio=inputs.ratio,
            candidate_width=int(self.source_blocks.shape[1]),
            device=dev,
        )
        self.plan = dict(
            scan_key=plan.cand_key,
            n_split=1,
            grid_ctas=int(plan.grid_ctas),
            cand_cap=int(plan.cand_cap),
            check_period=int(plan.check_period),
        )
        self.runner = nondet_backend.prepare_dsa_indexer_nondet_topk_in_candidates(
            *args,
            self.source_blocks,
            **inputs.kwargs(),
            **extra,
            workspace_buffer=self.workspace,
            indices=self.indices,
            scores=self.scores,
        )

    def __call__(self):
        return self.runner()

    def outputs(self, out):
        """None: the runner writes into ``indices`` / ``scores`` allocated before the measured call."""
        return ()


def _det(name, knobs=None, with_candidates=False, in_candidates=False):
    def make(inputs):
        arm = det_bench.CakeArm(
            inputs,
            knobs=knobs,
            name=name,
            with_candidates=with_candidates,
            in_candidates=in_candidates,
        )
        arm.package = "det"
        return arm

    return make


def _nondet(
    name, knobs=None, with_candidates=False, in_candidates=False, cb_scan=False
):
    def make(inputs):
        return NondetArm(
            inputs,
            knobs=knobs,
            name=name,
            with_candidates=with_candidates,
            in_candidates=in_candidates,
            cb_scan=cb_scan,
        )

    return make


# name -> factory(inputs) -> Arm.  ``det_aa`` / ``nondet_aa`` prepare the same arm a second time (the in-process A/A
# band); the ``_nosample`` arms run the same programs with the sampled first threshold off.
ARMS = {
    "det": _det("det"),
    "nondet": _nondet("nondet"),
    "det_aa": _det("det_aa"),
    "nondet_aa": _nondet("nondet_aa"),
    "det_nosample": _det("det_nosample", knobs=det_bench.NOSAMPLE),
    "nondet_nosample": _nondet("nondet_nosample", knobs=det_bench.NOSAMPLE),
    # the candidate-block call of the same top_k (``--rows candidates``)
    "det_cb": _det("det_cb", with_candidates=True),
    "nondet_cb": _nondet("nondet_cb", with_candidates=True),
    "nondet_cb_aa": _nondet("nondet_cb_aa", with_candidates=True),
    # the same call on its scan_blocks program where every row keeps all its blocks (the dispatch before iota)
    "nondet_cb_scan": _nondet("nondet_cb_scan", with_candidates=True, cb_scan=True),
    # the candidate-restricted call over the row's own candidate blocks (``--rows candidates``)
    "det_in": _det("det_in", in_candidates=True),
    "nondet_in": _nondet("nondet_in", in_candidates=True),
    "nondet_in_aa": _nondet("nondet_in_aa", in_candidates=True),
}
DEFAULT_ARMS = "det,nondet"
TIMING_BOUNDARY = (
    "det arms: the complete operator of the deterministic package (scan, merge, ordering pass) of a prepared runner "
    "into caller-owned outputs and workspace; nondet arms: the complete operator of the non-deterministic package "
    "(scan, and the plain merge on a split call) of a prepared runner into caller-owned outputs and workspace; "
    "CUPTI GPU span of every launched kernel per iteration, cold L2"
)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_row(name, inputs, arms, args):
    row = det_bench.ROWS[name]
    reps = args.reps if args.reps is not None else row["reps"]
    prepared = {}
    for arm_name in arms:
        try:
            prepared[arm_name] = ARMS[arm_name](inputs)
        except Exception as exc:
            # a comparison without one of its arms is not the A/B the row exists for
            raise SystemExit(
                f"[{name}] {arm_name}: {type(exc).__name__}: {exc}"
            ) from exc
    result = dict(
        row=name,
        T=inputs.num_queries,
        Tkv=inputs.num_keys,
        segments=inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        reps=reps,
        arms={},
    )
    samples = {a: {"AB": [], "BA": [], "IL": []} for a in prepared}
    order = list(prepared)
    for arm_name in order:
        samples[arm_name]["AB"] = det_bench.gpu_times(
            prepared[arm_name], reps, args.warmup
        )
    for arm_name in reversed(order):
        samples[arm_name]["BA"] = det_bench.gpu_times(
            prepared[arm_name], reps, args.warmup
        )
    for _ in range(reps):
        for arm_name in order:
            samples[arm_name]["IL"].extend(
                det_bench.gpu_times(prepared[arm_name], 1, 0)
            )
    outputs = {}
    for arm_name, arm in prepared.items():
        out = arm()
        torch.cuda.synchronize()
        outputs[arm_name] = (out[0].clone(), out[1].clone())
        every = (
            samples[arm_name]["AB"] + samples[arm_name]["BA"] + samples[arm_name]["IL"]
        )
        record = dict(
            package=arm.package,
            gpu_ms={
                phase: statistics.median(v) for phase, v in samples[arm_name].items()
            },
            wall_ms=det_bench.wall_ms(arm, min(reps, 5)),
            peak_extra_bytes=det_bench.peak_extra_bytes(arm),
            workspace_bytes=arm.workspace_bytes,
            plan=arm.plan,
            knobs=arm.knobs,
            # the structure checks of the reference on the rows put in ascending id order
            structure_failures=ref.structure_failures(
                inputs, *canonical(outputs[arm_name])
            )[:3],
        )
        record["gpu_ms"]["all"] = statistics.median(every)
        if getattr(arm, "blocks", None) is not None:
            cb = det_bench.candidate_block_record(
                inputs, outputs[arm_name][0], arm.blocks
            )
            record["candidate_blocks"] = cb
            if cb["bad_counts"] or cb["missing_newest"] or cb["ids_outside_blocks"]:
                record["structure_failures"].append(f"candidate_blocks {cb}")
        if getattr(arm, "source_blocks", None) is not None:
            rec = det_bench.in_candidates_record(
                inputs, outputs[arm_name][0], arm.source_blocks
            )
            plain = arm.package
            if plain in outputs:
                rec["rows_with_plain_set"] = int(
                    (canonical(outputs[arm_name])[0] == canonical(outputs[plain])[0])
                    .all(dim=1)
                    .sum()
                )
            record["in_candidates"] = rec
            if rec["ids_outside_blocks"]:
                record["structure_failures"].append(f"in_candidates {rec}")
        result["arms"][arm_name] = record
    if args.verify:
        reference = ref.select_reference(inputs)
        for arm_name, out in outputs.items():
            failures, rows, same = ref.compare_selection(canonical(out), reference)
            result["arms"][arm_name]["verdict"] = (
                f"structure {failures[0]}"
                if failures
                else f"{same}/{rows} rows hold the FP64 selection's ids"
            )
            result["arms"][arm_name]["verdict_passed"] = not failures
    # a row is correct when no arm breaks the output structure (each arm checked on its own, never against another
    # arm); no speedup is published for a row that is not
    failures = []
    for arm_name, rec in result["arms"].items():
        if rec["structure_failures"]:
            failures.append(f"{arm_name}: structure {rec['structure_failures'][0]}")
        if rec.get("verdict_passed") is False:
            failures.append(f"{arm_name}: {rec['verdict']}")
    result["failures"] = failures
    # the candidate-block call over the plain call of the same package and top_k (above 1: the candidate blocks cost)
    for cb_arm, plain in (
        ("nondet_cb", "nondet"),
        ("det_cb", "det"),
        ("nondet_in", "nondet"),
        ("det_in", "det"),
    ):
        if cb_arm in prepared and plain in prepared:
            result[f"{cb_arm}_over_{plain}"] = (
                None
                if failures
                else result["arms"][cb_arm]["gpu_ms"]["all"]
                / result["arms"][plain]["gpu_ms"]["all"]
            )
    if "det" in prepared and "nondet" in prepared:
        det_ms = result["arms"]["det"]["gpu_ms"]["all"]
        nondet_ms = result["arms"]["nondet"]["gpu_ms"]["all"]
        result["det_ms"] = det_ms
        result["nondet_ms"] = nondet_ms
        result["det_over_nondet"] = None if failures else det_ms / nondet_ms
        det_plan, nondet_plan = prepared["det"].plan, prepared["nondet"].plan
        result["same_scan_program"] = (
            det_plan["scan_key"] == nondet_plan["scan_key"]
            and det_plan["n_split"] == nondet_plan["n_split"]
        )
    return result


def print_row(result):
    for arm_name, rec in result["arms"].items():
        gpu = rec["gpu_ms"]
        bad = (
            f" STRUCTURE {rec['structure_failures'][0]}"
            if rec["structure_failures"]
            else ""
        )
        verdict = f" {rec['verdict']}" if "verdict" in rec else ""
        print(
            f"  [{result['row']}] {arm_name:<16} gpu {gpu['all']:10.3f} ms (AB {gpu['AB']:.3f} / BA {gpu['BA']:.3f} / IL {gpu['IL']:.3f})"
            f" wall {rec['wall_ms']:.3f} ms extra {rec['peak_extra_bytes'] / 2**20:.1f} MiB"
            f" workspace {rec['workspace_bytes'] / 2**20:.1f} MiB {rec['plan']['scan_key']} n_split {rec['plan']['n_split']}"
            f"{bad}{verdict}",
            flush=True,
        )
    if result.get("det_over_nondet") is not None:
        print(
            f"  [{result['row']}] det {result['det_ms']:.3f} ms  nondet {result['nondet_ms']:.3f} ms  "
            f"det/nondet {result['det_over_nondet']:.3f}  same scan program {result['same_scan_program']}",
            flush=True,
        )
    if result["failures"]:
        print(
            f"  [{result['row']}] FAILED (no speedup published): {result['failures']}",
            flush=True,
        )


def print_summary(results):
    """The det / nondet table; a row with a correctness failure is flagged FAILED and has no ratio."""
    rows = [r for r in results if "det_ms" in r]
    if not rows:
        return
    print("\n| row | T | Tkv | top_k | det ms | nondet ms | det / nondet |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        ratio = "FAILED" if r["failures"] else f"{r['det_over_nondet']:.3f}"
        print(
            f"| {r['row']} | {r['T']} | {r['Tkv']} | {r['top_k']} | {r['det_ms']:.3f} | {r['nondet_ms']:.3f} | "
            f"{ratio} |"
        )
    ratios = [r["det_over_nondet"] for r in rows if not r["failures"]]
    if ratios:
        geomean = math.exp(statistics.fmean(math.log(v) for v in ratios))
        print(
            f"\ndet / nondet over {len(ratios)} correct rows: geometric mean {geomean:.3f}, "
            f"min {min(ratios):.3f}, max {max(ratios):.3f}",
            flush=True,
        )
    failed = [r["row"] for r in rows if r["failures"]]
    if failed:
        print(f"FAILED rows (excluded): {failed}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--form",
        choices=det_bench.FORMS,
        default="bf16",
        help="input form of every arm: bf16 (default) or fp8 (float8_e4m3fn q / k with per-key scales on the plain "
        "iid rows; see the module docstring)",
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        default=None,
        help="rows of the deterministic benchmark to measure (default: its default rows with --form bf16, its fp8 "
        "group with --form fp8; the group names iid / sink / pool / all / fp8 expand)",
    )
    parser.add_argument(
        "--arms",
        default=DEFAULT_ARMS,
        help=f"comma-separated arms in AB order (default: {DEFAULT_ARMS}; known: {', '.join(ARMS)})",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="measured iterations per phase (default: the row's count)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--strided-k",
        action="store_true",
        help="hand k as the [:, :128] view of a [Tkv, 704] packed tensor",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare every arm with the FP64 selection: structure, rows identical (small rows)",
    )
    parser.add_argument(
        "--cupti-required",
        action="store_true",
        help="fail when the CUPTI timer (torch.profiler) falls back to CUDA events",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if torch.are_deterministic_algorithms_enabled():
        raise SystemExit(
            "torch.use_deterministic_algorithms(True) is set: every run of the nondet arms would alert"
        )
    det_bench.TIMER["required"] = bool(args.cupti_required)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit(
            f"an arm may be named once; use det_aa / nondet_aa for an A/A pair: {arms}"
        )
    fp8 = args.form == "fp8"
    if args.rows is None:
        args.rows = det_bench.FP8_ROWS if fp8 else det_bench.DEFAULT_ROWS
    rows = det_bench.expand_rows(args.rows)
    unknown_rows = [r for r in rows if r not in det_bench.ROWS]
    if unknown_rows:
        raise SystemExit(
            f"unknown rows {unknown_rows}; known: {sorted(det_bench.ROWS)} and the groups {sorted(det_bench.ROW_GROUPS)}"
        )
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    for package, backend in (("det", det_backend), ("nondet", nondet_backend)):
        used = any(a.split("_")[0] == package for a in arms)
        if used and not backend.generated_program_available(device, dtype=dtype):
            raise SystemExit(
                f"generated {package} programs of the {args.form} form not registered for this device"
            )
    print(
        f"{props.name} ({props.multi_processor_count} SMs, cc {props.major}.{props.minor}, CUDA {torch.version.cuda}), "
        f"rows {rows}, arms {arms}, form {args.form}"
    )
    print(f"timing boundary: {TIMING_BOUNDARY}", flush=True)
    results = []
    for name in rows:
        build = det_bench.build_fp8_inputs if fp8 else det_bench.build_inputs
        inputs = build(name, device, args.strided_k)
        print(
            f"[{name}] T={inputs.num_queries} Tkv={inputs.num_keys} segments={inputs.num_segments} "
            f"top_k={inputs.top_k} ratio={inputs.ratio} form={args.form}: {det_bench.ROWS[name]['note']}",
            flush=True,
        )
        result = measure_row(name, inputs, arms, args)
        print_row(result)
        results.append(result)
        del inputs
        torch.cuda.empty_cache()
    print_summary(results)
    print(f"timer: {det_bench.timer_record()}", flush=True)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(
                    device=props.name,
                    sms=props.multi_processor_count,
                    compute_capability=f"{props.major}.{props.minor}",
                    cuda=torch.version.cuda,
                    torch=torch.__version__,
                    packages=dict(
                        det=str(Path(det_backend.__file__).resolve().parent),
                        nondet=str(Path(nondet_backend.__file__).resolve().parent),
                    ),
                    results=results,
                    timer=det_bench.timer_record(),
                    arms=arms,
                    warmup=args.warmup,
                    reps=args.reps,
                    strided_k=args.strided_k,
                    form=args.form,
                    timing_boundary=TIMING_BOUNDARY,
                ),
                indent=1,
            )
        )
        print(f"wrote {args.json}")
    failed = {r["row"]: r["failures"] for r in results if r["failures"]}
    if failed:
        raise SystemExit(f"correctness failures (no speedup published): {failed}")


if __name__ == "__main__":
    main()
