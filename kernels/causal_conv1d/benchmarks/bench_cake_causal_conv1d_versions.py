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

Benchmark of two (or more) installed versions of the Cake causal conv1d training package against each other and
against the non-Cake baselines, on one GPU, single rank.

Each version lives in its own Python environment (``--cake LABEL=PYTHON``, e.g. ``old=/venvs/old/bin/python
new=/venvs/new/bin/python``): one worker process per version (``cake_causal_conv1d_bench_worker.py``, role ``cake``)
and one for the baselines (``--sota-python``, role ``sota``), all on the same GPU.  The driver (this process, which
imports no package) builds the same row in every worker, runs the accuracy gate there and then times the arms in
paired groups, so every arm of a row is timed in the same minutes on the same GPU:

* arms: ``<label>_det`` and ``<label>_nondet`` per version (the package's ``cake_causal_conv1d_det`` /
  ``cake_causal_conv1d_nondet`` autograd entries) and every baseline of ``--sota``;
* phases (``--phases``): ``fwd`` (the training forward, its graph recorded), ``bwd`` (the backward of a recorded
  forward), ``step`` (both) and ``ckpt`` (``torch.utils.checkpoint`` around the entry, then its backward: two forwards
  and one backward);
* modes (``--modes``): ``eager`` (each iteration behind a spin kernel that hides its host time) and ``graph`` (the
  replay of a CUDA graph of one call; an arm and phase that cannot be captured is reported with the reason);
* timing (``cake_causal_conv1d_timer.py``): per iteration an L2 flush, then the CUPTI span from the first kernel
  start to the last kernel end of the iteration's own launches;
* groups (``--groups``): per phase and mode, ``G`` groups; group ``g`` times every arm for ``iters`` iterations in
  the arm list rotated left by ``g mod n`` (reversed when ``g // n`` is odd), so each arm leads and closes groups
  equally often.  Each arm's value is the median over the groups of the group's median; each ratio between two arms
  is the median over the groups of the ratio of their group medians, with its ``[min, max]``.

Accuracy gate (``--draws``, at least 32): every arm, Cake modes included, is judged on the row's shape by its error
against a float64 oracle beside the error of PyTorch BF16 over 32 random inputs (``cake_causal_conv1d_numerics.py``,
the training tests' judge).  A baseline counts as the best baseline of a comparison only when it passes; a ``det``
comparison takes only baselines that claim and show run-to-run bitwise outputs.  A Cake arm that fails the gate is
reported as failed.

Rows (``--rows``): ``owner`` (31 shapes, ``S`` 16384 to 500680, ``C`` 128 to 10240), ``pplx`` (the single-rank
performance matrix of ``bench_cake_causal_conv1d_nondet_train.py``) or ``custom`` (``--tokens`` x ``--channels``),
each packed as four documents of 0.37 / 0.29 / 0.21 / rest of ``S`` (each nudged off a multiple of 64); ``--shard
I/N`` takes every N-th row from the I-th.  Every row is appended to ``--jsonl`` as one JSON object when done.
``--summarize FILE...`` prints the comparison table of finished rows instead of running.

Example::

    python benchmarks/bench_cake_causal_conv1d_versions.py --cake old=/v/old/bin/python --cake new=/v/new/bin/python \\
        --sota-python /v/new/bin/python --rows owner --jsonl rows.jsonl
    python benchmarks/bench_cake_causal_conv1d_versions.py --summarize rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
WORKER = HERE / "cake_causal_conv1d_bench_worker.py"
WIDTH = 4
PERF_FRACTIONS = (0.37, 0.29, 0.21)
# The owner's 31 single-rank shapes (S, C).
OWNER_ROWS = (
    [(16384, 8192), (16384, 10240), (31292, 640), (31292, 1024), (32768, 8192), (32768, 10240)]
    + [(s, 512) for s in (239718, 245028, 249866, 258574, 259042, 259584, 261393)]
    + [(250340, c) for c in (128, 224, 256, 512, 576, 640, 768, 1024, 1160, 1280, 1344, 1536, 2048, 4096, 8192)]
    + [(262144, 1280), (262144, 1536), (500680, 640)]
)
PPLX_ROWS = [(s, 512) for s in (239718, 245028, 249866, 250340, 258574, 259042, 259584, 261393)] + [
    (250340, c) for c in (128, 256, 1024, 1160, 1280, 1344, 1536, 2048, 4096, 8192)
]
SOTA_ARMS = ("fla", "fla_gluon", "dao", "dao_det", "cudnn_fe", "cudnn_fe_det", "torch_compile")
PHASES = ("fwd", "bwd", "step", "ckpt")
# Iterations per arm and group: about TARGET_MS of GPU time, within [MIN_ITERS, MAX_ITERS].
TARGET_MS = 25.0
MIN_ITERS = 10
MAX_ITERS = 100


def doc_lengths(tokens: int) -> list[int]:
    lens = []
    for fraction in PERF_FRACTIONS:
        n = int(fraction * tokens)
        if n % 64 == 0:
            n -= 1
        lens.append(n)
    lens.append(tokens - sum(lens))
    if not all(n > 0 for n in lens):
        raise ValueError(f"S={tokens} is too short for the perf packing")
    return lens


class WorkerProcess:
    """A worker (``cake_causal_conv1d_bench_worker.py``) on its own interpreter; one request at a time."""

    def __init__(self, label: str, python: str, role: str, log_dir: Path) -> None:
        self.label, self.role = label, role
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / f"worker-{label}.log", "a")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        self.proc = subprocess.Popen(
            [python, "-u", str(WORKER), role],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
        )
        hello = self._read()
        if not hello.get("ok"):
            raise RuntimeError(f"worker {label} did not start: {hello}")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker {self.label} exited (rc {self.proc.poll()}); see its log")
        return json.loads(line)

    def call(self, cmd: str, **kwargs) -> dict:
        self.proc.stdin.write(json.dumps({"cmd": cmd, **kwargs}) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"worker {self.label} {cmd}: {reply.get('error')}")
        return reply

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def _median(values) -> float:
    return float(statistics.median(list(values)))


def group_order(arms: list[str], g: int) -> list[str]:
    n = len(arms)
    shift = g % n
    order = arms[shift:] + arms[:shift]
    if (g // n) % 2:
        order.reverse()
    return order


def bench_row(S: int, C: int, args, workers: dict[str, WorkerProcess], arms_of: dict[str, list[str]]) -> dict:
    lens = doc_lengths(S)
    route = {
        "weight_dtype": args.weight_dtype,
        "bias_dtype": None if args.bias == "none" else args.bias,
        "silu_grad_preactivation": args.silu_grad_preactivation,
    }
    t0 = time.time()
    result: dict = {"S": S, "C": C, "lens": lens, "route": route, "arms": {}, "timing": {}, "notes": []}
    # Arm name -> worker label; entries by full name.
    owner: dict[str, str] = {}

    def full(label: str, name: str) -> str:
        return f"{label}_{name}" if workers[label].role == "cake" else name

    gated: dict[str, dict] = {}
    if args.draws > 0:
        # The gate, one worker at a time (its row built, judged and freed), so the float64 oracle of a large row
        # never shares the GPU with the other workers' rows.
        t1 = time.time()
        for label, worker in workers.items():
            worker.call("row", S=S, C=C, lens=lens, route=route, arms=arms_of[label], seed=args.seed)
            reply = worker.call("gate", draws=args.draws, seed=args.gate_seed)
            worker.call("free")
            for name, entry in reply["arms"].items():
                gated[full(label, name)] = entry
        result["gate_s"] = time.time() - t1
    else:
        result["notes"].append("accuracy gate not run (--draws 0): no arm is eligible")
    for label, worker in workers.items():
        reply = worker.call("row", S=S, C=C, lens=lens, route=route, arms=arms_of[label], seed=args.seed)
        for name, entry in reply["arms"].items():
            owner[full(label, name)] = label
            result["arms"][full(label, name)] = {"local": name, "worker": label, **entry}
    for name, entry in gated.items():
        if name in result["arms"]:
            for key in ("gate", "eligible", "measured_deterministic"):
                if key in entry:
                    result["arms"][name][key] = entry[key]
            if "unsupported" in entry and "unsupported" not in result["arms"][name]:
                result["arms"][name]["unsupported"] = entry["unsupported"]
    result["setup_s"] = time.time() - t0 - result.get("gate_s", 0.0)
    runnable = [name for name, e in result["arms"].items() if "unsupported" not in e]
    t2 = time.time()
    for mode in args.modes:
        for phase in args.phases:
            cell: dict = {"skipped": {}, "arms": {}, "group_orders": []}
            result["timing"][f"{phase}/{mode}"] = cell
            try:
                time_cell(cell, phase, mode, runnable, owner, result, args, workers)
            except Exception as exc:  # noqa: BLE001 -- keep the other cells; a dead worker ends the row
                cell["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                traceback.print_exc()
                if any(w.proc.poll() is not None for w in workers.values()):
                    raise
            if mode == "graph":
                for worker in workers.values():
                    worker.call("drop_graphs")
    result["timing_s"] = time.time() - t2
    for worker in workers.values():
        worker.call("free")
    return result


def time_cell(cell, phase, mode, runnable, owner, result, args, workers) -> None:
    """Time every runnable arm of a row in one phase and mode, in ``--groups`` paired groups (module docstring)."""

    prepared: dict[str, dict] = {}
    for label, worker in workers.items():
        local = [result["arms"][n]["local"] for n in runnable if owner[n] == label]
        if not local:
            continue
        reply = worker.call("prepare", mode=mode, phases=[phase], arms=local)
        for key, value in reply["prepared"].items():
            name, _ = key.split("/")
            full = f"{label}_{name}" if worker.role == "cake" else name
            prepared[full] = value
    timed = [n for n in runnable if n in prepared and "error" not in prepared[n]]
    cell["skipped"] = {n: prepared[n]["error"] for n in runnable if n in prepared and "error" in prepared[n]}
    if not timed:
        cell["error"] = "no arm could be timed in this phase and mode"
        return
    for n in timed:
        if "spin_ms" in prepared[n]:
            cell["arms"].setdefault(n, {})["spin_ms"] = prepared[n]["spin_ms"]
    iters: dict[str, int] = {}
    groups: dict[str, list[float]] = {n: [] for n in timed}
    late: dict[str, int] = {n: 0 for n in timed}
    for g in range(args.groups):
        order = group_order(timed, g)
        if not order:
            break
        cell["group_orders"].append(order)
        for n in order:
            worker = workers[owner[n]]
            local = result["arms"][n]["local"]
            if n not in timed:
                continue
            if n not in iters:
                probe = worker.call("time", arm=local, phase=phase, mode=mode, iters=3)
                est = max(_median(probe["ms"]), 1e-3)
                iters[n] = int(min(MAX_ITERS, max(MIN_ITERS, TARGET_MS / est)))
            try:
                reply = worker.call("time", arm=local, phase=phase, mode=mode, iters=iters[n])
            except RuntimeError as exc:
                if worker.proc.poll() is not None:
                    raise
                cell.setdefault("arm_errors", {})[n] = str(exc)[:400]
                timed = [m for m in timed if m != n]
                continue
            groups[n].append(_median(reply["ms"]) * 1e3)
            late[n] += reply["late"]
            cell["method"] = reply["method"]
    for n in timed:
        values = groups[n]
        cell["arms"].setdefault(n, {}).update(
            us=_median(values), us_range=[min(values), max(values)], groups_us=values, iters=iters[n], late=late[n]
        )


# ---------------------------------------------------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------------------------------------------------


def paired_ratio(cell: dict, num: str, den: str) -> Optional[dict]:
    """The median over the groups of ``num / den`` (above 1: ``den`` faster) with its ``[min, max]``."""

    a, b = cell["arms"].get(num), cell["arms"].get(den)
    if not a or not b or "groups_us" not in a or "groups_us" not in b:
        return None
    ratios = [x / y for x, y in zip(a["groups_us"], b["groups_us"], strict=True)]
    return {"median": _median(ratios), "range": [min(ratios), max(ratios)]}


def best_baseline(result: dict, cell: dict, det_only: bool) -> Optional[str]:
    best, best_us = None, None
    for name, entry in result["arms"].items():
        if entry.get("worker") in result.get("cake_labels", ()) or not entry.get("eligible"):
            continue
        if det_only and not entry.get("measured_deterministic"):
            continue
        timing = cell["arms"].get(name)
        if timing and "us" in timing and (best_us is None or timing["us"] < best_us):
            best, best_us = name, timing["us"]
    return best


def table_rows(result: dict, old: str, new: str) -> list[dict]:
    rows = []
    for key, cell in result["timing"].items():
        phase, mode = key.split("/")
        for kind in ("det", "nondet"):
            o, n = f"{old}_{kind}", f"{new}_{kind}"
            if n not in cell["arms"]:
                continue
            best = best_baseline(result, cell, det_only=kind == "det")
            row = {
                "S": result["S"], "C": result["C"], "kind": kind, "phase": phase, "mode": mode,
                "new_us": cell["arms"][n]["us"],
                "new_gate": result["arms"][n].get("gate", {}).get("passed"),
                "old_us": cell["arms"].get(o, {}).get("us"),
                "old_over_new": paired_ratio(cell, o, n),
                "sota": best,
                "sota_library": None if best is None else result["arms"][best].get("library"),
                "sota_us": None if best is None else cell["arms"][best]["us"],
                "sota_over_new": None if best is None else paired_ratio(cell, best, n),
            }  # fmt: skip
            if kind == "nondet":
                row["det_us"] = cell["arms"].get(f"{new}_det", {}).get("us")
                row["det_over_nondet"] = paired_ratio(cell, f"{new}_det", n)
            rows.append(row)
    return rows


def _us(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _fmt(r: Optional[dict]) -> str:
    return "-" if r is None else f"{r['median']:.3f} [{r['range'][0]:.3f}, {r['range'][1]:.3f}]"


def summarize(paths: list[str], old: str, new: str) -> int:
    for path in paths:
        header = None
        for line in open(path):
            record = json.loads(line)
            if record.get("kind") == "header":
                header = record
                print(f"# {path}: {record['device']} sm_{''.join(map(str, record['capability']))} "
                      f"SMs {record['sm_count']} torch {record['torch']} CUDA {record['cuda']} route {record['route']}")
                for label, info in record["workers"].items():
                    print(f"#   {label}: {json.dumps(info.get('package') or info.get('libraries'))}")
                print("| S | C | mode | phase | timing | old us | new us | old/new | best baseline | baseline us "
                      "| baseline/new | det us | det/nondet |")
                print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                continue
            if record.get("kind") != "row":
                continue
            record["cake_labels"] = list(header["cake_labels"]) if header else [old, new]
            for r in table_rows(record, old, new):
                sota = "-" if r["sota"] is None else f"{r['sota']} ({r['sota_library']})"
                new_cell = f"{_us(r['new_us'])}{'' if r['new_gate'] else ' (gate FAIL)'}"
                print(
                    f"| {r['S']} | {r['C']} | {r['kind']} | {r['phase']} | {r['mode']} | {_us(r['old_us'])} "
                    f"| {new_cell} "
                    f"| {_fmt(r['old_over_new'])} | {sota} | {_us(r['sota_us'])} | {_fmt(r['sota_over_new'])} "
                    f"| {_us(r.get('det_us'))} | {_fmt(r.get('det_over_nondet'))} |"
                )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cake", action="append", default=[], help="LABEL=PYTHON of one installed package version")
    parser.add_argument("--sota-python", default=None, help="the interpreter of the baselines (default: none)")
    parser.add_argument("--sota", nargs="*", default=["all"], help=f"baselines: all, none or names of {SOTA_ARMS}")
    parser.add_argument("--modes-of-cake", nargs="*", default=["det", "nondet"], help="Cake modes to time")
    parser.add_argument("--rows", choices=("owner", "pplx", "custom"), default="owner")
    parser.add_argument("--tokens", type=int, nargs="*", default=None)
    parser.add_argument("--channels", type=int, nargs="*", default=None)
    parser.add_argument("--shard", default="0/1", help="I/N: every N-th row from the I-th")
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--bias", choices=("none", "float32", "bfloat16"), default="none")
    parser.add_argument("--silu-grad-preactivation", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--phases", nargs="*", default=["fwd", "bwd", "ckpt"], choices=PHASES)
    parser.add_argument("--modes", nargs="*", default=["eager", "graph"], choices=("eager", "graph"))
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--draws", type=int, default=32, help="accuracy gate draws (0: no gate, nothing eligible)")
    parser.add_argument("--seed", type=int, default=0, help="the timed row's input seed")
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--log-dir", default=None, help="worker logs (default: next to --jsonl)")
    parser.add_argument("--summarize", nargs="*", default=None, help="print the table of these JSONL files")
    parser.add_argument("--old", default="old", help="label of the previous version in --summarize")
    parser.add_argument("--new", default="new", help="label of the new version in --summarize")
    args = parser.parse_args(argv)
    if args.summarize:
        return summarize(args.summarize, args.old, args.new)
    if args.draws and args.draws < 32:
        parser.error("--draws: the gate judges at least 32 draws (or 0: no gate)")
    sota = list(SOTA_ARMS) if args.sota == ["all"] else ([] if args.sota == ["none"] else args.sota)
    unknown = [a for a in sota if a not in SOTA_ARMS]
    if unknown:
        parser.error(f"unknown baselines {unknown}")
    if args.rows == "owner":
        rows = list(OWNER_ROWS)
    elif args.rows == "pplx":
        rows = list(PPLX_ROWS)
    else:
        rows = [(s, c) for s in args.tokens or [] for c in args.channels or []]
    index, count = (int(v) for v in args.shard.split("/"))
    rows = rows[index::count]
    log_dir = Path(args.log_dir or (Path(args.jsonl).parent if args.jsonl else "."))
    workers: dict[str, WorkerProcess] = {}
    arms_of: dict[str, list[str]] = {}
    try:
        for spec in args.cake:
            label, _, python = spec.partition("=")
            workers[label] = WorkerProcess(label, python, "cake", log_dir)
            arms_of[label] = list(args.modes_of_cake)
        if sota and args.sota_python:
            workers["sota"] = WorkerProcess("sota", args.sota_python, "sota", log_dir)
            arms_of["sota"] = sota
        infos = {label: w.call("info") for label, w in workers.items()}
        first = next(iter(infos.values()))
        header = {
            "kind": "header",
            "device": first["device"],
            "capability": first["capability"],
            "sm_count": first["sm_count"],
            "torch": first["torch"],
            "cuda": first["cuda"],
            "gpu_timer": first["gpu_timer"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "environment": {k: os.environ.get(k) for k in (
                "CUDA_HOME", "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "CAKE_PPLX_CACHE_DIR")},
            "host": os.uname().nodename,
            "route": {"weight_dtype": args.weight_dtype, "bias": args.bias,
                      "silu_grad_preactivation": args.silu_grad_preactivation},
            "cake_labels": [label for label, w in workers.items() if w.role == "cake"],
            "workers": infos,
            "args": vars(args),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }  # fmt: skip
        out = open(args.jsonl, "a") if args.jsonl else None
        if out:
            out.write(json.dumps(header) + "\n")
            out.flush()
        print(f"device {first['device']} capability {first['capability']} timer {first['gpu_timer']} rows {len(rows)}")
        for label, info in infos.items():
            print(f"  {label}: {json.dumps(info.get('package') or info.get('libraries'))}", flush=True)
        for S, C in rows:
            t0 = time.time()
            try:
                result = bench_row(S, C, args, workers, arms_of)
            except Exception as exc:  # noqa: BLE001 -- keep the other rows; a dead worker ends the run
                traceback.print_exc()
                result = {"S": S, "C": C, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                if any(w.proc.poll() is not None for w in workers.values()):
                    if out:
                        out.write(json.dumps({"kind": "row", **result}) + "\n")
                    raise
            result["kind"] = "row"
            result["wall_s"] = time.time() - t0
            if out:
                out.write(json.dumps(result) + "\n")
                out.flush()
            if "error" in result:
                print(f"S={S} C={C} ERROR {result['error']}", flush=True)
                continue
            result["cake_labels"] = header["cake_labels"]
            labels = header["cake_labels"]
            gates = " ".join(
                f"{n}:{'pass' if e.get('gate', {}).get('passed') else 'FAIL'}" for n, e in result["arms"].items()
                if "unsupported" not in e
            )
            print(f"S={S} C={C} {result['wall_s']:.0f}s gate {gates}", flush=True)
            for n, e in result["arms"].items():
                if "unsupported" in e:
                    print(f"  {n}: unsupported ({e['unsupported'][:200]})", flush=True)
                elif not e.get("gate", {}).get("passed"):
                    print(f"  {n}: gate {(e.get('gate', {}).get('failures') or [e.get('gate', {}).get('error')])[:2]}")
            if len(labels) >= 2:
                for r in table_rows(result, labels[0], labels[-1]):
                    print(
                        f"  {r['kind']:6s} {r['phase']:4s} {r['mode']:5s} new {r['new_us']:9.2f} "
                        f"old/new {_fmt(r['old_over_new'])} best {r['sota']} "
                        f"{'-' if r['sota_us'] is None else round(r['sota_us'], 2)} "
                        f"sota/new {_fmt(r['sota_over_new'])}",
                        flush=True,
                    )
    finally:
        for w in workers.values():
            w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
