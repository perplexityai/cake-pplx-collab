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

Benchmark of the Cake mHC training head (``cake_pplx.cake_mhc_head_det_train`` and
``cake_pplx.cake_mhc_head_nondet_train``): the stream mean ``Y = mean_i X[:, i, :]`` of BF16 ``X [T, 4, H]`` and its
gradient ``dX[:, i] = BF16(dY / 4)``, against PyTorch's own ``X.mean(dim=1)`` (the function the head computes) and
its ``torch.compile`` versions.

Arms: ``cake_head_det`` / ``cake_head_nondet`` (the explicit entry points, caller-owned outputs; ``--aa`` adds a
second instance of each, the noise band), ``torch_eager`` (``X.mean(dim=1)`` with autograd), ``compile_default`` and
``compile_max_autotune`` (``torch.compile`` of the same expression, ``fullgraph``, static shapes,
``max-autotune-no-cudagraphs`` for the second).  Operations: ``fwd`` (``Y``; the PyTorch arms record the autograd
graph), ``bwd`` (``dX`` written; the PyTorch arms' backward of a recorded forward into a cleared ``.grad``) and
``bwd_acc`` (``dX`` added into the existing gradient: the head's ``accumulate=True``, autograd's accumulation into an
existing ``.grad``).  The comparisons are fixed: every operation against ``torch_eager`` and, in a second column,
``compile_max_autotune``, the same arms on every architecture.

Timing, modes (eager launches and CUDA graph replays), groups, the cold L2 and the JSON record are those of the mHC
benchmarks (``cake_mhc_head_harness.py`` beside this file: this family's copy of their timer and helpers).  Accuracy
before timing: the family's judge (``tests/cake_naive.py``) over ``--judge-draws`` (32) draws of ``(--judge-tokens, H)``
per hidden size holds every arm to the naive tier (PyTorch BF16, i.e. ``torch_eager`` itself) against a float64 oracle;
on every row each arm's outputs are also compared bit for bit with ``torch_eager``'s (recorded only; the contract makes
no bit promise against torch).  Each arm's determinism is measured by ``--det-runs`` (5) runs of the same call compared
bit for bit.

Example::

    python benchmarks/bench_cake_mhc_head_train.py --json head.json
    python benchmarks/bench_cake_mhc_head_train.py --arms cake_head_det,cake_head_nondet --aa --judge-draws 0 \\
        --label old --json head-old.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch

import cake_mhc_head_harness as harness

_HERE = Path(__file__).resolve().parent

STREAMS = 4
BF16, F64 = torch.bfloat16, torch.float64
OPS = ("fwd", "bwd", "bwd_acc")
OUTPUTS = ("Y", "dX")
#: The fixed comparisons: (name, operations, arm).
COMPARISONS = (
    ("torch", OPS, "torch_eager"),
    ("compile", OPS, "compile_max_autotune"),
)
DEFAULT_ARMS = "cake_head_det,cake_head_nondet,torch_eager,compile_default,compile_max_autotune"


def load_judge():
    path = _HERE.parent / "tests" / "cake_naive.py"
    spec = importlib.util.spec_from_file_location("cake_mhc_head_benchmark_naive_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(T: int, hidden: int, *, seed: int, device) -> dict:
    """``X ~ N(0, 1)`` BF16 ``[T, 4, H]`` and ``dY ~ N(0, 1)`` BF16 ``[T, H]`` from a seeded device generator."""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(T, STREAMS, hidden, generator=g, device=device).to(BF16)
    dY = torch.randn(T, hidden, generator=g, device=device).to(BF16)
    return {"X": X, "dY": dY, "T": T, "hidden": hidden}


def oracle(i: dict) -> dict:
    """float64 ``Y`` and ``dX``."""
    Y = i["X"].to(F64).mean(dim=1)
    dX = (i["dY"].to(F64) / STREAMS).unsqueeze(1).expand(-1, STREAMS, -1).contiguous()
    return {"Y": Y, "dX": dX}


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class CakeHead(harness.Arm):
    cake = True
    ops = OPS

    def __init__(self, mode: str, label: Optional[str] = None) -> None:
        self.mode = mode
        self.name = label or f"cake_head_{mode}"
        self.library = f"cake_pplx.cake_mhc_head_{mode}_train"
        self.deterministic = mode == "det"
        self.determinism = (
            "deterministic by contract: run-to-run bitwise" if mode == "det"
            else "non-deterministic mode: the contract promises no run-to-run bits (elementwise, fixed-order kernels)"
        )
        self.mod = importlib.import_module(self.library)

    def f(self, name: str):
        return getattr(self.mod, f"cake_mhc_head_{self.mode}_train_{name}")

    def available(self, device, hidden):
        ok = bool(getattr(self.mod, f"is_cake_mhc_head_{self.mode}_train_supported")(device, hidden))
        return ok, "" if ok else f"the package exports no head route for hidden {hidden} on this device"

    def versions(self) -> dict:
        backend = importlib.import_module(f"cake_pplx._backends.cake_mhc_head_{self.mode}_train")
        d = Path(backend.__file__).resolve().parent
        out = {"distribution": "cake-pplx-mhc-head", "version": harness.dist_version("cake-pplx-mhc-head"),
               "module": str(Path(self.mod.__file__).resolve()), "backend": str(d),
               "build_json_sha256": _sha256(d / "build.json"), "cake_jit_sha256": _sha256(d / "cake_jit.py")}
        try:
            out["generator"] = json.loads((d / "build.json").read_text()).get("generator")
        except (OSError, ValueError) as exc:
            out["generator"] = f"unreadable: {exc}"
        return out

    def run_once(self, i: dict) -> dict:
        return {"Y": self.f("forward")(i["X"]), "dX": self.f("backward")(i["dY"])}

    def make(self, i: dict, op: str) -> Callable[[], None]:
        T, H, dev = i["T"], i["hidden"], i["X"].device
        X, dY = i["X"], i["dY"]
        Y = torch.empty(T, H, dtype=BF16, device=dev)
        dX = torch.empty(T, STREAMS, H, dtype=BF16, device=dev)
        fwd, bwd = self.f("forward"), self.f("backward")
        if op == "fwd":
            return lambda: fwd(X, out=Y)
        if op == "bwd":
            return lambda: bwd(dY, dX=dX)
        if op == "bwd_acc":
            bwd(dY, dX=dX)
            return lambda: bwd(dY, dX=dX, accumulate=True)
        raise ValueError(op)


def _mean(X):
    return X.mean(dim=1)


class TorchHead(harness.Arm):
    """PyTorch's ``X.mean(dim=1)`` with autograd: eager, or through ``torch.compile`` (``mode``)."""

    ops = OPS

    def __init__(self, name: str, compile_mode: Optional[str] = None, compiled: bool = False) -> None:
        self.name = name
        self.compiled = compiled
        self.compile_mode = compile_mode
        if compiled:
            self.library = (f"torch.compile of X.mean(dim=1) (mode={compile_mode or 'default'}, fullgraph, static "
                            "shapes, no buffer donation)")
            self.determinism = "not declared (measured by the benchmark)"
        else:
            self.library = "PyTorch eager X.mean(dim=1) with autograd"
            self.determinism = "not declared by PyTorch (measured by the benchmark)"
        self._fn = None
        self._count = 0

    def versions(self) -> dict:
        out = {"torch": torch.__version__, "torch_cuda": torch.version.cuda}
        if self.compiled:
            import triton

            out.update({"triton": triton.__version__, "mode": self.compile_mode or "default"})
        return out

    def prepare_shape(self, i: dict, timer=None) -> dict:
        if self.compiled:
            import torch._functorch.config as functorch_config

            functorch_config.donated_buffer = False
            self._count += 1
            kwargs = {"fullgraph": True, "dynamic": False}
            if self.compile_mode:
                kwargs["mode"] = self.compile_mode
            self._fn = torch.compile(harness._fresh_function(_mean, f"{self.name}_{self._count}"), **kwargs)
        else:
            self._fn = _mean
        return {}

    def forward(self, X):
        if self._fn is None:
            self.prepare_shape({})
        return self._fn(X)

    def run_once(self, i: dict) -> dict:
        X = i["X"].detach().requires_grad_(True)
        Y = self.forward(X)
        torch.autograd.backward(Y, i["dY"])
        return {"Y": Y.detach(), "dX": X.grad}

    def make(self, i: dict, op: str) -> Callable[[], None]:
        X = i["X"].detach().requires_grad_(True)
        dY = i["dY"]
        if op == "fwd":
            return lambda: self.forward(X)
        Y = self.forward(X)
        if op == "bwd":

            def call() -> None:
                X.grad = None
                torch.autograd.backward(Y, dY, retain_graph=True)

            return call
        if op == "bwd_acc":
            torch.autograd.backward(Y, dY, retain_graph=True)
            return lambda: torch.autograd.backward(Y, dY, retain_graph=True)
        raise ValueError(op)


def make_arms(names: list, aa: bool) -> list:
    arms = []
    for name in names:
        if name in ("cake_head_det", "cake_head_nondet"):
            mode = name.rsplit("_", 1)[1]
            arms.append(CakeHead(mode))
            if aa:
                arms.append(CakeHead(mode, f"{name}~aa"))
        elif name == "torch_eager":
            arms.append(TorchHead("torch_eager"))
        elif name == "compile_default":
            arms.append(TorchHead("compile_default", None, compiled=True))
        elif name == "compile_max_autotune":
            arms.append(TorchHead("compile_max_autotune", "max-autotune-no-cudagraphs", compiled=True))
        else:
            raise ValueError(f"unknown arm {name}")
    return arms


def judge_arms(arms: list, hidden: int, *, device, judge, draws: int, tokens: int, seed: int) -> dict:
    """Every arm against the float64 oracle beside the naive BF16 tier (``X.mean(dim=1)`` and its autograd
    gradient in BF16), over ``draws`` draws."""
    judges = {a.name: judge.TierJudge(draws) for a in arms}
    errors: dict = {}
    contract: dict = {a.name: [] for a in arms}
    for j in range(draws):
        s = judge.draw_seed(seed, j)
        judge.seed_all(s)
        i = make_inputs(tokens, hidden, seed=s, device=device)
        want = oracle(i)
        X = i["X"].detach().requires_grad_(True)
        Yn = X.mean(dim=1)
        torch.autograd.backward(Yn, i["dY"])
        naive = {"Y": Yn.detach(), "dX": X.grad}
        for arm in arms:
            if arm.name in errors:
                continue
            try:
                outs = arm.run_once(i)
            except Exception as exc:  # noqa: BLE001
                errors[arm.name] = f"{type(exc).__name__}: {exc}"[:400]
                continue
            for k in OUTPUTS:
                msg = judge.contract(k, outs[k], want[k])
                if msg:
                    contract[arm.name].append(f"draw {j}: {msg}")
                    continue
                judges[arm.name].add(k, j, outs[k], want[k], {"bf16": naive[k]})
    out = {}
    for arm in arms:
        if arm.name in errors:
            out[arm.name] = {"error": errors[arm.name], "meets_contract": False}
            continue
        fails = contract[arm.name] + judges[arm.name].failures()
        out[arm.name] = {"meets_contract": not fails, "failures": fails[:20], "verdicts": judges[arm.name].verdicts(),
                         "tier_lines": judges[arm.name].lines(f"judge[{arm.name}] H={hidden} T={tokens}")}
    return out


def run_row(hidden: int, tokens: int, arms: list, args, timer, device, eligible: dict) -> dict:
    row: dict = {"hidden": hidden, "tokens": tokens, "seed": args.seed + 7919 * hidden + tokens, "arms": {}, "ops": {}}
    t0 = time.time()
    i = make_inputs(tokens, hidden, seed=row["seed"], device=device)
    live = []
    for arm in arms:
        ok, note = arm.available(device, hidden)
        rec = row["arms"][arm.name] = {"available": ok, "note": note}
        if not ok:
            print(f"H={hidden} T={tokens} {arm.name}: unsupported: {note}", flush=True)
            continue
        try:
            arm.prepare_shape(i, timer)
            live.append(arm)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "prepare: " + harness._err(exc, f"{arm.name} prepare")
    # bit-for-bit comparison with torch_eager (recorded, not part of the contract) and the determinism runs
    ref = None
    try:
        ref = TorchHead("ref").run_once(i)
    except Exception as exc:  # noqa: BLE001
        row["reference_error"] = harness._err(exc, "reference")
    for arm in list(live):
        rec = row["arms"][arm.name]
        try:
            outs = arm.run_once(i)
            if ref is not None:
                rec["bitwise_vs_torch_eager"] = {
                    k: bool(torch.equal(harness._bits(outs[k]), harness._bits(ref[k]))) for k in OUTPUTS}
            del outs
            if args.det_runs > 1 and not arm.name.endswith("~aa"):
                rec["determinism"] = harness.determinism_runs(arm, i, args.det_runs)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "check: " + harness._err(exc, f"{arm.name} check")
            live.remove(arm)
        harness._free()
    del ref
    for op in args.ops:
        participants = [a for a in live if op in a.ops]
        for mode in args.modes:
            res = harness.time_op(participants, i, op, mode, args, timer, device)
            res["comparisons"] = comparisons(arms, row, res, op, hidden, eligible)
            row["ops"].setdefault(op, {})[mode] = res
            cells = [f"{a.name} {res[a.name]['active_us']['median']:.2f}" if "active_us" in res.get(a.name, {})
                     else f"{a.name} ERR" for a in participants]
            comp = "; ".join(f"{k} {v['arm']} {v['status'][:30]}" + (f" x{v['speedup']:.2f}" if "speedup" in v else "")
                             for k, v in res["comparisons"].items())
            print(f"H={hidden} T={tokens} {op} {mode} | " + " | ".join(cells) + (f" || {comp}" if comp else ""),
                  flush=True)
    row["seconds"] = round(time.time() - t0, 1)
    del i
    harness._free()
    return row


def comparisons(arms, row, res, op, hidden, eligible) -> dict:
    if not any(not a.cake for a in arms):
        return {}
    names = {a.name for a in arms}
    cake = next((a.name for a in arms if a.cake and "active_us" in res.get(a.name, {})), None)
    out = {}
    for cname, ops, aname in COMPARISONS:
        if op not in ops:
            continue
        cell = {"arm": aname, "label": aname, "reference": cake, "judge": eligible.get((hidden, aname))}
        rec, r = row["arms"].get(aname), res.get(aname, {})
        if aname not in names:
            cell["status"] = "not run: the arm was not selected"
        elif not rec or not rec.get("available"):
            cell["status"] = "unsupported: " + ((rec or {}).get("note") or "not available")
        elif rec.get("error"):
            cell["status"] = "error: " + rec["error"]
        elif "active_us" not in r:
            cell["status"] = "error: " + str(r.get("error") or "not timed")
        elif eligible.get((hidden, aname)) is False:
            cell["status"] = "different contract: the arm misses the naive-tier judge"
        else:
            cell["status"] = "ok"
        if "active_us" in r and cake:
            a_us, c_us = r["active_us"]["median"], res[cake]["active_us"]["median"]
            cell.update({"arm_us": a_us, "cake_us": c_us, "speedup": a_us / c_us,
                         "paired": (r.get(f"over_{cake}") or {}).get("median")})
        out[cname] = cell
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=DEFAULT_ARMS)
    p.add_argument("--aa", action="store_true")
    p.add_argument("--shapes", nargs="*", default=None, help="HxT items, e.g. 4096x1024,7168x65536")
    p.add_argument("--hidden", type=int, nargs="*", default=None)
    p.add_argument("--tokens", type=int, nargs="*", default=None)
    p.add_argument("--ops", default=",".join(OPS))
    p.add_argument("--modes", default="eager,graph")
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--calls", type=int, default=32)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--dry", type=int, default=2)
    p.add_argument("--judge-draws", type=int, default=32)
    p.add_argument("--judge-tokens", type=int, default=1024)
    p.add_argument("--det-runs", type=int, default=5)
    p.add_argument("--no-kernel-names", dest="kernel_names", action="store_false")
    p.add_argument("--timer", default="auto", choices=("auto", "cupti", "profiler"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    args.ops = [o for o in args.ops.split(",") if o]
    args.modes = [m for m in args.modes.split(",") if m]
    device = torch.device("cuda", torch.cuda.current_device())
    timer = harness.Timer(device, args.timer, args.calls, args.dry)
    try:  # a timer that cannot record fails the run here, not per arm
        timer.times(lambda: timer.flush.zero_(), calls=2, dry=0)
    except Exception as exc:  # noqa: BLE001
        print(f"TIMER SELF-TEST FAILED ({timer.backend}): {harness._err(exc)}", file=sys.stderr, flush=True)
        return 4
    judge = load_judge()
    arms = make_arms([a for a in args.arms.split(",") if a], args.aa)
    shapes = harness.parse_shapes(args)
    record: dict = {
        "label": args.label, "argv": sys.argv, "host": socket.gethostname(),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "family": "mhc_head", "device": harness.device_record(device), "software": harness.software_record(timer),
        "protocol": {"groups": args.groups, "calls": args.calls, "warmup": args.warmup, "dry": args.dry,
                     "ops": args.ops, "modes": args.modes, "flush_bytes": timer.flush_bytes,
                     "statistic": "median of group medians of per-call CUPTI active time (us)",
                     "judge_draws": args.judge_draws, "judge_tokens": args.judge_tokens, "det_runs": args.det_runs},
        "comparisons": [{"name": n, "ops": list(o), "arm": a} for n, o, a in COMPARISONS],
        "arms": {a.name: a.describe() for a in arms}, "versions": {}, "judge": {}, "rows": [],
    }
    for arm in arms:
        try:
            record["versions"][arm.name] = arm.versions()
        except Exception as exc:  # noqa: BLE001
            record["versions"][arm.name] = {"error": harness._err(exc)}
        print(f"arm {arm.name}: {arm.library} {record['versions'][arm.name]}", flush=True)

    def dump() -> None:
        if args.json:
            record["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            with open(args.json + ".tmp", "w") as handle:
                json.dump(record, handle, indent=1, default=str)
            os.replace(args.json + ".tmp", args.json)

    eligible: dict = {}
    if args.judge_draws > 0:
        for hidden in sorted({h for h, _ in shapes}):
            cands = []
            for arm in arms:
                if arm.name.endswith("~aa"):
                    continue
                ok, note = arm.available(device, hidden)
                if not ok:
                    record["judge"].setdefault(str(hidden), {})[arm.name] = {"unavailable": note}
                    print(f"judge H={hidden} {arm.name}: unsupported: {note}", flush=True)
                    continue
                arm.prepare_shape(make_inputs(args.judge_tokens, hidden, seed=args.seed, device=device))
                cands.append(arm)
            verdicts = judge_arms(cands, hidden, device=device, judge=judge, draws=args.judge_draws,
                                  tokens=args.judge_tokens, seed=args.seed)
            record["judge"].setdefault(str(hidden), {}).update(verdicts)
            for name, v in verdicts.items():
                eligible[(hidden, name)] = bool(v.get("meets_contract"))
                print(f"judge H={hidden} T={args.judge_tokens} {name}: "
                      f"{'meets the contract' if v.get('meets_contract') else 'DIFFERENT CONTRACT'} "
                      f"{v.get('error') or ''} {'; '.join(v.get('failures', [])[:3])}", flush=True)
                for line in v.get("tier_lines", []):
                    print("  " + line, flush=True)
            dump()
    for hidden, tokens in shapes:
        try:
            record["rows"].append(run_row(hidden, tokens, arms, args, timer, device, eligible))
        except Exception as exc:  # noqa: BLE001
            record["fatal"] = {"hidden": hidden, "tokens": tokens, "error": harness._err(exc, "row"),
                               "trace": traceback.format_exc(limit=8)[-3000:]}
            dump()
            return 3
        dump()
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
