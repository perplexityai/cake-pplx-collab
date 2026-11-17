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

One worker process of ``bench_cake_causal_conv1d_versions.py``: it holds the arms of one Python environment on one
GPU and answers the driver's requests, one JSON object per line on stdin, one reply per line on the stdout it had at
start (library output goes to stderr).

Roles:

* ``cake``: the two modes of the ``cake_pplx`` causal conv1d package installed in this interpreter's environment,
  ``det`` (``cake_causal_conv1d_det``) and ``nondet`` (``cake_causal_conv1d_nondet``), through their autograd entries;
* ``sota``: the non-Cake baselines of ``cake_causal_conv1d_baselines.py`` (``SINGLE_RANK_ARMS``).

Requests (``cmd``): ``info``; ``row`` (build the row's tensors and arms, warm every phase up); ``gate`` (the accuracy
gate of every built arm, ``cake_causal_conv1d_numerics.py``); ``prepare`` (per arm and phase: the eager spin length,
or the CUDA graph of ``graph`` mode, or the reason it cannot be captured); ``time`` (``iters`` timed iterations of
one arm, phase and mode, ``cake_causal_conv1d_timer.py``); ``drop_graphs``; ``free``; ``quit``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path

PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_causal_conv1d_baselines as baselines  # noqa: E402
import cake_causal_conv1d_numerics as numerics  # noqa: E402
import cake_causal_conv1d_timer as timer_lib  # noqa: E402

WIDTH = 4
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PHASES = ("fwd", "bwd", "step", "ckpt")
CAKE_DISTRIBUTION = "cake-pplx-causal-conv1d"


def _package_identity() -> dict:
    """The installed causal conv1d package: version, location and a digest of its files."""

    import cake_pplx.cake_causal_conv1d_det_train as det

    root = Path(det.__file__).resolve().parent
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and ("causal_conv1d" in p.name or any("causal_conv1d" in part for part in p.relative_to(root).parts))
    )
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(root)).encode())
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    try:
        version = importlib.metadata.version(CAKE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "distribution": CAKE_DISTRIBUTION,
        "version": version,
        "location": str(root),
        "files": len(files),
        "sha256": digest.hexdigest(),
    }


class Worker:
    def __init__(self, role: str) -> None:
        self.role = role
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.timer = timer_lib.GpuTimer(self.device)
        self.side = torch.cuda.Stream(self.device)
        self.row: dict | None = None
        if role == "cake":
            det = importlib.import_module("cake_pplx.cake_causal_conv1d_det_train")
            nondet = importlib.import_module("cake_pplx.cake_causal_conv1d_nondet_train")
            self.modes = {
                "det": (det.is_cake_causal_conv1d_det_train_supported, det.cake_causal_conv1d_det),
                "nondet": (nondet.is_cake_causal_conv1d_nondet_train_supported, nondet.cake_causal_conv1d_nondet),
            }

    # -- requests ------------------------------------------------------------------------------------------------

    def info(self, _req: dict) -> dict:
        props = torch.cuda.get_device_properties(self.device)
        out = {
            "role": self.role,
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(self.device),
            "capability": list(torch.cuda.get_device_capability(self.device)),
            "sm_count": props.multi_processor_count,
            "l2_bytes": props.L2_cache_size,
            "gpu_timer": self.timer.method,
            "pid": os.getpid(),
        }
        if self.role == "cake":
            out["package"] = _package_identity()
        else:
            versions = {}
            for name in (
                "fla-core",
                "flash-linear-attention",
                "causal-conv1d",
                "nvidia-cudnn-frontend",
                "nvidia-cutlass-dsl",
                "megatron-core",
                "triton",
                "cupti-python",
            ):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            out["libraries"] = versions
        return out

    def row_cmd(self, req: dict) -> dict:
        self.free({})
        S, C, lens = int(req["S"]), int(req["C"]), [int(n) for n in req["lens"]]
        route = req["route"]
        weight_dtype = DTYPES[route["weight_dtype"]]
        bias_dtype = None if route["bias_dtype"] is None else DTYPES[route["bias_dtype"]]
        policy = route["silu_grad_preactivation"]
        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=self.device)
        x, w, b, dy = numerics.draw_inputs(S, C, weight_dtype, bias_dtype, int(req.get("seed", 0)), self.device)
        arms: dict[str, dict] = {}
        if self.role == "cake":
            for mode, (supported, entry_point) in self.modes.items():
                if mode not in req["arms"]:
                    continue
                entry: dict = {"name": mode, "eligible": False}
                arm = {"entry": entry}
                arms[mode] = arm
                if not supported(self.device):
                    entry["unsupported"] = f"the {mode} package has no exported route on this device"
                    continue

                def layer(*t, _f=entry_point):
                    return (_f(t[0], t[1], cu, t[2] if len(t) > 2 else None, silu_grad_preactivation=policy),)

                leaves = [t.detach().clone().requires_grad_(True) for t in (x, w, b) if t is not None]
                arm["calls"] = baselines.autograd_calls(layer, leaves, [dy])
                arm["run"] = baselines.gate_run(layer, {})
                entry.update(
                    determinism="deterministic" if mode == "det" else "non-deterministic",
                    feature={"match": "exact", "notes": []},
                    env={},
                )
        else:
            for name in req["arms"]:
                entry, calls, run = baselines.single_rank_arm(
                    name, x=x, w=w, b=b, dy=dy, cu=cu, lens=lens, cake_preactivation=policy
                )
                arms[name] = {"entry": entry, "calls": calls, "run": run}
        for arm in arms.values():
            calls = arm.get("calls")
            if calls is None:
                continue
            try:
                with baselines.environment(arm["entry"].get("env", {})):
                    calls["record"]()
                    for phase in PHASES:
                        calls[phase]()
                    calls["record"]()
                torch.cuda.synchronize(self.device)
            except Exception as exc:  # noqa: BLE001 -- a failing arm is reported, the row goes on
                arm["entry"]["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                arm["calls"] = arm["run"] = None
        self.row = {
            "S": S, "C": C, "cu": cu, "lens": lens, "weight_dtype": weight_dtype, "bias_dtype": bias_dtype,
            "arms": arms, "graphs": {}, "spins": {},
        }  # fmt: skip
        return {"arms": {name: _jsonable(arm["entry"]) for name, arm in arms.items()}}

    def gate(self, req: dict) -> dict:
        row = self.row
        runs = {name: arm["run"] for name, arm in row["arms"].items() if arm.get("run") is not None}
        verdicts = numerics.gate(
            runs, S=row["S"], C=row["C"], cu=row["cu"], weight_dtype=row["weight_dtype"],
            bias_dtype=row["bias_dtype"], draws=int(req.get("draws", numerics.judge_lib.DRAWS)),
            seed=int(req.get("seed", 0)),
        )  # fmt: skip
        for name, verdict in verdicts.items():
            entry = row["arms"][name]["entry"]
            baselines.apply_gate(entry, verdict)
            if self.role == "cake":
                entry["measured_deterministic"] = name == "det" and bool(verdict.get("run_to_run_bitwise"))
        torch.cuda.empty_cache()
        return {"arms": {name: _jsonable(arm["entry"]) for name, arm in row["arms"].items()}}

    def prepare(self, req: dict) -> dict:
        """Per requested ``(arm, phase)``: the eager spin, or the CUDA graph (``mode == "graph"``).  A graph is
        captured from a second call table of the arm on fresh leaves whose every call runs on the capture stream (its
        forward and backward, so the backward's kernels and the leaves' gradient accumulation are on that stream)."""

        row, mode, out = self.row, req["mode"], {}
        for name in req["arms"]:
            arm = row["arms"].get(name)
            if arm is None or arm.get("calls") is None:
                continue
            calls, env = arm["calls"], arm["entry"].get("env", {})
            if mode == "graph" and "graph_calls" not in arm:
                try:
                    self.side.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(self.side), baselines.environment(env):
                        fresh = [leaf.detach().clone().requires_grad_(True) for leaf in calls.leaves]
                        gcalls = baselines.autograd_calls(calls.layer, fresh, calls.dys)
                        gcalls["record"]()
                        for phase in PHASES:
                            gcalls[phase]()
                    torch.cuda.synchronize(self.device)
                    arm["graph_calls"] = gcalls
                except Exception as exc:  # noqa: BLE001 -- recorded: no graph mode for this arm
                    torch.cuda.synchronize(self.device)
                    arm["graph_calls"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            for phase in req["phases"]:
                key = f"{name}/{phase}"
                try:
                    with baselines.environment(env):
                        if mode == "eager":
                            row["spins"][key] = self.timer.spin_ms(calls[phase])
                            out[key] = {"spin_ms": row["spins"][key]}
                            continue
                        gcalls = arm["graph_calls"]
                        if isinstance(gcalls, str):
                            raise RuntimeError(f"no call table on the capture stream: {gcalls}")
                        if phase == "bwd":
                            with torch.cuda.stream(self.side):
                                gcalls["record"]()
                        row["graphs"][key] = timer_lib.capture_graph(gcalls[phase], self.side)
                        out[key] = {"graph": True}
                except Exception as exc:  # noqa: BLE001 -- recorded: this arm and phase have no such mode
                    torch.cuda.synchronize(self.device)
                    out[key] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        torch.cuda.synchronize(self.device)
        return {"prepared": out}

    def time(self, req: dict) -> dict:
        row, name, phase, mode = self.row, req["arm"], req["phase"], req["mode"]
        arm = row["arms"][name]
        key = f"{name}/{phase}"
        with baselines.environment(arm["entry"].get("env", {})):
            if mode == "eager":
                fn, spin = arm["calls"][phase], row["spins"].get(key)
            else:
                fn, spin = row["graphs"][key].replay, None
            result = self.timer.times(fn, int(req["iters"]), spin)
        return result

    def drop_graphs(self, _req: dict) -> dict:
        if self.row is not None:
            self.row["graphs"].clear()
            for arm in self.row["arms"].values():
                arm.pop("graph_calls", None)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return {}

    def free(self, _req: dict) -> dict:
        self.row = None
        import gc

        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return {}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    role = sys.argv[1] if len(sys.argv) > 1 else "cake"
    try:
        worker = Worker(role)
    except Exception as exc:  # noqa: BLE001 -- reported to the driver
        PROTOCOL.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        return 1
    PROTOCOL.write(json.dumps({"ok": True, "ready": role}) + "\n")
    handlers = {
        "info": worker.info,
        "row": worker.row_cmd,
        "gate": worker.gate,
        "prepare": worker.prepare,
        "time": worker.time,
        "drop_graphs": worker.drop_graphs,
        "free": worker.free,
    }
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        try:
            reply = {"ok": True, **handlers[req["cmd"]](req)}
        except Exception as exc:  # noqa: BLE001 -- the driver decides
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        PROTOCOL.write(json.dumps(reply) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
