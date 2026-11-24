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

Benchmark of the Cake packed KDA chunked training kernels in both numerical modes against the strongest non-Cake
baselines on one GPU, on the same inputs (``cake_kda_baselines.py``):

* ``cake_det``: the deterministic package (``cake_pplx.cake_kda_det_train.cake_kda_det``);
* ``cake_nondet``: the non-deterministic package (``cake_pplx.cake_kda_nondet_train.cake_kda_nondet``);
* ``cudnn_bi1``: the cuDNN frontend's KDA (plan ``kda_frost``, ``batch_invariant=True``,
  ``checkpoint_every_n_tokens=16``), the baseline of ``cake_det``;
* ``cudnn_bi0``: the same with ``batch_invariant=False``, the baseline of ``cake_nondet``;
* ``fla``: flash-linear-attention's ``chunk_kda``, a reference column.

The cuDNN frontend and flash-linear-attention are optional: an arm whose library does not import (or that refuses
the row) is reported as unsupported with the reason, and the Cake arms run on.  The script prints each baseline's
library and version.

Timed boundaries, every arm through autograd on the same leaves (``q, k, v, g, beta`` and in the raw gate mode
``A_log, dt_bias``) and the same output gradient:

* ``fwd``: the training forward, its autograd graph recorded and dropped (each Cake arm on its own workspace, the
  saved set written);
* ``bwd``: the backward of one recorded forward (``torch.autograd.grad(..., retain_graph=True)``);
* ``step``: ``fwd`` + ``bwd``;
* ``ckpt``: ``torch.utils.checkpoint(..., use_reentrant=False)`` around the forward, then the backward: two
  forwards and the backward.

Timing: eager (no CUDA graph).  The figure of one call is the summed GPU time of the CUDA activity (kernels,
memsets, copies) whose start lies inside the call's profiler range, from CUPTI through the torch profiler, with the
L2 flushed (a write of twice its size) and the device synchronized before every call; host gaps between kernels do
not count (``bench_cake_kda_nondet_train.bench_gpu_time``).  Each measurement takes about 25 ms of untimed calls,
then at least 10 timed calls and 100 ms; its value is the median.  After ``--warmup`` untimed calls of every
boundary, the arms run interleaved for ``--rounds`` rounds, each timing every arm and boundary once, the arm order
reversed on every other round; a row reports per arm and boundary the median of the round medians and their
``[min, max]``.  Speedups are ``baseline time / Cake time`` (above 1: Cake is faster): ``cudnn_bi1 / cake_det``,
``cudnn_bi0 / cake_nondet``, ``fla / cake_det`` and ``fla / cake_nondet``; ``cake_det / cake_nondet`` is the
deterministic package against the non-deterministic one.

Each row also compares every arm's ``out`` and gradients with ``cake_det``'s on the row's inputs (relative L2
difference, one untimed call): an adapter that drops a gate term or a norm shows there.  Accuracy is the tests' job
(``tests/experimental/test_cake_kda_*_naive.py``).

Rows: every ``--shapes`` entry (``SxN``: ``N`` unequal documents of ``S`` tokens in total, as the other KDA
benchmarks split them; or explicit document lengths ``L1,L2,...``) with every ``--heads`` value, in every
``--gate`` form (``g``: the activated FP32 gate; ``raw``: the raw gate activated in the kernels, lower bound -5)
and ``--l2n`` mode (``0``: normalized ``q`` / ``k``; ``1``: raw rows normalized in the kernels).  ``--rows`` adds
explicit ``shape:heads`` rows.  The default rows are the KDA benchmarks' ``16384x3`` and ``65536x3`` at 4 heads,
one long pack of three documents (``16313,221356,2049``) at 4 heads and ``65536x3`` at 64 heads.

Official numbers: run one timing process per GPU node (other processes on the node share its power, clocks,
memory bandwidth and host).  The JSON records the GPU, its architecture and SM count, the driver, CUDA and the
library versions, the timer, the warm-up and rounds, and every round's medians.

Example::

    python benchmarks/bench_cake_kda_baselines.py --json kda_baselines.json
    python benchmarks/bench_cake_kda_baselines.py --shapes 16384x3 --heads 4 --gate raw --l2n 1 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cake_kda_baselines as baselines  # noqa: E402
from bench_cake_kda_nondet_train import (  # noqa: E402
    bench_gpu_time,
    split_docs,
    _timer,
)
from cake_pplx._backends.cake_kda_det_train import (  # noqa: E402
    cake_backend as det_backend,
)
from cake_pplx._backends.cake_kda_nondet_train import (  # noqa: E402
    cake_backend as nd_backend,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    D,
    LOWER_BOUND,
    make_inputs,
    make_raw_inputs,
)

CAKE_ARMS = ("cake_det", "cake_nondet")
ARMS = CAKE_ARMS + baselines.SINGLE_GPU_ARMS
BOUNDARIES = ("fwd", "bwd", "step", "ckpt")
DEFAULT_SHAPES = ("16384x3:4", "65536x3:4", "16313,221356,2049:4", "65536x3:64")
# (baseline, Cake arm) of the speedup columns: baseline time / Cake time
SPEEDUPS = (
    ("cudnn_bi1", "cake_det"),
    ("cudnn_bi0", "cake_nondet"),
    ("fla", "cake_det"),
    ("fla", "cake_nondet"),
    ("cake_det", "cake_nondet"),
)
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")


def shape_lengths(shape: str) -> tuple[int, ...]:
    """``SxN`` -> :func:`split_docs` ``(S, N)``; ``L1,L2,...`` -> the lengths."""
    if "x" in shape:
        S, N = shape.split("x")
        return split_docs(int(S), int(N))
    return tuple(int(x) for x in shape.split(","))


def raw_qk(S: int, H: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (not normalized) BF16 ``q`` / ``k`` ``[S, H, 128]`` of a few magnitudes per token (the caller's
    projections), on the host, as the non-deterministic benchmark draws them."""
    gen = torch.Generator().manual_seed(7817 + seed + 31 * S + H)
    mag = torch.exp(torch.randn(S, H, 1, generator=gen) * 1.5)
    q = (torch.randn(S, H, D, generator=gen) * mag).to(torch.bfloat16)
    k = (torch.randn(S, H, D, generator=gen) * mag.flip(0)).to(torch.bfloat16)
    return q, k


def make_row(lengths, H: int, gate: str, l2n: bool, device: torch.device) -> dict:
    inputs = {"g": make_inputs, "raw": make_raw_inputs}[gate](lengths, H, device)
    if l2n:
        q, k = raw_qk(int(sum(lengths)), H)
        inputs["q"], inputs["k"] = q.to(device), k.to(device)
    return inputs


def cake_arm(
    arm: str, inputs: dict, l2n: bool
) -> tuple[Callable[..., torch.Tensor], dict]:
    """``(call, info)`` of a Cake arm on its own loaded workspace: ``call(*leaves)`` returns ``out`` through
    autograd."""
    raw = "A_log" in inputs
    lengths = inputs["lengths"]
    S, N, H = int(sum(lengths)), len(lengths), inputs["q"].shape[-2]
    mode = {"gate": "raw" if raw else "g", "l2n": "fused" if l2n else "off"}
    if arm == "cake_det":
        ws = det_backend.CakeKDADetWorkspace.for_problem(
            S, N, H, inputs["q"].device, **mode
        )
        fn = det_backend.cake_kda_det
    else:
        ws = nd_backend.CakeKDANondetWorkspace.for_problem(
            S, N, H, inputs["q"].device, **mode
        )
        fn = nd_backend.cake_kda_nondet
    ws.load()
    cu = inputs["cu_seqlens"]

    def call(q, k, v, g, beta, *params):
        kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
        if raw:
            kw.update(
                use_gate_in_kernel=True,
                A_log=params[0],
                dt_bias=params[1],
                lower_bound=LOWER_BOUND,
            )
        return fn(q, k, v, g, beta, cu, workspace=ws, **kw)

    info = {
        "entry": f"cake_pplx.{'cake_kda_det_train.cake_kda_det' if arm == 'cake_det' else 'cake_kda_nondet_train.cake_kda_nondet'}",
        "workspace": "for_problem, save_mode light",
        **mode,
        "recurrence": getattr(getattr(ws, "plan", None), "recur", None),
        "segments_per_doc": getattr(getattr(ws, "plan", None), "P", None),
    }
    return call, info


def boundaries(
    call: Callable[..., torch.Tensor], inputs: dict
) -> dict[str, Callable[[], object]]:
    """The timed boundaries of one arm on fresh leaves of the row (see the module docstring)."""
    names = baselines.leaf_names(inputs)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
    dout = inputs["dout"]
    retained = {"o": call(*leaves)}

    def fwd() -> object:
        return call(*leaves)

    def bwd() -> object:
        return torch.autograd.grad(retained["o"], leaves, dout, retain_graph=True)

    def step() -> object:
        return torch.autograd.grad(call(*leaves), leaves, dout)

    def ckpt() -> object:
        o = checkpoint(call, *leaves, use_reentrant=False)
        return torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "step": step, "ckpt": ckpt}


def results(call: Callable[..., torch.Tensor], inputs: dict) -> dict[str, torch.Tensor]:
    """``out`` and every gradient of one untimed step."""
    names = baselines.leaf_names(inputs)
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
    o = call(*leaves)
    grads = torch.autograd.grad(o, leaves, inputs["dout"])
    grad_names = RAW_GRAD_NAMES if "A_log" in inputs else GRAD_NAMES
    return {
        "out": o.detach(),
        **dict(zip(grad_names, (g.detach() for g in grads), strict=True)),
    }


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """``||a - b|| / ||b||`` in FP64 (inf for a non-finite difference)."""
    a64, b64 = a.detach().double().reshape(b.shape), b.detach().double()
    diff = (a64 - b64).norm().item()
    norm = b64.norm().item()
    return diff / norm if norm > 0 else (0.0 if diff == 0 else float("inf"))


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def env_info(device: torch.device) -> dict:
    p = torch.cuda.get_device_properties(device)
    info = {
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_step": os.environ.get("SLURM_STEP_ID"),
        "gpu": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "arch": f"sm_{p.major}{p.minor}a",
        "sm_count": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "timer": _timer(),
        "baselines": baselines.library_versions(),
    }
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,clocks.max.sm,power.limit",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        info["nvidia_smi"] = smi.stdout.strip().splitlines()[:8]
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        info["gpu_uuid"] = str(getattr(p, "uuid", ""))
    except Exception as exc:  # noqa: BLE001
        info["nvidia_smi"] = _error(exc)
    try:
        nvcc = os.path.join(
            os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"
        )
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=60
        ).stdout
        info["nvcc"] = out.strip().splitlines()[-1] if out.strip() else None
    except Exception as exc:  # noqa: BLE001
        info["nvcc"] = _error(exc)
    return info


def run_row(
    lengths, H: int, gate: str, l2n: bool, arms, args, device: torch.device
) -> dict:
    inputs = make_row(lengths, H, gate, l2n, device)
    inputs["lengths"] = tuple(lengths)
    row = {
        "lengths": list(lengths),
        "tokens": int(sum(lengths)),
        "docs": len(lengths),
        "heads": H,
        "gate": gate,
        "l2n": l2n,
        "arms": {},
        "unsupported": {},
    }
    calls, closures = {}, {}
    for arm in arms:
        t0 = time.time()
        try:
            if arm in CAKE_ARMS:
                call, info = cake_arm(arm, inputs, l2n)
            else:
                call, info = baselines.single_gpu_arm(arm, inputs, LOWER_BOUND, l2n)
            fns = boundaries(call, inputs)
            for name in BOUNDARIES:
                for _ in range(args.warmup):
                    fns[name]()
            torch.cuda.synchronize(device)
        except baselines.Unsupported as exc:
            row["unsupported"][arm] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 -- a baseline's own failure leaves its arm out; a Cake failure stops
            if arm in CAKE_ARMS:
                raise
            row["unsupported"][arm] = _error(exc)
            row.setdefault("errors", {})[arm] = traceback.format_exc()[-3000:]
            torch.cuda.empty_cache()
            continue
        calls[arm], closures[arm] = call, fns
        row["arms"][arm] = {"info": info, "setup_s": round(time.time() - t0, 1)}
    if "cake_det" in calls:
        ref = results(calls["cake_det"], inputs)
        for arm, call in calls.items():
            if arm == "cake_det":
                continue
            got = results(call, inputs)
            row["arms"][arm]["vs_cake_det_rel_l2"] = {
                n: rel_l2(got[n], ref[n]) for n in ref
            }
        del ref
    live = list(closures)
    per = {a: {b: [] for b in BOUNDARIES} for a in live}
    for r in range(args.rounds):
        order = live if r % 2 == 0 else list(reversed(live))
        for arm in order:
            for name in BOUNDARIES:
                per[arm][name].append(
                    statistics.median(bench_gpu_time(closures[arm][name]))
                )
    med = {}
    for arm in live:
        row["arms"][arm]["ms"] = {
            b: {
                "median": statistics.median(v),
                "min": min(v),
                "max": max(v),
                "rounds": v,
            }
            for b, v in per[arm].items()
        }
        med[arm] = {b: statistics.median(v) for b, v in per[arm].items()}
    row["speedup"] = {
        f"{base}/{cake}": {b: med[base][b] / med[cake][b] for b in BOUNDARIES}
        for base, cake in SPEEDUPS
        if base in med and cake in med
    }
    del closures, calls, inputs
    torch.cuda.empty_cache()
    return row


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def print_row(row: dict) -> None:
    head = f"S={row['tokens']} N={row['docs']} H={row['heads']} gate={row['gate']} l2n={int(row['l2n'])}"
    for arm, reason in row["unsupported"].items():
        print(f"UNSUPPORTED {head} {arm}: {reason}", flush=True)
    cols = [a for a in ARMS if a in row["arms"] and "ms" in row["arms"][a]]
    print(
        f"| {head} | boundary | "
        + " | ".join(f"{a} ms" for a in cols)
        + " | "
        + " | ".join(k for k in row["speedup"])
        + " |",
        flush=True,
    )
    for b in BOUNDARIES:
        cells = []
        for a in cols:
            m = row["arms"][a]["ms"][b]
            cells.append(f"{m['median']:.3f} [{m['min']:.3f}, {m['max']:.3f}]")
        ratios = [_fmt(v[b]) for v in row["speedup"].values()]
        print(
            f"| {head} | {b} | "
            + " | ".join(cells)
            + " | "
            + " | ".join(ratios)
            + " |",
            flush=True,
        )
    for arm in cols:
        rel = row["arms"][arm].get("vs_cake_det_rel_l2")
        if rel:
            print(
                f"CHECK {head} {arm} vs cake_det rel_l2 "
                + " ".join(f"{n}={v:.2e}" for n, v in rel.items()),
                flush=True,
            )


def _write(path: str, doc: dict) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("limitations under the License.")[-1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Official numbers: run one timing process per GPU node.",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="SxN or L1,L2,... (each with every --heads); default: the rows of DEFAULT_SHAPES",
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--rows", nargs="+", default=None, help="explicit shape:heads rows"
    )
    parser.add_argument("--gate", nargs="+", default=["g", "raw"], choices=("g", "raw"))
    parser.add_argument("--l2n", type=int, nargs="+", default=[0, 1], choices=(0, 1))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed calls of every boundary before timing",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", type=str, default=None, help="write the rows here")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1:
        parser.error("--rounds and --warmup must be at least 1")
    if not torch.cuda.is_available():
        print("CUDA device required", file=sys.stderr)
        return 2
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows_spec = []
    if args.rows:
        for spec in args.rows:
            shape, heads = spec.rsplit(":", 1)
            rows_spec.append((shape, int(heads)))
    if args.shapes:
        rows_spec += [(shape, h) for shape in args.shapes for h in args.heads]
    if not rows_spec:
        rows_spec = [
            (s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in DEFAULT_SHAPES
        ]
    env = env_info(device)
    print("ENV " + json.dumps(env, default=str), flush=True)
    for arm in baselines.SINGLE_GPU_ARMS:
        if arm in args.arms:
            print(
                f"BASELINE {arm}: {baselines.arm_label(arm, env['baselines'])}",
                flush=True,
            )
    doc = {
        "env": env,
        "protocol": {
            "boundaries": list(BOUNDARIES),
            "execution": "eager",
            "timer": env["timer"] + ", cold L2, summed device activity per call",
            "warmup_calls_per_boundary": args.warmup,
            "rounds": args.rounds,
            "statistic": "median over rounds of the per-measurement median (>= 10 calls, >= 100 ms)",
            "order": "arms interleaved, reversed every other round",
            "speedup": "baseline ms / Cake ms (above 1: Cake faster)",
            "official": "one timing process per GPU node",
        },
        "rows": [],
    }
    status = 0
    for shape, H in rows_spec:
        lengths = shape_lengths(shape)
        for gate in args.gate:
            for l2n in args.l2n:
                try:
                    row = run_row(lengths, H, gate, bool(l2n), args.arms, args, device)
                except Exception as exc:  # noqa: BLE001 -- a Cake failure is reported and fails the run
                    row = {
                        "lengths": list(lengths),
                        "heads": H,
                        "gate": gate,
                        "l2n": bool(l2n),
                        "error": _error(exc),
                        "traceback": traceback.format_exc()[-4000:],
                    }
                    print(
                        f"ERROR S={sum(lengths)} H={H} gate={gate} l2n={l2n}: {_error(exc)}",
                        flush=True,
                    )
                    status = 1
                    torch.cuda.empty_cache()
                else:
                    print_row(row)
                doc["rows"].append(row)
                if args.json:
                    _write(args.json, doc)
    return status


if __name__ == "__main__":
    sys.exit(main())
