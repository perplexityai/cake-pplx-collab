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

Benchmark of the Cake non-deterministic dense multi-head attention backward (``cake_pplx.cake_fmha_nondet_train``)
against FlashAttention-4 CuTe (``flash_attn.cute``) and the cuDNN frontend SDPA backward, each when installed.

Arms (``--arms``; default: every arm whose library imports):

* ``cake``: ``cake_fmha_nondet_train_backward(q, k, v, out, lse, dout, softmax_scale=..., causal=...)``.
* ``fa4`` / ``fa4_det``: ``flash_attn.cute.interface._flash_attn_bwd(..., deterministic=False / True)``.
* ``cudnn`` / ``cudnn_det``: a dense cuDNN frontend ``sdpa_backward`` graph over the same BSHD tensors (FP32
  statistics = ``lse``; top-left causal mask), ``use_deterministic_algorithm`` False / True. The plan is the fastest
  that builds among those the heuristics offer (modes A and FALLBACK), timed on the row's own inputs before the row.

Every arm takes the same operands and returns contiguous BF16 ``dq [B, Sq, H, D]`` and ``dk``, ``dv [B, Skv, H, D]``.
Every arm allocates its outputs and its scratch inside the timed call (the torch caching allocator); graph and plan
building and kernel compilation happen before timing.

Inputs per shape: ``q``, ``k``, ``v``, ``dout`` unit ``randn`` BF16 ``[B, S, H, D]`` from one CUDA generator
(``--seed``); ``out`` and ``lse`` (FP32 natural-log ``logsumexp`` ``[B, H, S]``) from one untimed forward: FA4's when it
is installed, else a chunked FP32 torch forward. ``--inputs DIR`` loads ``DIR/<shape>.pt`` instead (a dict of the six
tensors), so separate processes time the same bits.

Shapes (``--shapes``): ``T1`` B4 H32 S8192 D128, ``T2`` the same causal, ``T3`` B8 H32 S8192 D64, ``T4`` the same
causal, ``S1`` B1 H1 S128 D128, ``S2`` B1 H1 S256 D128, ``S3`` B1 H1 S128 D64, ``S4`` B1 H1 S256 D64, ``L1`` B1 H1
S512 D128, ``L2`` B1 H1 S1024 D128, ``L3`` B1 H1 S512 D64, ``L4`` B1 H1 S1024 D64, or ``B:H:S:D[:causal]``, where
``S`` is the query and the key length, or ``SqxSkv`` gives them apart (``1:1:128x8192:128``: 128 queries, 8192 keys).

Timing: CUDA events, eager, warm L2. One sample is ``--loop`` back-to-back calls (default 1, or 50 for a shape with
fewer than 65536 query rows over all heads, whose call is bound by the host); the per-call time is reported, with the
host time of issuing the calls (``host_bound`` when that is at least 0.9 of the GPU time). Per arm and round: warm-up
samples until both ``--warmup`` samples and ``--warmup-ms`` of continuous calls have passed (the default 3000 ms ends
at the clock a long run settles to under the power limit), then ``--reps`` samples with no pause; the round's value is
their median. ``--rounds`` rounds run the arms in the given order, then reversed, and so on; an arm's value is the
median of its round values, with their min and max. ``--warmup-ms 0 --reps 10`` is the short method (its large rows
include the boost clock). TFLOPS = 10 B H Sq Skv D (halved when causal) / time.

    python benchmarks/bench_cake_fmha_nondet_train.py --shapes T1,T3,S1 --arms cake,fa4,cudnn --out rows.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

SHAPES = {
    "T1": (4, 32, 8192, 128, False),
    "T2": (4, 32, 8192, 128, True),
    "T3": (8, 32, 8192, 64, False),
    "T4": (8, 32, 8192, 64, True),
    "S1": (1, 1, 128, 128, False),
    "S2": (1, 1, 256, 128, False),
    "S3": (1, 1, 128, 64, False),
    "S4": (1, 1, 256, 64, False),
    "L1": (1, 1, 512, 128, False),
    "L2": (1, 1, 1024, 128, False),
    "L3": (1, 1, 512, 64, False),
    "L4": (1, 1, 1024, 64, False),
}
ARMS = ("cake", "fa4", "fa4_det", "cudnn", "cudnn_det")
FROST = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"


@contextlib.contextmanager
def _env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def shape(spec: str) -> dict[str, Any]:
    if spec in SHAPES:
        B, H, S, D, causal = SHAPES[spec]
        Sq = Skv = S
    else:
        parts = spec.split(":")
        B, H, D = int(parts[0]), int(parts[1]), int(parts[3])
        q_len, _, kv_len = parts[2].partition("x")  # S, or Sq x Skv
        Sq, Skv = int(q_len), int(kv_len or q_len)
        causal = len(parts) > 4 and parts[4].lower() in ("1", "true", "causal", "yes")
    return {"id": spec, "B": B, "H": H, "Sq": Sq, "Skv": Skv, "D": D, "causal": causal, "scale": D**-0.5}


def flops(s: dict[str, Any]) -> float:
    f = 10.0 * s["B"] * s["H"] * s["Sq"] * s["Skv"] * s["D"]
    return f / 2 if s["causal"] else f


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def _torch_forward(s: dict[str, Any], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """``out`` (BF16 [B, S, H, D]) and natural-log ``lse`` (FP32 [B, H, S]) of an FP32 forward, in row chunks."""
    B, H, S, D = s["B"], s["H"], s["Sq"], s["D"]
    qh, kh, vh = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    out = torch.empty(B, H, S, D, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(B, H, S, device=q.device)
    for a in range(0, S, 128):
        sc = (qh[:, :, a : a + 128] @ kh.transpose(-1, -2)) * s["scale"]
        if s["causal"]:
            rows = torch.arange(a, min(a + 128, S), device=q.device)[:, None]
            sc = sc.masked_fill(torch.arange(s["Skv"], device=q.device)[None, :] > rows, -math.inf)
        chunk = sc.logsumexp(-1)
        out[:, :, a : a + 128] = ((sc - chunk[..., None]).exp() @ vh).bfloat16()
        lse[:, :, a : a + 128] = chunk
    return out.permute(0, 2, 1, 3).contiguous(), lse.contiguous()


def make_inputs(s: dict[str, Any], seed: int, device: torch.device) -> tuple[dict[str, torch.Tensor], str]:
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(n: int) -> torch.Tensor:
        return torch.randn(s["B"], n, s["H"], s["D"], device=device, generator=g).bfloat16()

    x = {"q": randn(s["Sq"]), "k": randn(s["Skv"]), "v": randn(s["Skv"]), "dout": randn(s["Sq"])}
    try:
        from flash_attn.cute.interface import _flash_attn_fwd

        out, lse = _flash_attn_fwd(
            x["q"], x["k"], x["v"], softmax_scale=s["scale"], causal=s["causal"], return_lse=True
        )[:2]
        source = "flash_attn.cute forward"
    except ImportError:
        out, lse = _torch_forward(s, x["q"], x["k"], x["v"])
        source = "FP32 torch forward"
    x["out"], x["lse"] = out.contiguous(), lse.contiguous()
    torch.cuda.synchronize()
    return x, source


def load_inputs(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return {n: blob[n].to(device).contiguous() for n in ("q", "k", "v", "out", "lse", "dout")}


# ----------------------------------------------------------------------------------------------------------------
# Arms: setup(s, x) before timing, __call__(x) -> (dq, dk, dv), info() -> dict


class CakeArm:
    def __init__(self) -> None:
        from cake_pplx.cake_fmha_nondet_train import cake_fmha_nondet_train_backward

        self.f = cake_fmha_nondet_train_backward

    def setup(self, s: dict[str, Any], x: dict[str, torch.Tensor]) -> None:
        self.s = s

    def __call__(self, x: dict[str, torch.Tensor]):
        s = self.s
        return self.f(
            x["q"], x["k"], x["v"], x["out"], x["lse"], x["dout"], softmax_scale=s["scale"], causal=s["causal"]
        )

    def info(self) -> dict[str, Any]:
        return {"impl": "cake_fmha_nondet_train_backward", "version": _version("cake-pplx-fmha")}


class FlashAttentionArm:
    def __init__(self, deterministic: bool) -> None:
        from flash_attn.cute.interface import _flash_attn_bwd

        self.f, self.det = _flash_attn_bwd, deterministic

    def setup(self, s: dict[str, Any], x: dict[str, torch.Tensor]) -> None:
        self.s = s

    def __call__(self, x: dict[str, torch.Tensor]):
        s = self.s
        return self.f(
            x["q"], x["k"], x["v"], x["out"], x["dout"], x["lse"], softmax_scale=s["scale"], causal=s["causal"],
            deterministic=self.det,
        )[:3]  # fmt: skip

    def info(self) -> dict[str, Any]:
        return {"impl": "flash_attn.cute _flash_attn_bwd", "deterministic": self.det,
                "version": _version("flash-attn-4")}  # fmt: skip


class CudnnArm:
    UIDS = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # q, k, v, o, dO, stats, dQ, dK, dV

    def __init__(self, deterministic: bool) -> None:
        import cudnn

        self.c, self.det = cudnn, deterministic
        self.handle = cudnn.create_handle()
        self.rec: dict[str, Any] = {}

    def _graph(self, s: dict[str, Any]):
        c = self.c
        B, H, S, Skv, D = s["B"], s["H"], s["Sq"], s["Skv"], s["D"]
        g = c.pygraph(
            io_data_type=c.data_type.BFLOAT16, intermediate_data_type=c.data_type.FLOAT,
            compute_data_type=c.data_type.FLOAT, handle=self.handle,
        )  # fmt: skip

        def bshd(name: str, uid: int, n: int):
            return g.tensor(name=name, dim=[B, H, n, D], stride=[n * H * D, D, H * D, 1],
                            data_type=c.data_type.BFLOAT16, uid=uid)  # fmt: skip

        q, k, v, o, do = (bshd(n, u, m) for n, u, m in (("q", 1, S), ("k", 2, Skv), ("v", 3, Skv), ("o", 4, S),
                                                         ("dO", 5, S)))  # fmt: skip
        stats = g.tensor(name="stats", dim=[B, H, S, 1], stride=[H * S, S, 1, 1], data_type=c.data_type.FLOAT, uid=6)
        kw: dict[str, Any] = {"attn_scale": s["scale"], "use_deterministic_algorithm": self.det}
        if s["causal"]:
            kw.update(diagonal_alignment=c.diagonal_alignment.TOP_LEFT, diagonal_band_right_bound=0)
        grads = g.sdpa_backward(name="sdpa_bwd", q=q, k=k, v=v, o=o, dO=do, stats=stats, **kw)
        for t, uid, n in zip(grads, (7, 8, 9), (S, Skv, Skv)):
            t.set_uid(uid).set_output(True).set_dim([B, H, n, D]).set_stride([n * H * D, D, H * D, 1]).set_data_type(
                c.data_type.BFLOAT16
            )
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([c.heur_mode.A, c.heur_mode.FALLBACK])
        return g

    def _run(self, g, x: dict[str, torch.Tensor], ws_bytes: int):
        dq = torch.empty_like(x["q"])
        dk = torch.empty_like(x["k"])
        dv = torch.empty_like(x["v"])
        ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=x["q"].device)
        self.c.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        g.execute([x["q"], x["k"], x["v"], x["out"], x["dout"], x["lse"], dq, dk, dv], ws, handle=self.handle,
                  tensor_uids=list(self.UIDS))  # fmt: skip
        return dq, dk, dv

    def setup(self, s: dict[str, Any], x: dict[str, torch.Tensor]) -> None:
        with _env(FROST, "1"):  # the FROST engines are offered too while planning
            self._plan(s, x)

    def _plan(self, s: dict[str, Any], x: dict[str, torch.Tensor]) -> None:
        g0 = self._graph(s)
        names = [g0.get_plan_name_at_index(i) for i in range(g0.get_execution_plan_count())]
        best, declined, timed = None, [], {}
        for i, name in enumerate(names):
            try:
                g = self._graph(s)
                g.select_plan(i)
                g.check_support()
                g.build_plans()
                ws_bytes = int(g.get_workspace_size())
                ms = _median_ms(lambda g=g, b=ws_bytes: self._run(g, x, b), s["loop"], 5)
                timed[name] = round(ms, 5)
                if best is None or ms < best[0]:
                    best = (ms, name, g, ws_bytes)
            except Exception as e:  # a plan that does not build is recorded
                if isinstance(e, torch.OutOfMemoryError):
                    raise
                declined.append({"plan": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        if best is None:
            raise RuntimeError(f"no cuDNN plan builds for {s['id']}: {declined[:4]}")
        _, name, self.g, self.ws_bytes = best
        self.rec = {"plan": name, "offered": names, "timed_ms": timed, "declined": declined}

    def __call__(self, x: dict[str, torch.Tensor]):
        return self._run(self.g, x, self.ws_bytes)

    def info(self) -> dict[str, Any]:
        return {"impl": "cudnn frontend sdpa_backward", "deterministic": self.det,
                "frontend": getattr(self.c, "__version__", None), "backend": self.c.backend_version(),
                **self.rec}  # fmt: skip


def make_arm(name: str):
    if name == "cake":
        return CakeArm()
    if name in ("fa4", "fa4_det"):
        return FlashAttentionArm(name == "fa4_det")
    if name in ("cudnn", "cudnn_det"):
        return CudnnArm(name == "cudnn_det")
    raise ValueError(f"unknown arm {name!r} (one of {', '.join(ARMS)})")


# ----------------------------------------------------------------------------------------------------------------
# Timing


def _median_ms(call: Callable[[], Any], loop: int, reps: int) -> float:
    call()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(loop):
            call()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / loop)
    return statistics.median(times)


def time_round(arm, x: dict[str, torch.Tensor], loop: int, warmup: int, warmup_ms: float, reps: int) -> dict[str, Any]:
    start, done = time.perf_counter(), 0
    while done < warmup or (time.perf_counter() - start) * 1e3 < warmup_ms:
        for _ in range(loop):
            arm(x)
        torch.cuda.synchronize()
        done += 1
    gpu, host = [], []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        h0 = time.perf_counter()
        a.record()
        for _ in range(loop):
            arm(x)
        b.record()
        h1 = time.perf_counter()
        b.synchronize()
        gpu.append(a.elapsed_time(b) / loop)
        host.append((h1 - h0) * 1e3 / loop)
    return {"ms": statistics.median(gpu), "host_ms": statistics.median(host), "samples_ms": gpu,
            "warmup_samples": done}  # fmt: skip


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def identity(device: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(device)
    dists = ("cake-pplx-fmha", "apache-tvm-ffi", "flash-attn-4", "nvidia-cudnn-frontend", "nvidia-cudnn-cu13")
    out: dict[str, Any] = {
        "gpu": p.name,
        "uuid": f"GPU-{p.uuid}",
        "cc": list(torch.cuda.get_device_capability(device)),
        "sms": p.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "versions": {d: _version(d) for d in dists},
    }
    query = ["nvidia-smi", f"--id=GPU-{p.uuid}", "--query-gpu=driver_version,power.limit,clocks.max.sm"]
    with contextlib.suppress(Exception):
        q = subprocess.run([*query, "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
        out["driver_power_maxclock"] = q.stdout.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark of the Cake non-deterministic dense attention backward.")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--arms", default=None, help=f"comma list of {', '.join(ARMS)} (default: every importable arm)")
    ap.add_argument("--inputs", default=None, help="directory of <shape>.pt input files (q, k, v, out, lse, dout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loop", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--warmup-ms", type=float, default=3000.0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    names = args.arms.split(",") if args.arms else list(ARMS)
    arms = {}
    for name in names:
        try:
            arms[name] = make_arm(name)
        except ImportError as e:
            print(f"arm {name} unavailable: {e}", file=sys.stderr)
    ident = identity(device)
    print(json.dumps({"identity": ident, "arms": list(arms)}), flush=True)
    rows = []
    for spec in args.shapes.split(","):
        s = shape(spec)
        s["loop"] = args.loop or (1 if s["B"] * s["H"] * s["Sq"] >= 65536 else 50)
        if args.inputs:
            x, source = load_inputs(os.path.join(args.inputs, f"{spec}.pt"), device), f"{args.inputs}/{spec}.pt"
        else:
            x, source = make_inputs(s, args.seed, device)
        row: dict[str, Any] = {k: s[k] for k in ("id", "B", "H", "Sq", "Skv", "D", "causal", "loop")}
        row.update(inputs=source, flops=flops(s), arms={})
        ready = []
        for name, arm in arms.items():
            try:
                arm.setup(s, x)
                arm(x)  # compiles / loads the kernels outside the timing
                torch.cuda.synchronize()
                ready.append(name)
            except Exception as e:  # noqa: BLE001 -- an arm that refuses a row is reported, not timed
                row["arms"][name] = {"status": "unsupported", "reason": f"{type(e).__name__}: {str(e)[:500]}"}
        per: dict[str, list[dict[str, Any]]] = {name: [] for name in ready}
        for r in range(args.rounds):
            for name in ready if r % 2 == 0 else ready[::-1]:
                per[name].append(time_round(arms[name], x, s["loop"], args.warmup, args.warmup_ms, args.reps))
        for name in ready:
            meds = [p["ms"] for p in per[name]]
            ms = statistics.median(meds)
            row["arms"][name] = {
                "status": "ok", "ms": ms, "min_ms": min(meds), "max_ms": max(meds), "round_ms": meds,
                "host_bound": any(p["host_ms"] >= 0.9 * p["ms"] for p in per[name]),
                "tflops": flops(s) / (ms * 1e-3) / 1e12, "info": arms[name].info(),
            }  # fmt: skip
        cake = row["arms"].get("cake", {}).get("ms")
        if cake:
            row["speedup_vs_cake"] = {n: a["ms"] / cake for n, a in row["arms"].items() if n != "cake" and "ms" in a}
        brief = {n: (round(a["ms"], 4) if "ms" in a else a["status"]) for n, a in row["arms"].items()}
        print(json.dumps({"shape": spec, "ms": brief, "speedup_vs_cake": row.get("speedup_vs_cake")}), flush=True)
        rows.append(row)
        del x
        torch.cuda.empty_cache()
    if args.out:
        settings = {k: getattr(args, k) for k in ("warmup", "warmup_ms", "reps", "rounds", "loop", "seed", "inputs")}
        with open(args.out, "w") as f:
            json.dump({"identity": ident, "settings": settings, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
