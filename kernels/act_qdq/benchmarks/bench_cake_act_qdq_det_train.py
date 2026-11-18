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

Measure the Cake activation quantize-dequantize forward.

Two figures per row:

- ``gpu_us``: the median GPU time of one call, from CUPTI activity records
  (through ``torch.profiler``), with the L2 cache flushed before every call.
- ``loop_us``: the CUDA-event time per call of back-to-back calls (warm L2,
  host launch cost included), so eager and CUDA graph launches differ here.

Cases: ``[rows, 512]`` FP4 (E4M3 scale, group 16); ``[rows, 128]`` and
``[rows, 32, 128]`` MXFP4; ``[rows, 512]`` and ``[rows, 8, 4096]`` MXFP8; at
8,192 and 16,384 rows.  Each runs out of place and in place, eager and in a
CUDA graph, without and with the scale output.  The input is ``randn`` BF16.

Optional baselines, each imported only when its flag is given:

- ``--deepseek-kernel PATH``: the TileLang kernels of DeepSeek-V4.1-Flash
  (``inference/kernel.py`` at revision
  314113f8a7dadb388215396ba5fbee2c610a04d6; needs ``tilelang``), called
  directly with preallocated BF16 output and scale buffers, out of place and in
  place; plus the released in-place wrapper (``act_quant`` /
  ``fp4_act_quant``), which allocates and copies.
- ``--torch-compile``: ``torch.compile`` of an FP32 composite of the rule, with
  the strict pointwise numerics flags this torch offers (recorded per row).
- ``--cudnn``: the cuDNN frontend ``RopeQDQInplace`` with an identity RoPE
  cache.  It works in place only, on capability (10, 0) only, and fuses a RoPE
  on the last 64 channels of each 512-channel row (an identity here; the
  ``[rows, 8, 4096]`` input is viewed as ``[-1, 512]``).

Before timing, the Cake output is checked bit for bit against the FP64 rule
(``--no-oracle`` skips this), and every baseline output against the Cake
output, on the timed input; each row records the result in ``exact``.

Usage::

    python benchmarks/bench_cake_act_qdq_det_train.py --json out.json
    python benchmarks/bench_cake_act_qdq_det_train.py --rows 16384 \\
        --case mxfp8_e8m0_g32:8x4096 --deepseek-kernel kernel.py --torch-compile --cudnn
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from cake_pplx.cake_act_qdq_det_train import (
    GROUP_SIZES,
    cake_act_qdq_det_train_forward,
    is_cake_act_qdq_det_train_supported,
)

ROWS = (8192, 16384)
# (format, the shape after the row dimension)
CASES = (
    ("fp4_e4m3_g16", (512,)),
    ("mxfp4_e8m0_g32", (128,)),
    ("mxfp4_e8m0_g32", (32, 128)),
    ("mxfp8_e8m0_g32", (512,)),
    ("mxfp8_e8m0_g32", (8, 4096)),
)
MODES = ("out_of_place", "in_place")
EXECS = ("eager", "graph")
QMAX = {"fp4_e4m3_g16": 6.0, "mxfp4_e8m0_g32": 6.0, "mxfp8_e8m0_g32": 448.0}
DEEPSEEK_REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"


def _case_name(fmt: str, inner: tuple[int, ...]) -> str:
    return f"{fmt}:{'x'.join(map(str, inner))}"


CASE_NAMES = {_case_name(fmt, inner): (fmt, inner) for fmt, inner in CASES}


# --------------------------------------------------------------------------- timing


def _cupti_call_ms(fn, *, dry_run_iters: int = 10, repeat_iters: int = 100) -> list[float]:
    """Per-call GPU time of ``fn`` in milliseconds, the L2 cache flushed before
    every call: the summed duration of the GPU activities (kernels, copies,
    fills) that CUPTI records through ``torch.profiler`` between two sentinel
    kernels.

    Every timed call runs between two launches of ATen's one-cycle spin kernel;
    the first record after a sentinel is the L2 flush and is not counted.
    Device records are attributed by position, not by the host operation that
    launched them, so kernels launched through the driver API or a CUDA graph
    count too.
    """
    l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    flush = torch.empty(max(2 * l2_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
    for _ in range(dry_run_iters):
        flush.fill_(1)
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        torch.cuda._sleep(1)
        for _ in range(repeat_iters):
            flush.fill_(1)
            fn()
            torch.cuda._sleep(1)
        torch.cuda.synchronize()
    records = sorted(
        (event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda event: event.time_range.start,
    )
    call_us: list[float] = []
    current = None
    for event in records:
        if "::spin_kernel(" in event.name:
            if current is not None:
                call_us.append(sum(e.time_range.end - e.time_range.start for e in current[1:]))
            current = []
        elif current is not None:
            current.append(event)
    if len(call_us) != repeat_iters:
        raise RuntimeError(
            f"the profiler returned {len(call_us)} complete calls of {repeat_iters}; CUPTI lost device records"
        )
    if any(us <= 0 for us in call_us):
        raise RuntimeError("CUPTI recorded no GPU activity for a timed call")
    return [us / 1000.0 for us in call_us]


def _gpu_us(fn) -> float:
    return 1000.0 * float(statistics.median(_cupti_call_ms(fn)))


def _loop_us(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return 1000.0 * start.elapsed_time(end) / iters


def _graphed(fn: Callable[[], Any]) -> Callable[[], None]:
    """``fn`` captured in a CUDA graph after two warm-up calls on a side stream; returns the replay."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


# --------------------------------------------------------------------------- exactness


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(_bits(a), _bits(b)))


@functools.cache
def _reference() -> Any:
    """The FP64 rule of the family's tests (``tests/experimental/test_helpers``), or None when absent."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module("tests.experimental.test_helpers.cake_act_qdq_det_train_reference")
    except ImportError:
        return None


def _matches_rule(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, fmt: str) -> Optional[bool]:
    """Whether ``y`` and ``s`` equal the FP64 rule of ``x`` bit for bit, evaluated in row chunks."""
    ref = _reference()
    if ref is None:
        return None
    width = x.shape[-1]
    group = GROUP_SIZES[fmt]
    xf, yf, sf = x.reshape(-1, width), y.reshape(-1, width), s.reshape(-1, width // group)
    rows = max(1, (1 << 22) // max(width, 1))
    for start in range(0, xf.shape[0], rows):
        want_y, want_s = ref.qdq(xf[start : start + rows], fmt)
        if not torch.equal(ref.y_bits(want_y), _bits(yf[start : start + rows])):
            return False
        if not torch.equal(ref.scale_bits(want_s), _bits(sf[start : start + rows])):
            return False
    return True


# --------------------------------------------------------------------------- implementations


def _cake_calls(x: torch.Tensor, fmt: str, mode: str, scales: bool) -> Callable[[], Any]:
    shape = (*x.shape[:-1], x.shape[-1] // GROUP_SIZES[fmt])
    s = torch.empty(shape, dtype=torch.float32, device=x.device) if scales else None
    if mode == "in_place":
        return lambda: cake_act_qdq_det_train_forward(x, format=fmt, out=x, scales=s)
    return lambda: cake_act_qdq_det_train_forward(x, format=fmt, scales=s)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@functools.cache
def _deepseek(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("deepseek_v4_1_flash_inference_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deepseek_kernel(module: Any, fmt: str, width: int) -> tuple[Any, torch.dtype]:
    """The TileLang kernel of ``fmt`` for last dimension ``width`` (in-place form: BF16 output) and its scale dtype."""
    if fmt == "fp4_e4m3_g16":
        return module.fp4_quant_kernel(width, 16, scale_dtype=module.FP8, inplace=True), torch.float8_e4m3fn
    if fmt == "mxfp4_e8m0_g32":
        return module.fp4_quant_kernel(width, 32, scale_dtype=module.FE8M0, inplace=True), torch.float8_e8m0fnu
    return (
        module.act_quant_kernel(width, 32, scale_dtype=module.FE8M0, round_scale=True, inplace=True),
        torch.float8_e8m0fnu,
    )


def _deepseek_wrapper(module: Any, fmt: str) -> Callable[[torch.Tensor], Any]:
    """The released in-place wrapper of ``fmt`` (it allocates an output and copies it back into ``x``)."""
    if fmt == "fp4_e4m3_g16":
        return lambda x: module.fp4_act_quant(x, 16, True, scale_dtype=torch.float8_e4m3fn)
    if fmt == "mxfp4_e8m0_g32":
        return lambda x: module.fp4_act_quant(x, 32, True)
    return lambda x: module.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True)


def _strict_inductor() -> dict[str, Any]:
    """Turn on the strict pointwise numerics this torch offers (IEEE division, no flush to zero, exact casts);
    returns what was set.  The composite's exactness is measured either way."""
    import torch._inductor.config as config

    applied: dict[str, Any] = {}
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version >= (2, 15) and hasattr(config, "numerics"):
        config.numerics = "strict_pointwise"
        applied["numerics"] = "strict_pointwise"
    eager = getattr(config, "eager_numerics", None)
    for name in ("division_rounding", "disable_ftz"):
        if eager is not None and hasattr(eager, name):
            setattr(eager, name, True)
            applied[f"eager_numerics.{name}"] = True
    # Older spellings of the division and cast flags.
    for name in ("emulate_divison_rounding", "emulate_division_rounding", "emulate_precision_casts"):
        if hasattr(config, name):
            setattr(config, name, True)
            applied[name] = True
    return applied


def _composite_rule(fmt: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """The rule as FP32 torch operations (true division for the E4M3 scale, exact power-of-two scales)."""
    group, qmax = GROUP_SIZES[fmt], QMAX[fmt]
    e8m0 = {"mxfp4_e8m0_g32": (0x400000, 2, 1), "mxfp8_e8m0_g32": (0x600000, 8, 105)}.get(fmt)
    e2m1 = fmt != "mxfp8_e8m0_g32"

    def rule(x: torch.Tensor) -> torch.Tensor:
        g = x.float().reshape(-1, group)
        amax = g.abs().amax(-1, keepdim=True)
        if e8m0 is None:
            amax = torch.clamp_min(amax, 6.0 * 2.0**-9)
            s = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
            u = (g / s).clamp(-qmax, qmax)
        else:
            mantissa, exponent, floor = e8m0
            bits = amax.view(torch.int32)
            byte = (bits >> 23) - exponent + ((bits & 0x7FFFFF) > mantissa).to(torch.int32)
            byte = torch.clamp_min(byte, floor)
            s = (byte << 23).view(torch.float32)
            u = (g * ((254 - byte) << 23).view(torch.float32)).clamp(-qmax, qmax)
        if e2m1:
            a = u.abs()
            mag = torch.zeros_like(a)
            for th, strict, val in (
                (0.25, True, 0.5),
                (0.75, False, 1.0),
                (1.25, True, 1.5),
                (1.75, False, 2.0),
                (2.50, True, 3.0),
                (3.50, False, 4.0),
                (5.00, True, 6.0),
            ):
                mag = torch.where(a > th if strict else a >= th, val, mag)
            q = torch.copysign(mag, u)
        else:
            q = u.to(torch.float8_e4m3fn).float()
        return (q * s).to(torch.bfloat16).reshape(x.shape)

    return rule


def _cudnn_view(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, int, Optional[str]]:
    """``x`` in the layout ``RopeQDQInplace`` takes (group-32 FP4: ``[N, H, 128]`` with ``H`` in 1, 4, 32; the
    others: ``[..., 512]``), the number of position ids and a note when the layout is a view."""
    if fmt == "mxfp4_e8m0_g32":
        view = x.reshape(x.shape[0], -1, 128)
        note = None if x.dim() == 3 else f"viewed as [{view.shape[0]}, {view.shape[1]}, 128]"
        return view, view.shape[0], note
    view = x.reshape(-1, 512)
    return view, view.shape[0], None if x.shape[-1] == 512 else "viewed as [-1, 512]"


# --------------------------------------------------------------------------- driver


def _measure(fn: Callable[[], Any], execution: str) -> dict[str, float]:
    run = _graphed(fn) if execution == "graph" else fn
    run()
    torch.cuda.synchronize()
    return {"gpu_us": _gpu_us(run), "loop_us": _loop_us(run)}


def _row(base: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {**base, **fields}


def _try_measure(rows: list[dict[str, Any]], base: dict[str, Any], fn: Callable[[], Any], execution: str) -> None:
    try:
        rows.append(_row(base, execution=execution, **_measure(fn, execution)))
    except Exception as error:  # noqa: BLE001 - a baseline that cannot run in this form is recorded, not fatal
        rows.append(_row(base, execution=execution, error=f"{type(error).__name__}: {error}"[:400]))


class _Case:
    """One timed input: ``x0`` (BF16 randn), the Cake reference output and the fields every row of it carries."""

    def __init__(self, fmt: str, inner: tuple[int, ...], rows_count: int, env: dict[str, Any]) -> None:
        self.fmt = fmt
        self.inner = inner
        self.shape = (rows_count, *inner)
        self.group = GROUP_SIZES[fmt]
        generator = torch.Generator(device="cuda").manual_seed(83 + rows_count + sum(inner))
        self.x0 = torch.randn(self.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        self.want_y, self.want_s = cake_act_qdq_det_train_forward(self.x0, format=fmt, return_scales=True)
        self.common = {
            "format": fmt,
            "shape": "[" + ", ".join(map(str, self.shape)) + "]",
            "rows": rows_count,
            "gpu": env["gpu"],
            "arch": env["arch"],
        }


def _bench_cake(args: argparse.Namespace, case: _Case, out: list[dict[str, Any]]) -> None:
    exact = None if args.no_oracle else _matches_rule(case.x0, case.want_y, case.want_s, case.fmt)
    for scales in args.scales:
        for mode in args.mode:
            fn = _cake_calls(case.x0.clone(), case.fmt, mode, scales)
            for execution in args.exec:
                base = _row(case.common, impl="cake", mode=mode, scales=scales, exact=exact)
                _try_measure(out, base, fn, execution)


def _bench_deepseek(args: argparse.Namespace, case: _Case, out: list[dict[str, Any]]) -> None:
    module = _deepseek(args.deepseek_kernel)
    width, group = case.shape[-1], case.group
    kernel, scale_dtype = _deepseek_kernel(module, case.fmt, width)
    s = torch.empty((*case.shape[:-1], width // group), dtype=scale_dtype, device="cuda")
    y = torch.empty_like(case.x0)
    # Checked bit for bit on the timed input before timing.
    kernel(case.x0.view(-1, width), y.view(-1, width), s.view(-1, width // group))
    torch.cuda.synchronize()
    exact = _same_bits(y, case.want_y)
    scales_exact = bool(torch.equal(s.float(), case.want_s))
    for mode in args.mode:
        x = case.x0.clone()
        dst = x if mode == "in_place" else y
        fn = lambda x=x, dst=dst: kernel(x.view(-1, width), dst.view(-1, width), s.view(-1, width // group))
        for execution in args.exec:
            base = _row(
                case.common, impl="deepseek_tilelang", mode=mode, scales=True, exact=exact, scales_exact=scales_exact
            )
            _try_measure(out, base, fn, execution)
    wrapper = _deepseek_wrapper(module, case.fmt)
    x = case.x0.clone()
    wrapper(x)
    torch.cuda.synchronize()
    base = _row(case.common, impl="deepseek_wrapper", mode="in_place", scales=False, exact=_same_bits(x, case.want_y))
    x = case.x0.clone()
    _try_measure(out, base, lambda x=x: wrapper(x), "eager")


def _bench_torch_compile(args: argparse.Namespace, case: _Case, out: list[dict[str, Any]], env: dict) -> None:
    rule = _composite_rule(case.fmt)
    compiled = torch.compile(rule, dynamic=False, fullgraph=True)
    compiled_inplace = torch.compile(lambda x: x.copy_(rule(x)), dynamic=False, fullgraph=True)
    exact = {"out_of_place": _same_bits(compiled(case.x0), case.want_y)}
    x = case.x0.clone()
    compiled_inplace(x)
    exact["in_place"] = _same_bits(x, case.want_y)
    for mode in args.mode:
        x = case.x0.clone()
        fn = (lambda x=x: compiled_inplace(x)) if mode == "in_place" else (lambda x=x: compiled(x))
        for execution in args.exec:
            base = _row(
                case.common, impl="torch_compile", mode=mode, scales=False, exact=exact[mode], inductor=env["inductor"]
            )
            _try_measure(out, base, fn, execution)


def _bench_cudnn(args: argparse.Namespace, case: _Case, out: list[dict[str, Any]], env: dict) -> None:
    base = _row(case.common, impl="cudnn_rope_qdq", mode="in_place", scales=False)
    if env["capability"] != [10, 0]:
        out.append(_row(base, error="unsupported: RopeQDQInplace requires capability (10, 0)"))
        return
    import cudnn

    cache = torch.zeros(1, 64, dtype=torch.float32, device="cuda")
    cache[:, :32] = 1.0  # 32 cosines then 32 sines: cos 1, sin 0
    scale_format = "e4m3" if case.fmt == "fp4_e4m3_g16" else "ue8m0"
    quantization = "fp8" if case.fmt == "mxfp8_e8m0_g32" else "fp4"
    x = case.x0.clone()
    view, tokens, note = _cudnn_view(x, case.fmt)
    positions = torch.zeros(tokens, dtype=torch.int32, device="cuda")
    op = cudnn.RopeQDQInplace(
        view, cache, positions, quantization=quantization, group_size=case.group, scale_format=scale_format
    )
    op.check_support()
    op.compile()
    # Checked bit for bit on the timed input before timing.
    op.execute(view, cache, positions, current_stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    exact = _same_bits(x, case.want_y)
    x = case.x0.clone()
    view, _, _ = _cudnn_view(x, case.fmt)

    def fn(view=view):
        op.execute(view, cache, positions, current_stream=torch.cuda.current_stream().cuda_stream)

    base = _row(base, exact=exact, workspace_bytes=int(op.scratch_workspace_bytes()), note=note)
    for execution in args.exec:
        _try_measure(out, base, fn, execution)


def _run_case(args: argparse.Namespace, fmt: str, inner: tuple[int, ...], rows_count: int, env: dict) -> list:
    case = _Case(fmt, inner, rows_count, env)
    out: list[dict[str, Any]] = []
    _bench_cake(args, case, out)
    baselines = []
    if args.deepseek_kernel:
        baselines.append(("deepseek_tilelang", lambda: _bench_deepseek(args, case, out)))
    if args.torch_compile:
        baselines.append(("torch_compile", lambda: _bench_torch_compile(args, case, out, env)))
    if args.cudnn:
        baselines.append(("cudnn_rope_qdq", lambda: _bench_cudnn(args, case, out, env)))
    for impl, run in baselines:
        try:
            run()
        except Exception as error:  # noqa: BLE001 - a baseline that is missing or fails is recorded, not fatal
            out.append(_row(case.common, impl=impl, mode=None, scales=None, error=f"{type(error).__name__}: {error}"[:400]))
    return out


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    capability = list(torch.cuda.get_device_capability())
    env: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(),
        "capability": capability,
        "arch": f"sm_{capability[0]}{capability[1]}",
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": "CUPTI cold-L2 median of 100 calls (gpu_us); CUDA events over 200 back-to-back calls (loop_us)",
        "input": "torch.randn BF16",
        "inductor": None,
    }
    for distribution in ("cake-pplx-act-qdq", "apache-tvm-ffi", "tilelang", "nvidia-cudnn-frontend"):
        try:
            env[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            env[distribution] = None
    if args.deepseek_kernel:
        data = Path(args.deepseek_kernel).read_bytes()
        env["deepseek_kernel"] = {
            "path": str(Path(args.deepseek_kernel).resolve()),
            "revision": DEEPSEEK_REVISION,
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob": _git_blob_sha1(data),
        }
    if args.torch_compile:
        env["inductor"] = _strict_inductor()
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, action="append", help=f"row counts (default {ROWS})")
    parser.add_argument("--case", action="append", choices=sorted(CASE_NAMES), help="cases (default all)")
    parser.add_argument("--mode", action="append", choices=MODES, help="default both")
    parser.add_argument("--exec", action="append", choices=EXECS, help="default both")
    parser.add_argument("--scales", choices=("off", "on", "both"), default="both", help="Cake scale output")
    parser.add_argument("--deepseek-kernel", help="path of DeepSeek-V4.1-Flash inference/kernel.py")
    parser.add_argument("--torch-compile", action="store_true", help="time a torch.compile composite of the rule")
    parser.add_argument("--cudnn", action="store_true", help="time cuDNN frontend RopeQDQInplace (sm_100 only)")
    parser.add_argument("--no-oracle", action="store_true", help="skip the FP64 check of the Cake output")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not is_cake_act_qdq_det_train_supported(torch.device("cuda", 0)):
        raise RuntimeError("this benchmark needs an SM100, SM103 or SM107 device")
    args.mode = tuple(args.mode or MODES)
    args.exec = tuple(args.exec or EXECS)
    args.scales = {"off": (False,), "on": (True,), "both": (False, True)}[args.scales]
    cases = [CASE_NAMES[name] for name in args.case] if args.case else list(CASES)
    env = _environment(args)

    rows: list[dict[str, Any]] = []
    for rows_count in args.rows or ROWS:
        for fmt, inner in cases:
            rows.extend(_run_case(args, fmt, inner, rows_count, env))
            torch.cuda.empty_cache()

    print(f"{env['gpu']} ({env['arch']}), torch {env['torch']}, CUDA {env['cuda']}")
    print("| format | shape | impl | mode | exec | scales | exact | GPU (us) | loop (us) |")
    print("|---|---|---|---|---|---|---|---:|---:|")
    for row in rows:
        gpu = f"{row['gpu_us']:.2f}" if "gpu_us" in row else row.get("error", "")
        loop = f"{row['loop_us']:.2f}" if "loop_us" in row else ""
        print(
            f"| {row['format']} | {row['shape']} | {row['impl']} | {row['mode']} | {row.get('execution', '')} | "
            f"{row['scales']} | {row.get('exact')} | {gpu} | {loop} |"
        )
    if args.json:
        args.json.write_text(json.dumps({"environment": env, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
