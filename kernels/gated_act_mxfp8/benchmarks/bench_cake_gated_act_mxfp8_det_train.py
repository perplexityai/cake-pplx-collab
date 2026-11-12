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

Measure fused gated activation plus MXFP8 quantization. Timings use CUPTI GPU
kernel activity (through ``torch.profiler``) with a cold L2 cache.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from cake_pplx.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_backward,
    cake_gated_act_mxfp8_det_train_forward,
)


SHAPES = (
    (128, 128),
    (4096, 2048),
    (4096, 7168),
    (16384, 7168),
    (131072, 8192),
)

MODES = (
    ("fwd-row", "forward", True, False),
    ("fwd-col", "forward", False, True),
    ("fwd-both", "forward", True, True),
    ("bwd-row", "backward", True, False),
    ("bwd-col", "backward", False, True),
    ("bwd-both", "backward", True, True),
)


def _fused(
    direction: str,
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    rowwise: bool,
    colwise: bool,
):
    if direction == "forward":
        return cake_gated_act_mxfp8_det_train_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        )
    return cake_gated_act_mxfp8_det_train_backward(
        gated_input, grad_output, rowwise=rowwise, colwise=colwise
    )


def _cupti_call_ms(fn, *, dry_run_iters: int = 10, repeat_iters: int = 100) -> list[float]:
    """Per-call GPU time of ``fn`` in milliseconds, the L2 cache flushed before
    every call: the summed duration of the GPU activities (kernels, copies,
    fills) that CUPTI records through ``torch.profiler`` between two sentinel
    kernels, an autograd backward included.

    Every timed call runs between two launches of ATen's one-cycle spin kernel;
    the first record after a sentinel is the L2 flush and is not counted.
    Device records are attributed by position, not by the host operation that
    launched them, so kernels launched through the driver API count too.
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
        (
            event
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ),
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
            f"the profiler returned {len(call_us)} complete calls of {repeat_iters}; "
            "CUPTI lost device records"
        )
    if any(us <= 0 for us in call_us):
        raise RuntimeError("CUPTI recorded no GPU activity for a timed call")
    return [us / 1000.0 for us in call_us]


def _median_ms(fn) -> float:
    return float(
        statistics.median(_cupti_call_ms(fn, dry_run_iters=10, repeat_iters=100))
    )


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        m_text, k_text = value.lower().split("x", 1)
        shape = int(m_text), int(k_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must use MxK syntax") from error
    if shape not in SHAPES:
        raise argparse.ArgumentTypeError(f"shape must be one of {SHAPES}")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError("this benchmark requires SM100 or SM103")

    shapes = tuple(args.shape) if args.shape else SHAPES
    rows = []
    for m, k in shapes:
        generator = torch.Generator(device="cuda").manual_seed(20260820 + m + k)
        gated_input = torch.randn(
            (m, 2 * k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        grad_output = torch.randn(
            (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        for mode, direction, rowwise, colwise in MODES:
            candidate = lambda: _fused(
                direction, gated_input, grad_output, rowwise, colwise
            )
            candidate()
            torch.cuda.synchronize()
            candidate_ms = _median_ms(candidate)
            rows.append(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "architecture": f"sm_{capability[0]}{capability[1]}",
                    "timing": "CUPTI cold-L2",
                    "M": m,
                    "K": k,
                    "mode": mode,
                    "candidate_ms": candidate_ms,
                }
            )

    print("| M | K | Mode | Fused (ms) |")
    print("|---:|---:|:---|---:|")
    for row in rows:
        print(
            f"| {row['M']} | {row['K']} | {row['mode']} | "
            f"{row['candidate_ms']:.6f} |"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
