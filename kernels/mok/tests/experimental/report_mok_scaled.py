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

Memory of the 512-expert top-10 group (``router_weight_gradient="direct"``, ``shared_expert="scaled"``): a report, not
a test (no pass or fail; pytest does not collect it).  The workspace and the state a layer retains are reported
separately, since a caller sizes them differently: the workspace once per EP group and adapter, the retained state once
per layer.

Run from the directory that holds ``tests`` on an SM100a, SM103a or SM107a GPU::

    python -m tests.experimental.report_mok_scaled --tokens 32768 --hidden 4096 --intermediate 1024 --out report.txt

One step of the adapter at EP8 ``(8, 64)`` as rank 0 of its EP group emulated on one GPU
(:mod:`tests.experimental._mok_one_gpu`, every route on this rank, the worst-case schedule capacity), on inputs drawn on
the CPU.  Per part, the bytes of the distinct storages of its tensors (:func:`memory_report`; a storage reached from
several parts counts in the first):

* ``workspace``: the caller-owned symmetric workspace (``create_mok_bf16_workspace``);
* ``schedule``: ``build_schedule``'s schedule, the state a checkpointed layer retains from forward to backward;
* ``context``: the saved forward context, which a layer without checkpointing retains as well;
* ``forward_peak`` / ``backward_peak``: the peak device memory a call allocates beyond what is allocated before it
  (``torch.cuda.max_memory_allocated``), its transients: the backward's include the scaled ``dy`` ring and the two
  shared-scale operands ``BF16(s * dy)`` and ``BF16(s * x)``.
"""

from __future__ import annotations

import argparse
import dataclasses

import torch

#: The parts of :func:`memory_report`, in the order a storage is counted.
PARTS = ("workspace", "schedule", "context")


def _storages(value, seen, found):
    """Collect ``(device, data_ptr) -> nbytes`` of every tensor storage reachable from ``value`` through dataclass
    fields, tuples, lists and dict values; ``seen`` holds the ids of the objects already walked."""
    if value is None or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        found.setdefault((str(value.device), storage.data_ptr()), storage.nbytes())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _storages(getattr(value, field.name), seen, found)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _storages(item, seen, found)
    elif isinstance(value, dict):
        for item in value.values():
            _storages(item, seen, found)


def part_bytes(parts):
    """Bytes per part of ``parts`` (``(name, object)`` pairs, in order): the distinct storages reachable from each
    object that no earlier part reaches."""
    counted, out = {}, {}
    for name, value in parts:
        found = {}
        _storages(value, set(), found)
        out[name] = sum(size for key, size in found.items() if key not in counted)
        counted.update(found)
    return out


def memory_report(workspace, schedule, context=None):
    """The workspace, the schedule and the saved context in bytes (:func:`part_bytes`, in that order), and the state a
    layer retains: ``retained_checkpointed`` (the schedule) and ``retained_with_context`` (the schedule and the
    context)."""
    out = part_bytes(
        (("workspace", workspace), ("schedule", schedule), ("context", context))
    )
    out["retained_checkpointed"] = out["schedule"]
    out["retained_with_context"] = out["schedule"] + out["context"]
    return out


def _peak(run):
    """``run()`` and the peak bytes it allocates beyond what is allocated before it."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = run()
    torch.cuda.synchronize()
    return result, torch.cuda.max_memory_allocated() - before


def main(argv=None):
    from tests.cake.test_helpers.cake_package import cake_package
    from tests.experimental._mok_reference import rand, randn
    from tests.experimental._mok_one_gpu import (
        OneGpuGroup,
        bound_multiplier,
        one_gpu_workspace,
    )

    backend = cake_package("cake_mok_bf16", "backend")
    kernels = cake_package("cake_mok_bf16", "_kernels")
    workspace_module = cake_package("cake_mok_bf16", "workspace")
    group = kernels.MODE_GROUPS["scaled_top10"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--macro", type=int, default=32768)
    parser.add_argument("--mini", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    (ep, experts), topk = group.layouts[0], group.topk
    t, h, i = args.tokens, args.hidden, args.intermediate
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep,
        experts,
        topk,
        fp32_wgrad=True,
        router_weight_gradient="direct",
        shared_expert="scaled",
    )
    functional.communication = OneGpuGroup()
    multiplier = bound_multiplier(t, topk, ep, experts)
    capacity = workspace_module.schedule_capacity_for(t, topk, ep, multiplier)
    workspace = one_gpu_workspace(ep, t, h, topk, capacity, device)
    config = workspace_module.MoKConfig(
        fwd_num_comm_sms=40,
        bwd_num_comm_sms=40,
        minibatch_size=args.mini,
        macrobatch_size=args.macro,
        schedule_capacity_multiplier=multiplier,
    )
    g = torch.Generator().manual_seed(args.seed)
    x = (randn(t, h, generator=g) * 0.125).bfloat16().to(device)
    dy = (randn(t, h, generator=g) * 0.125).bfloat16().to(device)
    weights = [
        (randn(*shape, generator=g) * scale).bfloat16().to(device)
        for shape, scale in (
            ((i, h), h**-0.5), ((i, h), h**-0.5), ((h, i), i**-0.5),
            ((experts, i, h), h**-0.5), ((experts, i, h), h**-0.5), ((experts, h, i), i**-0.5),
        )
    ]  # fmt: skip
    scores = (rand(t, topk, generator=g) * 1.5).to(device)
    scale = (rand(t, generator=g) * 2.0).to(device)
    ids = rand(t, experts, generator=g).argsort(-1)[:, :topk].contiguous().to(device)
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    schedule = functional.build_schedule(
        workspace, config, ids, num_local_experts=experts
    )

    def forward():
        return functional.forward(
            config, workspace, schedule, x, scores, *weights, shared_scale=scale
        )

    def backward():
        return functional.backward(
            config, workspace, schedule, context, dy, x, scores, *weights, weight_grad_accumulators=accumulators,
            shared_scale=scale,
        )  # fmt: skip

    forward()
    (_, context), forward_peak = _peak(forward)
    backward()
    _, backward_peak = _peak(backward)
    report = memory_report(workspace, schedule, context)
    report.update(forward_peak=forward_peak, backward_peak=backward_peak)
    lines = [
        f"layout ep{ep}-local{experts} top{topk} tokens {t} hidden {h} intermediate {i} macro {args.macro} "
        f"multiplier {multiplier:.6g} capacity {capacity} routed rows {int(schedule.num_tokens.item())} "
        f"device {torch.cuda.get_device_name(device)}"
    ]
    lines += [
        f"{name} {value} B {value / 2**20:.1f} MiB" for name, value in report.items()
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
