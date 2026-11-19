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

Independent single-GPU fixtures for the MoK kernel tests.
One-GPU emulation of rank 0 of an EP group, for the adapter at the 384-expert layouts without a process group.

:class:`OneGpuGroup` stands in for the group's communication and :func:`one_gpu_workspace` builds rank 0's native
workspace with every peer pointer at this rank's own buffers; :func:`one_gpu_source_workspace` wraps it as the source
workspace of changing source counts.  With every token routed to this rank's experts (the gathered expert ids of every
rank are this rank's), each route reaches this rank once from every emulated peer, and the peers' writes of a combine
row, an input-gradient row or a router-weight gradient are equal bytes to one address: ``y``, ``dx`` and the
router-weight gradients are those of a single-rank layer over this rank's experts, and each routed weight gradient sums
EP copies of every route.  Not a distributed run: no peer barrier and no cross-rank race is exercised.
"""

import math

import torch

from tests.cake.test_helpers.cake_package import cake_package

backend = cake_package("cake_mok_bf16", "backend")
workspace_module = cake_package("cake_mok_bf16", "workspace")

#: Source rows of the top-6 steps over changing counts on one source workspace, in order: 1031, 257 and 513 rows are
#: not multiples of the 256-row tile (padded staging of rows ``[count, shared_rows)``), 257 and 513 come after a larger
#: count, and 768 is a multiple of the tile (unpadded staging).
CHANGED_COUNTS = (1031, 768, 257, 513)


def bound_multiplier(tokens, topk, ep, experts):
    """The requester's smallest no-overflow multiplier:
    ``ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP``."""
    worst = min(topk, experts) * tokens * ep + experts * 255
    return -(-worst // (tokens * topk)) / ep


class OneGpuGroup:
    """Single-GPU stand-in of an EP group's communication: every rank's gathered expert ids are this rank's and the
    barriers launch nothing.  An emulation, not a distributed run: with every peer pointer at this rank's own buffers
    (:func:`one_gpu_workspace`) the adapter runs rank 0's launches at the group's geometry, and since every route of
    the full skew is one of this rank's tokens, the EP writers of a combine row or router-weight gradient write equal
    bytes."""

    def all_gather_top_experts(self, routes, gathered, multicast_address, rank):
        gathered.copy_(routes.unsqueeze(0).expand_as(gathered))

    def barrier_all(self, counter, peer_pointers, multicast_address, target):
        return None

    def barrier_plan(self, counter, peer_pointers, multicast_address, target):
        return None


def one_gpu_workspace(ep, tokens, hidden, topk, capacity, device):
    """Rank 0's native workspace of an EP group on one GPU: every peer pointer is this rank's own buffer."""

    def buffer(*shape, dtype):
        value = torch.zeros(*shape, dtype=dtype, device=device)
        return value, None, [value.data_ptr()] * ep

    names = ("x_buffer", "combine_buffer", "d_y_buffer", "d_x_routed_buffer")
    shapes = ((tokens, hidden), (tokens * topk, hidden), (tokens, hidden), (tokens * topk, hidden))
    fields = {}
    for name, shape in zip(names, shapes, strict=True):
        fields[name], fields[f"{name}_handle"], fields[f"{name}_ptrs"] = buffer(*shape, dtype=torch.bfloat16)
    for name in ("router_weight_buffer", "d_router_weight_buffer"):
        fields[name], fields[f"{name}_handle"], fields[f"{name}_ptrs"] = buffer(tokens, topk, dtype=torch.float32)
    gathered = torch.full((ep, tokens, topk), -1, dtype=torch.int32, device=device)
    barrier, _, barrier_ptrs = buffer(1, dtype=torch.int32)
    return workspace_module.MoKWorkspace(
        group_name="one-gpu",
        ep_rank=0,
        ep_size=ep,
        device=device,
        num_local_tokens=tokens,
        hidden_size=hidden,
        topk=topk,
        schedule_capacity=capacity,
        **fields,
        all_gather_top_experts_buffer=gathered,
        all_gather_top_experts_buffer_handle=None,
        all_gather_top_experts_buffer_multicast_ptr=gathered.data_ptr(),
        barrier_buffer=barrier,
        barrier_buffer_handle=None,
        barrier_buffer_ptrs=barrier_ptrs,
        barrier_buffer_multicast_ptr=barrier.data_ptr(),
        barrier_target=torch.zeros(1, dtype=torch.int32, device=device),
    )


def one_gpu_source_workspace(ep, experts, counts, hidden, topk, device):
    """Rank 0's source workspace (:class:`MoKSourceWorkspace`, as ``create_mok_bf16_workspace`` returns it) on one GPU
    for the source counts ``counts``: the physical rows that factory reserves (the largest count rounded up to its row
    alignment, at least 512) and the worst-case multiplier at those rows.  Returns the workspace and the multiplier."""
    alignment = math.lcm(256, 512 // math.gcd(topk, 512))
    rows = -(-max(512, *counts) // alignment) * alignment
    multiplier = bound_multiplier(rows, topk, ep, experts)
    capacity = workspace_module.schedule_capacity_for(rows, topk, ep, multiplier)
    storage = one_gpu_workspace(ep, rows, hidden, topk, capacity, device)
    return backend.MoKSourceWorkspace(storage, (counts[0],) * ep), multiplier
