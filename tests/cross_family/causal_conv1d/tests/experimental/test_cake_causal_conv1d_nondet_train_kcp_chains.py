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

Context-parallel chains at 32 and 64 ranks: the per-document zigzag layout cuts every
document into ``2 x W`` pieces, so at W = 64 a document is a chain of 128 fragments, and a short document's pieces are
one or two tokens long, each reading its history rows from up to three predecessor ranks and its tail rows from up to
three successor ranks.  One GPU emulates the ``W`` ranks one after the other: the package's halo exchange
(``kcp._exchange``, one ``all_to_all_single`` per direction) is replaced by the receive pool the ``W`` ranks' send rows
give the rank (the rank-agreement check runs inside the exchange and needs a process group: its own tests cover it).
Per row:

- the ranks' ``y`` and ``dx`` (put back in global order), the sum of the ranks' ``dweight`` (in rank order) and the
  outputs of the single-rank call on the unpartitioned pack are judged by the naive judge
  (``tests/cake/test_helpers/cake_naive.py``: over 32 random draws against the float64 oracle and PyTorch's BF16
  naive convolution; no bit rule);
- every pool row a rank receives is one its peers sent it (the emulated exchange checks the split tables agree).

Skipped until the exporter renders the package.
"""

from __future__ import annotations

import pytest
import torch

import cake_pplx._backends.cake_causal_conv1d_nondet_train as pkg
from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_causal_conv1d_oracle import assert_judged, problem
from tests.test_cake_causal_conv1d_det_train_kcp import WIDTH
from tests.test_cake_causal_conv1d_det_train_kcp_dist import _local, _partition

# (documents, W, C): short documents cut into one- and two-token pieces, a long one, a one-token one (pieces of zero
# tokens on most ranks), at 32 and 64 ranks.
ROWS = [
    ((200, 5000, 1, 130), 32, 256),
    ((260, 6000, 3, 129), 64, 256),
    ((128, 3001), 64, 1024),
]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc"
    ),
)


class _Halo:
    """The package's halo exchange for ``W`` emulated ranks: :meth:`__call__` records the send rows of the calling
    rank and returns its receive pool, once :meth:`deliver` has assembled the pools from every rank's sends (before
    that, a zero pool of the right shape)."""

    def __init__(self, ctxs, direction: str) -> None:
        self.ctxs = ctxs
        self.direction = direction
        self.sends: dict[int, torch.Tensor] = {}
        self.pools: list[torch.Tensor] | None = None

    def __call__(self, send, *, recv_rows, send_splits, recv_splits, ctx, **_):
        rank = int(ctx.rank)
        assert sum(send_splits) == send.shape[0]
        assert sum(recv_splits) == recv_rows
        self.sends[rank] = send.clone()
        if self.pools is None:
            return send.new_zeros((recv_rows, send.shape[1])), None
        pool = self.pools[rank]
        assert pool.shape[0] == recv_rows, (rank, pool.shape, recv_rows)
        return pool.clone(), None

    def deliver(self) -> None:
        """``all_to_all_single``: rank ``r`` receives, in source-rank order, the block of every source's send rows
        destined to ``r``; the block sizes must equal ``r``'s receive splits."""
        W = len(self.ctxs)
        assert sorted(self.sends) == list(range(W)), sorted(self.sends)
        send_name, recv_name = (
            f"{self.direction}_send_rows",
            f"{self.direction}_recv_rows",
        )
        pools = []
        for r in range(W):
            blocks = []
            for s in range(W):
                splits = list(getattr(self.ctxs[s].halo, send_name))
                start = sum(splits[:r])
                blocks.append(self.sends[s][start : start + splits[r]])
                assert splits[r] == list(getattr(self.ctxs[r].halo, recv_name))[s], (
                    f"rank {s} sends {splits[r]} rows to rank {r}, which expects "
                    f"{list(getattr(self.ctxs[r].halo, recv_name))[s]}"
                )
            pools.append(torch.cat(blocks, 0))
        self.pools = pools


def _single_rank(lengths, x, w, dy, device):
    """``y``, ``dx`` and ``dweight`` of the unpartitioned pack: one rank, every document a fragment."""
    from cake_pplx._backends.cake_kda_det_train_kcp import (
        cake_kcp_plan as plan,
    )

    part, _ = _partition(
        plan.cake_kda_det_train_kcp_partition, lengths, 1, "contiguous"
    )
    ctx = plan.cake_kda_det_train_kcp_build_context(
        part, None, 0, device, conv_kernel_size=WIDTH
    )
    y, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)
    dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hist)
    return y, dx, dw


@pytest.mark.parametrize(
    "row", ROWS, ids=lambda row: f"w{row[1]}-c{row[2]}-{'-'.join(map(str, row[0]))}"
)
def test_chains_match_reference(row, monkeypatch) -> None:
    from cake_pplx._backends.cake_kda_det_train_kcp import (
        cake_kcp_plan as plan,
    )

    lengths, W, C = row
    device = torch.device("cuda", torch.cuda.current_device())
    part, cu = _partition(
        plan.cake_kda_det_train_kcp_partition, lengths, W, "zigzag_per_document"
    )
    chain = max(
        sum(
            1
            for r in range(W)
            for f in part.fragments[r]
            if f.doc == d and f.length > 0
        )
        for d in range(len(lengths))
    )
    assert chain >= min(2 * W, max(lengths)), (
        f"longest chain has {chain} non-empty fragments"
    )
    ctxs = [
        plan.cake_kda_det_train_kcp_build_context(
            part, None, r, device, conv_kernel_size=WIDTH
        )
        for r in range(W)
    ]
    T = cu[-1]
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    rows = [_local(part, r, cu, torch.arange(T)).to(device) for r in range(W)]
    assert torch.equal(torch.sort(torch.cat(rows)).values.cpu(), torch.arange(T))

    def launch(_draw: int, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x_all = torch.randn(T, C, generator=g).to(torch.bfloat16).to(device)
        w = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous().to(device)
        dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16).to(device)
        xs = [_local(part, r, cu, x_all) for r in range(W)]
        dys = [_local(part, r, cu, dy_all) for r in range(W)]
        y_one, dx_one, dw_one = _single_rank(lengths, x_all, w, dy_all, device)

        # forward: the send rows of every rank, then each rank's call on its pool
        fwd = _Halo(ctxs, "fwd")
        monkeypatch.setattr(conv_kcp, "_exchange", fwd)
        for r in range(W):
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(xs[r], w, ctxs[r])
        fwd.deliver()
        hists = []
        y_world = torch.full_like(x_all, float("nan"))
        for r in range(W):
            y, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                xs[r], w, ctxs[r]
            )
            y_world[rows[r]] = y
            hists.append(hist)
        # backward: the head rows every rank sends, then each rank's call on its tail pool
        bwd = _Halo(ctxs, "bwd")
        monkeypatch.setattr(conv_kcp, "_exchange", bwd)
        for r in range(W):
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                xs[r], w, ctxs[r], dys[r], hists[r]
            )
        bwd.deliver()
        dx_world = torch.full_like(x_all, float("nan"))
        dw_sum = None
        for r in range(W):
            dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                xs[r], w, ctxs[r], dys[r], hists[r]
            )
            dx_world[rows[r]] = dx
            dw_sum = dw.clone() if dw_sum is None else dw_sum + dw
        torch.cuda.synchronize()
        outputs = {
            "y": [y_world, y_one],
            "dx": [dx_world, dx_one],
            "dweight": [dw_sum, dw_one],
        }
        return outputs, problem(x_all, w, cu_dev, None, dy_all)

    assert_judged(launch, seed=7815, label=f"w{W}-c{C}")
