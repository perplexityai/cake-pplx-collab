"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The KDA Q/K L2-norm reference against the shared GDN reference (``cake_l2n`` of the ``cake_gdn_det_train_kcp``
package, family ``gdn``), bit for bit.  Moved from ``kernels/kda/tests/experimental/test_cake_kda_det_train_l2n.py``:
it needs the gdn family installed beside kda.
"""

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_kda_det_train", "cake_backend")

D = cake_backend.D


def _cake_l2n():
    try:
        from cake_pplx._backends.cake_gdn_det_train_kcp import cake_l2n
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "the shared GDN reference cake_gdn_det_train_kcp/cake_l2n.py is not installed (family cake-pplx-gdn)"
        )
    return cake_l2n


def _edge_rows(H: int = 3) -> torch.Tensor:
    """BF16 raw rows ``[T, H, 128]`` with the edge cases of the norm: random of several magnitudes, an all-zero row
    (``ss = 0``: ``r = 1000``), signed zeros, BF16 subnormals, a row whose ``ss`` overflows FP32 (``r = 0``), one
    large channel among tiny ones."""
    g = torch.Generator().manual_seed(7817)
    rows = [torch.randn(64, H, D, generator=g) * s for s in (1e-3, 1.0, 30.0, 3e3)]
    special = torch.zeros(6, H, D)
    special[1] = -0.0
    special[2] = torch.tensor(2.0**-130)  # BF16 subnormal
    special[3, :, ::2] = -(2.0**-133)
    special[4] = 3e19  # ss overflows FP32
    special[5] = 1e-20
    special[5, :, 17] = 5.0
    return torch.cat(rows + [special]).to(torch.bfloat16)


def test_kda_reference_is_bitwise_the_shared_gdn_reference() -> None:
    from tests.experimental.test_helpers import (
        cake_kda_det_train_reference_l2n as kda_l2n,
    )

    gdn = _cake_l2n()
    assert not gdn.flush_denormal_active()
    x = _edge_rows()
    assert torch.equal(
        _bits(kda_l2n.sum_of_squares(x)), _bits(gdn.sum_of_squares(x.float()))
    )
    qn_k, r_k = kda_l2n.qk_l2norm(x)
    qn_g, r_g = gdn.qk_l2norm_ref(x)
    assert torch.equal(_bits(r_k), _bits(r_g)) and torch.equal(_bits(qn_k), _bits(qn_g))
    # a strided view gives the contiguous copy's bits (contract C11 at the reference level)
    wide = torch.zeros(x.shape[0], 2 * x.shape[1] * D + 8, dtype=torch.bfloat16)
    wide[:, 8 : 8 + x.shape[1] * D] = x.reshape(x.shape[0], -1)
    view = wide[:, 8 : 8 + x.shape[1] * D].view(x.shape)
    qn_v, r_v = kda_l2n.qk_l2norm(view)
    assert torch.equal(_bits(qn_v), _bits(qn_k)) and torch.equal(_bits(r_v), _bits(r_k))
    g = torch.Generator().manual_seed(5747)
    for scale in (1e-6, 1.0, 1e4):
        Dq = torch.randn(x.shape, generator=g) * scale
        Dq[-6:-4] = 0.0
        a = kda_l2n.l2n_pullback(x, r_k, Dq)
        b = gdn.l2n_pullback(x, r_g, Dq)
        assert torch.equal(_bits(a), _bits(b))
        assert torch.equal(_bits(a.to(torch.bfloat16)), _bits(b.to(torch.bfloat16)))
    assert torch.equal(_bits(kda_l2n.l2n_dot(Dq)), _bits(gdn.l2n_dot(Dq)))



def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(
        torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    )
