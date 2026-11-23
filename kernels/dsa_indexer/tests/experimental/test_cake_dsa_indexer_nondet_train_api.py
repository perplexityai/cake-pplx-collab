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

Tests of the public entry points of the non-deterministic Cake DSA indexer top-k
(set-equality semantics):
``cake_pplx.cake_dsa_indexer_nondet_train.cake_dsa_indexer_nondet_train_topk``
and ``cake_dsa_indexer_nondet_train_topk_workspace_size`` against the backend
package, and their stable-API contract (no experimental warning).  A row's slot
order may differ between two calls, so results are compared as sets: on integer
inputs every call holds the exact selection's ids and valid count per row, in any
slot order (``naive.assert_exact`` with ``ordered=False``).  The backend contract itself
(registry, dispatch policy, validation, binding, the reference and the operator
tests) lives in ``test_cake_dsa_indexer_nondet_train.py``, which imports only the
package; the naming rules live in ``test_cake_dsa_indexer_nondet_train_names.py``.
The operator tests here need a compute capability 10.0 / 10.3 / 10.7 device with
registered generated programs and skip otherwise.
"""

import contextlib
import inspect
import re
import warnings
from pathlib import Path

import pytest
import torch
from cake_pplx.cake_dsa_indexer_nondet_train import (
    cake_dsa_indexer_nondet_train_topk,
    cake_dsa_indexer_nondet_train_topk_in_candidates,
    cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size,
    cake_dsa_indexer_nondet_train_topk_with_candidates,
    cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size,
    cake_dsa_indexer_nondet_train_topk_workspace_size,
)
from cake_pplx._backends.cake_dsa_indexer_nondet_train import cake_backend
from cake_pplx._backends.cake_dsa_indexer_nondet_train.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    TOPK_ENTRY,
    generated_program_available,
)

from tests.experimental.test_helpers import cake_dsa_indexer_naive as naive
from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref

ALERT = "does not have a deterministic implementation"


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")


def _run(inputs: ref.IndexerInputs, **extra):
    """The backend call that the public entry point wraps."""
    out = cake_backend.cake_dsa_indexer_nondet_train_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 2
    return out


def _public(inputs: ref.IndexerInputs, **extra):
    return cake_dsa_indexer_nondet_train_topk(
        inputs.q,
        inputs.k,
        inputs.w,
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_k,
        **inputs.kwargs(),
        **extra,
    )


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool):
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)


@pytest.mark.parametrize(
    "fn",
    [
        cake_dsa_indexer_nondet_train_topk,
        cake_dsa_indexer_nondet_train_topk_workspace_size,
    ],
)
def test_public_api_is_stable(fn):
    """The facades are stable APIs (no experimental marker, no warning)."""
    assert not getattr(fn, "is_experimental", False)


def test_public_entry_point_matches_backend_without_warning():
    _require_program()
    inputs = ref.make_integer_inputs(
        [200, 300], [200, 2500], top_k=128, seed=70, device="cuda"
    )
    expected = _run(inputs)
    reference = naive.assert_exact(expected, inputs, ordered=False)
    with warnings.catch_warnings():
        # no warning at all (cake_api is a no-op decorator)
        warnings.simplefilter("error")
        got = _public(inputs)
        again = _public(
            inputs,
            max_seqlen_q=max(inputs.seg_q_len),
            max_seqlen_k=max(inputs.seg_k_len),
        )
        indices_only = _public(inputs, return_scores=False)
    naive.assert_exact(got, inputs, reference, ordered=False)
    naive.assert_exact(again, inputs, reference, ordered=False)
    assert indices_only[1] is None
    # the ids of every row as a set (a row's ids are unique; its padding is -1)
    assert torch.equal(
        indices_only[0].sort(dim=1).values, reference.indices.sort(dim=1).values
    )
    assert cake_dsa_indexer_nondet_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    ) == cake_backend.cake_dsa_indexer_nondet_train_topk_workspace_size(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=128,
        device=inputs.device,
    )


def _bf16_host_args():
    return (
        torch.zeros((6, 32, 128), dtype=torch.bfloat16),
        torch.zeros((10, 128), dtype=torch.bfloat16),
        torch.zeros((6, 32), dtype=torch.float32),
        torch.tensor((0, 2, 6), dtype=torch.int32),
        torch.tensor((0, 3, 10), dtype=torch.int32),
    )


def test_public_entry_point_host_argument_rules():
    """The facade's own checks (before any device rule): segment boundaries are required, ``scores`` goes with
    ``return_scores=True``, and neither the keywords of the SM120 serving form (``backend``, ``q_scale``,
    ``block_table``, ``context_lens``, ``enable_pdl``) nor a ``deterministic`` switch are parameters of this entry
    point."""
    q, k, w, cu_q, cu_k = _bf16_host_args()
    with pytest.raises(ValueError, match="cu_seqlens_q and cu_seqlens_k"):
        cake_dsa_indexer_nondet_train_topk(q, k, w, None, cu_k, top_k=4)
    with pytest.raises(ValueError, match="return_scores=False"):
        cake_dsa_indexer_nondet_train_topk(
            q,
            k,
            w,
            cu_q,
            cu_k,
            top_k=4,
            return_scores=False,
            scores=torch.empty((6, 4), dtype=torch.float32),
        )
    for keyword in (
        "backend",
        "q_scale",
        "block_table",
        "context_lens",
        "enable_pdl",
        "deterministic",
    ):
        with pytest.raises(TypeError, match=keyword):
            cake_dsa_indexer_nondet_train_topk(
                q, k, w, cu_q, cu_k, top_k=4, **{keyword: None}
            )
    # host tensors with every rule above satisfied reach the backend's device rule
    with pytest.raises(ValueError, match="CUDA"):
        cake_dsa_indexer_nondet_train_topk(q, k, w, cu_q, cu_k, top_k=4)


def test_public_signatures_take_k_scale():
    """Every public and backend entry point takes the FP8 form's ``k_scale`` (default None, for BF16 calls) and no
    ``deterministic`` switch; the workspace bound takes the input form by its ``dtype`` keyword (default
    ``torch.bfloat16``)."""
    for fn in (
        cake_dsa_indexer_nondet_train_topk,
        cake_backend.cake_dsa_indexer_nondet_train_topk,
        cake_backend.prepare_dsa_indexer_nondet_topk,
        cake_backend.validate_dsa_indexer_nondet_inputs,
    ):
        parameters = inspect.signature(fn).parameters
        assert parameters["k_scale"].default is None, fn.__qualname__
        assert "deterministic" not in parameters, fn.__qualname__
    for fn in (
        cake_dsa_indexer_nondet_train_topk_workspace_size,
        cake_backend.cake_dsa_indexer_nondet_train_topk_workspace_size,
    ):
        parameters = inspect.signature(fn).parameters
        assert "k_scale" not in parameters and "deterministic" not in parameters
        assert parameters["dtype"].default is torch.bfloat16, fn.__qualname__


def _fp8_host_args():
    q = torch.zeros((6, 32, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    k = torch.zeros((10, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    w = torch.zeros((6, 32), dtype=torch.float32)
    cu_q = torch.tensor((0, 2, 6), dtype=torch.int32)
    cu_k = torch.tensor((0, 3, 10), dtype=torch.int32)
    return q, k, w, cu_q, cu_k


def test_public_entry_point_fp8_argument_rules():
    """The FP8 form takes ``k_scale`` for float8_e4m3fn q / k with ``softmax_scale=None``, before the device rule
    (host tensors here); a BF16 call takes no ``k_scale``, mixed q / k dtypes are rejected, and ``q_scale`` (the
    MXFP4 form of the SM120 serving entry) is not a parameter here.  The workspace bound takes
    ``dtype=float8_e4m3fn`` and names ``bfloat16`` for any other dtype, before it reads a device."""
    fq, fk, w, cu_q, cu_k = _fp8_host_args()
    q, k, *_ = _bf16_host_args()
    k_scale = torch.ones(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA"):
        cake_dsa_indexer_nondet_train_topk(
            fq, fk, w, cu_q, cu_k, top_k=4, k_scale=k_scale
        )
    with pytest.raises(TypeError, match="q_scale"):
        cake_dsa_indexer_nondet_train_topk(
            fq,
            fk,
            w,
            cu_q,
            cu_k,
            top_k=4,
            k_scale=k_scale,
            q_scale=torch.ones((6, 32), dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="softmax_scale=None"):
        cake_dsa_indexer_nondet_train_topk(
            fq, fk, w, cu_q, cu_k, top_k=4, k_scale=k_scale, softmax_scale=0.5
        )
    with pytest.raises(ValueError, match="need k_scale"):
        cake_dsa_indexer_nondet_train_topk(fq, fk, w, cu_q, cu_k, top_k=4)
    with pytest.raises(ValueError, match="bfloat16 q / k take none"):
        cake_dsa_indexer_nondet_train_topk(
            q, k, w, cu_q, cu_k, top_k=4, k_scale=k_scale
        )
    with pytest.raises(ValueError, match="k must be float8_e4m3fn"):
        cake_dsa_indexer_nondet_train_topk(
            fq, k, w, cu_q, cu_k, top_k=4, k_scale=k_scale
        )
    with pytest.raises(ValueError, match="k must be bfloat16"):
        cake_dsa_indexer_nondet_train_topk(q, fk, w, cu_q, cu_k, top_k=4)
    with pytest.raises(ValueError, match="bfloat16"):
        cake_dsa_indexer_nondet_train_topk_workspace_size(
            6, 10, 2, top_k=4, dtype=torch.float16
        )


def test_public_entry_point_fp8_matches_backend():
    """The FP8 form through the public entry point: on integer e4m3 inputs the backend call and the public call each
    hold the exact selection's ids and valid count per row, in any slot order, and both workspace bounds of the FP8
    form agree."""
    _require_program()
    if not generated_program_available(torch.device("cuda"), dtype=torch.float8_e4m3fn):
        pytest.skip("generated FP8 DSA indexer programs not registered for this device")
    inputs = ref.make_integer_fp8_inputs(
        [200, 300], [200, 2500], top_k=128, seed=71, device="cuda"
    )
    reference = naive.assert_exact(_run(inputs), inputs, ordered=False)
    naive.assert_exact(_public(inputs), inputs, reference, ordered=False)
    sizes = [
        fn(
            inputs.num_queries,
            inputs.num_keys,
            inputs.num_segments,
            top_k=128,
            device=inputs.device,
            dtype=torch.float8_e4m3fn,
        )
        for fn in (
            cake_dsa_indexer_nondet_train_topk_workspace_size,
            cake_backend.cake_dsa_indexer_nondet_train_topk_workspace_size,
        )
    ]
    assert sizes[0] == sizes[1]


def test_public_entry_point_under_torch_deterministic_mode():
    """Under ``torch.use_deterministic_algorithms(True)`` the public entry point raises PyTorch's ``RuntimeError``
    naming the backend's eager entry (no deterministic package named as a fallback); with ``warn_only=True`` it warns
    once, pointing at this file through the facade and its logging decorator, and returns the exact selection's sets
    (integer inputs), like a call outside the mode.  The workspace bound never alerts."""
    _require_program()
    inputs = ref.make_integer_inputs(
        [200, 300], [200, 2500], top_k=128, seed=72, device="cuda"
    )
    expected = _run(inputs)
    reference = naive.assert_exact(expected, inputs, ordered=False)
    size = cake_dsa_indexer_nondet_train_topk_workspace_size(
        inputs.num_queries, inputs.num_keys, inputs.num_segments, top_k=128
    )
    match = re.escape(f"{TOPK_ENTRY} {ALERT}")
    with _torch_deterministic(warn_only=False):
        with pytest.raises(RuntimeError, match=match) as info:
            _public(inputs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                cake_dsa_indexer_nondet_train_topk_workspace_size(
                    inputs.num_queries, inputs.num_keys, inputs.num_segments, top_k=128
                )
                == size
            )
    assert "_det_train" not in str(info.value)
    with (
        _torch_deterministic(warn_only=True),
        pytest.warns(UserWarning, match=ALERT) as record,
    ):
        got = _public(inputs)
    alerts = [w for w in record if str(w.message).startswith(f"{TOPK_ENTRY} {ALERT}")]
    assert len(alerts) == 1, [str(w.message) for w in record]
    assert Path(alerts[0].filename).name == Path(__file__).name, alerts[0].filename
    naive.assert_exact(got, inputs, reference, ordered=False)


# ---------------------------------------------------------------------------
# The candidate-block call (``*_topk_with_candidates``, sm_100a / sm_103a / sm_107a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        cake_dsa_indexer_nondet_train_topk_with_candidates,
        cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size,
    ],
)
def test_candidate_block_facades_are_stable(fn):
    """The candidate-block facades are stable APIs (no experimental marker)."""
    assert not getattr(fn, "is_experimental", False)


# key segments past 2048 blocks (the scan_blocks program) and of at most 2048 blocks (the plain programs and the rows
# written after them)
CB_FACADE_SHAPES = {
    "scan_blocks": ([48, 16], [17000, 900]),
    "iota": ([48, 16], [16000, 384]),
}


@pytest.mark.parametrize("shape", sorted(CB_FACADE_SHAPES))
def test_candidate_block_facade_matches_backend_and_owns_no_policy(shape):
    """The public candidate-block call returns the backend call's ``(indices, scores, candidate_blocks)`` without a
    warning (the same sets), ``None`` in place of the scores with ``return_scores=False`` (``scores`` given then is a
    ``ValueError``), the caller's output tensors when given, and the backend's workspace bound."""
    _require_program()
    inputs = ref.make_random_inputs(
        *CB_FACADE_SHAPES[shape], top_k=512, seed=7810, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    kwargs = inputs.kwargs()
    expected = tuple(
        t.clone()
        for t in cake_backend.cake_dsa_indexer_nondet_train_topk_with_candidates(
            *args, **kwargs
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = tuple(
            t.clone()
            for t in cake_dsa_indexer_nondet_train_topk_with_candidates(*args, **kwargs)
        )
    assert torch.equal(got[0].sort(1).values, expected[0].sort(1).values)
    assert torch.equal(got[2].sort(1).values, expected[2].sort(1).values)
    ids, none, blocks = cake_dsa_indexer_nondet_train_topk_with_candidates(
        *args, **kwargs, return_scores=False
    )
    assert (
        none is None
        and ids.shape == (inputs.num_queries, 512)
        and blocks.shape == (inputs.num_queries, 2048)
    )
    with pytest.raises(ValueError, match="return_scores=False"):
        cake_dsa_indexer_nondet_train_topk_with_candidates(
            *args, **kwargs, return_scores=False, scores=torch.empty_like(expected[1])
        )
    owned = (
        torch.empty_like(expected[0]),
        torch.empty_like(expected[1]),
        torch.empty_like(expected[2]),
    )
    out = cake_dsa_indexer_nondet_train_topk_with_candidates(
        *args, **kwargs, indices=owned[0], scores=owned[1], candidate_blocks=owned[2]
    )
    assert [t.data_ptr() for t in out] == [t.data_ptr() for t in owned]
    T, Tkv, S = inputs.num_queries, inputs.num_keys, inputs.num_segments
    assert cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size(
        T, Tkv, S, top_k=512
    ) == cake_backend.cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size(
        T, Tkv, S, top_k=512, device=torch.device("cuda")
    )


# ---------------------------------------------------------------------------
# The candidate-restricted top-k (``*_topk_in_candidates``, sm_100a / sm_103a / sm_107a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        cake_dsa_indexer_nondet_train_topk_in_candidates,
        cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size,
    ],
)
def test_candidate_restricted_facades_are_stable(fn):
    """The candidate-restricted facades are stable APIs (no experimental marker)."""
    assert not getattr(fn, "is_experimental", False)


def test_candidate_restricted_facade_matches_backend_and_owns_no_policy():
    """The public candidate-restricted call takes the candidate-block call's ``candidate_blocks`` as is and returns
    the backend call's ``(indices, scores)`` without a warning (the same sets), ``None`` in place of the scores with
    ``return_scores=False`` (``scores`` given then is a ``ValueError``), the caller's output tensors when given, and
    the backend's workspace bound."""
    _require_program()
    inputs = ref.make_random_inputs(
        [48, 16], [17000, 900], top_k=512, seed=7820, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    kwargs = inputs.kwargs()
    blocks = cake_dsa_indexer_nondet_train_topk_with_candidates(*args, **kwargs)[
        2
    ].clone()
    expected = tuple(
        t.clone()
        for t in cake_backend.cake_dsa_indexer_nondet_train_topk_in_candidates(
            *args, blocks, **kwargs
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = tuple(
            t.clone()
            for t in cake_dsa_indexer_nondet_train_topk_in_candidates(
                *args, blocks, **kwargs
            )
        )
    assert torch.equal(got[0].sort(1).values, expected[0].sort(1).values)
    ids, none = cake_dsa_indexer_nondet_train_topk_in_candidates(
        *args, blocks, **kwargs, return_scores=False
    )
    assert none is None and ids.shape == (inputs.num_queries, 512)
    with pytest.raises(ValueError, match="return_scores=False"):
        cake_dsa_indexer_nondet_train_topk_in_candidates(
            *args,
            blocks,
            **kwargs,
            return_scores=False,
            scores=torch.empty_like(expected[1]),
        )
    owned = (torch.empty_like(expected[0]), torch.empty_like(expected[1]))
    out = cake_dsa_indexer_nondet_train_topk_in_candidates(
        *args, blocks, **kwargs, indices=owned[0], scores=owned[1]
    )
    assert [t.data_ptr() for t in out] == [t.data_ptr() for t in owned]
    T, Tkv, S = inputs.num_queries, inputs.num_keys, inputs.num_segments
    assert cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size(
        T, Tkv, S, top_k=512
    ) == cake_backend.cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size(
        T, Tkv, S, top_k=512, device=torch.device("cuda")
    )
