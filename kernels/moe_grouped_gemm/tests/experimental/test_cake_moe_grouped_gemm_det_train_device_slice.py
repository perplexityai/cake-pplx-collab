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

Device slices of the deterministic ragged BF16 MoE grouped GEMM: ``m_indptr[E + 1]``
row tables whose first entry is above 0 (``tests/cake/test_helpers/cake_moe_device_slice.py``
has the checks and the two references).

On the CPU: the registry's lead twins mirror the routes, their builds share the
CUDA floor, the row table is read on the shape alone and passed as views, an ``m_indptr``
never falls back to a route without its lead twin, ``_bind`` refuses a wrong pairing of
route and row table, the availability query needs the lead twins, and a sweep
of architectures and SM counts binds the lead twins with reference 1's plan.  On a GPU:
reference 1's bits always and reference 2's where it plans the same, every
forced stage from an odd row and from row 0 (the split tail with group 0 in it, the
probed queue split), CUDA graphs over device-copied routings, edges and whole-empty
chunks with the rows outside poisoned (the ``-0.0`` exception of
equal values), padded rows, two streams, the bits of the judged ``[E]`` cases, the
autograd function, accumulation and the binding before row tables
(negative control).
"""

import pytest
import torch

from tests.cake.test_helpers import cake_moe_device_slice as ds
from tests.experimental import test_cake_moe_grouped_gemm_det_train as suite

P = ds.Package("cake_moe_grouped_gemm_det_train")

# The cases test_{fwd,dgrad,wgrad}_matches_reference judge
# (test_judged_cases_are_the_suite_cases).
JUDGED = {
    "fwd": [
        (suite.RAGGED_E4, 512, 256),
        (suite.RAGGED_E6, 1024, 512),
        (suite.TINY_E5, 256, 64),
    ],
    "dgrad": [
        (suite.RAGGED_E4, 256, 512),
        (suite.RAGGED_E6, 1024, 512),
        (suite.TINY_E5, 64, 256),
    ],
    "wgrad": [
        (suite.RAGGED_E4, 512, 256),
        (suite.RAGGED_E6, 512, 256),
        (suite.TINY_E5, 256, 512),
    ],
}
JUDGED_TESTS = {
    "fwd": suite.test_fwd_matches_reference,
    "dgrad": suite.test_dgrad_matches_reference,
    "wgrad": suite.test_wgrad_matches_reference,
}
WGRAD_FORMS = tuple(form for form in ds.WGRAD_FORMS if form != "k512_queue")


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def test_lead_routes_mirror_their_routes():
    ds.check_registry_mirror(P)


def test_lead_builds_share_the_cuda_floor():
    ds.check_lead_builds(P)


def test_row_table_is_read_on_the_shape_as_views():
    ds.check_row_table(P)


def test_m_indptr_never_falls_back_to_the_route(monkeypatch):
    ds.check_bound_route(monkeypatch, P)


def test_availability_needs_the_lead_twins(monkeypatch):
    ds.check_available_needs_lead(monkeypatch, P)


def test_bind_refuses_a_wrong_row_table(monkeypatch):
    ds.check_bind_mismatch(monkeypatch, P)


@pytest.mark.parametrize("op", ds.OPS)
def test_host_binds_the_lead_twins(op, monkeypatch):
    ds.check_host_binding(monkeypatch, P, op)


def test_two_reference_chunks_plan_as_reference_2(monkeypatch):
    ds.check_reference_2_plans(monkeypatch, P)


@pytest.mark.parametrize("op", ds.OPS)
def test_judged_cases_are_the_suite_cases(op):
    assert JUDGED[op] == ds.judged_cases(JUDGED_TESTS[op])


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,out_dtype",
    [
        ("fwd", None),
        ("dgrad", None),
        ("wgrad", torch.bfloat16),
        ("wgrad", torch.float32),
    ],
)
def test_slices_give_both_references_bits(op, out_dtype):
    ds.check_two_references(P, op, out_dtype or torch.bfloat16)


@pytest.mark.parametrize("out_dtype", ds.OUT_DTYPES)
def test_plan_divergent_slice_gives_reference_1_bits(out_dtype, record_property):
    ds.check_plan_divergent(P, out_dtype, record_property)


@pytest.mark.parametrize("form", ds.ROW_FORMS)
@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_forced_row_stage_slices(op, form, monkeypatch):
    ds.check_row_form(monkeypatch, P, op, form)


@pytest.mark.parametrize("out_dtype", ds.OUT_DTYPES)
@pytest.mark.parametrize("form", WGRAD_FORMS)
def test_forced_wgrad_form_slices(form, out_dtype, monkeypatch, record_property):
    ds.check_wgrad_form(monkeypatch, P, form, out_dtype, record_property)


@pytest.mark.parametrize("out_dtype", ds.OUT_DTYPES)
def test_split_tail_holds_group_0(out_dtype, monkeypatch):
    ds.check_split_tail_holds_group_0(monkeypatch, P, out_dtype)


def test_slice_takes_the_probed_queue_split(monkeypatch):
    ds.check_probed_queue_split(monkeypatch, P)


@pytest.mark.parametrize(
    "op,out_dtype",
    [
        ("fwd", None),
        ("dgrad", None),
        ("wgrad", torch.bfloat16),
        ("wgrad", torch.float32),
    ],
)
def test_graph_follows_device_copied_routings(op, out_dtype, monkeypatch):
    ds.check_graph(monkeypatch, P, op, out_dtype or torch.bfloat16)


@pytest.mark.parametrize(
    "op,out_dtype",
    [
        ("fwd", None),
        ("dgrad", None),
        ("wgrad", torch.bfloat16),
        ("wgrad", torch.float32),
    ],
)
def test_edges_with_poisoned_rows(op, out_dtype):
    ds.check_edges(P, op, out_dtype or torch.bfloat16)


@pytest.mark.parametrize("route", ["fused", "composed"])
@pytest.mark.parametrize("out_dtype", ds.OUT_DTYPES)
def test_whole_empty_accumulate_signed_zero(out_dtype, route, monkeypatch):
    ds.check_empty_accumulate(monkeypatch, P, out_dtype, route)


@pytest.mark.parametrize(
    "op,form",
    [
        ("fwd", None),
        ("dgrad", None),
        ("wgrad", "k256_main"),
        ("wgrad", "k512_main_halves"),
    ],
)
def test_unaligned_row_padded_slices(op, form, monkeypatch):
    ds.check_row_padded(monkeypatch, P, op, form)


@pytest.mark.parametrize("op", ds.OPS)
def test_two_streams(op, monkeypatch):
    ds.check_two_streams(monkeypatch, P, op)


@pytest.mark.parametrize(
    "op,case",
    [(op, case) for op, cases in JUDGED.items() for case in range(len(cases))],
)
def test_slices_give_the_judged_bits(op, case, record_property):
    sizes, n, k = JUDGED[op][case]
    dtypes = ds.OUT_DTYPES if op == "wgrad" else (torch.bfloat16,)
    for out_dtype in dtypes:
        lead_rows = ds.check_judged_bits(P, suite._inputs, op, sizes, n, k, out_dtype)
        record_property(f"lead_rows_{out_dtype}", lead_rows)


def test_autograd_slice():
    ds.check_autograd_slice(P)


def test_autograd_chunks_of_one_weight():
    ds.check_autograd_chunks(P)


@pytest.mark.parametrize("route", ["fused", "composed"])
@pytest.mark.parametrize("out_dtype", ds.OUT_DTYPES)
def test_accumulate_slices(out_dtype, route, monkeypatch):
    ds.check_accumulate(monkeypatch, P, out_dtype, route)


@pytest.mark.parametrize("op", ds.OPS)
def test_binding_before_row_tables_misses_group_0(op, monkeypatch):
    ds.check_base_host(monkeypatch, P, op)
