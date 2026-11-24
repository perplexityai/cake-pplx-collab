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

Wide outputs of the non-deterministic ragged BF16 MoE grouped GEMM: forward and data-gradient
outputs whose span passes 2^31 - 1 elements launch in row chunks on the architectures of
``wide_rows_archs`` (``tests/cake/test_helpers/cake_moe_wide_rows.py`` has the checks).

On the CPU: the shipped constants, the chunk arithmetic, the row tables of the chunks, a call
within the span planned and bound as one launch, a wide call's chunks, rows and row tables (meta
tensors over sm_100a, sm_103a and sm_107a), and the refusals (an architecture taken out of
``wide_rows_archs``, a row pitch past the span).  On a GPU:
forced chunks of every served stage against one launch (bitwise), the issue's 655,360 rows over 64
and 32 skewed groups and the boundary row counts 524,287 / 524,288 / 524,289 / 1,048,576 judged
against the naive BF16 tier and windowed bitwise against independent calls within the span, an
``m_indptr`` device slice into a slice of a larger ``out``, dynamic routings eager and from one
CUDA graph with no host synchronization, a checkpointed MoE block, a weight gradient whose operands
pass 2^31 elements, and repeats.
"""

import pytest

from tests.cake.test_helpers import cake_moe_device_slice as ds
from tests.cake.test_helpers import cake_moe_wide_rows as wr

P = ds.Package("cake_moe_grouped_gemm_nondet_train")


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_wide_rows_constants():
    wr.check_constants(P)


@pytest.mark.cpu_only
def test_row_chunks_fit_the_span():
    wr.check_row_chunks(P)


@pytest.mark.cpu_only
def test_row_tables_hold_each_chunks_rows():
    wr.check_row_tables(P)


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_within_span_binds_one_launch(op, monkeypatch):
    wr.check_within_span(monkeypatch, P, op)


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_wide_call_binds_row_chunks(op, monkeypatch):
    wr.check_wide_binding(monkeypatch, P, op)


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_wide_call_refusals(op, monkeypatch):
    wr.check_refusals(monkeypatch, P, op)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ds.ROW_FORMS)
@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_forced_chunks_give_one_launch_bits(op, form, monkeypatch, record_property):
    wr.check_forced_chunks(monkeypatch, P, op, form, record_property)


@pytest.mark.parametrize("groups", wr.ISSUE_GROUPS)
@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_issue_rows_meet_the_naive_tier(op, groups, monkeypatch, record_property):
    wr.check_issue_rows(monkeypatch, P, op, wr.ISSUE_ROWS, groups, record_property)


@pytest.mark.parametrize("rows", wr.BOUNDARY_ROWS)
@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_boundary_rows_meet_the_naive_tier(op, rows, monkeypatch, record_property):
    wr.check_issue_rows(monkeypatch, P, op, rows, 64, record_property)


@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_device_slice_into_a_slice_of_out(op, record_property):
    wr.check_slice(P, op, record_property)


@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_dynamic_rows_graph_and_no_host_sync(op, record_property):
    wr.check_dynamic_rows(P, op, record_property)


def test_checkpointed_block_gives_the_plain_bits(record_property):
    wr.check_checkpoint_block(P, record_property)


def test_wgrad_operands_past_the_span_meet_the_naive_tier(record_property):
    wr.check_wgrad_operands(P, record_property)


@pytest.mark.parametrize("op", wr.ROW_OPS)
def test_wide_launch_repeats_bit_for_bit(op, record_property):
    wr.check_repeat(P, op, record_property)
