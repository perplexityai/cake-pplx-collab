# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""The suite's terminal summary: one ``TIER`` line per output a GPU test judged against the
naive BF16 tier (``tests/cake/test_helpers/cake_naive.py``), passing tests included."""

from tests.cake.test_helpers.cake_naive import summarize_tiers


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)
