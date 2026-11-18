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

The causal_conv1d suites' accuracy verdicts: each test that judges a training output against the naive BF16 tier
(``tests.cake.test_helpers.cake_causal_conv1d_oracle``) records it on its item, and the terminal summary prints one
``TIER`` line per recorded output, passing tests included.
"""

from __future__ import annotations

import pytest

from tests.cake.test_helpers import cake_causal_conv1d_oracle
from tests.cake.test_helpers.cake_naive import summarize_tiers


@pytest.fixture(autouse=True)
def _record_naive_tiers(request):
    """Point the judge at the running item, so its verdicts land in the item's user properties."""

    cake_causal_conv1d_oracle.set_node(request.node)
    yield
    cake_causal_conv1d_oracle.set_node(None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)
