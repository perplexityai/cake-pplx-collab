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

The terminal summary prints one ``TIER`` line per output the tests judged against the naive BF16 tier
(:mod:`.cake_naive`), passing tests included.
"""

from .cake_naive import summarize_tiers


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summarize_tiers(terminalreporter)
