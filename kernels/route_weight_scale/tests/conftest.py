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

The family's pytest hooks: the terminal summary prints the measured lines the tests reported
(:data:`tests.cake.test_helpers.route_weight_scale_reference.REPORT`: the ``dw`` error against FP64 beside the FP32
row sum's, and the peak extra device memory), passing tests included.
"""

from tests.cake.test_helpers.route_weight_scale_reference import REPORT


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if REPORT:
        terminalreporter.section("route_weight_scale measurements")
        for line in REPORT:
            terminalreporter.write_line(line)
