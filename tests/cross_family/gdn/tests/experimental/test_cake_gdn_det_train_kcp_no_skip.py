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

The cross-family GDN det suites and benchmark import the det packages without ``pytest.importorskip``: a moved module
fails the test instead of skipping it.  The in-family suites have the same check
(``kernels/gdn/tests/experimental/test_cake_gdn_det_train_names.py``); each tree checks its own files.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _skipped_det_imports(path: Path) -> list[int]:
    """Lines of ``path`` that import a GDN det module through ``pytest.importorskip`` (a literal path naming
    ``gdn_det``, or a computed one): such a test skips instead of failing when the module moves."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        func = getattr(node, "func", None)
        if isinstance(node, ast.Call) and getattr(func, "attr", None) == "importorskip":
            arg = node.args[0] if node.args else None
            literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            if not literal or "gdn_det" in arg.value:
                out.append(node.lineno)
    return out


def test_det_tests_import_the_det_packages_without_a_skip() -> None:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "tests" / "experimental").glob("test_cake_gdn_det_train*.py"))
    files.append(root / "benchmarks" / "bench_cake_gdn_det_train_kcp.py")
    assert len(files) > 3, files
    offenders = {p.name: lines for p in files if (lines := _skipped_det_imports(p))}
    assert not offenders, offenders
