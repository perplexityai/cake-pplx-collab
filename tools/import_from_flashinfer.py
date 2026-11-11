#!/usr/bin/env python3
"""Import one Cake training family from a FlashInfer checkout into ``kernels/<family>/``.

The import is mechanical and repeatable:

1. Copy the family's backend packages (``flashinfer/experimental/cake/<module>/``) to
   ``kernels/<family>/cake_pplx/_backends/<module>/`` and its facades
   (``flashinfer/<module>.py``) to ``kernels/<family>/cake_pplx/<module>.py``.
2. Copy the family's tests and benchmarks, plus the transitive closure of the
   ``tests.*`` helper modules they import, keeping their upstream paths.
3. Rewrite import paths (``flashinfer.experimental.cake.`` -> ``cake_pplx._backends.``,
   ``flashinfer.cake_*`` -> ``cake_pplx.cake_*``), install the per-package JIT shim
   and the per-family API decorator, and point ``build.json`` at the new package.
4. Write ``pyproject.toml``, ``VERSION``, ``README.md`` and ``UPSTREAM.json``.
5. Report every remaining ``flashinfer`` reference, split into runtime and test code.

Usage::

    python tools/import_from_flashinfer.py --src /path/to/flashinfer --family kda --dest kernels
    python tools/import_from_flashinfer.py --src /path/to/flashinfer --all --dest kernels
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TEMPLATES = TOOLS / "templates"

# Longest names first: ``rmsnorm`` must not claim ``rmsnorm_silu_gate``, ``mhc`` must not claim ``mhc_head``.
FAMILIES = [
    "rmsnorm_silu_gate",
    "rmsnorm",
    "causal_conv1d",
    "mhc_head",
    "mhc",
    "gdn",
    "kda",
    "lm_head",
    "moe_grouped_gemm",
    "dense_gemm",
    "dsa_sparse_attention",
    "dsa_indexer",
    "gqa",
    "gated_act_mxfp8",
]

# Extra top-level FlashInfer modules that belong to one family.
EXTRA_MODULES = {"lm_head": ["_cake_lm_head_train_knobs"]}

# Upstream tests that exercise several families; they go to ``integration/``.
CROSS_FAMILY_TESTS = {
    "tests/cpu/cake/test_cake_kcp_partition_parity.py",
}

MODULE_RE = r"cake_{family}_(?:det|nondet)_train(?:_kcp)?"
SKIP_DIRS = {"__pycache__", "aot", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".rst", ".txt", ".toml", ".cfg"}
ALLOWED_RUNTIME = {"torch", "tvm_ffi", "numpy", "cuda"}


def module_regex(family: str) -> re.Pattern[str]:
    return re.compile(r"^" + MODULE_RE.format(family=family) + r"$")


def family_of_name(name: str) -> str | None:
    for family in FAMILIES:
        if re.search(r"(?:^|[^a-z0-9])cake_" + family + r"_(?:det|nondet|train|kcp)", name) or re.search(
            r"(?:^|/)(?:test|bench|diag)_cake_" + family + r"(?:_|\.)", name
        ):
            return family
    return None


def git_head(src: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def tracked_files(src: Path, *paths: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(src), "ls-files", "--", *paths], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


# --------------------------------------------------------------------------- selection


def select(src: Path, family: str) -> dict[str, list[str]]:
    """Upstream files of ``family``: backend packages, facades, extra modules, tests, benchmarks."""
    pattern = module_regex(family)
    cake_root = src / "flashinfer" / "experimental" / "cake"
    packages = sorted(p.name for p in cake_root.iterdir() if p.is_dir() and pattern.match(p.name))
    facades = sorted(
        p.stem for p in (src / "flashinfer").glob("cake_*.py") if pattern.match(p.stem)
    ) + EXTRA_MODULES.get(family, [])
    tests = [
        f
        for f in tracked_files(src, "tests")
        if f.endswith(".py") and f not in CROSS_FAMILY_TESTS and family_of_name(f) == family
    ]
    benches = [
        f
        for f in tracked_files(src, "benchmarks")
        if f.endswith(".py") and family_of_name(f) == family
    ]
    return {"packages": packages, "facades": facades, "tests": sorted(tests), "benchmarks": sorted(benches)}


def test_closure(src: Path, roots: list[str]) -> list[str]:
    """``roots`` plus every ``tests.*`` module they import, transitively, plus the packages' ``__init__.py``."""
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = src / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level > 0:
                base = Path(rel).parent
                for _ in range(node.level - 1):
                    base = base.parent
                module = ".".join(base.parts) + (f".{node.module}" if node.module else "")
                names = [module] + [f"{module}.{alias.name}" for alias in node.names]
            for name in names:
                if not name.startswith("tests."):
                    continue
                candidate = Path(*name.split("."))
                for option in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if (src / option).is_file():
                        todo.append(str(option))
    files = set(seen)
    for rel in list(seen):
        parent = Path(rel).parent
        while parent.parts:
            init = parent / "__init__.py"
            if (src / init).is_file():
                files.add(str(init))
            parent = parent.parent
    return sorted(files)


# --------------------------------------------------------------------------- rewriting

REWRITES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bflashinfer\.experimental\.cake\."), "cake_pplx._backends."),
    (re.compile(r"\bflashinfer/experimental/cake/"), "cake_pplx/_backends/"),
    (re.compile(r"from \.experimental\.cake\."), "from ._backends."),
    (re.compile(r"from \.experimental\.cake import"), "from ._backends import"),
    (re.compile(r"\bflashinfer\.(_?cake_[a-z0-9_]+_train)"), r"cake_pplx.\1"),
    (re.compile(r"\bflashinfer\.(_cake_lm_head_train_knobs)"), r"cake_pplx.\1"),
    (re.compile(r"\bflashinfer/(_?cake_[a-z0-9_]+_train[a-z0-9_]*\.py)"), r"cake_pplx/\1"),
    (re.compile(r"from flashinfer import (_?cake_[a-z0-9_]+_train)"), r"from cake_pplx import \1"),
]

JIT_IMPORT_OLD = (
    "        from flashinfer.jit.core import gen_jit_spec\n"
    "        from flashinfer.jit.cpp_ext import get_cuda_version\n"
)
JIT_IMPORT_NEW = "        from ._cake_pplx_jit import gen_jit_spec, get_cuda_version\n"


def rewrite_text(text: str) -> str:
    for pattern, replacement in REWRITES:
        text = pattern.sub(replacement, text)
    return text


def rewrite_backend_py(path: Path, package_root: Path, family: str) -> None:
    text = rewrite_text(path.read_text(encoding="utf-8"))
    depth = len(path.relative_to(package_root).parts) - 1
    dots = "." * (depth + 1)
    if path.name == "cake_build.py":
        if JIT_IMPORT_OLD not in text:
            raise RuntimeError(f"{path}: the FlashInfer JIT import block changed upstream")
        text = text.replace(JIT_IMPORT_OLD, JIT_IMPORT_NEW)
        text = text.replace("and flashinfer's JIT is not available", "and the cake_pplx JIT shim is not available")
    text = re.sub(
        r"from flashinfer\.jit\.(?:core|cpp_ext) import ",
        f"from {dots}_cake_pplx_jit import ",
        text,
    )
    text = text.replace('sys.modules["flashinfer.jit.cpp_ext"]', f'sys.modules[__package__ + "._cake_pplx_jit"]')
    path.write_text(text, encoding="utf-8")


def rewrite_facade(path: Path, family: str) -> None:
    text = rewrite_text(path.read_text(encoding="utf-8"))
    text = re.sub(
        r"from \.api_logging import flashinfer_api\b",
        f"from ._cake_pplx_api_{family} import cake_api",
        text,
    )
    text = re.sub(r"\bflashinfer_api\b", "cake_api", text)
    path.write_text(text, encoding="utf-8")


def fix_module_imports(text: str) -> str:
    """After rewriting, ``cake_pplx.<module>`` attribute use needs ``import cake_pplx.<module>``."""
    used = sorted(set(re.findall(r"(?<![\w.])cake_pplx\.(_?cake_[a-z0-9_]+)\b", text)))
    if not used:
        return text
    tree = ast.parse(text)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    missing = [m for m in used if f"cake_pplx.{m}" not in imported]
    flashinfer_left = re.search(r"(?<![\w.])flashinfer\.", text) is not None
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if re.fullmatch(r"(\s*)import flashinfer(\s*#.*)?", line):
            indent = re.match(r"\s*", line).group(0)
            new = [f"{indent}import cake_pplx.{m}" for m in missing]
            if flashinfer_left:
                new.insert(0, line)
            lines[index : index + 1] = new
            missing = []
            break
    text = "\n".join(lines)
    if missing:
        tree = ast.parse(text)
        insert_at = 0
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) or (
                isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
            ):
                insert_at = node.end_lineno
            else:
                break
        lines = text.split("\n")
        lines[insert_at:insert_at] = [f"import cake_pplx.{m}" for m in missing]
        text = "\n".join(lines)
    return text


def rewrite_test(path: Path) -> None:
    text = rewrite_text(path.read_text(encoding="utf-8"))
    text = re.sub(
        r"from flashinfer import (?:\(\s*)?((?:_?cake_[a-z0-9_]+_train[a-z0-9_]*\s*,?\s*)+)\)?",
        lambda m: "from cake_pplx import " + m.group(1).strip().rstrip(","),
        text,
    )
    try:
        text = fix_module_imports(text)
    except SyntaxError:
        pass
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- metadata


def tvm_ffi_requirement(packages: list[Path]) -> str:
    specs = set()
    for package in packages:
        document = json.loads((package / "build.json").read_text(encoding="utf-8"))
        specs.add(document["common"]["requires"]["tvm_ffi"]["python"])
    if len(specs) != 1:
        raise RuntimeError(f"packages disagree on the tvm_ffi requirement: {sorted(specs)}")
    return specs.pop()


def write_metadata(dest: Path, family: str, selection: dict[str, list[str]], src_commit: str) -> None:
    backends = dest / "cake_pplx" / "_backends"
    packages = [backends / name for name in selection["packages"]]
    runtime_text = "\n".join(p.read_text(encoding="utf-8") for pkg in packages for p in pkg.rglob("*.py"))
    dependencies = ["torch", tvm_ffi_requirement(packages)]
    if "cuda.bindings" in runtime_text:
        dependencies.append("cuda-python")
    distribution = "cake-pplx-" + family.replace("_", "-")
    if not (dest / "VERSION").exists():
        (dest / "VERSION").write_text("0.1.0\n", encoding="utf-8")
    deps = ",\n".join(f'    "{d}"' for d in dependencies)
    (dest / "pyproject.toml").write_text(
        f"""[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "{distribution}"
dynamic = ["version"]
description = "Cake {family} training kernels (det / nondet) for Perplexity"
requires-python = ">=3.10"
license = {{ text = "Apache-2.0" }}
dependencies = [
{deps},
]

[project.optional-dependencies]
test = ["pytest", "numpy"]

[tool.setuptools.dynamic]
version = {{ file = "VERSION" }}

[tool.setuptools.packages.find]
where = ["."]
include = ["cake_pplx*"]
namespaces = true

[tool.setuptools.package-data]
"*" = ["build.json", "build.test.json", "README.md", "csrc/**/*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
""",
        encoding="utf-8",
    )
    generators = {}
    for package in packages:
        document = json.loads((package / "build.json").read_text(encoding="utf-8"))
        generators[package.name] = document.get("generator")
    upstream = {
        "source": "https://github.com/flashinfer-ai/flashinfer",
        "commit": src_commit,
        "packages": {name: f"flashinfer/experimental/cake/{name}" for name in selection["packages"]},
        "facades": {name: f"flashinfer/{name}.py" for name in selection["facades"]},
        "tests": selection["tests"],
        "benchmarks": selection["benchmarks"],
        "generators": generators,
    }
    (dest / "UPSTREAM.json").write_text(json.dumps(upstream, indent=2) + "\n", encoding="utf-8")
    facade_list = "\n".join(f"- `cake_pplx.{name}`" for name in selection["facades"] if not name.startswith("_"))
    (dest / "README.md").write_text(
        f"""# {distribution}

Cake `{family}` training kernels. Public modules:

{facade_list}

Install this family alone, pinned to a commit:

```bash
pip install "{distribution} @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/{family}"
```

Imported from FlashInfer `{src_commit[:12]}`; see `UPSTREAM.json`. The repository rules are in
[`ARCHITECTURE.md`](../../ARCHITECTURE.md).
""",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- report


def residual_report(dest: Path) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"runtime": [], "tests": []}
    for path in sorted(dest.rglob("*.py")):
        rel = path.relative_to(dest)
        kind = "runtime" if rel.parts[0] == "cake_pplx" else "tests"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bflashinfer\b", line) and not line.lstrip().startswith("#"):
                report[kind].append(f"{rel}:{number}: {line.strip()[:160]}")
    return report


# --------------------------------------------------------------------------- main


def copy_tree(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, ignore=lambda d, names: [n for n in names if n in SKIP_DIRS or n.endswith(".so")])


def import_family(src: Path, family: str, kernels: Path) -> dict[str, object]:
    selection = select(src, family)
    if not selection["packages"]:
        raise RuntimeError(f"{family}: no backend packages under {src}/flashinfer/experimental/cake")
    dest = kernels / family
    version = (dest / "VERSION").read_text(encoding="utf-8") if (dest / "VERSION").exists() else None
    for sub in ("cake_pplx", "tests", "benchmarks"):
        shutil.rmtree(dest / sub, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    if version is not None:
        (dest / "VERSION").write_text(version, encoding="utf-8")
    backends = dest / "cake_pplx" / "_backends"
    backends.mkdir(parents=True)
    for name in selection["packages"]:
        target = backends / name
        copy_tree(src / "flashinfer" / "experimental" / "cake" / name, target)
        shutil.copy2(TEMPLATES / "_cake_pplx_jit.py", target / "_cake_pplx_jit.py")
        for path in target.rglob("*.py"):
            rewrite_backend_py(path, target, family)
        for path in list(target.rglob("*.md")) + [target / "build.json", target / "build.test.json"]:
            if path.is_file():
                path.write_text(rewrite_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    for name in selection["facades"]:
        target = dest / "cake_pplx" / f"{name}.py"
        shutil.copy2(src / "flashinfer" / f"{name}.py", target)
        rewrite_facade(target, family)
    api = (TEMPLATES / "_cake_pplx_api.py").read_text(encoding="utf-8")
    (dest / "cake_pplx" / f"_cake_pplx_api_{family}.py").write_text(api, encoding="utf-8")
    tests = test_closure(src, selection["tests"])
    for rel in tests:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
        rewrite_test(target)
    for rel in selection["benchmarks"]:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
        rewrite_test(target)
    write_metadata(dest, family, selection, git_head(src))
    report = residual_report(dest)
    foreign = sorted(
        {
            f
            for f in tests
            if family_of_name(f) not in (None, family)
        }
    )
    return {
        "family": family,
        "packages": selection["packages"],
        "facades": selection["facades"],
        "tests": len(selection["tests"]),
        "test_closure": len(tests),
        "foreign_test_files": foreign,
        "benchmarks": len(selection["benchmarks"]),
        "residual_runtime": report["runtime"],
        "residual_tests": report["tests"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="FlashInfer checkout")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent.parent / "kernels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", choices=FAMILIES, action="append")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--report", type=Path, help="write the JSON import report here")
    args = parser.parse_args()
    families = FAMILIES if args.all else args.family
    results = [import_family(args.src.resolve(), family, args.dest.resolve()) for family in families]
    for result in results:
        print(
            f"{result['family']}: {len(result['packages'])} packages, {len(result['facades'])} facades, "
            f"{result['tests']} tests ({result['test_closure']} files with helpers), {result['benchmarks']} benchmarks, "
            f"residual flashinfer refs: runtime {len(result['residual_runtime'])}, tests {len(result['residual_tests'])}"
        )
        if result["foreign_test_files"]:
            print(f"  helpers named after other families: {result['foreign_test_files']}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
