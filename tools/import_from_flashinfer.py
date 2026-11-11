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

# Upstream tests that exercise several families; they go to ``tests/cross_family/``.
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


def is_training(path: str) -> bool:
    """A training test or benchmark (FlashInfer also has inference ``cake_gdn`` / ``cake_kda`` suites)."""
    return re.search(r"cake_.*(?:train|_det|nondet|kcp)", path) is not None


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
        if f.endswith(".py") and f not in CROSS_FAMILY_TESTS and is_training(f) and family_of_name(f) == family
    ]
    benches = [
        f
        for f in tracked_files(src, "benchmarks")
        if f.endswith(".py") and is_training(f) and family_of_name(f) == family
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
    (re.compile(r"from flashinfer\.experimental\.cake import"), "from cake_pplx._backends import"),
    (re.compile(r"from flashinfer import (_?cake_[a-z0-9_]*_train(?:_kcp)?)\b(?=\s+as\b|\s*$)", re.M), r"from cake_pplx import \1"),
    # regex-escaped module paths inside test patterns, e.g. r"flashinfer\.cake_rmsnorm_det_train"
    (re.compile(r"\bflashinfer\\\.experimental\\\.cake\\\."), lambda m: "cake_pplx\\._backends\\."),
    (re.compile(r"\bflashinfer\\\.(?=_?cake_[a-z0-9_]*_train)"), lambda m: "cake_pplx\\."),
    (re.compile(r"\bflashinfer\.(_?cake_[a-z0-9_]*_train(?:_kcp)?)\b"), r"cake_pplx.\1"),
    (re.compile(r"\bflashinfer\.(_cake_lm_head_train_knobs)"), r"cake_pplx.\1"),
    (re.compile(r"\bflashinfer/(_?cake_[a-z0-9_]+_train[a-z0-9_]*\.py)"), r"cake_pplx/\1"),
]

def new_module(upstream: str) -> str:
    """``.cake_x`` / ``.experimental.cake.y`` (relative to ``flashinfer``) -> the ``cake_pplx`` module."""
    name = upstream.lstrip(".")
    if name.startswith("experimental.cake."):
        return "cake_pplx._backends." + name[len("experimental.cake.") :]
    return "cake_pplx." + name


def export_map(src: Path) -> dict[str, tuple[str, str | None]]:
    """Top-level ``flashinfer.<name>`` Cake training exports -> (``cake_pplx`` module, attribute or None)."""
    tree = ast.parse((src / "flashinfer" / "__init__.py").read_text(encoding="utf-8"))
    exports: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_LAZY_EXPORTS":
            value = node.value
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_LAZY_EXPORTS" for t in node.targets):
            value = node.value
        else:
            value = None
        if isinstance(value, ast.Dict):
            for key, item in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and isinstance(item, ast.Tuple) and len(item.elts) == 2:
                    module, attribute = (getattr(e, "value", None) for e in item.elts)
                    if isinstance(module, str) and re.search(r"cake_[a-z0-9_]+_train", module):
                        exports[key.value] = (new_module(module), attribute)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module and re.match(
            r"(?:experimental\.cake\.)?_?cake_[a-z0-9_]+_train", node.module
        ):
            for alias in node.names:
                exports[alias.asname or alias.name] = (new_module("." + node.module), alias.name)
    return exports


def rewrite_flashinfer_imports(text: str, exports: dict[str, tuple[str, str | None]]) -> str:
    """``from flashinfer import <cake export>`` -> imports from the owning ``cake_pplx`` module."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = text.split("\n")
    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "flashinfer"):
            continue
        moved = [a for a in node.names if a.name in exports]
        if not moved:
            continue
        kept = [a for a in node.names if a.name not in exports]
        indent = re.match(r"\s*", lines[node.lineno - 1]).group(0)
        new = []
        if kept:
            new.append(
                indent
                + "from flashinfer import "
                + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in kept)
            )
        for alias in moved:
            module, attribute = exports[alias.name]
            local = alias.asname or alias.name
            if attribute is None:
                new.append(f"{indent}import {module} as {local}")
            else:
                new.append(f"{indent}from {module} import {attribute}" + (f" as {local}" if local != attribute else ""))
        edits.append((node.lineno, node.end_lineno, new))
    for start, end, new in sorted(edits, reverse=True):
        lines[start - 1 : end] = new
    text = "\n".join(lines)
    for name, (module, attribute) in sorted(exports.items(), key=lambda kv: -len(kv[0])):
        target = module if attribute is None else f"{module}.{attribute}"
        text = re.sub(r"(?<![\w.])flashinfer\." + re.escape(name) + r"\b", target, text)
    return text


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
    # Stack-level helpers skip the frames of "this package and the facade layer": that layer is cake_pplx now.
    text = text.replace('name == "flashinfer"', 'name == "cake_pplx"').replace(
        'name.startswith("flashinfer.")', 'name.startswith("cake_pplx.")'
    )
    path.write_text(text, encoding="utf-8")


def rewrite_facade(path: Path, family: str) -> None:
    text = rewrite_text(path.read_text(encoding="utf-8"))
    text = re.sub(
        r"from \.api_logging import flashinfer_api\b",
        f"from ._cake_pplx_api_{family} import cake_api",
        text,
    )
    text = re.sub(r"\bflashinfer_api\b", "cake_api", text)
    text = re.sub(r"from \.utils import (?=[\w, (]*register_(?:custom|fake)_op)", f"from ._cake_pplx_api_{family} import ", text)
    text = re.sub(r"from \.trace\.templates\.\w+ import", f"from ._cake_pplx_api_{family} import", text)
    path.write_text(text, encoding="utf-8")


def attribute_modules(tree: ast.AST) -> set[str]:
    """``cake_x`` / ``_backends.cake_x`` for every ``cake_pplx.<...>`` attribute chain in code."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if not (isinstance(current, ast.Name) and current.id == "cake_pplx"):
            continue
        chain.reverse()
        if chain[0] == "_backends" and len(chain) > 1:
            found.add("_backends." + chain[1])
        elif re.fullmatch(r"_?cake_[a-z0-9_]+_train(?:_kcp)?|_cake_lm_head_train_knobs", chain[0]):
            found.add(chain[0])
    return found


def fix_module_imports(text: str) -> str:
    """After rewriting, ``cake_pplx.<module>`` attribute use needs ``import cake_pplx.<module>``."""
    used = sorted(attribute_modules(ast.parse(text)))
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


def rewrite_test(path: Path, exports: dict[str, tuple[str, str | None]]) -> None:
    text = rewrite_flashinfer_imports(path.read_text(encoding="utf-8"), exports)
    text = rewrite_text(text)
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
    if "load_inline" in runtime_text:
        dependencies.append("ninja")
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
markers = [
    "cpu_only: runs without a GPU",
    "node_exclusive: needs the whole node",
]
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

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
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


def module_owners(src: Path) -> dict[str, str]:
    """Every upstream Cake training module name (facade or backend package) -> its family."""
    owners = {}
    for family in FAMILIES:
        selection = select(src, family)
        for name in selection["packages"] + selection["facades"]:
            owners[name] = family
    return owners


def data_files(src: Path, files: list[str]) -> list[str]:
    """Tracked non-Python files next to a copied test whose names start with its stem
    (golden data such as ``test_x_golden.json`` beside ``test_x.py``)."""
    found = set()
    for rel in files:
        path = Path(rel)
        if path.suffix != ".py" or path.name == "__init__.py":
            continue
        for other in tracked_files(src, str(path.parent)):
            candidate = Path(other)
            if candidate.parent == path.parent and candidate.suffix != ".py" and candidate.name.startswith(path.stem):
                found.add(other)
    return sorted(found - set(files))


def bench_closure(src: Path, roots: list[str]) -> list[str]:
    """``roots`` plus the sibling ``benchmarks/<name>.py`` modules they import by bare name, transitively."""
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            tree = ast.parse((src / rel).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                candidate = Path("benchmarks") / f"{name.split('.')[0]}.py"
                if (src / candidate).is_file():
                    todo.append(str(candidate))
    return sorted(seen)


MODULE_PATH = re.compile(r"cake_pplx\.(?:_backends\.)?(_?cake_[a-z0-9_]+_train(?:_kcp)?)(?:\.\w+)*")


def referenced_modules(path: Path) -> set[str]:
    """Cake modules ``path`` really uses: import statements (at any depth) and string
    literals that are exactly a module path (``importlib.import_module(...)``, tables)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            if node.module in ("cake_pplx", "cake_pplx._backends"):
                names += [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MODULE_PATH.fullmatch(node.value):
                names.append(node.value)
    found = set()
    for name in names:
        match = MODULE_PATH.match(name)
        if match:
            found.add(match.group(1))
    return found


def foreign_imports(path: Path, family: str, owners: dict[str, str]) -> list[str]:
    """``cake_pplx`` modules of other families that ``path`` uses."""
    return sorted(m for m in referenced_modules(path) if owners.get(m) not in (None, family))


TABLE_ENTRY = re.compile(r'^\s*"[^"]+"\s*:\s*"((?:flashinfer|cake_pplx)[\w.]*)"\s*,\s*$')


def trim_tables(path: Path, family: str, owners: dict[str, str]) -> None:
    """Drop ``"key": "<module path>",`` table rows that name another family's module or a
    FlashInfer module (shared test-helper registries list every package)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    kept = []
    for line in lines:
        match = TABLE_ENTRY.match(line)
        if match:
            value = match.group(1)
            own = MODULE_PATH.match(value)
            if value.startswith("flashinfer") or (own and owners.get(own.group(1)) not in (None, family)):
                continue
        kept.append(line)
    if kept != lines:
        path.write_text("\n".join(kept), encoding="utf-8")


def local_test_imports(path: Path, rel: str) -> set[str]:
    """Repository-relative files of the ``tests.*`` / sibling-benchmark modules ``path`` imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for name in names:
            if name.startswith("tests."):
                out.add(str(Path(*name.split(".")).with_suffix(".py")))
            elif rel.startswith("benchmarks/"):
                out.add(f"benchmarks/{name.split('.')[0]}.py")
    return out


def import_family(
    src: Path, family: str, kernels: Path, exports: dict[str, tuple[str, str | None]], owners: dict[str, str]
) -> dict[str, object]:
    selection = select(src, family)
    if not selection["packages"]:
        raise RuntimeError(f"{family}: no backend packages under {src}/flashinfer/experimental/cake")
    dest = kernels / family
    cross_family = kernels.parent / "tests" / "cross_family" / family
    version = (dest / "VERSION").read_text(encoding="utf-8") if (dest / "VERSION").exists() else None
    for sub in ("cake_pplx", "tests", "benchmarks"):
        shutil.rmtree(dest / sub, ignore_errors=True)
    shutil.rmtree(cross_family, ignore_errors=True)
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
            path.write_text(rewrite_flashinfer_imports(path.read_text(encoding="utf-8"), exports), encoding="utf-8")
        for path in list(target.rglob("*.md")) + [target / "build.json", target / "build.test.json"]:
            if path.is_file():
                path.write_text(rewrite_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    for name in selection["facades"]:
        target = dest / "cake_pplx" / f"{name}.py"
        shutil.copy2(src / "flashinfer" / f"{name}.py", target)
        rewrite_facade(target, family)
        target.write_text(rewrite_flashinfer_imports(target.read_text(encoding="utf-8"), exports), encoding="utf-8")
    api = (TEMPLATES / "_cake_pplx_api.py").read_text(encoding="utf-8")
    (dest / "cake_pplx" / f"_cake_pplx_api_{family}.py").write_text(api, encoding="utf-8")

    files = test_closure(src, selection["tests"]) + bench_closure(src, selection["benchmarks"])
    files += data_files(src, files)
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
        if not rel.endswith(".py"):
            continue
        rewrite_test(target, exports)
        if "/test_helpers/" in rel:
            trim_tables(target, family, owners)

    # Tests and benchmarks that need another family move to tests/cross_family/<family>/, together with
    # every file that imports a moved one; the helpers they use are copied along.
    moved: dict[str, list[str]] = {}
    for rel in files:
        foreign = foreign_imports(dest / rel, family, owners)
        if foreign:
            moved[rel] = foreign
    changed = True
    while changed:
        changed = False
        for rel in files:
            if rel in moved or not (dest / rel).is_file() or rel.endswith("__init__.py"):
                continue
            if local_test_imports(dest / rel, rel) & set(moved):
                moved[rel] = ["(imports a moved module)"]
                changed = True
    for rel in list(moved):
        stem, parent = Path(rel).stem, Path(rel).parent
        for other in files:
            if not other.endswith(".py") and Path(other).parent == parent and Path(other).name.startswith(stem):
                moved.setdefault(other, [f"(data of {rel})"])
    for rel in moved:
        target = cross_family / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dest / rel), target)
    for rel in moved:
        for helper in local_test_imports(cross_family / rel, rel):
            if (dest / helper).is_file() and not (cross_family / helper).exists():
                (cross_family / helper).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest / helper, cross_family / helper)
    for init in [f for f in files if f.endswith("__init__.py")]:
        for rel in moved:
            if Path(rel).is_relative_to(Path(init).parent) and not (cross_family / init).exists():
                (cross_family / init).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest / init, cross_family / init)

    write_metadata(dest, family, selection, git_head(src))
    report = residual_report(dest)
    return {
        "family": family,
        "packages": selection["packages"],
        "facades": selection["facades"],
        "tests": len(selection["tests"]),
        "files": len(files),
        "moved_to_cross_family": moved,
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
    src = args.src.resolve()
    exports = export_map(src)
    owners = module_owners(src)
    results = [import_family(src, family, args.dest.resolve(), exports, owners) for family in families]
    for result in results:
        print(
            f"{result['family']}: {len(result['packages'])} packages, {len(result['facades'])} facades, "
            f"{result['tests']} tests + {result['benchmarks']} benchmarks ({result['files']} files with helpers), "
            f"{len(result['moved_to_cross_family'])} moved to tests/cross_family/, "
            f"residual flashinfer refs: runtime {len(result['residual_runtime'])}, tests {len(result['residual_tests'])}"
        )
    if args.report:
        args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
