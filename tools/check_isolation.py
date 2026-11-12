#!/usr/bin/env python3
"""Check that every family under ``kernels/`` stands alone (ARCHITECTURE.md, "Isolation rules").

Checks:

1. No ``__init__.py`` at the namespace levels ``cake_pplx/`` and ``cake_pplx/_backends/``.
2. Runtime code (``cake_pplx/``) imports only its own family's ``cake_pplx`` modules, the
   standard library, ``torch``, ``tvm_ffi``, ``numpy`` and ``cuda`` (cuda-python).
3. Tests and benchmarks may import third-party reference implementations (``pytest``,
   ``fla``, ``flash_attn``, ...), but never a module of another family.  A ``cake_pplx``
   name that no family owns is allowed: it is a test that an old name stays removed.
   Tests that need several families live in ``tests/cross_family/``, which this check skips.
4. With ``--base REF``: the change since ``REF`` touches at most one family.

Usage::

    python tools/check_isolation.py
    python tools/check_isolation.py --base origin/main
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"
RUNTIME_ALLOWED = {"torch", "tvm_ffi", "numpy", "cuda"}


def own_modules(family_dir: Path) -> set[str]:
    package = family_dir / "cake_pplx"
    names = {p.stem for p in package.glob("*.py")}
    names |= {p.name for p in (package / "_backends").iterdir() if p.is_dir()} if (package / "_backends").is_dir() else set()
    return names


def imports_of(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"<syntax error: {exc.msg}>")]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and path.parent.name == "cake_pplx":
            # a facade's relative import: ``from ._backends.<module> import``, ``from .<module> import``
            module = node.module or ""
            if module == "_backends":
                found += [(node.lineno, f"cake_pplx._backends.{alias.name}") for alias in node.names]
            elif module:
                found.append((node.lineno, "cake_pplx." + module))
            else:
                found += [(node.lineno, f"cake_pplx.{alias.name}") for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "cake_pplx":
                found += [(node.lineno, f"cake_pplx.{alias.name}") for alias in node.names]
            else:
                found.append((node.lineno, node.module))
    return found


def check_family(family_dir: Path, owners: dict[str, str]) -> list[str]:
    errors = []
    for level in ("cake_pplx", "cake_pplx/_backends"):
        if (family_dir / level / "__init__.py").exists():
            errors.append(f"{family_dir.name}: {level}/__init__.py breaks the shared namespace")
    mine = own_modules(family_dir)
    for path in sorted(family_dir.rglob("*.py")):
        rel = path.relative_to(family_dir)
        runtime = rel.parts[0] == "cake_pplx"
        for line, name in imports_of(path):
            top = name.split(".")[0]
            if top == "cake_pplx":
                parts = name.split(".")
                target = parts[2] if len(parts) > 2 and parts[1] == "_backends" else (parts[1] if len(parts) > 1 else "")
                owner = owners.get(target)
                if path.parent.name == "cake_pplx" and target not in mine and owner is None and not target.startswith("_cake_pplx_api"):
                    errors.append(f"{family_dir.name}/{rel}:{line}: facade imports {name}, which no family provides")
                elif target not in mine and owner is not None:
                    errors.append(f"{family_dir.name}/{rel}:{line}: imports {name}, which family {owner} owns")
            elif not runtime or top in sys.stdlib_module_names or top == "__future__":
                continue
            elif top not in RUNTIME_ALLOWED:
                errors.append(f"{family_dir.name}/{rel}:{line}: runtime code imports {name}")
    return errors


def changed_families(base: str) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", f"{base}...HEAD"], capture_output=True, text=True, check=True
    ).stdout
    return {Path(line).parts[1] for line in out.splitlines() if line.startswith("kernels/") and len(Path(line).parts) > 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="fail if the change since this ref touches more than one family")
    args = parser.parse_args()
    errors = []
    families = sorted(p for p in KERNELS.iterdir() if p.is_dir()) if KERNELS.is_dir() else []
    owners = {module: family_dir.name for family_dir in families for module in own_modules(family_dir)}
    for family_dir in families:
        errors += check_family(family_dir, owners)
    if args.base:
        touched = changed_families(args.base)
        if len(touched) > 1:
            errors.append(f"one family per change: this change touches {sorted(touched)}")
    for error in errors:
        print(error)
    print(f"checked {len(families)} families: {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
