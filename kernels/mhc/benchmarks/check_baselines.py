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

Check that every fixed comparison of the mHC benchmarks used the same non-Cake arm, at the same versions, on every
architecture.

Input: the JSON records of ``bench_cake_mhc_det_train.py`` / ``bench_cake_mhc_nondet_train.py`` (any number, from any
architectures), or the small identity files this script writes with ``--extract`` (the arms, their versions and each
comparison cell's arm and status, without timings: they can be moved between machines instead of the records).

For each comparison (``cake_mhc_baselines.COMPARISONS``) and architecture (``sm_<major><minor>a`` of the record's
device) the check collects the arms the comparison cells name and the version record of each (``versions`` of the
arm; library paths removed).  It fails (exit status 1) when a comparison names another arm than its fixed one, when
the arm's versions differ between architectures or between the records of one architecture, or when the records
disagree on the comparison table itself.  Cells whose arm cannot run on an architecture (``unsupported: <why>``) are
allowed and listed with their reasons; cells with errors or a different contract are listed as findings.

Example::

    python benchmarks/check_baselines.py gb300/*.json gb200/*.json vr200/*.json
    python benchmarks/check_baselines.py --extract vr200-identity.json vr200/*.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

MARKER = "cake_mhc_baseline_identity"


def _arch(device: dict) -> str:
    cap = device.get("capability") or [0, 0]
    return f"sm_{cap[0]}{cap[1]}a"


def _clean(value):
    """A version record without machine-specific paths (``... (/path/libcublas.so.13)`` -> ``...``)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in sorted(value.items())}
    if isinstance(value, str):
        return re.sub(r"\s*\(/[^)]*\)$", "", value)
    return value


def extract(path: Path) -> dict:
    """The identity of one record (or the identity file itself)."""
    rec = json.loads(path.read_text())
    if rec.get(MARKER):
        return rec
    cells = []
    for row in rec.get("rows", []):
        for op, modes in row.get("ops", {}).items():
            for mode, res in modes.items():
                for cname, cell in (res.get("comparisons") or {}).items():
                    cells.append({"H": row["hidden"], "T": row["tokens"], "iters": row.get("iters", 20), "op": op,
                                  "mode": mode, "comparison": cname, "arm": cell.get("arm"),
                                  "status": cell.get("status")})
    soft = rec.get("software", {})
    return {
        MARKER: 1,
        "source": str(path),
        "label": rec.get("label"),
        "host": rec.get("host"),
        "arch": _arch(rec.get("device", {})),
        "device": {k: rec.get("device", {}).get(k) for k in ("name", "capability", "sm_count", "driver_version")},
        "software": {k: soft.get(k) for k in ("python", "torch", "torch_cuda", "triton", "pytorch-triton",
                                              "liger-kernel", "tilelang", "tile-kernels", "deep_gemm",
                                              "deep_gemm.__version__", "cupti-python", "apache-tvm-ffi", "nvcc",
                                              "cublas")},
        "comparisons_spec": rec.get("comparisons"),
        "arms": rec.get("arms", {}),
        "versions": rec.get("versions", {}),
        "cells": cells,
    }


def check(identities: list) -> tuple[bool, list, dict]:
    """``(ok, report lines, result)`` over the identities of several records."""
    lines: list = []
    problems: list = []
    specs = {json.dumps(i.get("comparisons_spec"), sort_keys=True) for i in identities if i.get("comparisons_spec")}
    if len(specs) > 1:
        problems.append("the records disagree on the comparison table")
    spec = json.loads(specs.pop()) if specs else []
    fixed = {c["name"]: c["arm"] for c in spec}
    arches = sorted({i["arch"] for i in identities})
    result: dict = {"arches": arches, "comparisons": {}}
    for cname, arm in fixed.items():
        named = collections.defaultdict(set)  # arch -> arms named by the cells
        versions = collections.defaultdict(set)  # arch -> version records of the fixed arm
        status = collections.defaultdict(collections.Counter)  # arch -> status class counts
        reasons = collections.defaultdict(set)  # arch -> unsupported / error / contract messages
        for ident in identities:
            cells = [c for c in ident["cells"] if c["comparison"] == cname]
            if not cells:
                continue
            for c in cells:
                named[ident["arch"]].add(c["arm"])
                st = c.get("status") or "?"
                cls = st.split(":", 1)[0]
                status[ident["arch"]][cls] += 1
                if cls != "ok":
                    reasons[ident["arch"]].add(st[:300])
            if any(not (c.get("status") or "").startswith(("unsupported", "not run")) for c in cells):
                v = ident["versions"].get(arm)
                if v is not None:
                    versions[ident["arch"]].add(json.dumps(_clean(v), sort_keys=True))
        entry = {"arm": arm, "per_arch": {}, "problems": []}
        for a in arches:
            entry["per_arch"][a] = {"arms_named": sorted(named.get(a, ())), "versions": sorted(versions.get(a, ())),
                                    "status": dict(status.get(a, {})), "reasons": sorted(reasons.get(a, ()))}
            other = named.get(a, set()) - {arm}
            if other:
                entry["problems"].append(f"{a}: cells name {sorted(other)} instead of {arm}")
            if len(versions.get(a, ())) > 1:
                entry["problems"].append(f"{a}: {len(versions[a])} different version records of {arm}")
        distinct = {v for a in arches for v in versions.get(a, ())}
        if len(distinct) > 1:
            entry["problems"].append(f"{arm}: the version records differ between architectures")
        problems.extend(f"{cname}: {p}" for p in entry["problems"])
        result["comparisons"][cname] = entry
    lines.append("| comparison | arm | " + " | ".join(arches) + " | same arm and versions |")
    lines.append("|---|---|" + "---|" * len(arches) + "---|")
    for cname, entry in result["comparisons"].items():
        cells = []
        for a in arches:
            pa = entry["per_arch"][a]
            st = ", ".join(f"{k} {n}" for k, n in sorted(pa["status"].items())) or "no cells"
            cells.append(st)
        lines.append(f"| {cname} | {entry['arm']} | " + " | ".join(cells) + " | "
                     + ("yes" if not entry["problems"] else "NO: " + "; ".join(entry["problems"])) + " |")
    lines.append("")
    for cname, entry in result["comparisons"].items():
        for a, pa in entry["per_arch"].items():
            for r in pa["reasons"]:
                lines.append(f"- {cname} on {a}: {r}")
            for v in pa["versions"]:
                lines.append(f"- {cname} on {a}: {entry['arm']} versions {v}")
    lines.append("")
    lines.append("Software per record:")
    for ident in identities:
        lines.append(f"- {ident['arch']} {ident.get('host')} {ident.get('label')} {Path(ident['source']).name}: "
                     + json.dumps({k: v for k, v in ident.get("software", {}).items() if v is not None}, sort_keys=True))
    result["problems"] = problems
    return not problems, lines, result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("records", nargs="+", type=Path)
    p.add_argument("--extract", type=Path, default=None,
                   help="write the identities of the records to this file (a list) and stop")
    p.add_argument("--json", type=Path, default=None, help="write the check result to this file")
    args = p.parse_args(argv)
    identities = []
    for path in args.records:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            identities.extend(data)
        else:
            identities.append(extract(path))
    if args.extract:
        args.extract.write_text(json.dumps(identities, indent=1, sort_keys=True))
        print(f"wrote {len(identities)} identities to {args.extract}")
        return 0
    ok, lines, result = check(identities)
    print("\n".join(lines))
    for problem in result["problems"]:
        print(f"PROBLEM {problem}")
    print("CHECK " + ("PASS: every comparison used the same arm and versions on " if ok else "FAIL on ")
          + ", ".join(result["arches"]))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
