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

Two installed versions of the Cake causal conv1d KCP packages against each other, through one of the KCP benchmarks
(``--bench nondet``: ``bench_cake_causal_conv1d_nondet_train_kcp.py``, both modes and the FLA / Attention Gym
baselines; ``--bench det``: ``bench_cake_causal_conv1d_det_train_kcp.py``, both launch forms of the deterministic
package and every KCP baseline).  A process group imports one version only, so the versions alternate as whole
``torchrun`` launches on the same GPUs: ``--groups`` paired groups, group ``g`` running the versions in the label order
rotated by ``g`` (reversed in odd blocks), each launch timing every row in its own interleaved rounds.  Each version's
value of a row, arm and phase is the median over the groups of the launch's value (the slowest rank's ``ahead`` time:
the host enqueue hidden); ``old / new`` is the median over the groups of the per-group ratio, with its ``[min, max]``.

The first group runs the accuracy gate (``--draws``) and the baselines (``--sota`` in the ``--sota-label`` version's
launch only); later groups time the Cake arms alone with the gate off (their launches exit with status 1, which is
expected and recorded).

Example::

    python bench_cake_causal_conv1d_kcp_versions.py --cake old=/v/old/bin/python --cake new=/v/new/bin/python \\
        --world 4 --bench nondet --groups 4 --out kcp_w4 -- --rows customer --layout zigzag --rounds 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
BENCHES = {
    "nondet": HERE / "bench_cake_causal_conv1d_nondet_train_kcp.py",
    "det": HERE / "bench_cake_causal_conv1d_det_train_kcp.py",
}


def group_order(labels: list[str], g: int) -> list[str]:
    n = len(labels)
    order = labels[g % n :] + labels[: g % n]
    if (g // n) % 2:
        order.reverse()
    return order


def launch(python: str, bench: Path, world: int, json_path: Path, log: Path, extra: list[str]) -> int:
    cmd = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}", str(bench)]
    cmd += [*extra, "--json", str(json_path)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for key in [k for k in env if k.startswith("SLURM_")]:
        env.pop(key)
    with open(log, "w") as handle:
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        return subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)


def values_of(doc: dict, bench: str) -> dict[tuple[int, str, str], float]:
    """``{(row, arm, phase): us}`` of one launch's JSON: the slowest rank's ``ahead`` time per arm and phase."""

    out = {}
    for index, row in enumerate(doc.get("rows", [])):
        arms = row.get("modes", {}) if bench == "nondet" else row.get("arms", {})
        for arm, entry in arms.items():
            if bench == "det":
                for key, samples in entry.get("rounds_us", {}).items():
                    if key.endswith("_ahead") and samples:
                        out[(index, arm, key.removesuffix("_ahead"))] = statistics.median(samples)
            else:
                for key, value in entry.items():
                    if key.endswith("_ahead_us") and isinstance(value, (int, float)):
                        out[(index, arm, key.removesuffix("_ahead_us"))] = value
    return out


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cake", action="append", default=[], help="LABEL=PYTHON of one installed version")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="nondet")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--sota", default="all", help="the baselines of the --sota-label launch of the first group")
    parser.add_argument("--sota-label", default=None, help="default: the last --cake label")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)
    cakes = dict(spec.split("=", 1) for spec in args.cake)
    labels = list(cakes)
    sota_label = args.sota_label or labels[-1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for g in range(args.groups):
        for label in group_order(labels, g):
            first = g == 0
            flags = ["--draws", str(args.draws if first else 0)]
            flags += ["--sota", *(args.sota.split() if first and label == sota_label else ["none"])]
            path = out / f"g{g}-{label}.json"
            t0 = time.time()
            rc = launch(cakes[label], BENCHES[args.bench], args.world, path, out / f"g{g}-{label}.log", extra + flags)
            runs.append({"group": g, "label": label, "rc": rc, "seconds": time.time() - t0, "json": str(path)})
            print(f"group {g} {label}: rc {rc} {time.time() - t0:.0f}s", flush=True)
    values: dict[str, dict] = {label: {} for label in labels}
    for run in runs:
        try:
            doc = json.load(open(run["json"]))
        except Exception as exc:  # noqa: BLE001 -- a failed launch has no JSON
            run["error"] = f"{type(exc).__name__}: {exc}"
            continue
        for key, us in values_of(doc, args.bench).items():
            values[run["label"]].setdefault(key, {})[run["group"]] = us
    summary = []
    if len(labels) >= 2:
        old, new = labels[0], labels[-1]
        for key in sorted(values[new]):
            a, b = values[old].get(key, {}), values[new][key]
            groups = sorted(set(a) & set(b))
            if not groups:
                continue
            ratios = [a[g] / b[g] for g in groups]
            summary.append({
                "row": key[0], "arm": key[1], "phase": key[2],
                f"{old}_us": statistics.median(a[g] for g in groups),
                f"{new}_us": statistics.median(b[g] for g in groups),
                f"{old}_over_{new}": statistics.median(ratios), "range": [min(ratios), max(ratios)],
                "groups": len(groups),
            })  # fmt: skip
    json.dump({"bench": args.bench, "world": args.world, "extra": extra, "runs": runs, "summary": summary},
              open(out / "summary.json", "w"), indent=1)  # fmt: skip
    for s in summary:
        print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
