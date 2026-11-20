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

Independent single-GPU fixtures for the MoK kernel tests.
Self-spawning launcher of the multi-GPU MoK tests: one process per GPU of this node, one NCCL group.

``run_ranks(world, target, *args)`` starts ``world`` processes (``torch.multiprocessing`` spawn; process ``r`` on GPU
``r``), initializes the WORLD group over NCCL, calls ``target(*args)`` in each and returns every rank's return value.
The test skips when this process sees fewer than ``world`` GPUs or a GPU outside ``sm_100a`` / ``sm_103a`` /
``sm_107a``.  The ``SLURM_*`` task variables are removed from the children's environment (one process sees every GPU of
the node); ``target`` must be a module-level function.  A rank that raises fails the test with its traceback.

``run_ranks_isolated(world, target, *args, timeout=...)`` runs the same group from a child process in a session of its
own and returns what each rank reported instead of failing: for a step expected to fail on one rank (a kernel trap)
while its peers wait for it.  The child exits nonzero when a rank fails; the parent then stops every process left in
the session (``os.killpg``), and on the timeout too.
"""

from __future__ import annotations

import datetime
import os
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

SUPPORTED = ((10, 0), (10, 3), (10, 7))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rank_main(rank, world, port, out, target, args):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world),
    )
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    status = ("error", "init")
    try:
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", rank), timeout=datetime.timedelta(seconds=900)
        )
        status = ("ok", target(*args))
        torch.cuda.synchronize()
        dist.barrier()
    except BaseException:  # noqa: BLE001 - reported to the parent
        status = ("error", traceback.format_exc())
    Path(out, f"rank{rank}.pkl").write_bytes(pickle.dumps(status))
    if status[0] == "error":
        os._exit(1)  # the peers may wait in a collective; the parent stops them
    dist.destroy_process_group()


def _require(world: int) -> None:
    """Skip unless this process sees ``world`` GPUs of the supported architectures."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} CUDA devices visible to one process")
    for index in range(world):
        if torch.cuda.get_device_capability(index) not in SUPPORTED:
            pytest.skip("Requires SM100a, SM103a or SM107a CUDA devices")


def run_ranks(world: int, target, *args):
    """Run ``target(*args)`` on ``world`` ranks; returns the list of their return values (rank order)."""
    _require(world)
    saved = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    for key in saved:
        del os.environ[key]
    try:
        with tempfile.TemporaryDirectory(prefix="mok-ranks-") as out:
            error = None
            try:
                mp.start_processes(
                    _rank_main,
                    args=(world, _free_port(), out, target, args),
                    nprocs=world,
                    join=True,
                    start_method="spawn",
                )
            except Exception as exc:  # noqa: BLE001 - the ranks' own reports follow
                error = exc
            results, failures = [], []
            for rank in range(world):
                path = Path(out, f"rank{rank}.pkl")
                if not path.exists():
                    failures.append(f"rank {rank}: no result ({error!r})")
                    continue
                kind, value = pickle.loads(path.read_bytes())
                if kind == "error":
                    failures.append(f"rank {rank}:\n{value}")
                results.append(value)
            if failures or error is not None:
                pytest.fail("\n".join(failures) or repr(error))
            return results
    finally:
        os.environ.update(saved)


def _isolated_main(out: str) -> None:
    """The child of :func:`run_ranks_isolated`: the group run of the target in ``out/spec.pkl``; each rank writes its
    report to ``out``.  Exits 1 when a rank failed (the others were stopped)."""
    world, target, args = pickle.loads(Path(out, "spec.pkl").read_bytes())
    try:
        mp.start_processes(
            _rank_main, args=(world, _free_port(), out, target, args), nprocs=world, join=True, start_method="spawn"
        )
    except Exception:  # noqa: BLE001 - the ranks' own reports follow
        traceback.print_exc()
        sys.exit(1)


def run_ranks_isolated(world: int, target, *args, timeout: float = 600.0) -> dict:
    """Run ``target(*args)`` on ``world`` ranks from a child process in a new session (as :func:`run_ranks`, with the
    same skips), waiting at most ``timeout`` seconds; then stop every process left in the session.  Returns
    ``returncode`` (the child's; negative when it was stopped), ``timed_out``, ``records`` (per rank ``("ok",
    value)``, ``("error", traceback)`` or None for a rank that wrote nothing) and ``log`` (the child's output)."""
    _require(world)
    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    with tempfile.TemporaryDirectory(prefix="mok-isolated-") as out:
        Path(out, "spec.pkl").write_bytes(pickle.dumps((world, target, args)))
        command = "import sys; from tests.experimental._mok_spawn import _isolated_main; _isolated_main(sys.argv[1])"
        with open(Path(out, "child.log"), "wb") as log:
            child = subprocess.Popen(
                [sys.executable, "-c", command, out],
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = child.wait()
        records = []
        for rank in range(world):
            path = Path(out, f"rank{rank}.pkl")
            records.append(pickle.loads(path.read_bytes()) if path.exists() else None)
        text = Path(out, "child.log").read_text(errors="replace")
    return dict(returncode=returncode, timed_out=timed_out, records=records, log=text)
