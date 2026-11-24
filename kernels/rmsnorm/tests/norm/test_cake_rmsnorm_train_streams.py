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

The CUDA stream of the launches, in both packages.

A forward or backward call launches its kernels on the current torch stream
of the current CUDA device, read on every call.  While the loader's tvm_ffi
is the one the package imported, the package sets the FFI stream of that
device itself, with tvm_ffi's ``core._env_set_current_stream`` and torch's
``torch._C._cuda_getCurrentRawStream``, and restores the previous FFI stream
after the launches.  The GPU tests run H = 128 forward and backward calls
(BF16 weights, no residual) at two token counts and check:

* The package holds those two functions, its launches set the stream
  themselves, and the raw stream it reads is the ``cuda_stream`` of
  ``torch.cuda.current_stream()``, on the default stream and on a side
  stream.
* Every launch of a call sees the call's current torch stream as the FFI
  stream of the call's device -- the default stream, a side stream, a side
  stream of a second device (two devices) -- and the FFI stream is the one
  before the call afterwards.  The launches run in that stream's order: a
  call on a side stream behind a long sleep and the copy of its inputs
  returns the bits of the same call on the copied inputs.
* While a CUDA graph captures a forward and a backward call, every launch
  sees the capture stream; each replay gives the eager call's ``y``,
  ``rstd`` and ``dx`` (and the deterministic package's ``dw``) bit for bit.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

MODES = ("det", "nondet")
HIDDEN = 128
ROWS = (16, 4096)
EPS = 1e-6
# Cycles a side stream sleeps before its inputs are copied (tens of ms).
SLEEP_CYCLES = 1 << 27
SEED = 128


def _package(mode: str):
    return cake_package(f"cake_rmsnorm_{mode}_train")


def _loader(mode: str):
    return cake_package(f"cake_rmsnorm_{mode}_train", "cake_jit")


def _api(mode: str):
    pkg = _package(mode)
    return (
        getattr(pkg, f"cake_rmsnorm_{mode}_train_forward"),
        getattr(pkg, f"cake_rmsnorm_{mode}_train_backward"),
    )


def _devices_or_skip(mode: str) -> list[torch.device]:
    """The CUDA devices with an H = 128 BF16 forward and backward route
    (``cuda:0`` first); skips when ``cuda:0`` has none."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    loader = _loader(mode)
    devices = []
    for index in range(torch.cuda.device_count()):
        capability = tuple(torch.cuda.get_device_capability(index))
        if all(
            loader.route_applies(
                device_capability=capability,
                hidden=HIDDEN,
                kind=kind,
                residual=False,
                w_dtype="bf16",
            )
            for kind in ("fwd", "bwd")
        ):
            devices.append(torch.device("cuda", index))
    if not devices or devices[0].index != 0:
        pytest.skip(f"no H = {HIDDEN} route on this device")
    return devices


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def _same(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Bitwise equality (a NaN equals a NaN of the same bits)."""
    return got.shape == want.shape and torch.equal(_bits(got), _bits(want))


def _ffi_stream(mode: str):
    """A reader of the FFI stream of a CUDA device index."""
    tvm_ffi = _loader(mode).tvm_ffi
    getter = getattr(tvm_ffi, "get_raw_stream", None)

    def stream_of(index: int) -> int:
        device = tvm_ffi.device(f"cuda:{index}")
        if getter is not None:
            return int(getter(device))
        return int(
            tvm_ffi.core._env_get_current_stream(
                device.dlpack_device_type(), device.index
            )
        )

    return stream_of


@contextlib.contextmanager
def _recorded_launches(mode: str, patch: pytest.MonkeyPatch):
    """Records ``(FFI stream of the call's device, current torch stream of
    that device)`` at every launch of the package's launch plans (the run
    functions of each plan wrapped; the loader is left in place)."""
    pkg = _package(mode)
    stream_of = _ffi_stream(mode)
    launch = pkg._launch
    seen: list[tuple[int, int]] = []

    def recording(plan, values, rows, device_index):
        def recorded(run):
            def run_and_record(*args):
                seen.append(
                    (
                        stream_of(device_index),
                        torch.cuda.current_stream(device_index).cuda_stream,
                    )
                )
                return run(*args)

            return run_and_record

        plan = tuple(
            (name, recorded(run), keys, grid) for name, run, keys, grid in plan
        )
        return launch(plan, values, rows, device_index)

    patch.setattr(pkg, "_launch", recording)
    yield seen


def _inputs(rows: int, device: torch.device, draw: int) -> tuple[torch.Tensor, ...]:
    """``(x, w, g)`` of a call: BF16 normal values, a seed per draw."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + 7 * rows + draw)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device, torch.bfloat16)

    return normal(rows, HIDDEN), normal(HIDDEN), normal(rows, HIDDEN)


def _step(mode: str, x, w, g) -> dict[str, torch.Tensor]:
    """A forward call and the backward call on its ``rstd``."""
    forward, backward = _api(mode)
    y, rstd = forward(x, w, EPS)[:2]
    dx, dw = backward(g, x, w, rstd)
    return {"y": y, "rstd": rstd, "dx": dx, "dw": dw}


def _compared(mode: str) -> tuple[str, ...]:
    """The outputs a repeated call gives bit for bit (the non-deterministic
    package's ``dw`` may differ from run to run)."""
    return ("y", "rstd", "dx", "dw") if mode == "det" else ("y", "rstd", "dx")


@pytest.mark.parametrize("mode", MODES)
def test_the_package_sets_the_stream_with_torch_and_tvm_ffi_functions(
    mode: str,
) -> None:
    """The package's stream functions are torch's raw stream reader and
    tvm_ffi's stream setter, its launches set the stream themselves, and the
    raw stream is the ``cuda_stream`` of the current torch stream."""
    _devices_or_skip(mode)
    pkg, tvm_ffi = _package(mode), _loader(mode).tvm_ffi
    assert pkg._RAW_STREAM is torch._C._cuda_getCurrentRawStream, mode
    assert pkg._SET_STREAM is tvm_ffi.core._env_set_current_stream, mode
    assert pkg._stream_in_place() and pkg._loader_in_place(), mode
    index = torch.cuda.current_device()
    assert pkg._RAW_STREAM(index) == torch.cuda.current_stream().cuda_stream
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert pkg._RAW_STREAM(index) == side.cuda_stream != 0, mode


@pytest.mark.parametrize("mode", MODES)
def test_every_launch_runs_on_the_current_torch_stream(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the default stream, on a side stream and on a side stream of a
    second device: every launch's FFI stream is the call's current torch
    stream, the FFI stream after the call is the one before it, and a call
    on a side stream behind a sleep and the copy of its inputs returns the
    bits of the same call made on the copied inputs beforehand."""
    devices = _devices_or_skip(mode)
    stream_of = _ffi_stream(mode)
    with _recorded_launches(mode, monkeypatch) as seen:
        for device in devices[:2]:
            with torch.cuda.device(device):
                for rows in ROWS:
                    label = (mode, str(device), rows)
                    fresh = _inputs(rows, device, draw=0)
                    want = _step(mode, *fresh)
                    torch.cuda.synchronize(device)
                    x, w, g = (torch.zeros_like(tensor) for tensor in fresh)
                    side = torch.cuda.Stream(device)
                    for stream in (torch.cuda.current_stream(device), side):
                        before = stream_of(device.index)
                        seen.clear()
                        with torch.cuda.stream(stream):
                            current = torch.cuda.current_stream(device).cuda_stream
                            torch.cuda._sleep(SLEEP_CYCLES)
                            for static, value in zip((x, w, g), fresh, strict=True):
                                static.copy_(value)
                            got = _step(mode, x, w, g)
                        torch.cuda.synchronize(device)
                        assert current == stream.cuda_stream, label
                        assert len(seen) >= 2, label
                        assert seen == [(current, current)] * len(seen), label
                        assert stream_of(device.index) == before, label
                        for name in _compared(mode):
                            assert _same(got[name], want[name]), (*label, name)
                        for tensor in (x, w, g):
                            tensor.zero_()
                        torch.cuda.synchronize(device)
                    assert (
                        side.cuda_stream
                        != torch.cuda.default_stream(device).cuda_stream
                    )


@pytest.mark.parametrize("mode", MODES)
def test_a_captured_call_launches_on_the_capture_stream(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA graph captures a forward and a backward call (given no
    workspace): every launch sees the capture stream as its FFI stream, the
    FFI stream after the capture is the one before it, and each replay gives
    the eager call's outputs bit for bit (the non-deterministic ``dw``
    aside)."""
    device = _devices_or_skip(mode)[0]
    stream_of = _ffi_stream(mode)
    with _recorded_launches(mode, monkeypatch) as seen:
        for rows in ROWS:
            label = (mode, rows)
            x, w, g = _inputs(rows, device, draw=1)
            want = _step(mode, x, w, g)  # also loads the modules
            torch.cuda.synchronize()
            before = stream_of(device.index)
            graph = torch.cuda.CUDAGraph()
            seen.clear()
            with torch.cuda.graph(graph):
                capture = torch.cuda.current_stream().cuda_stream
                got = _step(mode, x, w, g)
            assert capture != torch.cuda.default_stream(device).cuda_stream, label
            assert len(seen) >= 2, label
            assert seen == [(capture, capture)] * len(seen), label
            assert stream_of(device.index) == before, label
            for _replay in range(2):
                for tensor in got.values():
                    tensor.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for name in _compared(mode):
                    assert _same(got[name], want[name]), (*label, name)
