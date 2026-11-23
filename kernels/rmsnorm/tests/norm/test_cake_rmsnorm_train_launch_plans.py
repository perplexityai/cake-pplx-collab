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

The launch plans of both packages.  A forward or backward call launches its
modules from a plan memoised per call shape (``_launch_plan``: per module the
``run`` function of the loaded module, the logical names of its kernel
arguments and its grid), inside one ``tvm_ffi.use_torch_stream()``, where
``cake_jit.run_module`` enters one per module and computes the rest on every
launch.  The tests need no device: the loader's device-facing functions
(``sm_count``, ``device_arch``, ``load`` and ``tvm_ffi.use_torch_stream``) are
fakes that record every launch and stream context, put both into the
loader and into the package's copies that its plans are checked against, so
the plans stay in use; ``torch.cuda.get_device_capability`` is a fake where a
test reads a device's architecture.

* Every window of the H = 128 routes (both edges and inside; BF16 and FP32
  weights) and the lowest token count of every window of every other route,
  on each architecture at the SM counts of its boards: a call through its plan
  launches the modules, arguments and grids ``run_module`` launches, loaded
  for the same architecture, in the same order, inside one stream context.
* Repeated calls of each of those call shapes, each with arguments of its own
  (a new object for every tensor, other scalar values): the repeats are served
  by the memoised plan, and every launch passes the tensors and scalars of its
  own call, the ones ``run_module`` passes for it.
* A replaced ``run_module``, ``launch_grid`` or ``sm_count`` (a test's
  recorder, an emulated SM count) takes effect after a plan of the same call
  shape was memoised: every launch goes through ``run_module``; with the
  package's copy replaced as well, the call uses a plan keyed by the replaced
  function, with the launches ``run_module`` makes under it.
* A replaced ``load`` or ``device_arch`` alone (the package's copy kept) takes
  effect after a plan was memoised: every launch goes through ``run_module``
  and runs the replacement's module, or the module of the replacement's
  architecture.
* Calls that change the token count, the device or the weight dtype, and come
  back, each launch what ``run_module`` launches for them.
* The deterministic package's memoised backward plan is the plan its backward
  computed per call (modules, launched workspace owner, capacity owner and
  layout) at every backward call above, also below ``ACCURACY_MIN_SMS``, and
  follows the ``selected_modules`` it is given.
* Two devices, in both packages: each device's architecture follows its own
  compute capability; at the SM counts of two boards of one architecture,
  every backward call gets the backward plan computed for its own device.
* With the package's own tvm_ffi and stream functions in place, a call sets
  the FFI stream of the current device to its current torch stream once
  (read on every call), the stream ``tvm_ffi.use_torch_stream()`` sets, and
  restores it, also when a launch raises; otherwise (a function missing, the
  loader's tvm_ffi or its ``use_torch_stream`` replaced) it runs inside that
  ``use_torch_stream()``.
* A replaced ``load`` with the package's copy replaced as well gets a plan
  keyed by it; a failing ``load`` is attempted once per call and its call
  raises what ``run_module`` raises.
* The registry token changes with each replaced registry object, never
  repeats, and does not follow a freed table's ``id``; the memoised route
  answer and forward modules follow the token and the function they are
  given.
* The non-deterministic backward given no workspace binds the views of the
  path through its workspace helpers (or calls a replaced helper), at the
  H = 128 placement; ``x``'s own row view and the companion inputs are read
  as the checked path reads them.
* Forward and backward calls of H = 128 (the device functions faked) return
  outputs of the call's shapes holding what the launches wrote; a caller's
  output, written in place or through a temporary copied back, is returned.
"""

from __future__ import annotations

import collections
import contextlib
import re
import types

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pytestmark = pytest.mark.cpu_only

MODES = ("det", "nondet")
# The SM counts of the boards of each architecture.
BOARD_SMS = {"sm_100a": (148, 152), "sm_103a": (152, 160), "sm_107a": (212, 216)}
# The width whose every window the tests visit at both edges and inside.
HIDDEN = 128
# The grid of the replaced ``launch_grid`` (no launch of the registry has it).
ODD_GRID = (99991, 1, 1)
_ROUTE = re.compile(r"(fwd|bwd)(_w32)?(_residual)?_h(\d+)")


def _package(mode: str):
    return cake_package(f"cake_rmsnorm_{mode}_train")


def _loader(mode: str):
    return cake_package(f"cake_rmsnorm_{mode}_train", "cake_jit")


def _route(route: str) -> tuple[str, int, bool, str]:
    """``(kind, hidden, residual, weight dtype token)`` of a route key."""
    match = _ROUTE.fullmatch(route)
    assert match is not None, route
    kind, w32, residual, hidden = match.groups()
    return kind, int(hidden), residual is not None, "f32" if w32 else "bf16"


def _calls(loader, arch: str) -> list[tuple[str, int, bool, str, int]]:
    """``(kind, hidden, residual, w dtype token, rows)`` of the calls the tests
    make on ``arch``: every window of the ``HIDDEN`` routes at its lower and
    upper edge and inside (an open window up to twice its lower edge), the
    lower edge of every window of every other route."""
    calls = []
    for route in sorted(loader.ROUTES[arch]):
        kind, hidden, residual, w_dtype = _route(route)
        for above, at_most, _library in loader.program_windows(
            arch, hidden, kind, residual, w_dtype
        ):
            rows = {above + 1}
            if hidden == HIDDEN:
                top = 2 * above + 2 if at_most is None else int(at_most)
                rows |= {top, (above + 1 + top) // 2}
            calls += [
                (kind, hidden, residual, w_dtype, count) for count in sorted(rows)
            ]
    return calls


class _Module:
    """A loaded module whose ``run`` records its launch."""

    def __init__(self, events: list, name: str, arch: str) -> None:
        self._events, self._name, self._arch = events, name, arch

    def run(self, *args) -> None:
        self._events.append((self._name, self._arch, args))


class _Fakes:
    """The device-facing functions of a package's loader replaced by fakes
    (in the loader and in the package's copies of them): ``sm_count`` of
    ``sms`` (device index -> SM count), ``device_arch`` of ``arch``, ``load``
    of :class:`_Module` and (unless ``ffi`` is false: the loader's tvm_ffi is
    kept) a ``tvm_ffi.use_torch_stream`` that records its contexts.
    ``events`` holds each launch, ``(name, arch, args)``, and each stream
    context's ``"enter"`` and ``"exit"``."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        arch: str,
        sms: dict,
        ffi: bool = True,
    ) -> None:
        self.pkg, self.loader, self.arch = _package(mode), _loader(mode), arch
        self.events: list = []
        self.sm_count = _sm_count(dict(sms))

        def device_arch(device_index: int) -> str:
            return arch

        def load(name: str, module_arch: str) -> _Module:
            return _Module(self.events, name, module_arch)

        @contextlib.contextmanager
        def use_torch_stream():
            self.events.append("enter")
            yield
            self.events.append("exit")

        for attr, fake in (
            ("sm_count", self.sm_count),
            ("device_arch", device_arch),
            ("load", load),
        ):
            monkeypatch.setattr(self.loader, attr, fake)
            monkeypatch.setattr(self.pkg, f"_{attr.upper()}", fake)
        if ffi:
            monkeypatch.setattr(
                self.loader,
                "tvm_ffi",
                types.SimpleNamespace(use_torch_stream=use_torch_stream),
            )

    def names(self, kind, hidden, residual, w_dtype, rows) -> tuple[str, ...]:
        return self.loader.selected_modules(
            self.arch, hidden, kind, residual, rows=rows, w_dtype=w_dtype
        )

    def through_plan(
        self, names, rows: int, device_index: int = 0, values: dict | None = None
    ):
        """``(events, plan lookups)`` of the package's launch of ``names`` with
        the logical arguments ``values`` (default :func:`_values`)."""
        self.events.clear()
        before = _lookups(self.pkg)
        self.pkg._run(
            names,
            _values(self.loader, rows) if values is None else values,
            arch=self.arch,
            rows=rows,
            device_index=device_index,
        )
        return list(self.events), _lookups(self.pkg) - before

    def through_run_module(
        self, names, rows: int, device_index: int = 0, values: dict | None = None
    ) -> list:
        """The events of ``run_module`` launching ``names`` one by one with the
        logical arguments ``values`` (default :func:`_values`)."""
        self.events.clear()
        if values is None:
            values = _values(self.loader, rows)
        for name in names:
            self.loader.run_module(name, values, rows=rows, device_index=device_index)
        return list(self.events)


def _sm_count(sms: dict):
    def sm_count(device_index: int) -> int:
        return sms[device_index]

    return sm_count


def _values(loader, rows: int) -> dict[str, object]:
    """The logical arguments of a call: a distinct stand-in for every tensor
    and scalar name of the loader's vocabulary (``rows`` the call's)."""
    values: dict[str, object] = {name: f"<{name}>" for name in loader.LOGICAL_TENSORS}
    values.update(
        {name: 1000 + index for index, name in enumerate(loader.LOGICAL_SCALARS)}
    )
    values["rows"] = rows
    return values


class _Arg:
    """The stand-in for one tensor of one call: equal to itself only."""

    def __init__(self, name: str, call: int) -> None:
        self.name, self.call = name, call

    def __repr__(self) -> str:
        return f"<{self.name} of call {self.call}>"


def _call_values(loader, rows: int, call: int) -> dict[str, object]:
    """The logical arguments of the ``call``-th call of a call shape: a new
    :class:`_Arg` for every tensor name and scalar values of that call alone
    (``rows`` the shape's)."""
    values: dict[str, object] = {
        name: _Arg(name, call) for name in loader.LOGICAL_TENSORS
    }
    values.update(
        {
            name: 1000 * (call + 1) + index
            for index, name in enumerate(loader.LOGICAL_SCALARS)
        }
    )
    values["rows"] = rows
    return values


def _lookups(pkg) -> int:
    info = pkg._launch_plan.cache_info()
    return info.hits + info.misses


def _launches(events: list) -> list:
    return [event for event in events if isinstance(event, tuple)]


def _one_context_each(launches: list) -> list:
    """The events of ``launches`` with one stream context each (``run_module``'s)."""
    return [event for launch in launches for event in ("enter", launch, "exit")]


def _recording(run_module, recorded: list):
    """A ``run_module`` that appends each module name to ``recorded``, then
    calls ``run_module``."""

    def recording(name, values, *, rows, device_index):
        recorded.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    return recording


def _loading(events: list, prefix: str):
    """A ``load`` whose modules record their launches in ``events`` under
    their name after ``prefix``."""

    def load(name: str, module_arch: str) -> _Module:
        return _Module(events, f"{prefix}{name}", module_arch)

    return load


def _arch_of(arch: str):
    """A ``device_arch`` that gives ``arch`` for every device."""

    def device_arch(device_index: int) -> str:
        return arch

    return device_arch


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch):
    """A factory of :class:`_Fakes`; the plans memoised against its fakes are
    dropped afterwards."""
    made: list[_Fakes] = []

    def make(mode: str, arch: str, sms: dict, ffi: bool = True) -> _Fakes:
        made.append(_Fakes(monkeypatch, mode, arch, sms, ffi))
        return made[-1]

    yield make
    for fake in made:
        fake.pkg._launch_plan.cache_clear()


def _pair_call(fake: _Fakes) -> tuple[tuple[str, ...], int] | None:
    """``(names, rows)`` of the largest token count of :func:`_calls` at which
    the BF16 H = 128 backward launches a pair (two modules), or ``None``."""
    found = None
    for kind, hidden, residual, w_dtype, rows in _calls(fake.loader, fake.arch):
        if (kind, hidden, residual, w_dtype) == ("bwd", HIDDEN, False, "bf16"):
            names = fake.names(kind, hidden, residual, w_dtype, rows)
            if len(names) == 2:
                found = (names, rows)
    return found


@pytest.mark.parametrize("mode", MODES)
def test_a_plan_launches_what_run_module_launches(mode: str, fakes) -> None:
    """Every call of :func:`_calls` on every architecture, at each SM count of
    its boards: through the package's plan (one plan lookup) the call launches
    the modules, arguments and grids that ``run_module`` launches one by one,
    loaded for the same architecture, in the same order, inside one stream
    context (``run_module`` enters one per module)."""
    checked = pairs = 0
    for arch in _loader(mode).ARCHES:
        for sms in BOARD_SMS[arch]:
            fake = fakes(mode, arch, {0: sms})
            for call in _calls(fake.loader, arch):
                names = fake.names(*call)
                label = (mode, arch, sms, *call, names)
                events, lookups = fake.through_plan(names, call[-1])
                reference = fake.through_run_module(names, call[-1])
                launches = _launches(reference)
                assert lookups == 1, label
                assert [launch[:2] for launch in launches] == [
                    (name, arch) for name in names
                ], label
                assert reference == _one_context_each(launches), label
                assert events == ["enter", *launches, "exit"], label
                checked += 1
                pairs += len(names) == 2
    assert checked and pairs, (checked, pairs)


@pytest.mark.parametrize("mode", MODES)
def test_a_plan_launches_the_arguments_of_its_own_call(mode: str, fakes) -> None:
    """Three calls of every call shape of :func:`_calls` on every architecture,
    each with arguments of its own (:func:`_call_values`): the second and the
    third are served by the plan the first memoised, and every launch passes
    the tensors and scalars of its own call, the ones ``run_module`` passes for
    it, never those of an earlier call."""
    checked = 0
    for arch in _loader(mode).ARCHES:
        fake = fakes(mode, arch, {0: BOARD_SMS[arch][0]})
        for call in _calls(fake.loader, arch):
            names, rows = fake.names(*call), call[-1]
            seen: list[list] = []
            for index in range(3):
                values = _call_values(fake.loader, rows, index)
                hits = fake.pkg._launch_plan.cache_info().hits
                events, lookups = fake.through_plan(names, rows, values=values)
                hits = fake.pkg._launch_plan.cache_info().hits - hits
                launches = _launches(
                    fake.through_run_module(names, rows, values=values)
                )
                label = (mode, arch, *call, names, index)
                assert lookups == 1 and (index == 0 or hits == 1), label
                assert events == ["enter", *launches, "exit"], label
                tensors = [
                    arg
                    for _name, _arch, args in _launches(events)
                    for arg in args
                    if isinstance(arg, _Arg)
                ]
                assert tensors and all(arg.call == index for arg in tensors), label
                assert all(launches != earlier for earlier in seen), label
                seen.append(launches)
            checked += 1
    assert checked, mode


@pytest.mark.parametrize("mode", MODES)
def test_a_replaced_loader_function_takes_effect_after_a_plan(mode: str, fakes) -> None:
    """After the plan of an H = 128 pair call was memoised: with ``run_module``
    replaced (a recorder calling the loader's) every launch goes through it, one
    by one; with ``launch_grid`` replaced the launches take its grid; with
    ``sm_count`` replaced (half the SM count) they take the grids of that SM
    count, which differ from the memoised plan's.  With the package's copy of
    ``launch_grid`` or ``sm_count`` replaced as well, the call uses a plan
    again, keyed by the replaced function, with the same launches."""
    found = 0
    for arch in _loader(mode).ARCHES:
        sms = BOARD_SMS[arch][0]
        fake = fakes(mode, arch, {0: sms})
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        names, rows = call
        memoised, lookups = fake.through_plan(names, rows)
        assert lookups == 1, (mode, arch)
        planned = _launches(memoised)
        assert memoised == ["enter", *planned, "exit"], (mode, arch)

        recorded: list[str] = []
        recording = _recording(fake.loader.run_module, recorded)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fake.loader, "run_module", recording)
            events, lookups = fake.through_plan(names, rows)
        assert recorded == list(names), (mode, arch, recorded)
        assert lookups == 0 and events == _one_context_each(planned), (mode, arch)

        def odd_grid(record, rows, *, device_index):
            return ODD_GRID

        for both in (False, True):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(fake.loader, "launch_grid", odd_grid)
                if both:
                    patch.setattr(fake.pkg, "_LAUNCH_GRID", odd_grid)
                events, lookups = fake.through_plan(names, rows)
                reference = fake.through_run_module(names, rows)
            label = (mode, arch, "launch_grid", both)
            assert lookups == int(both), label
            assert _launches(events) == _launches(reference), label
            assert [args[-3:] for _name, _arch, args in _launches(events)] == [
                ODD_GRID
            ] * len(names), label

        fewer = _sm_count({0: sms // 2})
        for both in (False, True):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(fake.loader, "sm_count", fewer)
                if both:
                    patch.setattr(fake.pkg, "_SM_COUNT", fewer)
                events, lookups = fake.through_plan(names, rows)
                reference = fake.through_run_module(names, rows)
            label = (mode, arch, "sm_count", both)
            assert lookups == int(both), label
            assert _launches(events) == _launches(reference), label
            assert _launches(reference) != planned, label
        # the original functions back: the memoised plan again
        events, lookups = fake.through_plan(names, rows)
        assert lookups == 1 and events == memoised, (mode, arch)
    assert found, mode


@pytest.mark.parametrize("mode", MODES)
def test_a_replaced_load_or_device_arch_takes_effect_after_a_plan(
    mode: str, fakes
) -> None:
    """After the plan of an H = 128 pair call was memoised, with the package's
    copies of ``load`` and ``device_arch`` kept: with the loader's ``load``
    replaced (modules that launch under another name) every launch goes
    through ``run_module`` and runs the replacement's module; with the
    loader's ``device_arch`` replaced (another architecture) every launch goes
    through ``run_module`` and runs the module loaded for that architecture.
    With both back, the memoised plan again."""
    found = 0
    for arch in _loader(mode).ARCHES:
        fake = fakes(mode, arch, {0: BOARD_SMS[arch][0]})
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        names, rows = call
        memoised, lookups = fake.through_plan(names, rows)
        assert lookups == 1, (mode, arch)
        other = next(each for each in fake.loader.ARCHES if each != arch)
        for attr, replacement, want in (
            (
                "load",
                _loading(fake.events, "replaced "),
                [(f"replaced {name}", arch) for name in names],
            ),
            ("device_arch", _arch_of(other), [(name, other) for name in names]),
        ):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(fake.loader, attr, replacement)
                events, lookups = fake.through_plan(names, rows)
                reference = fake.through_run_module(names, rows)
            label = (mode, arch, attr)
            assert lookups == 0 and events == reference, label
            assert [launch[:2] for launch in _launches(events)] == want, label
        events, lookups = fake.through_plan(names, rows)
        assert lookups == 1 and events == memoised, (mode, arch)
    assert found, mode


@pytest.mark.parametrize("mode", MODES)
def test_a_changed_call_shape_gets_its_own_launches(mode: str, fakes) -> None:
    """H = 128 calls that change the token count (from a pair window to the
    first window), the device (two devices, the second with half the SM count)
    and the weight dtype, then the forward, then the first call again: each
    launches through a plan what ``run_module`` launches for it; each change
    changes the launches, and the first call's come back."""
    found = 0
    for arch in _loader(mode).ARCHES:
        sms = BOARD_SMS[arch][0]
        fake = fakes(mode, arch, {0: sms, 1: sms // 2})
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        _names, pair_rows = call
        first_rows = min(
            rows
            for kind, hidden, _residual, w_dtype, rows in _calls(fake.loader, arch)
            if (kind, hidden, w_dtype) == ("bwd", HIDDEN, "bf16")
        )
        sequence = (
            ("bwd", "bf16", pair_rows, 0),
            ("bwd", "bf16", first_rows, 0),
            ("bwd", "bf16", pair_rows, 1),
            ("bwd", "f32", pair_rows, 0),
            ("fwd", "bf16", pair_rows, 1),
            ("bwd", "bf16", pair_rows, 0),
        )
        seen = []
        for kind, w_dtype, rows, device_index in sequence:
            names = fake.names(kind, HIDDEN, False, w_dtype, rows)
            label = (mode, arch, kind, w_dtype, rows, device_index, names)
            events, lookups = fake.through_plan(names, rows, device_index)
            reference = fake.through_run_module(names, rows, device_index)
            assert lookups == 1, label
            assert events == ["enter", *_launches(reference), "exit"], label
            seen.append(_launches(reference))
        assert all(seen[0] != other for other in seen[1:4]), (mode, arch)
        assert seen[5] == seen[0], (mode, arch)
    assert found, mode


def _per_call_plan(
    pkg, loader, arch, hidden, residual, w_dtype, rows, select, device_index=0
):
    """The deterministic backward's plan as it computed it on every call
    (on device ``device_index``): the capacity owner, the selection rows, the
    modules ``select`` names, the launched workspace owner and its layout."""
    owner = pkg._workspace_record(arch, hidden, residual, rows, w_dtype)
    selection = pkg._backward_selection_rows(
        arch, hidden, residual, rows, w_dtype, device_index
    )
    names = tuple(
        select(arch, hidden, "bwd", residual, rows=selection, w_dtype=w_dtype)
    )
    launched = [
        loader.MODULES[name]
        for name in names
        if loader.MODULES[name].get("workspace_rule") is not None
    ]
    assert len(launched) == 1, (arch, hidden, residual, w_dtype, rows, names)
    layout = loader.workspace_layout(
        launched[0],
        rows,
        device_index=device_index,
        capacity_record=launched[0] if owner is None else owner,
    )
    return names, launched[0], owner, layout


def _nondet_per_call_plan(
    pkg, loader, device_index, arch, hidden, residual, w_dtype, rows
):
    """The non-deterministic backward's plan computed for one call on device
    ``device_index``: the modules of ``rows``, their workspace owner and its
    layout, with the partial capacity of the route's largest pair record."""
    names = tuple(
        pkg.selected_modules(arch, hidden, "bwd", residual, rows=rows, w_dtype=w_dtype)
    )
    owners = [
        loader.MODULES[name]
        for name in names
        if loader.MODULES[name]["workspace_rule"] is not None
    ]
    assert len(owners) == 1, (arch, hidden, residual, w_dtype, rows, names)
    capacity = pkg._capacity_record(
        pkg._backward_owners(arch, hidden, residual, rows), rows, device_index
    )
    layout = loader.workspace_layout(
        owners[0], rows, device_index=device_index, capacity_record=capacity
    )
    return names, owners[0], layout


def test_the_det_backward_plan_is_the_per_call_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deterministic package's memoised backward plan is the plan its
    backward computed on every call -- modules, launched workspace owner,
    capacity owner and layout -- at every backward call of :func:`_calls` on
    every architecture, at the SM counts of its boards and one below
    ``ACCURACY_MIN_SMS``, the same objects on a repeat; with a replaced
    ``selected_modules`` (the plan's ``select``: the backward passes the
    package's) a pair call's plan names the replacement's modules and the
    per-call plan under it, and the package's again without it."""
    pkg, loader = _package("det"), _loader("det")
    windows = cake_package("cake_rmsnorm_det_train", "cake_windows")
    checked = 0
    for arch in loader.ARCHES:
        for sms in (*BOARD_SMS[arch], windows.ACCURACY_MIN_SMS[arch] - 1):
            monkeypatch.setattr(loader, "sm_count", _sm_count({0: sms}))
            token = pkg._registry_token()
            for kind, hidden, residual, w_dtype, rows in _calls(loader, arch):
                if kind != "bwd":
                    continue
                key = (0, arch, hidden, residual, w_dtype, rows)
                plan = pkg._backward_plan(*key, pkg.selected_modules, token)
                want = _per_call_plan(
                    pkg, loader, *key[1:], select=pkg.selected_modules
                )
                label = (arch, sms, *key[1:])
                assert plan[0] == want[0], label
                assert plan[1] is want[1] and plan[2] is want[2], label
                assert plan[3] == want[3], label
                assert pkg._backward_plan(*key, pkg.selected_modules, token) is plan
                checked += 1
    assert checked
    pairs = 0
    for arch in loader.ARCHES:
        monkeypatch.setattr(loader, "sm_count", _sm_count({0: BOARD_SMS[arch][0]}))
        token = pkg._registry_token()
        calls = [
            rows
            for kind, hidden, residual, w_dtype, rows in _calls(loader, arch)
            if (kind, hidden, residual, w_dtype) == ("bwd", HIDDEN, False, "bf16")
        ]
        pair = [
            rows
            for rows in calls
            if len(loader.selected_modules(arch, HIDDEN, "bwd", False, rows=rows)) == 2
        ]
        if not pair:
            continue
        pairs += 1
        key = (0, arch, HIDDEN, False, "bf16", pair[0])
        own = pkg._backward_plan(*key, pkg.selected_modules, token)
        # the modules of the first window that launches others (a column task)
        twin = next(
            names
            for names in (
                loader.selected_modules(arch, HIDDEN, "bwd", False, rows=rows)
                for rows in calls
            )
            if names != own[0]
        )

        def select(arch, hidden, kind, residual, *, rows, w_dtype="bf16", twin=twin):
            return twin

        replaced = pkg._backward_plan(*key, select, token)
        assert replaced[0] == twin, (arch, replaced[0])
        want = _per_call_plan(pkg, loader, *key[1:], select=select)
        assert replaced[1] is want[1] and replaced[2] is want[2], arch
        assert replaced[3] == want[3], arch
        assert pkg._backward_plan(*key, pkg.selected_modules, token) is own
    assert pairs


@pytest.mark.parametrize("mode", MODES)
def test_the_backward_plan_and_architecture_are_per_device(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two devices.  Of two architectures: each device's architecture follows
    its own compute capability, queried once per device.  Of one architecture
    at the SM counts of its two boards: every backward call of :func:`_calls`,
    alternating between the devices, gets the memoised backward plan that
    equals the plan computed for that call on its own device (modules,
    workspace records and layout), the same object on a repeat; the two
    devices' plans differ where the layout follows the SM count (its slots)."""
    pkg, loader = _package(mode), _loader(mode)
    capabilities = {arch: cap for cap, arch in loader.ARCH_BY_CAPABILITY.items()}
    archs = loader.ARCHES[:2]
    queried: list[int] = []

    def get_device_capability(device_index: int) -> tuple[int, int]:
        queried.append(device_index)
        return capabilities[archs[device_index]]

    pkg._capability.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.cuda, "get_device_capability", get_device_capability)
            for _repeat in range(2):
                devices = [torch.device("cuda", index) for index in (0, 1)]
                assert [pkg._device_arch(device) for device in devices] == list(
                    archs
                ), mode
    finally:
        pkg._capability.cache_clear()
    assert sorted(queried) == [0, 1], (mode, queried)

    checked = differ = 0
    for arch in loader.ARCHES:
        sms = dict(enumerate(BOARD_SMS[arch]))
        monkeypatch.setattr(loader, "sm_count", _sm_count(sms))
        token = pkg._registry_token()
        for kind, hidden, residual, w_dtype, rows in _calls(loader, arch):
            if kind != "bwd":
                continue
            plans: dict[int, tuple] = {}
            for device_index in (0, 1, 0, 1):
                key = (device_index, arch, hidden, residual, w_dtype, rows)
                if mode == "det":
                    plan = pkg._backward_plan(*key, pkg.selected_modules, token)
                    want = _per_call_plan(
                        pkg,
                        loader,
                        *key[1:],
                        select=pkg.selected_modules,
                        device_index=device_index,
                    )
                else:
                    plan = pkg._backward_plan(*key, token)
                    want = _nondet_per_call_plan(pkg, loader, *key)
                label = (mode, arch, sms, *key)
                records = zip(plan[1:-1], want[1:-1], strict=True)
                assert plan[0] == want[0] and plan[-1] == want[-1], label
                assert all(mine is its for mine, its in records), label
                assert plans.setdefault(device_index, plan) is plan, label
            checked += 1
            differ += plans[0] != plans[1]
    assert checked and differ, (mode, checked, differ)


# The FFI streams of the fakes of :class:`_Streams` before a call (device ->
# stream) and the current torch stream of each device.
FFI_STREAMS = {0: 0xA000, 1: 0xB000}
TORCH_STREAMS = {0: 0x1000, 1: 0x2000}
# DLPack's device type of a CUDA device.
CUDA = 2


class _Streams:
    """The functions the launches set the FFI stream with, replaced by fakes,
    and the loader's tvm_ffi (the fake of :class:`_Fakes`) taken as the
    package's copy, so that a call sets the stream itself: the package's
    ``_SET_STREAM`` records ``("set", device type, device, stream)`` in the
    events and keeps the FFI stream of each device in ``env``;
    ``_RAW_STREAM`` gives ``torch[device]``, the device's current torch stream;
    ``torch.cuda.current_device`` gives ``current[0]``."""

    def __init__(self, patch: pytest.MonkeyPatch, fake: _Fakes) -> None:
        self.env = {(CUDA, device): stream for device, stream in FFI_STREAMS.items()}
        self.torch = dict(TORCH_STREAMS)
        self.current = [0]

        def set_stream(device_type: int, device_id: int, stream: int) -> int:
            fake.events.append(("set", device_type, device_id, stream))
            previous = self.env[device_type, device_id]
            self.env[device_type, device_id] = stream
            return previous

        patch.setattr(fake.pkg, "_SET_STREAM", set_stream)
        patch.setattr(fake.pkg, "_RAW_STREAM", lambda index: self.torch[index])
        patch.setattr(fake.pkg, "_TVM_FFI", fake.loader.tvm_ffi)
        patch.setattr(
            fake.pkg, "_USE_TORCH_STREAM", fake.loader.tvm_ffi.use_torch_stream
        )
        patch.setattr(torch.cuda, "current_device", lambda: self.current[0])


def _recording_stream(events: list, label: str):
    """A ``use_torch_stream`` whose contexts record ``label`` on entry and exit."""

    @contextlib.contextmanager
    def use_torch_stream():
        events.append(f"enter {label}")
        yield
        events.append(f"exit {label}")

    return use_torch_stream


@pytest.mark.parametrize("mode", MODES)
def test_the_launches_set_the_torch_stream_of_the_current_device(
    mode: str, fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While the package's tvm_ffi and both stream functions are in place, an
    H = 128 pair call through its plan sets the FFI stream of the current CUDA
    device to that device's current torch stream once, launches what
    ``run_module`` launches, and restores the previous stream: read on every
    call (another torch stream, another current device -- not the device the
    call's tensors name -- take effect), restored when a launch raises.  With
    either stream function missing, or the loader's tvm_ffi or its
    ``use_torch_stream`` replaced, the launches run inside that
    ``use_torch_stream()`` instead."""
    found = 0
    for arch in _loader(mode).ARCHES:
        sms = BOARD_SMS[arch][0]
        fake = fakes(mode, arch, {0: sms, 1: sms})
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        names, rows = call
        streams = _Streams(monkeypatch, fake)
        reference = _launches(fake.through_run_module(names, rows))
        for current, stream in ((0, 0x1000), (0, 0x1010), (1, 0x2000), (1, 0x2020)):
            streams.current[0] = current
            streams.torch[current] = stream
            before = dict(streams.env)
            events, lookups = fake.through_plan(names, rows, device_index=0)
            label = (mode, arch, current, stream)
            assert lookups == 1, label
            assert events == [
                ("set", CUDA, current, stream),
                *reference,
                ("set", CUDA, current, FFI_STREAMS[current]),
            ], label
            assert streams.env == before, label

        def failing(name: str, module_arch: str, fake=fake, names=names) -> _Module:
            module = _Module(fake.events, name, module_arch)
            if name == names[-1]:

                def run(*args) -> None:
                    raise RuntimeError(f"launch of {name} failed")

                module.run = run
            return module

        streams.current[0] = 1
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fake.loader, "load", failing)
            patch.setattr(fake.pkg, "_LOAD", failing)
            with pytest.raises(RuntimeError, match="failed"):
                fake.through_plan(names, rows)
        assert fake.events == [
            ("set", CUDA, 1, streams.torch[1]),
            *reference[:-1],
            ("set", CUDA, 1, FFI_STREAMS[1]),
        ], (mode, arch, "raised")
        assert streams.env[CUDA, 1] == FFI_STREAMS[1], (mode, arch, "raised")

        for attr in ("_SET_STREAM", "_RAW_STREAM"):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(fake.pkg, attr, None)
                events, lookups = fake.through_plan(names, rows)
            assert lookups == 1, (mode, arch, attr)
            assert events == ["enter", *reference, "exit"], (mode, arch, attr)
        replaced = types.SimpleNamespace(
            use_torch_stream=_recording_stream(fake.events, "replaced")
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fake.loader, "tvm_ffi", replaced)
            events, lookups = fake.through_plan(names, rows)
        assert lookups == 1, (mode, arch, "tvm_ffi")
        assert events == ["enter replaced", *reference, "exit replaced"], (mode, arch)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                fake.loader.tvm_ffi,
                "use_torch_stream",
                _recording_stream(fake.events, "function"),
            )
            events, lookups = fake.through_plan(names, rows)
        assert events == ["enter function", *reference, "exit function"], (mode, arch)
        events, lookups = fake.through_plan(names, rows)
        assert events[0] == ("set", CUDA, 1, streams.torch[1]), (mode, arch, "back")
    assert found, mode


class _StreamModule:
    """A loaded module whose ``run`` records its launch with the FFI stream of
    device ``index`` (``stream_of(index)``) at the launch."""

    def __init__(self, events: list, name: str, stream_of, index: list) -> None:
        self._events, self._name = events, name
        self._stream_of, self._index = stream_of, index

    def run(self, *args) -> None:
        self._events.append((self._name, self._stream_of(self._index[0])))


@pytest.mark.parametrize("mode", MODES)
def test_the_stream_a_launch_sets_is_the_one_use_torch_stream_sets(
    mode: str, fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the loader's own tvm_ffi: the package sets the FFI stream with
    tvm_ffi's own function, and for each current device (with its own torch
    stream) every launch of an H = 128 pair call runs on the FFI stream that
    ``tvm_ffi.use_torch_stream()`` sets for that device and stream; after the
    call the device's FFI stream is the one before it.  (``torch.cuda``'s
    current device and stream, the current stream of
    ``torch.get_device_module()`` and the package's raw stream reader are
    fakes that agree.)"""
    pkg, loader = _package(mode), _loader(mode)
    tvm_ffi = loader.tvm_ffi
    assert pkg._TVM_FFI is tvm_ffi and pkg._USE_TORCH_STREAM is tvm_ffi.use_torch_stream
    assert pkg._SET_STREAM is tvm_ffi.core._env_set_current_stream, mode
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

    current = [0]

    def current_stream(device=None) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            device=torch.device("cuda", current[0]),
            cuda_stream=TORCH_STREAMS[current[0]],
        )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    # ``use_torch_stream()`` reads the current stream of the device module of
    # the current accelerator (``torch.cuda`` on a CUDA build)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device=None: types.SimpleNamespace(current_stream=current_stream),
    )
    monkeypatch.setattr(pkg, "_RAW_STREAM", lambda index: TORCH_STREAMS[index])
    found = 0
    for arch in loader.ARCHES:
        fake = fakes(mode, arch, {0: BOARD_SMS[arch][0]}, ffi=False)
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        names, rows = call

        def load(name: str, module_arch: str, fake=fake) -> _StreamModule:
            return _StreamModule(fake.events, name, stream_of, current)

        monkeypatch.setattr(loader, "load", load)
        monkeypatch.setattr(pkg, "_LOAD", load)
        for index in (0, 1):
            current[0] = index
            device = tvm_ffi.device(f"cuda:{index}")
            with tvm_ffi.use_raw_stream(device, FFI_STREAMS[index]):
                with tvm_ffi.use_torch_stream():
                    want = stream_of(index)
                assert stream_of(index) == FFI_STREAMS[index], (mode, arch, index)
                events, lookups = fake.through_plan(names, rows, device_index=0)
                label = (mode, arch, index)
                assert want == TORCH_STREAMS[index], label
                assert lookups == 1, label
                assert events == [(name, want) for name in names], label
                assert stream_of(index) == FFI_STREAMS[index], label
    assert found, mode


@pytest.mark.parametrize("mode", MODES)
def test_a_replaced_load_with_the_package_copy_gets_a_plan_of_its_own(
    mode: str, fakes
) -> None:
    """After the plan of an H = 128 pair call was memoised, with the loader's
    ``load`` and the package's copy of it both replaced (modules that launch
    under another name): the call uses a plan again, keyed by the replaced
    ``load``, whose launches run the replacement's modules, those
    ``run_module`` launches under it.  With both back, the memoised plan's
    launches again."""
    found = 0
    for arch in _loader(mode).ARCHES:
        fake = fakes(mode, arch, {0: BOARD_SMS[arch][0]})
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        names, rows = call
        memoised, lookups = fake.through_plan(names, rows)
        assert lookups == 1, (mode, arch)
        replaced = _loading(fake.events, "replaced ")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fake.loader, "load", replaced)
            patch.setattr(fake.pkg, "_LOAD", replaced)
            events, lookups = fake.through_plan(names, rows)
            reference = fake.through_run_module(names, rows)
        label = (mode, arch)
        assert lookups == 1, label
        assert events == ["enter", *_launches(reference), "exit"], label
        assert [launch[:2] for launch in _launches(events)] == [
            (f"replaced {name}", arch) for name in names
        ], label
        events, lookups = fake.through_plan(names, rows)
        assert lookups == 1 and events == memoised, label
    assert found, mode


@pytest.mark.parametrize("mode", MODES)
def test_a_failing_load_is_attempted_once_per_call(mode: str, fakes) -> None:
    """With ``load`` (the loader's and the package's copy) failing for one
    module of an H = 128 pair call, the first or the last: every call through
    the plan path attempts that load once, launches the modules before it
    (through ``run_module``), and raises the error ``run_module`` raises for
    the same call, the launches before it the same.  With an argument of the
    failing module missing as well, the call raises ``run_module``'s error
    that names the argument, still with one attempt."""
    found = 0
    for arch in _loader(mode).ARCHES:
        fake = fakes(mode, arch, {0: BOARD_SMS[arch][0]})
        call = _pair_call(fake)
        if call is None:
            continue
        found += 1
        names, rows = call
        attempts: collections.Counter = collections.Counter()
        for failing in (names[0], names[-1]):

            def load(
                name: str,
                module_arch: str,
                failing=failing,
                fake=fake,
                attempts=attempts,
            ) -> _Module:
                attempts[name] += 1
                if name == failing:
                    raise RuntimeError(f"cannot load {name} for {module_arch}")
                return _Module(fake.events, name, module_arch)

            record = fake.loader.MODULES[failing]
            first_key = next(
                record["logical"][key]
                for kind, key in record["arg_plan"]
                if kind != "grid"
            )
            missing = _values(fake.loader, rows)
            del missing[first_key]
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(fake.loader, "load", load)
                patch.setattr(fake.pkg, "_LOAD", load)
                for values, error in (
                    (None, RuntimeError),
                    (None, RuntimeError),
                    (missing, ValueError),
                ):
                    label = (mode, arch, failing, error.__name__)
                    attempts.clear()
                    with pytest.raises(error) as planned:
                        fake.through_plan(names, rows, values=values)
                    events = list(fake.events)
                    assert attempts[failing] == 1, (label, attempts)
                    with pytest.raises(error) as reference:
                        fake.through_run_module(names, rows, values=values)
                    assert str(planned.value) == str(reference.value), label
                    assert type(planned.value) is type(reference.value), label
                    assert events == fake.events, label
                    if error is ValueError:
                        assert repr(first_key) in str(planned.value), label
    assert found, mode


def _route_tables(loader, drop: tuple[str, str] | None = None) -> dict:
    """A copy of the loader's route table, without route ``drop[1]`` of
    architecture ``drop[0]``."""
    return {
        arch: {key: names for key, names in routes.items() if (arch, key) != drop}
        for arch, routes in loader.ROUTES.items()
    }


@pytest.mark.parametrize("mode", MODES)
def test_the_registry_token_follows_the_objects_not_their_ids(mode: str) -> None:
    """The registry token stays while the registry objects stay, changes to
    a token no other registry state had when any one of them is replaced (the
    module, route and capability tables, the SM-count function, and the
    workspace constants or the accuracy-guard tables), and changes again
    when it comes back.  A route table that replaced one which was freed -- at the freed
    table's address where the interpreter reuses it -- gets another token,
    and the memoised route answer of the freed table is not served for it."""
    pkg, loader = _package(mode), _loader(mode)
    windows = cake_package(f"cake_rmsnorm_{mode}_train", "cake_windows")
    first = pkg._registry_token()
    assert pkg._registry_token() == first, mode
    seen = {first}
    owners = [
        (loader, "MODULES"),
        (loader, "ROUTES"),
        (loader, "ARCH_BY_CAPABILITY"),
        (loader, "sm_count"),
    ]
    if mode == "nondet":
        owners += [(loader, "SUMS_BYTES"), (loader, "PAIR_ROWS_OFFSET")]
    else:
        owners += [(windows, "ACCURACY_EDGES"), (windows, "ACCURACY_MIN_SMS")]
    for owner, attr in owners:
        original = getattr(owner, attr)
        if isinstance(original, dict):
            replacement = dict(original)
        elif isinstance(original, int):
            replacement = original + 256
        else:

            def replacement(*args, original=original, **kwargs):
                return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(owner, attr, replacement)
            token = pkg._registry_token()
            assert token not in seen and pkg._registry_token() == token, attr
            seen.add(token)
        back = pkg._registry_token()
        assert back != token and pkg._registry_token() == back, attr
        seen.add(back)

    capability, arch = min(loader.ARCH_BY_CAPABILITY.items())
    route = min(loader.ROUTES[arch])
    kind, hidden, residual, w_dtype = _route(route)
    shape = (capability, hidden, kind, residual, w_dtype)
    items = list(_route_tables(loader, (arch, route)).items())
    with_route = _route_tables(loader)
    freed = id(with_route)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "ROUTES", with_route)
        held = pkg._registry_token()
        assert pkg._route_ok(*shape, held, pkg.route_applies) is True, mode
    pkg._registry_token()  # the shipped tables again; the copy is not held
    del with_route
    # a new dict built here takes the most recently freed one where the
    # interpreter keeps freed dicts for reuse
    reused = {arch: routes for arch, routes in items}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "ROUTES", reused)
        token = pkg._registry_token()
        label = (mode, "same address" if id(reused) == freed else "other address")
        assert token != held, label
        assert pkg._route_ok(*shape, token, pkg.route_applies) is False, label


@pytest.mark.parametrize("mode", MODES)
def test_the_route_answer_follows_the_registry_and_the_function(mode: str) -> None:
    """The memoised route answer is ``route_applies``'s for every route of
    every architecture and for a width with no route; another function's
    answer for that function; with the route table replaced by one without a
    route (a new registry token), that table's answer, and the shipped
    table's again with it back."""
    pkg, loader = _package(mode), _loader(mode)
    token = pkg._registry_token()
    checked = 0
    for capability, arch in sorted(loader.ARCH_BY_CAPABILITY.items()):
        for route in sorted(loader.ROUTES.get(arch, {})):
            kind, hidden, residual, w_dtype = _route(route)
            shape = (capability, hidden, kind, residual, w_dtype)
            want = loader.route_applies(
                device_capability=capability,
                hidden=hidden,
                kind=kind,
                residual=residual,
                w_dtype=w_dtype,
            )
            assert want is True, (arch, route)
            assert pkg._route_ok(*shape, token, pkg.route_applies) is want
            assert pkg._route_ok(*shape, token, pkg.route_applies) is want
            checked += 1
        assert (
            pkg._route_ok(capability, 7, "fwd", False, "bf16", token, pkg.route_applies)
            is False
        )
    assert checked, mode
    capability, arch = min(loader.ARCH_BY_CAPABILITY.items())
    route = min(loader.ROUTES[arch])
    kind, hidden, residual, w_dtype = _route(route)
    shape = (capability, hidden, kind, residual, w_dtype)
    asked: list[dict] = []

    def refuse(**kwargs) -> bool:
        asked.append(kwargs)
        return False

    assert pkg._route_ok(*shape, token, refuse) is False and asked, mode
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "ROUTES", _route_tables(loader, (arch, route)))
        replaced = pkg._registry_token()
        assert replaced != token, mode
        assert pkg._route_ok(*shape, replaced, pkg.route_applies) is False, mode
    assert pkg._route_ok(*shape, pkg._registry_token(), pkg.route_applies) is True


@pytest.mark.parametrize("mode", MODES)
def test_the_forward_modules_follow_the_registry_and_the_function(mode: str) -> None:
    """The memoised forward modules are ``selected_modules``'s for every
    forward call of :func:`_calls` on every architecture (the same object on a
    repeat); another function's for that function; with the route table
    replaced by one whose H = 128 forward route lists another forward route's
    modules (a new registry token), the modules that table selects."""
    pkg, loader = _package(mode), _loader(mode)
    checked = changed = 0
    for arch in loader.ARCHES:
        token = pkg._registry_token()
        for kind, hidden, residual, w_dtype, rows in _calls(loader, arch):
            if kind != "fwd":
                continue
            key = (arch, hidden, residual, w_dtype, rows, token)
            names = pkg._forward_names(*key, pkg.selected_modules)
            want = loader.selected_modules(
                arch, hidden, "fwd", residual, rows=rows, w_dtype=w_dtype
            )
            assert names == tuple(want), key
            assert pkg._forward_names(*key, pkg.selected_modules) is names, key
            checked += 1
        own = loader.route_key(HIDDEN, "fwd", False, "bf16")
        if own not in loader.ROUTES[arch]:
            continue
        rows = min(
            count
            for kind, hidden, residual, w_dtype, count in _calls(loader, arch)
            if (kind, hidden, residual, w_dtype) == ("fwd", HIDDEN, False, "bf16")
        )
        key = (arch, HIDDEN, False, "bf16", rows)
        shipped = pkg._forward_names(*key, token, pkg.selected_modules)
        twin = ("<twin>",)

        def select(arch, hidden, kind, residual, *, rows, w_dtype="bf16", twin=twin):
            return twin

        assert pkg._forward_names(*key, token, select) == twin, arch
        for other in sorted(loader.ROUTES[arch]):
            if other == own or not other.startswith("fwd"):
                continue
            tables = _route_tables(loader)
            tables[arch][own] = list(loader.ROUTES[arch][other])
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(loader, "ROUTES", tables)
                try:
                    want = tuple(
                        loader.selected_modules(arch, HIDDEN, "fwd", False, rows=rows)
                    )
                except ValueError:
                    continue
                if want == shipped:
                    continue
                replaced = pkg._registry_token()
                assert replaced != token, (arch, other)
                assert pkg._forward_names(*key, replaced, pkg.selected_modules) == want
            changed += 1
            break
    assert checked and changed, (mode, checked, changed)


def _placed_zeros(residue: int, made: list):
    """A ``torch.zeros`` of ``uint8`` bytes on the CPU whose allocation starts
    at an address of ``residue`` modulo 1024 (where a device allocation may
    start), each ``(size, address)`` appended to ``made``."""
    empty = torch.empty

    def zeros(size, *, dtype, device):
        assert dtype == torch.uint8, dtype
        size = int(size)
        block = empty(size + 2048, dtype=torch.uint8)
        skip = (residue - block.data_ptr()) % 1024
        tensor = block[skip : skip + size].zero_()
        made.append((size, tensor.data_ptr()))
        return tensor

    return zeros


def _view_of(tensor: torch.Tensor, base: int) -> tuple:
    """A workspace view as its offset from the allocation (``None`` for a view
    of no elements, whose address is not its place), shape, strides, dtype and
    whether it is all zero."""
    return (
        tensor.data_ptr() - base if tensor.numel() else None,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        bool((tensor == 0).all()),
    )


def _bind(pkg, call: dict, residue: int, seam: bool) -> tuple:
    """``(outcome, allocations, helper calls)`` of the package's binding of the
    workspace of a backward ``call`` given none, its allocations at
    ``residue`` modulo 1024: the views (:func:`_view_of`) or the error;
    ``seam``: with the package's ``_package_workspace`` replaced by a recorder
    of its calls (that calls the package's)."""
    made: list = []
    calls: list = []
    helper = pkg._package_workspace

    def recording(nbytes: int, device: torch.device, hidden: int) -> torch.Tensor:
        calls.append((nbytes, hidden))
        return helper(nbytes, device, hidden)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "zeros", _placed_zeros(residue, made))
        if seam:
            patch.setattr(pkg, "_package_workspace", recording)
        try:
            partial, counters = pkg._bound_workspace(None, **call)
        except (ValueError, RuntimeError) as error:
            return ("error", type(error), str(error)), made, calls
    base = made[0][1]
    return ("views", _view_of(partial, base), _view_of(counters, base)), made, calls


def test_the_package_workspace_views_are_the_placed_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-deterministic backward given no workspace, at every H = 128
    backward call of :func:`_calls` (single kernels and pairs, both weight
    dtypes, with and without residual) on every architecture, and at the
    lowest window of another width: with the package's workspace helpers in
    place it binds the same ``partial`` and ``counters`` views -- offset in
    the allocation, shape, strides, dtype, zeroed -- of the same single
    allocation as the path through ``_package_workspace`` and
    ``_workspace_views`` (the replaced helper is called, with the call's
    bytes); H = 128 views sit at the 256 mod 1024 placement; an allocation
    whose placement does not fit (or, for another width, is not 256-byte
    aligned) raises that path's error."""
    pkg, loader = _package("nondet"), _loader("nondet")
    cpu = torch.device("cpu")
    checked = others = errors = 0
    for arch in loader.ARCHES:
        monkeypatch.setattr(loader, "sm_count", _sm_count({0: BOARD_SMS[arch][0]}))
        token = pkg._registry_token()
        other = False
        for kind, hidden, residual, w_dtype, rows in _calls(loader, arch):
            if kind != "bwd" or (hidden != HIDDEN and other):
                continue
            _names, record, layout = pkg._backward_plan(
                0, arch, hidden, residual, w_dtype, rows, token
            )
            nbytes = int(layout["total_bytes"])
            if hidden != HIDDEN:
                other = True
                nbytes += 512  # the size that serves every window up to ``rows``
                monkeypatch.setattr(pkg, "_workspace_bytes", lambda *args, n=nbytes: n)
            call = {
                "record": record,
                "layout": layout,
                "rows": rows,
                "hidden": hidden,
                "residual": residual,
                "device": cpu,
            }
            residues = (0, 256, 512, 768, 64, 300) if hidden == HIDDEN else (0, 512, 64)
            for residue in residues:
                label = (arch, hidden, residual, w_dtype, rows, residue)
                got, made, calls = _bind(pkg, call, residue, seam=False)
                want, made_seam, calls_seam = _bind(pkg, call, residue, seam=True)
                assert got == want, label
                assert calls == [] and calls_seam == [(nbytes, hidden)], label
                slack = 768 if hidden == HIDDEN else 0
                assert [size for size, _ in made] == [nbytes + slack], label
                assert [size for size, _ in made_seam] == [nbytes + slack], label
                if got[0] == "error":
                    errors += 1
                    continue
                # the workspace's start, from the counters (never empty)
                start = got[2][0] - layout["counters_offset"]
                if hidden == HIDDEN:
                    assert (made[0][1] + start) % 1024 == 256, label
                    checked += 1
                else:
                    assert start == 0, label
                    others += 1
    assert checked and others and errors, (checked, others, errors)


def _outcome(function, given, *args, **kwargs) -> tuple:
    """What ``function(given, *args, **kwargs)`` gives: its error, or the
    tensor it returns described by whether it is ``given``, its shape,
    strides, dtype and values."""
    try:
        tensor = function(given, *args, **kwargs)
    except (TypeError, ValueError) as error:
        return ("error", type(error), str(error))
    return (
        "tensor",
        tensor is given,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.numel() == given.numel()
        and bool(torch.equal(tensor.reshape(-1), given.reshape(-1))),
    )


def _p7_companion(tensor, name: str, *, pkg, lead, hidden: int, copy: bool):
    """``_companion_view`` as every call ran it before (the shape check, then
    the row view)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected = (*lead, hidden)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )
    return pkg._row_view(tensor, name, copy=copy)[0]


@pytest.mark.parametrize("mode", MODES)
def test_a_call_s_own_rows_and_companions_take_the_checked_path(mode: str) -> None:
    """``x``'s own row view of a call is read as the checked path reads it:
    in place, re-viewed or copied, for every layout (contiguous, row-strided,
    odd column offset, odd row stride, one row, no rows, an expanded row,
    transposed) at both alignments, and the wrong dtype raises the same
    error.  A companion (``g``, ``residual``) of a ``[T, H]`` call is the
    tensor itself, and every other companion -- of another shape, not a
    tensor, ``[..., H]`` collapsible or not, copied or not -- gives what the
    shape check and the row view gave."""
    pkg, loader = _package(mode), _loader(mode)
    hidden, rows = HIDDEN, 8
    wide = torch.randn(rows, hidden + 32).to(torch.bfloat16)
    odd = torch.randn(rows, hidden + 8).to(torch.bfloat16)
    inputs = [
        torch.randn(rows, hidden).to(torch.bfloat16),
        wide[:, :hidden],
        wide[:, 1 : hidden + 1],
        odd[:, :hidden],
        wide[:1, :hidden],
        torch.empty(0, hidden, dtype=torch.bfloat16),
        torch.randn(1, hidden).to(torch.bfloat16).expand(rows, hidden),
        torch.randn(hidden, rows).to(torch.bfloat16).t(),
        torch.randn(rows, hidden),
    ]
    checked = 0
    for alignment in (loader.FORWARD_ALIGNMENT_BYTES, loader.BACKWARD_ALIGNMENT_BYTES):
        for tensor in inputs:
            count = int(tensor.shape[0])
            kwargs = {"rows": count, "hidden": hidden, "alignment": alignment}
            got = _outcome(pkg._own_rows, tensor, "x", dtype=torch.bfloat16, **kwargs)
            want = _outcome(
                pkg._kernel_input,
                tensor,
                "x",
                device=tensor.device,
                dtype=torch.bfloat16,
                **kwargs,
            )
            assert got == want, (mode, alignment, tuple(tensor.shape), tensor.stride())
            checked += 1
    heads = torch.randn(rows, 2, hidden + 32).to(torch.bfloat16)
    companions = [
        (torch.randn(rows, hidden).to(torch.bfloat16), (rows,)),
        (torch.randn(rows, hidden // 2).to(torch.bfloat16), (rows,)),
        (torch.randn(rows - 1, hidden).to(torch.bfloat16), (rows,)),
        (torch.randn(rows, hidden).to(torch.bfloat16), (rows // 2, 2)),
        (torch.randn(rows * hidden).to(torch.bfloat16), (rows,)),
        (heads[:, :, :hidden], (rows, 2)),
        (heads[:, :, :hidden].transpose(0, 1), (2, rows)),
        ([[0.0] * hidden] * rows, (rows,)),
    ]
    for tensor, lead in companions:
        for copy in (False, True):
            args = {"lead": lead, "hidden": hidden, "copy": copy}
            got = _outcome(pkg._companion_view, tensor, "g", **args)
            want = _outcome(_p7_companion, tensor, "g", pkg=pkg, **args)
            assert got == want, (mode, lead, copy, getattr(tensor, "shape", None))
            checked += 1
    assert checked, mode


# The value each output a launch binds receives from :func:`_writing`.
MARKS = {"y": 1.0, "r": 2.0, "h_new": 3.0, "dx": 4.0, "dw": 5.0}
# The value a caller's output holds before a call.
UNWRITTEN = -7.0
# Per output argument of each call: (the logical name it binds, its dtype,
# whether it has the input's shape; else the leading shape, or ``[H]``).
FWD_OUTPUTS = {
    "out": ("y", torch.bfloat16, True),
    "rstd": ("r", torch.float32, False),
    "residual_out": ("h_new", torch.bfloat16, True),
}
BWD_OUTPUTS = {"dx": ("dx", torch.bfloat16, True), "dw": ("dw", torch.float32, False)}


def _writing(loader, events: list):
    """A ``load`` whose modules fill the outputs their launch binds with
    :data:`MARKS` (a forward module's -- it binds ``y`` -- ``y``, ``r`` and
    ``h_new``, not an ``h_new`` bound to ``y``, the plain forward's; a
    backward module's ``dx`` and ``dw``) and record each launch in
    ``events``."""

    def load(name: str, module_arch: str) -> types.SimpleNamespace:
        record = loader.MODULES[name]
        keys = [
            record["logical"][key] for kind, key in record["arg_plan"] if kind != "grid"
        ]

        def run(*args) -> None:
            events.append((name, module_arch))
            bound = dict(zip(keys, args, strict=False))
            outputs = ("y", "r", "h_new") if "y" in bound else ("dx", "dw")
            for key in outputs:
                if key in bound and not (key == "h_new" and bound[key] is bound["y"]):
                    bound[key].fill_(MARKS[key])

        return types.SimpleNamespace(run=run)

    return load


def _buffer(shape: tuple[int, ...], dtype: torch.dtype, layout: str) -> torch.Tensor:
    """A caller's output of ``shape`` holding :data:`UNWRITTEN`: contiguous, or
    every other element of a buffer twice as wide (strided)."""
    if layout == "contiguous":
        return torch.full(shape, UNWRITTEN, dtype=dtype)
    wide = torch.full((*shape[:-1], 2 * shape[-1]), UNWRITTEN, dtype=dtype)
    return wide[..., ::2]


def _check_outputs(results, buffers: dict, outputs: dict, shapes, label) -> None:
    """Each result: its shape (``shapes``: with the input's shape, without),
    dtype and :data:`MARKS` value; the caller's buffer of its output when one
    was given, else a tensor of its own."""
    assert len(results) == len(outputs), label
    for name, result in zip(outputs, results, strict=True):
        key, dtype, full = outputs[name]
        assert tuple(result.shape) == shapes[0 if full else 1], (label, name)
        assert result.dtype == dtype, (label, name)
        assert bool((result == MARKS[key]).all()), (label, name)
        if name in buffers:
            assert result is buffers[name], (label, name)
        else:
            assert all(result is not each for each in buffers.values()), (label, name)


@pytest.mark.parametrize("mode", MODES)
def test_a_call_returns_and_writes_back_its_outputs(
    mode: str, fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward and backward calls without residual (H = 128) and with residual
    (the narrowest width of a residual route), as ``[T, H]`` and as
    ``[..., H]`` calls, on every architecture (the device functions faked,
    modules that fill the outputs they bind): with no output
    given, the call returns fresh outputs of its shapes and dtypes holding
    what the launches wrote; with one caller's output given -- contiguous
    (written in place) or strided (written through a temporary and copied
    back), one output at a time -- that output is the tensor returned and
    holds what the launches wrote, and the others are fresh."""
    pkg = _package(mode)
    forward = getattr(pkg, f"cake_rmsnorm_{mode}_train_forward")
    backward = getattr(pkg, f"cake_rmsnorm_{mode}_train_backward")
    rows = 8
    layouts = ("contiguous", "strided")
    checked = residuals = 0
    for arch in _loader(mode).ARCHES:
        fake = fakes(mode, arch, {0: BOARD_SMS[arch][0]})
        loader = fake.loader
        capability = min(
            each for each, its in loader.ARCH_BY_CAPABILITY.items() if its == arch
        )
        load = _writing(loader, fake.events)
        monkeypatch.setattr(loader, "load", load)
        monkeypatch.setattr(pkg, "_LOAD", load)
        monkeypatch.setattr(pkg, "_device_arch", lambda device, arch=arch: arch)
        monkeypatch.setattr(pkg, "_device_capability", lambda device, c=capability: c)
        monkeypatch.setattr(pkg, "_device_index", lambda device: 0)
        monkeypatch.setattr(pkg, "_capability", lambda index, c=capability: c)
        for residual in (False, True):
            widths = {
                _route(route)[1]
                for route in loader.ROUTES[arch]
                if _route(route)[2] == residual
            }
            if not widths:
                continue
            hidden = HIDDEN if HIDDEN in widths else min(widths)
            for lead in ((rows,), (2, rows // 2)):
                shape = (*lead, hidden)
                shapes = (shape, lead)
                x = torch.randn(shape).to(torch.bfloat16)
                w = torch.randn(hidden).to(torch.bfloat16)
                if loader.route_key(hidden, "fwd", residual) in loader.ROUTES[arch]:
                    outputs = dict(FWD_OUTPUTS)
                    extra = {}
                    if residual:
                        extra["residual"] = torch.randn(shape).to(torch.bfloat16)
                    else:
                        del outputs["residual_out"]
                    for given in (None, *((n, k) for n in outputs for k in layouts)):
                        buffers = {}
                        if given is not None:
                            _key, dtype, full = outputs[given[0]]
                            buffers[given[0]] = _buffer(
                                shapes[0 if full else 1], dtype, given[1]
                            )
                        results = forward(x, w, 1e-6, **extra, **buffers)
                        label = (mode, arch, "fwd", residual, lead, given)
                        _check_outputs(results, buffers, outputs, shapes, label)
                        checked += 1
                        residuals += residual
                if loader.route_key(hidden, "bwd", residual) in loader.ROUTES[arch]:
                    g = torch.randn(shape).to(torch.bfloat16)
                    rstd = torch.rand(lead) + 0.5
                    extra = {}
                    if residual:
                        extra["g_residual"] = torch.randn(shape).to(torch.bfloat16)
                    for given in (
                        None,
                        *((n, k) for n in BWD_OUTPUTS for k in layouts),
                    ):
                        buffers = {}
                        if given is not None:
                            _key, dtype, full = BWD_OUTPUTS[given[0]]
                            size = shape if full else (hidden,)
                            buffers[given[0]] = _buffer(size, dtype, given[1])
                        with pytest.MonkeyPatch.context() as patch:
                            # the workspace the call allocates, at an address
                            # where a device allocation may start
                            patch.setattr(torch, "zeros", _placed_zeros(0, []))
                            results = backward(g, x, w, rstd, **extra, **buffers)
                        label = (mode, arch, "bwd", residual, lead, given)
                        _check_outputs(
                            results, buffers, BWD_OUTPUTS, (shape, (hidden,)), label
                        )
                        checked += 1
    assert checked and residuals, (mode, checked, residuals)
