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
"""

from __future__ import annotations

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
    of :class:`_Module` and a ``tvm_ffi.use_torch_stream`` that records its
    contexts.  ``events`` holds each launch, ``(name, arch, args)``, and each
    stream context's ``"enter"`` and ``"exit"``."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, mode: str, arch: str, sms: dict
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

    def make(mode: str, arch: str, sms: dict) -> _Fakes:
        made.append(_Fakes(monkeypatch, mode, arch, sms))
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
