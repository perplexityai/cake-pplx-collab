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

The opt-in two-word accumulation of the mHC training registries (``accumulation="fp32x2"``), on the host rules alone
(no device): which records a call selects in that mode, at which hidden sizes and Sinkhorn counts the mode exists,
and what it refuses.

* The default mode (``"fp32"``) selects exactly what the registry selected without the keyword.
* In the two-word mode every record of a role of ``ACC_WORDS_ROLES`` is replaced by the two-word program of its
  architecture, role and parameter-row layout (``ACC_WORDS_PROGRAMS``; the ``"<role>/rows"`` program where
  ``bwd_group_rows`` holds: hidden 4096 above 14927 tokens on sm_100a / sm_103a); every other record stays.  The
  two-word records are in no route (only the mode selects them), ship for the architectures of their table, have the
  role they replace and keep the buffers of the records they replace (the same logical arguments; in the
  nondeterministic registry on sm_107a the two-word column slab, a fold, also binds the ``dfn`` slots, the arrival
  counts and ``prow``, which the reduction form it replaces leaves out).
* The mode exists at hidden 4096 and the Sinkhorn count of record only; elsewhere it is a ``NotImplementedError``,
  and an unknown mode is a ``ValueError``.
"""

from __future__ import annotations

import pytest

from tests.cake.test_helpers.cake_package import cake_package

DET = cake_package("cake.cake_mhc_det_train", "cake_jit")
ND = cake_package("cake.cake_mhc_nondet_train", "cake_jit")
REGISTRIES = {"det": DET, "nondet": ND}
TWO_WORD = "fp32x2"
ROWS = (
    1,
    7,
    64,
    65,
    255,
    1000,
    2049,
    4096,
    4097,
    8192,
    14927,
    14928,
    16263,
    20001,
    32768,
    65536,
)
KINDS = ("fwd_prepare", "fwd_recompute", "fwd_finish", "bwd_prepare", "bwd_finish")
DET_PATHS = ("forward", "backward", "checkpoint")
EXPECTED_ROLES = {
    "det": ("fwd_map", "bwd_token", "bwd_slab", "bwd_reduce"),
    "nondet": ("fwd_map", "bwd_token", "bwd_slab"),
}


def _boards(reg) -> list[tuple[str, int]]:
    return sorted(
        set(reg.RULES_BOARDS) | {("sm_100a", 148), ("sm_103a", 148), ("sm_107a", 212)}
    )


def _select(
    reg, arch: str, hidden: int, kind: str, rows: int, sm: int, path, **kwargs
) -> tuple[str, ...]:
    extra = {"path": path} if path is not None else {}
    return tuple(
        reg.selected_modules(
            arch, hidden, kind, rows=rows, sm_count=sm, **extra, **kwargs
        )
    )


def _paths(name: str):
    return DET_PATHS if name == "det" else (None,)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(REGISTRIES))
def test_tables_of_the_two_word_programs(name: str) -> None:
    reg = REGISTRIES[name]
    assert reg.ACCUMULATIONS == ("fp32", "fp32x2") and reg.ACCUMULATION == "fp32"
    assert reg.ACC_WORDS_HIDDEN == 4096
    assert reg.ACC_WORDS_ROLES == EXPECTED_ROLES[name]
    routed = {
        n for arch in reg.ROUTES for names in reg.ROUTES[arch].values() for n in names
    }
    for key, by_arch in reg.ACC_WORDS_PROGRAMS.items():
        role = key.split("/")[0]
        assert role in reg.ACC_WORDS_ROLES, key
        for arch, program in by_arch.items():
            record = reg.MODULES[program]
            assert program.endswith("_pair"), program
            assert (
                record["role"] == role and record["hidden"] == reg.ACC_WORDS_HIDDEN
            ), program
            assert arch in record["arches"], (program, arch)
            assert program not in routed, (
                f"{program} is in a route: only the two-word mode may select it"
            )
    # every role has a program on every architecture of the registry (the group-row key only where group rows run)
    for role in reg.ACC_WORDS_ROLES:
        assert set(reg.ACC_WORDS_PROGRAMS[role]) == set(reg.ARCHES), role
    for key in reg.ACC_WORDS_PROGRAMS:
        if key.endswith("/rows"):
            assert set(reg.ACC_WORDS_PROGRAMS[key]) == {
                arch
                for arch in reg.ARCHES
                if any(reg.bwd_group_rows(T, 148, arch, 4096) for T in ROWS)
            }, key


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(REGISTRIES))
def test_two_word_selection(name: str) -> None:
    reg = REGISTRIES[name]
    swapped = 0
    for arch, sm in _boards(reg):
        hidden = reg.ACC_WORDS_HIDDEN
        bound = {
            (arch, kind): {
                v
                for n in reg.ROUTES[arch][f"{kind}_h{hidden}"]
                for v in reg.MODULES[n]["logical"].values()
            }
            for kind in KINDS
        }
        for kind in KINDS:
            for path in _paths(name):
                for T in ROWS:
                    default = _select(reg, arch, hidden, kind, T, sm, path)
                    assert (
                        _select(
                            reg, arch, hidden, kind, T, sm, path, accumulation="fp32"
                        )
                        == default
                    )
                    two = _select(
                        reg, arch, hidden, kind, T, sm, path, accumulation=TWO_WORD
                    )
                    assert len(two) == len(default), (arch, kind, path, T)
                    rows = reg.bwd_group_rows(T, sm, arch, hidden)
                    for one, pair in zip(default, two, strict=True):
                        role = reg.MODULES[one]["role"]
                        if role not in reg.ACC_WORDS_ROLES:
                            assert pair == one, (arch, kind, path, T, one, pair)
                            continue
                        key = (
                            f"{role}/rows"
                            if rows and f"{role}/rows" in reg.ACC_WORDS_PROGRAMS
                            else role
                        )
                        assert pair == reg.ACC_WORDS_PROGRAMS[key][arch], (
                            arch,
                            kind,
                            path,
                            T,
                            role,
                        )
                        # the two-word program reads only values the call already binds for its route's records
                        assert (
                            set(reg.MODULES[pair]["logical"].values())
                            <= bound[(arch, kind)]
                        ), (one, pair)
                        swapped += 1
    assert swapped


@pytest.mark.cpu_only
def test_group_row_two_word_programs_on_sm_100a_and_sm_103a() -> None:
    """Above 14927 tokens at hidden 4096 on sm_100a / sm_103a the token owner (and the det reduction) write group
    rows: the two-word mode runs the group-row programs there and the per-CTA-row programs below; sm_107a has no
    group rows."""
    for name, reg in REGISTRIES.items():
        path = {"path": "backward"} if name == "det" else {}
        for arch in ("sm_100a", "sm_103a"):
            low = reg.selected_modules(arch, 4096, "bwd_finish", rows=14927, sm_count=148, accumulation=TWO_WORD,
                                       **path)  # fmt: skip
            high = reg.selected_modules(arch, 4096, "bwd_finish", rows=14928, sm_count=148, accumulation=TWO_WORD,
                                        **path)  # fmt: skip
            assert reg.ACC_WORDS_PROGRAMS["bwd_token"][arch] in low, (name, arch, low)
            assert reg.ACC_WORDS_PROGRAMS["bwd_token/rows"][arch] in high, (
                name,
                arch,
                high,
            )
            if name == "det":
                assert reg.ACC_WORDS_PROGRAMS["bwd_reduce"][arch] in low, (arch, low)
                assert reg.ACC_WORDS_PROGRAMS["bwd_reduce/rows"][arch] in high, (
                    arch,
                    high,
                )
        for T in (14928, 65536):
            sel = reg.selected_modules("sm_107a", 4096, "bwd_finish", rows=T, sm_count=212, accumulation=TWO_WORD,
                                       **path)  # fmt: skip
            assert reg.ACC_WORDS_PROGRAMS["bwd_token"]["sm_107a"] in sel, (name, sel)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(REGISTRIES))
def test_two_word_mode_coverage_and_refusals(name: str) -> None:
    reg = REGISTRIES[name]
    path = {"path": "backward"} if name == "det" else {}
    for arch in reg.ARCHES:
        for hidden in reg.supported_hidden_sizes(arch):
            if hidden == reg.ACC_WORDS_HIDDEN:
                continue
            with pytest.raises(
                NotImplementedError, match=f"hidden {reg.ACC_WORDS_HIDDEN} only"
            ):
                reg.selected_modules(arch, hidden, "bwd_finish", rows=64, sm_count=148, accumulation=TWO_WORD,
                                     **path)  # fmt: skip
            # the default mode serves that size
            assert reg.selected_modules(
                arch, hidden, "bwd_finish", rows=64, sm_count=148, **path
            )
        with pytest.raises(NotImplementedError, match="sinkhorn_iters"):
            reg.selected_modules(arch, 4096, "fwd_prepare", rows=64, sm_count=148, sinkhorn_iters=7,
                                 accumulation=TWO_WORD, **({"path": "forward"} if path else {}))  # fmt: skip
    for value in ("fp64", "FP32X2", "fp32x4", "", None, 2, 2.0):
        with pytest.raises(ValueError, match="accumulation"):
            reg.check_accumulation(value)
        with pytest.raises(ValueError, match="accumulation"):
            reg.selected_modules("sm_100a", 4096, "fwd_prepare", rows=64, sm_count=148, accumulation=value,
                                 **({"path": "forward"} if path else {}))  # fmt: skip
    assert reg.check_accumulation("fp32", 7168, 7) == "fp32"
    assert reg.check_accumulation(TWO_WORD, 4096, 20) == TWO_WORD
    assert reg.check_accumulation(TWO_WORD) == TWO_WORD


@pytest.mark.cpu_only
def test_the_checkpoint_carry_keeps_the_mode() -> None:
    """The det checkpoint carry (``carried_modules``) runs the two-word programs of the call's mode: only its fn
    split differs from the plain selection."""
    for arch, entry in DET.FN_SPLIT_CARRY.items():
        hidden = DET.MODULES[entry["record"]]["hidden"]
        if hidden != DET.ACC_WORDS_HIDDEN:
            continue
        sm = dict(_boards(DET)).get(arch, 148)
        for kind in ("fwd_recompute", "bwd_finish"):
            plain = DET.selected_modules(arch, hidden, kind, rows=129, sm_count=sm, path="checkpoint",
                                         accumulation=TWO_WORD)  # fmt: skip
            carried = DET.carried_modules(arch, hidden, kind, rows=129, sm_count=sm, path="checkpoint",
                                          accumulation=TWO_WORD)  # fmt: skip
            assert (
                carried[1:] == plain[1:]
                if kind == "fwd_recompute"
                else carried == plain[1:]
            ), (arch, kind)
            assert any(
                DET.MODULES[n]["role"] in DET.ACC_WORDS_ROLES and n.endswith("_pair")
                for n in carried
            )
