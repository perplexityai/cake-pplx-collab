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

Tests of the in-kernel Q/K L2 norm (``use_qk_l2norm_in_kernel=True``) of the Cake non-deterministic packed KDA
chunked training package (``cake_pplx._backends.cake_kda_nondet_train``).

The package runs the deterministic package's l2n programs in the same order on every row: its l2n records are the
deterministic package's, the fused gradient's l2n class copies the deterministic records (no S1 pair: no GLM intra
program of an l2n mode adds the ``q * dq`` term), raw mode's parameter-gradient reduce is the deterministic
``param_finalize`` (``param_finalize_l2n``; ``param_finalize``, the same program, serves the flag-off mode), the gate
epilogue never runs and the checkpoint hand-off never engages.  The one program of this package's own an l2n mode may
run is the tape WY twin, in place of the WY program on a tape backward.  These are mode contracts (records, plans,
launch lists, bytes, refusals); no result of this package is compared with the deterministic package's.

On the CPU (no kernel): the host rules this package shares with the deterministic package (the raw q / k operand
rule, the l2n mode keyword, the saved set's ``qk_rstd`` / ``qk_l2norm`` and its check, the layouts' transient ``qn`` /
``kn``, the byte queries, the registry's l2n leaf and the token grid, the values of the l2n launches, the test hook,
the autograd node's arity) and the plans of the l2n modes against the deterministic package's (records, grids,
launches, bytes, the pairing guard, no hand-off).  On a GPU: ``out`` and every result of every path (the saved set,
repeats, the replay, the light-states and tape sets, caller workspaces, lean tail passes, document groups, packed q /
k views, autograd and ``torch.utils.checkpoint``, the fused and the fallback routes, the tape WY twin and record) judged
against the exact FP64 reference beside the naive BF16 tier (``tests/cake_naive.py``), each path under its own name
over the judge's draws; the launch lists against the deterministic package's (a mode contract), the test hook's rows
bitwise the shared bitwise device-order reference's, a checkpointed forward that keeps no workspace and the refusals
(each package's backward refuses the other package's sets).  The package has no bit contract: no result is compared
with another result.  The deterministic package's l2n suite is ``test_cake_kda_det_train_l2n.py``; its FP64 reference
helper is shared.
"""

import dataclasses
import inspect
import types
from pathlib import Path

import pytest
import torch

from tests.experimental.test_helpers import cake_kda_det_train_reference_l2n as ref_l2n
from tests.experimental.test_helpers.cake_kda_det_train_reference import LOWER_BOUND
from tests.experimental.test_helpers.cake_kda_naive import (
    GRAD_NAMES,
    RAW_GRAD_NAMES,
    draw_case,
    judge_arms,
)
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_kda_nondet_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")
# the deterministic package: the source of the copied l2n records (mode contracts: records, plans, launch lists,
# bytes, refusals); none of its results is compared with this package's
det_backend = cake_package("cake.cake_kda_det_train", "cake_backend")
det_jit = cake_package("cake.cake_kda_det_train", "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent

D = cake_backend.D
SCALE = cake_backend.SCALE
CPU = torch.device("cpu")
ARCHES = tuple(cake_jit.ARCHES)
L2N_MODES = ("off", "fused", "fallback")
L2N_ON = ("fused", "fallback")
GATES = ("g", "raw")
# the fused gradient's raw class: the deterministic dqkg records with the class suffix (its l2n twins: the
# deterministic l2n records with ``_raw_l2n`` in place of ``_l2n``)
DQKG_RAW_SUFFIX, DQKG_RAW_L2N_SUFFIX = "_raw", "_raw_l2n"
# the tape path's w-only WY twin of this package (launched in place of its record where its rule admits it)
TAPE_TWINS = {"wy_bwd": "wy_bwd_v9"}
# the raw class's parameter finalize of the l2n modes -> the deterministic record it copies
L2N_PARAM = {"param_finalize_l2n": "param_finalize"}
# the backward's return order (g mode; raw mode adds the parameter gradients)
RETURN_NAMES = ("dq", "dk", "dv", "dbeta", "dg")
RAW_RETURN_NAMES = RETURN_NAMES + ("dA_log", "ddt_bias")


# the package's entries (they take no mode argument)
cake_kda_nondet_train_forward = cake_backend.cake_kda_nondet_train_forward
cake_kda_nondet_train_backward = cake_backend.cake_kda_nondet_train_backward
cake_kda_nondet = cake_backend.cake_kda_nondet
# each package's entries refuse the other package's saved set with ``ValueError`` naming the mode: this package's
# message, then the deterministic package's
ND_REFUSES = "is not a non-deterministic-package"
DET_REFUSES = "is not a deterministic-package"


def _det_name(name: str | None) -> str | None:
    """The deterministic record a record of this package copies (the tape WY twin: its WY record)."""
    if name is None:
        return None
    if name in L2N_PARAM:
        return L2N_PARAM[name]
    for parent, twin in TAPE_TWINS.items():
        if name == twin:
            return parent
    if name.startswith("dqkg"):
        if name.endswith(DQKG_RAW_L2N_SUFFIX):
            return name.removesuffix(DQKG_RAW_L2N_SUFFIX) + "_l2n"
        return name.removesuffix(DQKG_RAW_SUFFIX)
    return name


def _serves(name: str, mode: str, jit=cake_jit) -> bool:
    record = jit.MODULES[name]
    return record["stage"] not in jit.L2N_STAGES or mode in record.get("l2n", ["off"])


def _packed(S: int, H: int, c0: int, width: int) -> torch.Tensor:
    """The ``q`` columns of a packed projection output ``[S, width]``: ``out[:, c0:c0 + 128 * H].view(S, H, 128)``."""
    out = torch.randn(S, width, dtype=torch.float32).to(torch.bfloat16)
    return out[:, c0 : c0 + D * H].view(S, H, D)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality of tensors of any element size (signed zeros and NaN payloads count)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        a.element_size()
    ]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


# ---------------------------------------------------------------------------------------------------------------
# the raw q / k operand rule, the l2n mode keyword (the deterministic package's rules, this package's host)


def test_qk_raw_accepts_contiguous_and_packed_views_without_a_copy() -> None:
    S, H = 5, 3
    x = torch.randn(S, H, D).to(torch.bfloat16)
    assert cake_backend._check_qk_raw(x, "q", S, H, CPU) is x
    for c0, width in ((0, 3 * D * H), (8, 3 * D * H + 16), (D * H, 2 * D * H)):
        view = _packed(S, H, c0, width)
        got = cake_backend._check_qk_raw(view, "q", S, H, CPU)
        assert got.data_ptr() == view.data_ptr()
        assert got.stride() == view.stride() == (width, D, 1)


def test_qk_raw_canonical_axes_of_extent_one_are_restrided_without_a_copy() -> None:
    base = torch.zeros(4096, dtype=torch.bfloat16)
    one_head = base.as_strided((3, 1, D), (512, 7, 1))  # H == 1: any head stride
    got = cake_backend._check_qk_raw(one_head, "k", 3, 1, CPU)
    assert got.data_ptr() == one_head.data_ptr() and got.stride() == (512, D, 1)
    one_token = base.as_strided((1, 2, D), (3, D, 1))  # S == 1: any token stride
    got = cake_backend._check_qk_raw(one_token, "k", 1, 2, CPU)
    assert got.data_ptr() == one_token.data_ptr() and got.stride() == (2 * D, D, 1)


@pytest.mark.parametrize(
    "case",
    [
        "inner_stride",
        "head_stride",
        "token_stride_mod8",
        "overlapping_rows",
        "misaligned_base",
        "shape",
        "dtype",
    ],
)
def test_qk_raw_refusals_are_the_deterministic_packages(case: str) -> None:
    """Each view the deterministic package refuses is refused here with the same message."""
    S, H = 4, 2
    base = torch.zeros(1 << 16, dtype=torch.bfloat16)
    views = {
        "inner_stride": base.as_strided((S, H, D), (H * D * 2, D * 2, 2)),
        "head_stride": base.as_strided((S, H, D), (4 * D, 2 * D, 1)),
        "token_stride_mod8": base.as_strided((S, H, D), (H * D + 4, D, 1)),
        "overlapping_rows": base.as_strided((S, H, D), (D, D, 1)),
        "misaligned_base": base[4:].as_strided((S, H, D), (H * D, D, 1)),
        "shape": torch.zeros(S, H, 64, dtype=torch.bfloat16),
        "dtype": torch.zeros(S, H, D, dtype=torch.float32),
    }
    x = views[case]
    with pytest.raises(ValueError) as nd:
        cake_backend._check_qk_raw(x, "q", S, H, CPU)
    with pytest.raises(ValueError) as det:
        det_backend._check_qk_raw(x, "q", S, H, CPU)
    assert str(nd.value) == str(det.value), case


def test_operands_take_raw_views_only_with_the_flag() -> None:
    S, H = 6, 2
    q = _packed(S, H, 8, 3 * D * H + 8)
    k = _packed(S, H, D * H + 8, 3 * D * H + 8)
    v = torch.zeros(S, H, D, dtype=torch.bfloat16)
    g = torch.zeros(S, H, D, dtype=torch.float32)
    beta = torch.zeros(S, H, dtype=torch.bfloat16)
    cu = torch.tensor([0, S], dtype=torch.int32)
    for mode in L2N_ON:
        got = cake_backend._operands(q, k, v, g, beta, cu, l2n=mode)
        assert got[0].data_ptr() == q.data_ptr() and got[1].data_ptr() == k.data_ptr()
    with pytest.raises(ValueError, match="contiguous"):
        cake_backend._operands(q, k, v, g, beta, cu)


@pytest.mark.parametrize("bad", [1, 0, None, "fused", 1.0])
def test_mode_keyword_must_be_a_bool(bad) -> None:
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel must be a bool"):
        cake_backend._l2n_mode(bad)
    assert cake_backend._l2n_mode(True) == "fused"
    assert cake_backend._l2n_mode(False) == "off"


def test_l2n_vocabulary_is_the_deterministic_packages() -> None:
    """The modes, the token-local stages and the logical names of the l2n programs are the deterministic package's;
    the l2n stages are the deterministic package's (the output backward and the fused gradient among them; the
    fused gradient's S1 records serve the flag-off mode alone) and the parameter finalize (``param_finalize``
    serves the flag-off mode, ``param_finalize_l2n`` the l2n modes: one deterministic program)."""
    assert (
        tuple(cake_backend.L2N_MODES)
        == tuple(cake_jit.L2N_MODES)
        == tuple(det_jit.L2N_MODES)
        == L2N_MODES
    )
    assert {"dav", "dqkg"} <= set(det_jit.L2N_STAGES)
    nd_only = {"param_finalize"}
    assert set(cake_jit.L2N_STAGES) == set(det_jit.L2N_STAGES) | nd_only
    assert not nd_only & set(det_jit.L2N_STAGES)
    for stage in ("qk_l2norm", "qk_apply", "qk_pullback"):
        assert stage in cake_jit.STAGES and stage in cake_jit.OPTIONAL_STAGES
        assert stage in cake_jit.L2N_STAGES
    assert tuple(cake_jit.STAGES) == tuple(det_jit.STAGES)
    assert tuple(cake_jit.LOGICAL_TENSORS) == tuple(det_jit.LOGICAL_TENSORS)
    assert tuple(cake_jit.LOGICAL_SCALARS) == tuple(det_jit.LOGICAL_SCALARS)
    for name in ("q_raw", "k_raw", "qn", "kn", "qk_rstd", "qn_int", "kn_int"):
        assert name in cake_jit.LOGICAL_TENSORS
    for name in (
        "num_tokens",
        "q_raw_row_stride",
        "q_raw_head_stride",
        "k_raw_row_stride",
        "k_raw_head_stride",
        "dq_row_stride",
        "dk_row_stride",
        "read_rstd",
        "write_rstd",
        "write_qkn",
    ):
        assert name in cake_jit.LOGICAL_SCALARS


# ---------------------------------------------------------------------------------------------------------------
# saved set and bytes


def _saved(saved_type, S: int, H: int, **fields):
    a = torch.zeros(S, H, 64, dtype=torch.bfloat16)
    return saved_type(4, a, a, None, S, H, 1, SCALE, **fields)


def test_saved_set_fields_and_their_checks_read_either_package() -> None:
    """``qk_rstd`` (FP32 ``[2, S, H]``) and ``qk_l2norm`` default off and ride last in the autograd extras; this
    package's ``_check_saved_l2n`` and the deterministic package's apply the same rule to a set of either package's
    type (the entries refuse the other package's type before this check)."""
    S, H = 3, 2
    fields = {
        f.name: f.default
        for f in dataclasses.fields(cake_backend.CakeKDANondetChunkSaved)
    }
    assert fields["qk_rstd"] is None and fields["qk_l2norm"] is False
    for version in (cake_backend.SAVED_VERSION, cake_backend.SAVED_VERSION_TAPE):
        assert cake_backend._EXTRA_FIELDS[version][-1] == "qk_rstd"
    rstd = torch.zeros(2, S, H, dtype=torch.float32)
    checked = 0
    for saved_type in (
        cake_backend.CakeKDANondetChunkSaved,
        det_backend.CakeKDADetChunkSaved,
    ):
        on = _saved(saved_type, S, H, qk_rstd=rstd, qk_l2norm=True)
        off = _saved(saved_type, S, H)
        for check in (cake_backend._check_saved_l2n, det_backend._check_saved_l2n):
            check(on, "fused", S, H, CPU)
            check(on, "fallback", S, H, CPU)
            check(off, "off", S, H, CPU)
            checked += 3
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel=True; this backward"
            ):
                check(on, "off", S, H, CPU)
            with pytest.raises(
                ValueError, match="use_qk_l2norm_in_kernel=False; this backward"
            ):
                check(off, "fused", S, H, CPU)
            with pytest.raises(ValueError, match="carries none|has none"):
                check(_saved(saved_type, S, H, qk_l2norm=True), "fused", S, H, CPU)
            with pytest.raises(ValueError, match="without the in-kernel"):
                check(_saved(saved_type, S, H, qk_rstd=rstd), "off", S, H, CPU)
            for bad in (
                torch.zeros(2, S + 1, H),
                torch.zeros(2, S, H, dtype=torch.float64),
                torch.zeros(2, H, S).transpose(1, 2),
            ):
                with pytest.raises(ValueError):
                    check(
                        _saved(saved_type, S, H, qk_rstd=bad, qk_l2norm=True),
                        "fused",
                        S,
                        H,
                        CPU,
                    )
    assert checked == 2 * 2 * 3
    # the autograd extras: qk_rstd rides last, absent with the flag off
    on = _saved(
        cake_backend.CakeKDANondetChunkSaved, S, H, qk_rstd=rstd, qk_l2norm=True
    )
    extra, names = cake_backend._saved_extra(on)
    assert names == ("qk_rstd",) and extra[0] is rstd
    assert cake_backend._saved_extra(
        _saved(cake_backend.CakeKDANondetChunkSaved, S, H)
    ) == (
        (),
        (),
    )


def test_check_saved_l2n_agrees_in_both_directions() -> None:
    """Each package's ``_check_saved_l2n`` on the other package's saved sets and on its own: over both set types, every
    mode and every ``qk_rstd`` / ``qk_l2norm`` state, the two checks accept the same sets and refuse the others with
    the same message: the same rule (each package's backward reads its own type only and refuses the other's
    before this check)."""
    S, H = 3, 2
    rstd = torch.zeros(2, S, H, dtype=torch.float32)
    states = {
        "on": {"qk_rstd": rstd, "qk_l2norm": True},
        "off": {},
        "flag without rstd": {"qk_l2norm": True},
        "rstd without flag": {"qk_rstd": rstd},
        "short rstd": {"qk_rstd": torch.zeros(2, S + 1, H), "qk_l2norm": True},
        "fp64 rstd": {
            "qk_rstd": torch.zeros(2, S, H, dtype=torch.float64),
            "qk_l2norm": True,
        },
        "strided rstd": {
            "qk_rstd": torch.zeros(2, H, S).transpose(1, 2),
            "qk_l2norm": True,
        },
    }

    def outcome(check, saved, mode: str) -> str | None:
        try:
            check(saved, mode, S, H, CPU)
        except ValueError as exc:
            return str(exc)
        return None

    compared = accepted = 0
    for saved_type in (
        cake_backend.CakeKDANondetChunkSaved,
        det_backend.CakeKDADetChunkSaved,
    ):
        for state, fields in states.items():
            saved = _saved(saved_type, S, H, **fields)
            for mode in ("off", *L2N_ON):
                nd = outcome(cake_backend._check_saved_l2n, saved, mode)
                det = outcome(det_backend._check_saved_l2n, saved, mode)
                assert nd == det, (saved_type.__name__, state, mode, nd, det)
                compared += 1
                accepted += nd is None
    assert compared == 2 * len(states) * (1 + len(L2N_ON)) > 0
    # per set type: the "on" set in both l2n modes, the "off" set in the flag-off mode
    assert accepted == 2 * (len(L2N_ON) + 1)


def test_saved_and_output_bytes_are_the_deterministic_packages() -> None:
    S, H = 1000, 8
    # (the tape set's bytes need the document count)
    for N, save_mode in ((None, "light"), (3, "light"), (3, "tape")):
        kw = dict(save_mode=save_mode, sm_count=148, arch="sm_100a")
        off = cake_backend.saved_bytes(S, H, N, **kw)
        for mode in L2N_ON:
            nd = cake_backend.saved_bytes(S, H, N, **kw, l2n=mode)
            assert nd == off + 8 * S * H, (N, save_mode, mode)
            assert nd == det_backend.saved_bytes(S, H, N, **kw, l2n=mode)
    for backward in (False, True):
        for gate in GATES:
            for mode in L2N_MODES:
                nd = cake_backend.output_bytes(
                    S, H, backward=backward, gate=gate, l2n=mode
                )
                assert nd == cake_backend.output_bytes(
                    S, H, backward=backward, gate=gate
                )
                assert nd == det_backend.output_bytes(
                    S, H, backward=backward, gate=gate, l2n=mode
                )
    with pytest.raises(ValueError, match="l2n must be one of"):
        cake_backend.saved_bytes(S, H, l2n="on")


# ---------------------------------------------------------------------------------------------------------------
# layouts


def _rows(mode: str, **kwargs) -> dict[str, dict]:
    return {r["buffer"]: r for r in cake_backend.lifetime_table(mode, **kwargs)}


@pytest.mark.parametrize(
    "mode", ["fwd", "fwd_tape", "fwd_kcp", "replay", "replay_tape"]
)
@pytest.mark.parametrize("l2n", ["fused", "fallback"])
def test_layouts_carry_transient_qn_kn_int_on_the_u_w_slots(
    mode: str, l2n: str
) -> None:
    """The fused intra program's internal rows qn_int / kn_int ride at sub-offset 0 of the u / w slots of the
    forward and replay layouts, lifetime (0, 1), bytes unchanged; the layouts are the deterministic package's."""
    off = _rows(mode)
    on = _rows(mode, l2n=l2n)
    assert "qn_int" not in off and "kn_int" not in off
    for name, host in (("qn_int", "u"), ("kn_int", "w")):
        r = on[name]
        assert r["slot"] == on[host]["slot"] and r["kind"] == "B16"
        assert r["sub_offset_per_th"] == 0
        assert (r["first_write"], r["last_read"]) == (0, 1)
    assert cake_backend.lifetime_table(mode, l2n=l2n) == det_backend.lifetime_table(
        mode, l2n=l2n
    )
    R, H, NC = 128, 2, 2
    assert cake_backend._mode_bytes(
        R, H, NC, 0, mode, 1, True, l2n
    ) == cake_backend._mode_bytes(R, H, NC, 0, mode, 1, True, "off")
    assert cake_backend._mode_bytes(R, H, NC, 0, mode, 1, True, l2n) == (
        det_backend._mode_bytes(R, H, NC, 0, mode, 1, True, l2n)
    )


@pytest.mark.parametrize("mode", ["bwd", "replay", "bwd_tape", "replay_tape"])
def test_fallback_backward_keeps_dq_f_dk_f_until_the_pullback(mode: str) -> None:
    fused = _rows(mode, l2n="fused")
    fallback = _rows(mode, l2n="fallback")
    for name in ("dq_f", "dk_f"):
        assert (fused[name]["first_write"], fused[name]["last_read"]) == (6, 7)
        assert (fallback[name]["first_write"], fallback[name]["last_read"]) == (6, 8)
    # every l2n mode drops the gate epilogue's buffers (the l2n GLM intra programs fuse it)
    for rows in (fused, fallback):
        assert "dg_int" not in rows and "db_int" not in rows
    assert "qn_int" not in _rows("bwd", l2n="fused")
    with pytest.raises(ValueError, match="gate_fused must be True"):
        cake_backend.lifetime_table(mode, l2n="fused", gate_fused=False)


@pytest.mark.parametrize("arch", ARCHES)
def test_saved_set_backward_keeps_qn_kn_in_the_arena(arch: str) -> None:
    """The saved-set backward of an l2n mode keeps qn / kn on the arena's do_int / dv2 bytes (dv2 on the dq / dk
    bytes) as the deterministic package's does (no record of this package's saved-set paths binds do_int, the tape
    WY twin binds a subset of its record's operands): memory_report counts no per-call qn / kn there and is the
    deterministic package's; the replay, a lean layout, two tail passes and 2 S < R keep them per call."""
    rows = 2 * D * 2  # per-call qn + kn bytes per token-head
    checked = 0
    for mode in L2N_ON:
        for gate in GATES:
            for S, N, H in ((5000, 3, 16), (16384, 1, 64), (16384, 4, 64)):
                p = cake_backend.plan(S, N, H, 148, arch, gate=gate, l2n=mode)
                for path in (
                    x for x in ("light", "light_states", "tape") if x in p.records
                ):
                    layout = "bwd_tape" if path == "tape" else "bwd"
                    names = {r.name for r in p.layout(layout)}
                    assert {"do_int", "dv2"} <= names
                    assert cake_backend._l2n_rows_in_arena(p, path, names, 1), (
                        path,
                        mode,
                        gate,
                    )
                    assert not cake_backend._l2n_rows_in_arena(p, path, names, 2)
                    lean = {r.name for r in p.layout(layout + "_lean")}
                    assert not cake_backend._l2n_rows_in_arena(p, path, lean, 1)
                    checked += 1
                replay = {r.name for r in p.layout("replay")}
                assert not cake_backend._l2n_rows_in_arena(p, "replay", replay, 1)
                for save_mode in ("light", "tape"):
                    q = dict(sm_count=148, arch=arch, gate=gate, l2n=mode)
                    report = cake_backend.memory_report(S, N, H, save_mode, **q)
                    det = det_backend.memory_report(S, N, H, save_mode, **q)
                    assert report["bwd_qn_kn"] == 0, (save_mode, mode, gate)
                    assert report["replay_qn_kn"] == rows * S * H
                    assert {key: report[key] for key in det} == det, (
                        S,
                        N,
                        H,
                        save_mode,
                        mode,
                        gate,
                    )
                    assert (
                        report["handoff_bytes"] == report["handoff_budget_bytes"] == 0
                    )
                lean_report = cake_backend.memory_report(
                    S,
                    N,
                    H,
                    sm_count=148,
                    arch=arch,
                    gate=gate,
                    l2n=mode,
                    arena=cake_backend.CakeKDANondetArena(lean=True),
                )
                assert lean_report["bwd_qn_kn"] == rows * S * H
        # 64 tokens in 8 documents: R = 640 internal rows > 2 S, dv2 does not fit the dq / dk bytes
        p = cake_backend.plan(64, 8, 2, 148, arch, l2n=mode)
        assert 2 * p.S < p.R
        names = {r.name for r in p.layout("bwd")}
        assert not cake_backend._l2n_rows_in_arena(p, "light", names, 1)
        report = cake_backend.memory_report(64, 8, 2, sm_count=148, arch=arch, l2n=mode)
        assert report["bwd_qn_kn"] == rows * 64 * 2
    assert checked > 0


def test_gate_fused_on_every_arch_in_l2n_modes() -> None:
    for arch in ARCHES:
        for gate in GATES:
            for mode in L2N_ON:
                assert cake_backend._gate_fused(arch, gate, mode) is True
                assert det_backend._gate_fused(arch, gate, mode) is True


# ---------------------------------------------------------------------------------------------------------------
# registry


def test_select_record_refuses_an_unknown_mode_and_serves_each_class() -> None:
    """The registry's selection with an l2n mode: an unknown mode is refused; the token-local stages select nothing
    outside the fallback mode; the fused gradient selects the deterministic record's l2n copy (never an S1 record)
    in both l2n modes."""
    scalars = dict(
        num_chunks_pad=2, num_seqs=1, num_heads=1, sm_count=148, arch=ARCHES[0]
    )
    with pytest.raises(ValueError, match="l2n 'on' is not one of"):
        cake_jit.select_record("dqkg", **scalars, path="light", l2n="on")
    for stage in ("qk_l2norm", "qk_apply", "qk_pullback"):
        path = "fwd" if stage == "qk_l2norm" else "light"
        assert cake_jit.select_record(stage, **scalars, path=path, l2n="fused") is None
        assert cake_jit.select_record(stage, **scalars, path=path) is None
        assert cake_jit.select_record(stage, **scalars, path=path, l2n="fallback") == (
            det_jit.select_record(stage, **scalars, path=path, l2n="fallback")
        )
    for mode in L2N_ON:
        for path in ("light", "tape", "replay"):
            name = cake_jit.select_record("dqkg", **scalars, path=path, l2n=mode)
            assert name not in cake_jit.S1_RECORDS and name.endswith("_l2n"), name
            assert _det_name(name) == det_jit.select_record(
                "dqkg", **scalars, path=path, l2n=mode
            )


def test_the_l2n_leaf_and_the_tokens_grid_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain record kept on some paths of the fused route narrows its rule with ``l2n_in``; the twin serves the
    others: exactly one record per (mode, path).  The token-local programs take one CTA per 64 tokens and head."""
    fake = {
        "x_plain": {
            "stage": "qk_l2norm",
            "l2n": ["off", "fused"],
            "select_rule": {
                "kind": "any_of",
                "rules": [
                    {"kind": "l2n_in", "modes": ["off"]},
                    {"kind": "path_in", "paths": ["replay"]},
                ],
            },
            "grid_rule": {"kind": "tokens"},
        },
        "x_l2n": {
            "stage": "qk_l2norm",
            "l2n": ["fused"],
            "select_rule": {"kind": "path_in", "paths": ["fwd", "fwd_nosave"]},
            "grid_rule": {"kind": "tokens"},
        },
    }
    for name, rec in fake.items():
        monkeypatch.setitem(cake_jit.MODULES, name, rec)
    scalars = dict(
        num_chunks_pad=2, num_seqs=1, num_heads=1, sm_count=148, arch=ARCHES[0]
    )

    def sel(path: str, mode: str) -> str | None:
        return cake_jit.select_record("qk_l2norm", **scalars, path=path, l2n=mode)

    assert sel("fwd", "off") == "x_plain" and sel("replay", "off") == "x_plain"
    assert sel("fwd", "fused") == "x_l2n" and sel("fwd_nosave", "fused") == "x_l2n"
    assert sel("replay", "fused") == "x_plain"
    assert sel("fwd", "fallback") == "qk_l2norm"
    common = dict(num_chunks_pad=4, num_seqs=1, num_heads=3, sm_count=148)
    assert cake_jit.launch_grid("x_l2n", **common, num_tokens=129) == (3, 3, 1)
    assert cake_jit.launch_grid("x_l2n", **common, num_tokens=64) == (1, 3, 1)
    assert cake_jit.launch_grid("x_l2n", **common, num_tokens=0) == (1, 3, 1)


# ---------------------------------------------------------------------------------------------------------------
# plans


PLAN_ROWS = (
    ((256,), 2),
    ((5000, 3000, 2), 16),
    ((64,) * 40 + (3000,), 64),
    ((63, 1, 65), 4),
    ((2048,), 1),
    ((4000, 96), 32),
    ((129,), 128),
)


@pytest.mark.parametrize("arch", ARCHES)
def test_l2n_plans_are_the_deterministic_l2n_plans(arch: str) -> None:
    """``plan(..., l2n=mode)`` is the deterministic package's plan of the mode, in both gate modes: the same geometry,
    on every path the same record per stage (the fused gradient's raw and l2n classes and ``param_finalize_l2n`` by
    their deterministic names, never an S1 record or a hand-off twin) with the same launches, grids and launch
    counts, the same layouts and bytes; a separate cache entry from the flag-off plan; no gate epilogue; the
    token-local stages dispatched in the fallback mode alone; the checkpoint hand-off never engages on it; the flag-off
    plan is the default plan."""
    checked = 0
    for lengths, heads in PLAN_ROWS:
        S, N = sum(lengths), len(lengths)
        for sm in (148, 160):
            default = cake_backend.plan(S, N, heads, sm, arch)
            assert cake_backend.plan(S, N, heads, sm, arch, l2n="off") is default
            assert default.l2n == "off"
            assert not any(
                stage in ("qk_l2norm", "qk_apply", "qk_pullback")
                for by_stage in default.records.values()
                for stage in by_stage
            )
            for gate in GATES:
                for mode in L2N_ON:
                    where = (lengths, heads, sm, gate, mode)
                    p = cake_backend.plan(S, N, heads, sm, arch, gate=gate, l2n=mode)
                    dp = det_backend.plan(S, N, heads, sm, arch, gate=gate, l2n=mode)
                    assert p is not default
                    assert (p.gate, p.l2n) == (dp.gate, dp.l2n) == (gate, mode)
                    assert (
                        cake_backend.plan(S, N, heads, sm, arch, gate=gate, l2n=mode)
                        is p
                    )
                    assert p.gate_fused is True and dp.gate_fused is True
                    for field in (
                        "S",
                        "N",
                        "H",
                        "sm_count",
                        "arch",
                        "NC_max",
                        "NC_pad",
                        "R",
                        "recur",
                        "P",
                        "NSEG",
                        "proportional",
                        "seg_arg",
                        "gsize",
                        "light_path",
                        "seg_save",
                        "light_saves_qg",
                        "light_saves_wu",
                    ):
                        assert getattr(p, field) == getattr(dp, field), (where, field)
                    assert set(p.records) == set(dp.records), where
                    for path, by_stage in p.records.items():
                        at = (*where, path)
                        assert {s: _det_name(n) for s, n in by_stage.items()} == (
                            dp.records[path]
                        ), at
                        names = set(by_stage.values()) - {None}
                        assert not names & set(cake_jit.S1_RECORDS), at
                        assert not names & set(cake_jit.HANDOFF_TWINS.values()), at
                        assert not p.dispatched("gate_epilogue", path), at
                        for stage in ("qk_l2norm", "qk_apply", "qk_pullback"):
                            if mode == "fused":
                                assert stage not in by_stage, (*at, stage)
                        for stage, name in by_stage.items():
                            if name is None:
                                continue
                            assert _serves(name, mode), (*at, stage, name)
                            assert p.launches(stage, path) == dp.launches(stage, path)
                            for i in range(p.launches(stage, path)):
                                assert p.grid(stage, path, i) == dp.grid(
                                    stage, path, i
                                ), (
                                    *at,
                                    stage,
                                    i,
                                )
                        for tables in (True, False):
                            assert p.launch_count(path, tables=tables) == (
                                dp.launch_count(path, tables=tables)
                            ), (*at, tables)
                        checked += 1
                    for layout in cake_backend.MODES:
                        assert p.workspace_bytes(layout) == dp.workspace_bytes(
                            layout
                        ), (
                            where,
                            layout,
                        )
                    for handoff in ("on", "auto", "off"):
                        assert not cake_backend._handoff_engages(p, handoff, 0), (
                            where,
                            handoff,
                        )
                    assert p.handoff_twin is None or not cake_backend._handoff_engages(
                        p, "on", 0
                    )
    assert checked > 0


def test_l2n_plans_refuse_an_unknown_mode_and_the_pairing_guard_refuses_s1() -> None:
    with pytest.raises(ValueError, match="l2n must be one of"):
        cake_backend.plan(256, 1, 2, 148, "sm_100a", l2n="on")
    # an S1 record in an l2n plan is a silent dg error: the guard refuses it, in either gate mode
    for arch, glm in (
        ("sm_100a", "glm_intra_v21"),
        ("sm_103a", "glm_intra_v22"),
        ("sm_107a", "glm_intra_v23"),
    ):
        records = {"light": {"dqkg": "dqkg_davq_v24", "glm_intra": glm}}
        for mode in L2N_ON:
            with pytest.raises(ValueError, match=f"l2n mode {mode!r}"):
                cake_backend._check_pairs(records, arch=arch, H=16, l2n=mode)
            with pytest.raises(ValueError, match="raw gate mode"):
                cake_backend._check_pairs(
                    records, arch=arch, H=16, gate="raw", l2n=mode
                )
        # the flag-off g mode takes the S1 pair
        cake_backend._check_pairs(records, arch=arch, H=16)


@pytest.mark.parametrize("arch", ARCHES)
def test_l2n_plans_refuse_a_backward_path_without_an_l2n_dav_record(
    arch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_check_l2n_dav``: the fused gradient of an l2n mode (``davq`` 0) stores no
    ``dAqk``, so an l2n plan whose backward path launches no ``dav`` record of its mode
    (none, or a flag-off one) is refused with ``ValueError`` naming the path; a path
    without a ``dav`` stage (the forward) and the flag-off plans are not checked.
    Through ``plan``: a registry whose ``dav_l2n_replay`` rule admits no problem leaves
    the replay of an H 12 row without a ``dav`` record, and every l2n plan of the row
    is refused in both gate modes while its flag-off plan still builds."""
    check = cake_backend._check_l2n_dav
    fwd = {"fwd": {"fwd_intra": "fwd_intra_l2n"}}
    match = "path 'replay' launches no dav record"
    for mode in L2N_ON:
        check(fwd, arch=arch, H=12, l2n=mode)
        check({**fwd, "replay": {"dav": "dav_l2n_replay"}}, arch=arch, H=12, l2n=mode)
        for name in (None, "dav_serial_replay"):
            records = {**fwd, "replay": {"dav": name}}
            check(records, arch=arch, H=12, l2n="off")
            with pytest.raises(ValueError, match=match):
                check(records, arch=arch, H=12, l2n=mode)
    narrowed = dict(cake_jit.MODULES["dav_l2n_replay"])
    narrowed["select_rule"] = {"kind": "heads_at_least", "min": 1 << 20}
    monkeypatch.setitem(cake_jit.MODULES, "dav_l2n_replay", narrowed)
    cake_backend._plan.cache_clear()  # no plan of the real registry answers below
    try:
        for gate in GATES:
            for mode in L2N_ON:
                with pytest.raises(ValueError, match=match):
                    cake_backend.plan(2000, 3, 12, 148, arch, gate=gate, l2n=mode)
        p = cake_backend.plan(2000, 3, 12, 148, arch)
        assert p.record("dav", "replay") == "dav_serial_replay"
    finally:
        monkeypatch.undo()
        cake_backend._plan.cache_clear()  # nor a plan of the narrowed one after it


@pytest.mark.parametrize("arch", ARCHES)
def test_raw_l2n_plans_finish_the_parameter_gradients_with_the_deterministic_finalize(
    arch: str,
) -> None:
    """Raw mode with the in-kernel Q/K L2 norm: on every architecture and head count every backward path pairs the
    deterministic l2n GLM intra record (``glm_intra_raw_l2n``; the fallback route: ``glm_intra_raw_l2n_capture``)
    with ``param_finalize_l2n``, the deterministic ``param_finalize`` (the same program, launch count and grid); the
    flag-off raw plan pairs ``glm_intra_raw`` with ``param_finalize`` on every architecture, as the deterministic
    package's flag-off raw plan does.  The pairing guard refuses the flag-off finalize or none in an l2n mode, and
    ``param_finalize_l2n`` in the flag-off mode."""
    checked = 0
    for H in (1, 4, 16, 31, 32, 64):
        for mode in L2N_ON:
            glm = (
                "glm_intra_raw_l2n" if mode == "fused" else "glm_intra_raw_l2n_capture"
            )
            p = cake_backend.plan(5000, 3, H, 148, arch, gate="raw", l2n=mode)
            dp = det_backend.plan(5000, 3, H, 148, arch, gate="raw", l2n=mode)
            for path, by_stage in p.records.items():
                if by_stage.get("glm_intra") is None:
                    continue
                at = (H, mode, path)
                assert by_stage["glm_intra"] == glm == dp.records[path]["glm_intra"], at
                assert by_stage["param_finalize"] == "param_finalize_l2n", at
                assert dp.records[path]["param_finalize"] == "param_finalize", at
                launches = p.launches("param_finalize", path)
                assert launches == dp.launches("param_finalize", path) == 1, at
                assert p.grid("param_finalize", path, 0) == dp.grid(
                    "param_finalize", path, 0
                ), at
                checked += 1
    assert checked > 0
    # the flag-off raw plan of the same architecture: the deterministic pair, as the deterministic package's
    off = cake_backend.plan(5000, 3, 16, 148, arch, gate="raw")
    det_off = det_backend.plan(5000, 3, 16, 148, arch, gate="raw")
    paired = 0
    for path, by_stage in off.records.items():
        if by_stage.get("glm_intra") is None:
            continue
        pair = (by_stage["glm_intra"], by_stage["param_finalize"])
        assert pair == ("glm_intra_raw", "param_finalize"), (path, pair)
        assert det_off.records[path]["glm_intra"] == "glm_intra_raw", path
        assert det_off.records[path]["param_finalize"] == "param_finalize", path
        launches = off.launches("param_finalize", path)
        assert launches == det_off.launches("param_finalize", path) == 1, path
        assert off.grid("param_finalize", path, 0) == det_off.grid(
            "param_finalize", path, 0
        ), path
        paired += 1
    assert paired > 0
    for mode in L2N_ON:
        glm = "glm_intra_raw_l2n" if mode == "fused" else "glm_intra_raw_l2n_capture"
        base = {
            "dqkg": "dqkg_raw",
            "glm_intra": glm,
            "param_finalize": "param_finalize_l2n",
        }
        cake_backend._check_pairs(
            {"light": base}, arch=arch, H=16, gate="raw", l2n=mode
        )
        with pytest.raises(ValueError, match="it must be 'param_finalize_l2n'"):
            cake_backend._check_pairs(
                {"light": {**base, "param_finalize": "param_finalize"}},
                arch=arch,
                H=16,
                gate="raw",
                l2n=mode,
            )
        with pytest.raises(ValueError, match="it must be 'param_finalize_l2n'"):
            cake_backend._check_pairs(
                {"light": {**base, "param_finalize": None}},
                arch=arch,
                H=16,
                gate="raw",
                l2n=mode,
            )
    with pytest.raises(ValueError, match="it must be 'param_finalize',"):
        cake_backend._check_pairs(
            {
                "light": {
                    "dqkg": "dqkg_raw",
                    "glm_intra": "glm_intra_raw",
                    "param_finalize": "param_finalize_l2n",
                }
            },
            arch=arch,
            H=16,
            gate="raw",
        )


@pytest.mark.parametrize("arch", ARCHES)
def test_fallback_replay_counts_no_qk_apply(arch: str) -> None:
    """The fallback plan selects a ``qk_apply`` record on the replay path too, but the replay normalizes q / k with
    ``qk_l2norm`` and never launches ``qk_apply`` (the deterministic package's rule)."""
    for S, N, H in ((256, 1, 2), (5000, 3, 16), (64 * 40, 300, 64)):
        p = cake_backend.plan(S, N, H, 148, arch, l2n="fallback")
        assert p.dispatched("qk_apply", "replay") and p.dispatched(
            "qk_l2norm", "replay"
        )
        selected = {s for s, r in p.records["replay"].items() if r is not None}
        assert set(p.stages("replay")) == selected - {"qk_apply"}
        for path in ("light", "tape"):
            assert "qk_apply" in p.stages(path) and "qk_l2norm" not in p.stages(path)
        fused = cake_backend.plan(S, N, H, 148, arch, l2n="fused")
        assert p.launch_count("replay") == fused.launch_count("replay") + p.launches(
            "qk_l2norm", "replay"
        ) + p.launches("qk_pullback", "replay")


# ---------------------------------------------------------------------------------------------------------------
# values, test hook, autograd arity


def test_l2n_values_bind_normalized_rows_under_q_and_k() -> None:
    S, H = 4, 2
    q = _packed(S, H, 8, 3 * D * H + 8)
    k = torch.zeros(S, H, D, dtype=torch.bfloat16)
    qn, kn = torch.empty_like(k), torch.empty_like(k)
    rstd = torch.empty(2, S, H)
    v = cake_backend._l2n_values(q, k, qn, kn, rstd, read_rstd=1, write_rstd=0)
    assert v["q"] is qn and v["k"] is kn and v["q_raw"] is q and v["k_raw"] is k
    assert v["qk_rstd"] is rstd and v["num_tokens"] == S
    assert v["q_raw_row_stride"] == 3 * D * H + 8 and v["k_raw_row_stride"] == H * D
    assert v["q_raw_head_stride"] == v["k_raw_head_stride"] == D
    assert v["dq_row_stride"] == v["dk_row_stride"] == H * D
    assert (v["read_rstd"], v["write_rstd"]) == (1, 0)
    assert v["qn_int"] is qn and v["kn_int"] is kn and v["write_qkn"] == 1
    qi = torch.empty(64, H, D, dtype=torch.bfloat16)
    ki = torch.empty(64, H, D, dtype=torch.bfloat16)
    v = cake_backend._l2n_values(
        q, k, qn, kn, rstd, read_rstd=0, write_rstd=1, write_qkn=0, qn_int=qi, kn_int=ki
    )
    assert v["qn_int"] is qi and v["kn_int"] is ki and v["write_qkn"] == 0
    det = det_backend._l2n_values(
        q, k, qn, kn, rstd, read_rstd=0, write_rstd=1, write_qkn=0, qn_int=qi, kn_int=ki
    )
    assert v.keys() == det.keys()
    assert all(v[key] is det[key] or v[key] == det[key] for key in v)


def test_internal_rows_scatter_to_token_rows() -> None:
    H = 1
    lengths = (3, 64, 70)
    bos, ln = [], []
    t = 0
    for n in lengths:
        for c0 in range(0, n, 64):
            bos.append(t + c0)
            ln.append(min(64, n - c0))
        t += n
    bos += [0]  # a pad chunk
    ln += [0]
    S, nc = t, len(bos)
    tok = torch.arange(S, dtype=torch.float32)
    rows = torch.full((nc * 64, H, D), -1.0)
    for c in range(nc):
        rows[64 * c : 64 * c + ln[c], :, :] = tok[bos[c] : bos[c] + ln[c], None, None]
    out = torch.full((S, H, D), -2.0)
    cake_backend._internal_to_tokens(
        rows,
        torch.tensor(bos, dtype=torch.int32),
        torch.tensor(ln, dtype=torch.int32),
        out,
    )
    assert torch.equal(out[:, 0, 0], tok)


def test_record_hook_clones_and_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cake_backend._l2n_record is None
    x = torch.arange(4.0)
    cake_backend._record_l2n("fwd", qn=x)  # no hook: nothing recorded, no error
    store: dict = {}
    monkeypatch.setattr(cake_backend, "_l2n_record", store)
    cake_backend._record_l2n("bwd", qn=x, path="light")
    assert store["bwd.path"] == "light" and torch.equal(store["bwd.qn"], x)
    assert store["bwd.qn"].data_ptr() != x.data_ptr()


@pytest.mark.parametrize("gate", GATES)
def test_autograd_node_returns_one_gradient_slot_per_input_and_takes_no_handoff(
    monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    """The node's backward of an l2n set: one gradient slot per forward input (the raw mode's parameter gradients in
    theirs), the backward called with the flag and the set's ``qk_rstd``, and no checkpoint hand-off record taken."""
    n_inputs = (
        len(inspect.signature(cake_backend.CakeKDANondetFunction.forward).parameters)
        - 1
    )
    raw = gate == "raw"
    seen: dict = {}

    def fake_bwd(*args, **kwargs):
        seen.update(kwargs)
        return tuple(torch.full((1,), float(i)) for i in range(7 if raw else 5))

    def no_handoff(*args, **kwargs):
        raise AssertionError("an l2n backward took a checkpoint hand-off record")

    monkeypatch.setattr(cake_backend, "_bwd_call", fake_bwd)
    monkeypatch.setattr(cake_backend, "_handoff_take", no_handoff)
    t = torch.zeros(1)
    rstd = torch.zeros(1)
    params = (t, t) if raw else ()  # A_log, dt_bias
    ctx = types.SimpleNamespace(
        saved_tensors=(
            t,
            t,
            t,
            t,
            t,
            t,
            *params,
            t,
            t,
            rstd,
        ),  # q k v g beta cu [A dt] Aqk Akk qk_rstd
        gate=gate,
        lower_bound=LOWER_BOUND if raw else None,
        qk_l2norm=True,
        scale=SCALE,
        workspace_ref=None,
        arena=None,
        doc_lengths=None,
        version=cake_backend.SAVED_VERSION,
        n_tables=0,
        extra_fields=("qk_rstd",),
        problem=(1, 1, 1),
    )
    grads = cake_backend.CakeKDANondetFunction.backward(ctx, torch.zeros(1))
    assert len(grads) == n_inputs
    assert seen["use_qk_l2norm_in_kernel"] is True and seen["handoff"] is None
    assert seen["saved"].qk_l2norm is True and seen["saved"].qk_rstd is rstd
    # q k v g beta: the gradients in the backward's order dq dk dv dbeta dg mapped to the inputs
    assert [float(x) for x in grads[:5]] == [0.0, 1.0, 2.0, 4.0, 3.0]
    if raw:
        assert seen["use_gate_in_kernel"] is True and seen["lower_bound"] == LOWER_BOUND
        params_at = list(
            inspect.signature(cake_backend.CakeKDANondetFunction.forward).parameters
        )[1:]
        a, b = params_at.index("A_log"), params_at.index("dt_bias")
        assert (float(grads[a]), float(grads[b])) == (5.0, 6.0)
        assert sum(x is not None for x in grads) == 7
    else:
        assert sum(x is not None for x in grads) == 5


# ---------------------------------------------------------------------------------------------------------------
# device rows (skip without an exported device, without the build or without the l2n records of its architecture)
#
# Accuracy: every result is judged against the exact FP64 reference beside the naive BF16 tier over the judge's
# draws (test_helpers/cake_kda_naive.py :func:`judge_arms`, tests/cake_naive.py), each path under its own arm name.
# The package has no bit contract: no result is compared with another result of this package or with the
# deterministic package's.  The deterministic package runs here for mode contracts only (its launch lists, its
# refusals of this package's sets, the path its backward takes), never for a result.

SHAPES_L2N = (
    ((64,), 1),
    ((1000, 37, 300), 4),
    ((2048,), 16),
    ((4000, 96), 32),
    ((5000, 3000), 64),
)
# A judged draw runs the FP64 reference and the naive tier, token by token: rows of up to this many tokens are judged
# at any head count; a larger row keeps its exact checks only (its results finite), and the test names the row that
# judges its route.
JUDGE_MAX_TOKENS = 4096
# One production-size row per route may be judged up to this many tokens (at up to four heads).
PRODUCTION_MAX_TOKENS = 16384
# The seed of every judged case and of the inputs of the exact-only rows (each draw's seed derives from it).
SEED = 7817


def _shape_id(shape) -> str:
    lengths, heads = shape
    return f"{len(lengths)}docs-S{sum(lengths)}-H{heads}"


def _l2n_device(mode: str = "fused") -> torch.device:
    """The current device when it is an exported architecture of both packages, this package's build exists and
    both registries ship the l2n mode ``mode`` of the three non-optional l2n stages there; else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if not (PACKAGE_DIR / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    index = torch.cuda.current_device()
    arch = cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
    if arch is None:
        pytest.skip("not an exported compute capability")
    for jit in (cake_jit, det_jit):
        for stage in ("fwd_intra", "bwd_prep", "glm_intra"):
            if not any(
                mode in jit.MODULES[n].get("l2n", ["off"])
                and arch in jit.MODULES[n]["arches"]
                for n in jit.records_of_stage(stage)
            ):
                pytest.skip(f"no l2n={mode!r} record of {stage} for {arch}")
    return torch.device("cuda", index)


def _inputs(lengths, H: int, device, gate: str = "g", seed: int = SEED) -> dict:
    """The operands of draw seed ``seed`` with raw (not normalized) BF16 q / k rows of a few magnitudes per token
    (the caller's projections): :func:`draw_case`'s l2n form."""
    return draw_case(lengths, H, gate, True, seed, device)


def _gate_kw(inputs: dict, gate: str) -> dict:
    if gate != "raw":
        return {}
    return {
        "use_gate_in_kernel": True,
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lower_bound": LOWER_BOUND,
    }


def _names(gate: str) -> tuple[str, ...]:
    return RAW_RETURN_NAMES if gate == "raw" else RETURN_NAMES


def _fwd(inputs, gate="g", *, det=False, q=None, k=None, **kwargs):
    """The l2n forward of this package (``det``: of the deterministic package, for a mode contract only); ``q`` /
    ``k``: other views of the same raw rows."""
    fn = (
        det_backend.cake_kda_det_train_forward if det else cake_kda_nondet_train_forward
    )
    return fn(
        inputs["q"] if q is None else q,
        inputs["k"] if k is None else k,
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        scale=SCALE,
        use_qk_l2norm_in_kernel=True,
        **_gate_kw(inputs, gate),
        **kwargs,
    )


def _bwd(inputs, saved, gate="g", *, det=False, q=None, k=None, **kwargs):
    """The l2n backward of this package (``det``: of the deterministic package, for a mode contract only) from
    ``saved`` (``None``: the replay)."""
    fn = (
        det_backend.cake_kda_det_train_backward
        if det
        else cake_kda_nondet_train_backward
    )
    return fn(
        inputs["q"] if q is None else q,
        inputs["k"] if k is None else k,
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["dout"],
        saved=saved,
        scale=SCALE,
        use_qk_l2norm_in_kernel=True,
        **_gate_kw(inputs, gate),
        **kwargs,
    )


def _grads(got, gate: str) -> dict[str, torch.Tensor]:
    """The results of a backward (in the return order) by name, cloned: the judge reads them after the draw's later
    calls."""
    got = list(got)
    names = _names(gate)
    assert len(got) == len(names)
    return {name: t.clone() for name, t in zip(names, got, strict=True)}


def _step(out: torch.Tensor, grads: dict) -> dict[str, torch.Tensor]:
    """One arm's results: ``out`` (cloned) and the backward's results by name."""
    return {"out": out.clone(), **grads}


def _run(inputs, gate="g", **kwargs):
    """The forward with its saved set and the backward from it: ``out``, the results by name (cloned) and the set."""
    bwd_kwargs = {
        n: kwargs[n] for n in ("workspace", "arena", "doc_lengths") if n in kwargs
    }
    out, saved = _fwd(inputs, gate, **kwargs)
    grads = _grads(_bwd(inputs, saved, gate, **bwd_kwargs), gate)
    return out, grads, saved


def _leaf_names(gate: str) -> tuple[str, ...]:
    """The gradients of the autograd leaves ``q, k, v, g, beta`` (and ``A_log``, ``dt_bias``), in the leaves'
    order."""
    return RAW_GRAD_NAMES if gate == "raw" else GRAD_NAMES


def _check_finite(arms: dict) -> None:
    """Every result of ``arms`` is finite: the contract of a row too large to judge (its oracle is finite)."""
    checked = 0
    for arm, results in arms.items():
        for res in results if isinstance(results, list) else (results,):
            for name, t in res.items():
                assert bool(torch.isfinite(t).all()), (arm, name)
                checked += 1
    assert checked > 0


def _judge(record_property, draw, judged: bool) -> None:
    """Judge the case's arms over the judge's draws (:func:`judge_arms`, the l2n reference), or for a row too large
    to judge run its draw once (its exact checks) and check every result finite."""
    if judged:
        judge_arms(record_property, draw, seed=SEED, scale=SCALE, l2n=True)
        return
    arms, _ = draw(0, SEED)
    _check_finite(arms)


def _plan_of(lengths, H: int, device, gate: str, l2n: str = "fused"):
    """The plan of a call without a workspace on ``device`` (the default entry's): of the problem, the device's
    SM count and architecture, the gate mode and the l2n mode."""
    return cake_backend.plan(
        sum(lengths),
        len(lengths),
        H,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
        gate=gate,
        l2n=l2n,
    )


def _check_set(saved, p, lengths, gate: str, *, tape: bool = False) -> None:
    """A saved set of an l2n call of plan ``p`` (the call's plan) on documents ``lengths``: this package's type; the
    version of its save mode (``tape``: the tape set) and the problem's metadata (``S``, ``H``, ``num_docs``, the
    scale, the call's gate mode and lower bound, the l2n mode); every field present exactly where the save mode and
    the plan say, of its dtype and shape: ``Aqk`` / ``Akk`` BF16 ``[S, H, 64]``, the tables (int32, the plan's
    layout), ``qk_rstd`` FP32 ``[2, S, H]``; a light set ``qg`` where the plan saves it, ``w`` / ``u`` where its
    light backward reads them from the set (BF16 ``[R, H, 128]``), ``hseg`` / ``seg_state`` where it saves the
    segment states (FP32 ``[NSEG, H, 128, 128]`` / ``[2, NSEG, H, 128, 128]``) and no tape; a tape set the ``h`` tape
    over the plan's padded chunks (BF16 ``[NC_pad, H, 128, 128]``, the documents' written chunks among them),
    ``v_new`` (BF16 ``[R, H, 128]``), on the segmented route the transfers ``seg_m`` (FP32 ``[NSEG, H, 128, 128]``)
    and none of the light fields.  The values are judged through the backward that reads the set."""
    S, N, H = sum(lengths), len(lengths), p.H
    assert (p.S, p.N) == (S, N), (p.S, p.N, S, N)
    assert type(saved) is cake_backend.CakeKDANondetChunkSaved
    want = {
        "version": (
            cake_backend.SAVED_VERSION_TAPE if tape else cake_backend.SAVED_VERSION
        ),
        "S": S,
        "H": H,
        "num_docs": N,
        "scale": SCALE,
        "gate": gate,
        "lower_bound": LOWER_BOUND if gate == "raw" else None,
        "qk_l2norm": True,
    }
    got = {name: getattr(saved, name) for name in want}
    assert got == want, (got, want)
    bf16, fp32 = torch.bfloat16, torch.float32
    rows, state = (p.R, H, D), (p.NSEG, H, D, D)
    light = not tape
    seg_save = light and p.seg_save
    layout = {
        "Aqk": (bf16, (S, H, cake_backend.BT)),
        "Akk": (bf16, (S, H, cake_backend.BT)),
        "qk_rstd": (fp32, (2, S, H)),
        "qg": (bf16, rows) if light and p.light_saves_qg else None,
        "w": (bf16, rows) if light and p.light_saves_wu else None,
        "u": (bf16, rows) if light and p.light_saves_wu else None,
        "hseg": (fp32, state) if seg_save else None,
        "seg_state": (fp32, (2, *state)) if seg_save else None,
        "h": (bf16, (p.NC_pad, H, D, D)) if tape else None,
        "v_new": (bf16, rows) if tape else None,
        "seg_m": (fp32, state) if tape and p.recur == "seg" else None,
    }
    for name, spec in layout.items():
        t = getattr(saved, name)
        if spec is None:
            assert t is None, (name, "present", tape, p.recur, p.light_path)
        else:
            assert isinstance(t, torch.Tensor), (name, "absent", tape, p.light_path)
            assert (t.dtype, tuple(t.shape)) == spec, (name, t.dtype, t.shape, spec)
    words = [words for _name, _offset, words in p.table_layout()]
    assert isinstance(saved.tables, tuple), type(saved.tables)
    assert [(t.dtype, tuple(t.shape)) for t in saved.tables] == [
        (torch.int32, (w,)) for w in words
    ], ([(t.dtype, tuple(t.shape)) for t in saved.tables], words)
    if tape:
        # the documents' chunks, one per 64 tokens, are among the tape's padded chunks
        written = sum(-(-n // cake_backend.BT) for n in lengths)
        assert 0 < written <= saved.h.shape[0], (written, saved.h.shape)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", SHAPES_L2N, ids=_shape_id)
def test_l2n_is_judged_against_the_exact_reference(
    shape, gate: str, record_property
) -> None:
    """``out`` and every result of the saved-set path, judged against the exact FP64 recurrence whose q / k are
    normalized in FP64 inside the graph (``dq`` / ``dk`` the raw rows' gradients) beside the naive BF16 tier; on every
    draw the saved set is this package's light set of the call's plan (:func:`_check_set`: version, metadata, gate
    and l2n mode, fields, dtypes, shapes and tables).  The row of 8000 tokens at 64 heads keeps these exact checks
    only (its results finite): the serial route it takes is judged on the row of 4096 tokens at 32 heads."""
    device = _l2n_device()
    lengths, H = shape
    S = sum(lengths)
    p = _plan_of(lengths, H, device, gate)

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        out, grads, saved = _run(inputs, gate)
        torch.cuda.synchronize(device)
        _check_set(saved, p, lengths, gate)
        return {"": _step(out, grads)}, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", SHAPES_L2N[1:4], ids=_shape_id)
def test_l2n_paths_are_judged(
    shape, gate: str, record_property, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every path on one call's inputs, each judged under its own name: the saved-set path (``""``), two repeats
    (``repeat``: the worse kept), the replay (``saved=None``), a set without its tables and the set itself on a caller
    workspace of the mode, the forward and backward on that workspace and its replay, the tape set from the default
    entry, the forward and backward on a tape workspace and its replay, a light set on the tape workspace, and the tape
    backward with the plan's WY record and with the tape WY twin forced (``_TAPE_WY``: launched wherever the tape plan
    has one).  The workspaces' modes and bytes and the saved sets (the light set, the tape set from the default entry
    and from the tape workspace: :func:`_check_set`) are checked on every draw; on the first, each package's backward
    refuses the other package's light and tape sets (``ValueError`` naming the mode)."""
    device = _l2n_device()
    lengths, H = shape
    S, N = sum(lengths), len(lengths)
    p = _plan_of(lengths, H, device, gate)

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        out, grads, saved = _run(inputs, gate)
        _check_set(saved, p, lengths, gate)
        arms = {"": _step(out, grads), "repeat": []}
        for _ in range(2):
            o, gr, _ = _run(inputs, gate)
            arms["repeat"].append(_step(o, gr))
        arms["replay"] = _grads(_bwd(inputs, None, gate), gate)
        bare = dataclasses.replace(saved, tables=None)
        for label, other in (
            ("bare_set_on_workspace", bare),
            ("set_on_workspace", saved),
        ):
            ws = cake_backend.CakeKDANondetWorkspace.for_problem(
                S, N, H, device, gate=gate, l2n="fused"
            )
            assert ws.l2n == ws.plan.l2n == "fused" and ws.gate == gate
            arms[label] = _grads(_bwd(inputs, other, gate, workspace=ws), gate)
        ws = cake_backend.CakeKDANondetWorkspace.for_problem(
            S, N, H, device, gate=gate, l2n="fused"
        )
        assert ws.bytes == cake_backend.workspace_bytes(
            S, N, H, sm_count=ws.plan.sm_count, gate=gate, l2n="fused"
        )
        out_w, grads_w, _ = _run(inputs, gate, workspace=ws)
        arms["workspace"] = _step(out_w, grads_w)
        arms["workspace_replay"] = _grads(_bwd(inputs, None, gate, workspace=ws), gate)
        out_t, grads_t, saved_t = _run(inputs, gate, save_mode="tape")
        _check_set(saved_t, p, lengths, gate, tape=True)
        arms["tape"] = _step(out_t, grads_t)
        tape_ws = cake_backend.CakeKDANondetWorkspace.for_problem(
            S, N, H, device, save_mode="tape", gate=gate, l2n="fused"
        )
        out_tw, grads_tw, saved_tw = _run(inputs, gate, workspace=tape_ws)
        _check_set(saved_tw, tape_ws.plan, lengths, gate, tape=True)
        arms["tape_workspace"] = _step(out_tw, grads_tw)
        arms["tape_workspace_replay"] = _grads(
            _bwd(inputs, None, gate, workspace=tape_ws), gate
        )
        arms["light_set_on_tape_workspace"] = _grads(
            _bwd(inputs, saved, gate, workspace=tape_ws), gate
        )
        for label, forced in (("tape_wy_record", False), ("tape_wy_twin", True)):
            monkeypatch.setattr(cake_backend, "_TAPE_WY", forced)
            arms[label] = _grads(_bwd(inputs, saved_tw, gate, workspace=tape_ws), gate)
        monkeypatch.setattr(cake_backend, "_TAPE_WY", None)
        if j == 0:
            # each package's backward refuses the other's l2n sets (light and tape)
            # mode contract: the deterministic package's sets, for the refusals
            _, det_saved = _fwd(inputs, gate, det=True)
            _, det_saved_t = _fwd(inputs, gate, det=True, save_mode="tape")
            for nd_set, det_set in ((saved, det_saved), (saved_t, det_saved_t)):
                with pytest.raises(ValueError, match=ND_REFUSES):
                    _bwd(inputs, det_set, gate)
                with pytest.raises(ValueError, match=DET_REFUSES):
                    _bwd(inputs, nd_set, gate, det=True)  # mode contract
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize(
    ("num_docs", "heads"), [(1, 1), (2, 2)], ids=["n1-h1", "n2-h2"]
)
def test_l2n_saved_states_are_judged(
    num_docs, heads, gate: str, record_property
) -> None:
    """The segmented rows that save the segment-boundary states (``light_states``) in an l2n mode: ``out`` and the
    gradients from that set (``""``), the replay's (``replay``) and the tape set's (``tape``), each judged; every
    draw's light set carries the states and both sets are checked (:func:`_check_set`).  The one-document row (about
    14k tokens at one head) is the path's judged production-size row; the two-document row (about 28k tokens at two
    heads) keeps these exact checks only (its results finite)."""
    device = _l2n_device()
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    nseg = cake_backend.plan(1, num_docs, heads, sm, arch).NSEG
    S = cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * max(nseg, 1)
    base, extra = divmod(S, num_docs)
    lengths = tuple(base + (1 if i < extra else 0) for i in range(num_docs))
    p = cake_backend.plan(S, num_docs, heads, sm, arch, gate=gate, l2n="fused")
    assert p.recur == "seg" and p.seg_save and p.light_path == "light_states", arch
    judged = num_docs == 1
    assert not judged or S <= PRODUCTION_MAX_TOKENS, S

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, gate, s)
        out, grads, saved = _run(inputs, gate)
        assert saved.hseg is not None
        _check_set(saved, p, lengths, gate)
        arms = {"": _step(out, grads)}
        arms["replay"] = _grads(_bwd(inputs, None, gate), gate)
        out_t, grads_t, saved_t = _run(inputs, gate, save_mode="tape")
        _check_set(saved_t, p, lengths, gate, tape=True)
        arms["tape"] = _step(out_t, grads_t)
        return arms, inputs

    _judge(record_property, draw, judged)


def _recorder(log: list, jit):
    """A workspace launcher that records ``(record, grid, num_seqs, num_chunks_pad, num_tokens, read_rstd,
    write_rstd, write_qkn)`` and runs the record through ``jit``."""

    def recording(name, values, *, grid, device_index):
        log.append(
            (
                name,
                tuple(grid),
                *(
                    int(values[key]) if key in values else None
                    for key in (
                        "num_seqs",
                        "num_chunks_pad",
                        "num_tokens",
                        "read_rstd",
                        "write_rstd",
                        "write_qkn",
                    )
                ),
            )
        )
        jit.run_module(name, values, grid=grid, device_index=device_index)

    return recording


L2N_LAUNCH_ROWS = (
    ((63, 1, 65), 2),
    ((129,), 16),
    ((63, 1, 65), 64),
    ((63, 1, 65, 30), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
    ((300, 200), 4),
)


@pytest.mark.parametrize("mode", L2N_ON)
@pytest.mark.parametrize("gate", GATES)
def test_l2n_launch_sequences_are_the_deterministic_l2n(
    gate: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mode contract (records, not results): the l2n launch lists on every path are the deterministic package's
    (the fused gradient's raw and l2n classes by their deterministic names) with the same grids and l2n scalars
    (``num_tokens``, ``read_rstd``, ``write_rstd``, ``write_qkn``); each list has the plan's launch count, launches the
    l2n records of the mode (the first q / k reader and the GLM intra program of the class) and no S1 record, hand-off
    twin or gate epilogue; the tape backward with the tape WY twin's rule and with the twin forced launches the tape
    list (the twin in place of its WY record wherever the tape plan has one, on some row at least).  No result is
    compared: the results of these paths are judged in :func:`test_l2n_paths_are_judged`."""
    device = _l2n_device(mode)
    launches = twins = 0
    for lengths, heads in L2N_LAUNCH_ROWS:
        S, N = sum(lengths), len(lengths)
        inputs = _inputs(lengths, heads, device, gate)
        nd_log: list = []
        det_log: list = []
        workspaces = {}
        for save_mode in ("light", "tape"):
            workspaces[save_mode] = (
                cake_backend.CakeKDANondetWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    save_mode=save_mode,
                    gate=gate,
                    l2n=mode,
                    launcher=_recorder(nd_log, cake_jit),
                ),
                det_backend.CakeKDADetWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    save_mode=save_mode,
                    gate=gate,
                    l2n=mode,
                    launcher=_recorder(det_log, det_jit),
                ),
            )
        ws, det_ws = workspaces["light"]
        p = ws.plan
        assert (p.gate, p.l2n, ws.l2n) == (gate, mode, mode)
        nd_only = set(cake_jit.S1_RECORDS) | set(cake_jit.HANDOFF_TWINS.values())

        def taken(path: str, **kw) -> list:
            nonlocal launches
            nd = [(_det_name(n), *rest) for n, *rest in nd_log]
            names = [n for n, *_ in nd_log]
            det = list(det_log)
            nd_log.clear()
            det_log.clear()
            assert nd == det, (lengths, heads, path)
            assert names and not set(names) & nd_only, (lengths, heads, path)
            assert "gate_epilogue" not in names
            assert all(_serves(n, mode) for n in names), (path, names)
            launches += len(names)
            return names

        def first_reader(path: str) -> str:
            if path in ("fwd", "fwd_nosave"):
                return (
                    p.record("qk_l2norm", path)
                    if mode == "fallback"
                    else p.record("fwd_intra", path)
                )
            return p.record("glm_intra", path)

        monkeypatch.setattr(cake_backend, "_TAPE_WY", False)  # the plan's WY record
        out, saved = _fwd(inputs, gate, workspace=ws)
        _, det_saved = _fwd(inputs, gate, det=True, workspace=det_ws)  # mode contract
        seq = taken("fwd")
        assert len(seq) == p.launch_count("fwd") and first_reader("fwd") in seq
        if mode == "fused":
            assert not _serves(p.record("fwd_intra", "fwd"), "off"), p.record(
                "fwd_intra", "fwd"
            )
        _fwd(inputs, gate, workspace=ws, save_for_backward=False)
        # mode contract
        _fwd(inputs, gate, det=True, workspace=det_ws, save_for_backward=False)
        seq = taken("fwd_nosave")
        assert len(seq) == p.launch_count("fwd_nosave")
        _bwd(inputs, det_saved, gate, det=True, workspace=det_ws)  # mode contract
        _bwd(inputs, saved, gate, workspace=ws)
        seq = taken(p.light_path)
        assert len(seq) == p.launch_count(p.light_path)
        glm = p.record("glm_intra", p.light_path)
        assert glm in seq and not _serves(glm, "off"), glm
        assert p.record("dqkg", p.light_path) in seq
        _bwd(inputs, None, gate, workspace=ws)
        _bwd(inputs, None, gate, det=True, workspace=det_ws)  # mode contract
        seq = taken("replay")
        assert len(seq) == p.launch_count("replay")
        tape_ws, det_tape_ws = workspaces["tape"]
        _, saved_t = _fwd(inputs, gate, workspace=tape_ws)
        # mode contract
        _, det_saved_t = _fwd(inputs, gate, det=True, workspace=det_tape_ws)
        taken("fwd tape")
        _bwd(inputs, saved_t, gate, workspace=tape_ws)
        # mode contract
        _bwd(inputs, det_saved_t, gate, det=True, workspace=det_tape_ws)
        seq = taken("tape")
        assert len(seq) == p.launch_count("tape")
        monkeypatch.setattr(cake_backend, "_TAPE_WY", None)  # the twin's rule
        _bwd(inputs, saved_t, gate, workspace=tape_ws)
        names = [n for n, *_ in nd_log]
        nd_log.clear()
        assert [_det_name(n) for n in names] == [_det_name(n) for n in seq], (
            lengths,
            heads,
            "tape wy rule",
        )
        # the twin forced (the device, build and load checks skipped): launched in place of its WY record
        twin = tape_ws.plan.tape_wy_twin
        monkeypatch.setattr(cake_backend, "_TAPE_WY", True)
        _bwd(inputs, saved_t, gate, workspace=tape_ws)
        names = [n for n, *_ in nd_log]
        nd_log.clear()
        assert [_det_name(n) for n in names] == [_det_name(n) for n in seq], (
            lengths,
            heads,
            "tape wy twin",
        )
        assert (twin in names) == (twin is not None), (lengths, heads, twin, names)
        twins += twin is not None
        monkeypatch.setattr(cake_backend, "_TAPE_WY", None)
    assert launches > 0 and twins > 0


@pytest.mark.parametrize("mode", L2N_ON)
@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", L2N_LAUNCH_ROWS, ids=_shape_id)
def test_l2n_handoff_poison_hook_is_flag_off_only(
    shape, gate: str, mode: str, record_property, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hand-off's test hook (``_HANDOFF_POISON``: ``dv2`` NaN-filled before the prep) fills nothing in an l2n
    mode, whose ``kn`` rows may sit on ``dv2``'s bytes and where the hand-off never engages: the backward from a light
    set, from a tape set and the replay, each with the hook off and with it on (``<save mode>_<source>`` and
    ``..._poisoned``), are judged; a NaN the hook wrote under ``kn`` would reach the results."""
    device = _l2n_device(mode)
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)

    def draw(j, s):
        inputs = _inputs(lengths, heads, device, gate, s)
        arms = {}
        for save_mode in ("light", "tape"):
            ws = cake_backend.CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                save_mode=save_mode,
                gate=gate,
                l2n=mode,
            )
            _, saved = _fwd(inputs, gate, workspace=ws)
            sources = [("set", saved)]
            if save_mode == "light":
                sources.append(("replay", None))
            for label, source in sources:
                for poisoned in (False, True):
                    monkeypatch.setattr(cake_backend, "_HANDOFF_POISON", poisoned)
                    arm = f"{save_mode}_{label}" + ("_poisoned" if poisoned else "")
                    arms[arm] = _grads(_bwd(inputs, source, gate, workspace=ws), gate)
            monkeypatch.setattr(cake_backend, "_HANDOFF_POISON", False)
            del ws
        assert len(arms) == 6
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


L2N_ARENA_ROWS = [
    ((63, 1, 65), 4),
    ((100, 640, 77, 1000, 129), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
    ((129, 300), 16),
]


def _partitions(lengths):
    """``(label, CakeKDANondetArena keywords)`` of the document groupings: two groups and three (where the greedy
    partition gives exactly that many) and one document per group."""
    lengths = tuple(int(n) for n in lengths)
    S, longest = sum(lengths), max(lengths)
    out = []
    for k in (2, 3):
        lo, hi = longest, S
        while lo < hi:
            mid = (lo + hi) // 2
            if len(cake_backend.doc_batches(lengths, mid)) <= k:
                hi = mid
            else:
                lo = mid + 1
        if len(cake_backend.doc_batches(lengths, lo)) == k:
            out.append((f"g{k}", {"batch_tokens": lo}))
    out.append(("gdoc", {"batch_tokens": S, "max_docs_per_batch": 1}))
    return out


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", L2N_ARENA_ROWS, ids=_shape_id)
def test_l2n_lean_and_batched_arenas_are_judged(
    shape, gate: str, record_property
) -> None:
    """Lean workspaces of one to three tail passes and document groups (default and lean layouts), each arm judged:
    per tail pass (``lean<t>``) the forward and the backward from its own set, the backward from this package's
    default set on the workspace and per call with the arena option; per grouping (``<grouping>`` /
    ``<grouping>_lean``) the forward and the backward from its own grouped set on the workspace, the backward from
    this package's default set on the workspace, the default backward from the grouped set and (not lean) the replay
    group by group.  The workspaces' bytes and group counts are checked on every draw; on the first, a lean
    workspace refuses the replay, this package's backward refuses the deterministic package's default set and the
    deterministic package's backward refuses a grouped set of this package (``ValueError`` naming the mode)."""
    device = _l2n_device()
    lengths, H = shape
    S, N = sum(lengths), len(lengths)
    parts = _partitions(lengths)
    assert parts

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        _, saved = _fwd(inputs, gate)
        # mode contract: the deterministic package's set, for this package's refusals
        det_saved = _fwd(inputs, gate, det=True)[1] if j == 0 else None
        arms = {}
        for t in (1, 2, 3):
            arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
            ws = cake_backend.CakeKDANondetWorkspace.for_problem(
                S, N, H, device, arena=arena, gate=gate, l2n="fused"
            )
            assert ws.bytes == cake_backend.workspace_bytes(
                S, N, H, sm_count=ws.plan.sm_count, arena=arena, gate=gate, l2n="fused"
            )
            o, saved_l = _fwd(inputs, gate, workspace=ws)
            arms[f"lean{t}"] = _step(
                o, _grads(_bwd(inputs, saved_l, gate, workspace=ws), gate)
            )
            arms[f"lean{t}_default_set"] = _grads(
                _bwd(inputs, saved, gate, workspace=ws), gate
            )
            arms[f"lean{t}_default_set_per_call"] = _grads(
                _bwd(inputs, saved, gate, arena=arena), gate
            )
            if j == 0:
                with pytest.raises(ValueError, match=ND_REFUSES):
                    _bwd(inputs, det_saved, gate, workspace=ws)
                with pytest.raises(ValueError, match=ND_REFUSES):
                    _bwd(inputs, det_saved, gate, arena=arena)
                with pytest.raises(ValueError, match="lean=False"):
                    _bwd(inputs, None, gate, workspace=ws)
            del ws
        for label, kw in parts:
            for lean in (False, True):
                arena = cake_backend.CakeKDANondetArena(
                    lean=lean, tail_passes=3 if lean else 1, **kw
                )
                tag = f"{label}_lean" if lean else label
                ws = cake_backend.CakeKDANondetWorkspace.for_problem(
                    S,
                    N,
                    H,
                    device,
                    arena=arena,
                    doc_lengths=lengths,
                    gate=gate,
                    l2n="fused",
                )
                assert len(ws.plan.batches(arena, lengths)) >= 2, tag
                o, saved_b = _fwd(inputs, gate, workspace=ws, doc_lengths=lengths)
                arms[tag] = _step(
                    o,
                    _grads(
                        _bwd(inputs, saved_b, gate, workspace=ws, doc_lengths=lengths),
                        gate,
                    ),
                )
                arms[f"{tag}_default_set"] = _grads(
                    _bwd(inputs, saved, gate, workspace=ws, doc_lengths=lengths), gate
                )
                arms[f"{tag}_default_backward"] = _grads(
                    _bwd(inputs, saved_b, gate), gate
                )
                if not lean:
                    arms[f"{tag}_replay"] = _grads(
                        _bwd(inputs, None, gate, workspace=ws, doc_lengths=lengths),
                        gate,
                    )
                if j == 0:
                    with pytest.raises(ValueError, match=ND_REFUSES):
                        _bwd(inputs, det_saved, gate, workspace=ws, doc_lengths=lengths)
                    with pytest.raises(ValueError, match=DET_REFUSES):
                        _bwd(inputs, saved_b, gate, det=True)  # mode contract
                del ws
        # per tail pass three arms; per grouping four on the default layout (the replay among them), three on the
        # lean one
        assert len(arms) == 3 * 3 + len(parts) * (4 + 3)
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize(
    "lengths, batch_tokens",
    [((100, 126, 50), 226), ((100, 128, 50), 228)],
    ids=["overhang", "aligned"],
)
def test_l2n_batched_replay_reads_no_unwritten_row_of_the_next_group(
    lengths,
    batch_tokens: int,
    H: int,
    gate: str,
    record_property,
) -> None:
    """The batched arena's replay with use_qk_l2norm_in_kernel=True on the serial route (H 32 on every
    architecture, H 16 on sm_100a): the plain intra program of group 0 box-reads up to BT - 1 rows past the group
    end of the per-call qn / kn (``"overhang"``: document 1 ends 62 rows into its last chunk), which group 1's prep
    writes only later.  With every per-call float tensor and the arena filled with NaN, the replay before
    (``nan_replay_first``) and after (``nan_replay_after``) a forward and a saved-set backward (``nan_saved``) on the
    workspace give finite gradients, each judged, as is the per-call whole problem's replay (``replay``)
    (``"aligned"``: the group ends on a chunk boundary, the control)."""
    device = _l2n_device()
    S, N = sum(lengths), len(lengths)
    assert cake_backend.doc_batches(lengths, batch_tokens) == ((0, 2), (2, 3))
    names = _names(gate)
    alloc = cake_backend._alloc

    def nan_alloc(shape, dtype, dev):
        t = alloc(shape, dtype, dev)
        if t.is_floating_point() and t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    arena = cake_backend.CakeKDANondetArena(batch_tokens=batch_tokens)

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        arms = {"replay": _grads(_bwd(inputs, None, gate), gate)}
        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_backend, "_alloc", nan_alloc)
            for case in ("replay_first", "replay_after"):
                ws = cake_backend.CakeKDANondetWorkspace.for_problem(
                    S,
                    N,
                    H,
                    device,
                    arena=arena,
                    doc_lengths=lengths,
                    gate=gate,
                    l2n="fused",
                )
                ws.arena.fill_(0xFF)
                kw = {"workspace": ws, "doc_lengths": lengths}
                runs = []
                if case == "replay_after":
                    _, saved = _fwd(inputs, gate, **kw)
                    runs.append(("nan_saved", _bwd(inputs, saved, gate, **kw)))
                runs.append((f"nan_{case}", _bwd(inputs, None, gate, **kw)))
                for tag, got in runs:
                    for name, a in zip(names, got, strict=True):
                        assert bool(torch.isfinite(a).all()), (H, gate, tag, name)
                    arms[tag] = _grads(got, gate)
                del ws
        # the per-call whole problem's replay, the fresh replay, then a saved-set backward and the replay after it
        assert len(arms) == 4
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


PACKED_ROWS = (((1000, 37, 300), 4), ((2048,), 16))


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", PACKED_ROWS, ids=_shape_id)
def test_l2n_packed_views_are_judged(shape, gate: str, record_property) -> None:
    """q / k as column slices of one packed projection output (NaN outside the views), each arm judged: the forward
    and the backward from its set (``views``) and the replay (``views_replay``) on the views; the set saved from the
    views serving a call on contiguous rows (``contiguous_call``) and a set saved from contiguous rows serving a call
    on the views (``contiguous_set_views_call``).  The set saved from the views is checked on every draw
    (:func:`_check_set`); on the first draw the call on the views refuses the deterministic package's set
    (``ValueError`` naming the mode)."""
    device = _l2n_device()
    lengths, H = shape
    S = sum(lengths)
    p = _plan_of(lengths, H, device, gate)

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        packed = torch.full(
            (S, 3 * H * D + 8), float("nan"), dtype=torch.bfloat16, device=device
        )
        packed[:, 8 : 8 + H * D] = inputs["q"].reshape(S, -1)
        packed[:, 8 + H * D : 8 + 2 * H * D] = inputs["k"].reshape(S, -1)
        qv = packed[:, 8 : 8 + H * D].view(S, H, D)
        kv = packed[:, 8 + H * D : 8 + 2 * H * D].view(S, H, D)
        assert not qv.is_contiguous() and not kv.is_contiguous()
        out_v, saved_v = _fwd(inputs, gate, q=qv, k=kv)
        _check_set(saved_v, p, lengths, gate)
        arms = {
            "views": _step(
                out_v, _grads(_bwd(inputs, saved_v, gate, q=qv, k=kv), gate)
            ),
            "views_replay": _grads(_bwd(inputs, None, gate, q=qv, k=kv), gate),
            "contiguous_call": _grads(_bwd(inputs, saved_v, gate), gate),
        }
        _, saved_c = _fwd(inputs, gate)
        arms["contiguous_set_views_call"] = _grads(
            _bwd(inputs, saved_c, gate, q=qv, k=kv), gate
        )
        if j == 0:
            _, det_saved = _fwd(inputs, gate, det=True)  # mode contract
            with pytest.raises(ValueError, match=ND_REFUSES):
                _bwd(inputs, det_saved, gate, q=qv, k=kv)
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


def _leaves(inputs, gate: str) -> list[torch.Tensor]:
    names = ("q", "k", "v", "g", "beta") + (
        ("A_log", "dt_bias") if gate == "raw" else ()
    )
    return [inputs[n].detach().clone().requires_grad_(True) for n in names]


def _autograd(leaves, cu, gate: str, **kwargs) -> torch.Tensor:
    q, k, v, g, beta, *params = leaves
    gate_kw = (
        {
            "use_gate_in_kernel": True,
            "A_log": params[0],
            "dt_bias": params[1],
            "lower_bound": LOWER_BOUND,
        }
        if gate == "raw"
        else {}
    )
    return cake_kda_nondet(
        q, k, v, g, beta, cu, use_qk_l2norm_in_kernel=True, **gate_kw, **kwargs
    )


def _leaf_grads(leaves, gate: str) -> dict[str, torch.Tensor]:
    return {
        name: leaf.grad for name, leaf in zip(_leaf_names(gate), leaves, strict=True)
    }


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize(
    "shape", [((63, 1, 65), 2), ((1000, 37, 300), 16)], ids=_shape_id
)
def test_l2n_autograd_and_checkpoint_are_judged_and_take_no_handoff(
    shape, gate: str, record_property, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The autograd entry with every ``checkpoint_handoff`` value under each value of the environment override
    (nine calls per draw): ``out`` and the leaves' gradients directly (``autograd``), under non-reentrant
    (``checkpoint``) and reentrant (``checkpoint_reentrant``) ``torch.utils.checkpoint``, each arm judged with its nine
    calls as repeats (the worst kept).  An l2n call never makes, publishes or takes a hand-off record, and the
    hand-off counters of the device stay zero."""
    from torch.utils.checkpoint import checkpoint

    device = _l2n_device()
    lengths, H = shape

    def no_handoff(*args, **kwargs):
        raise AssertionError(
            "an l2n call made, published or took a checkpoint hand-off record"
        )

    monkeypatch.setattr(cake_backend, "_handoff_request", no_handoff)
    monkeypatch.setattr(cake_backend, "_handoff_take", no_handoff)
    monkeypatch.setattr(cake_backend, "_handoff_publish", no_handoff)
    cake_backend._handoff_reset()

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        arms = {"autograd": [], "checkpoint": [], "checkpoint_reentrant": []}
        for env in (None, "on", "off"):
            if env is None:
                monkeypatch.delenv("CAKE_KDA_CKPT_HANDOFF", raising=False)
            else:
                monkeypatch.setenv("CAKE_KDA_CKPT_HANDOFF", env)
            for mode in ("auto", "on", "off"):
                leaves = _leaves(inputs, gate)
                o = _autograd(leaves, cu, gate, checkpoint_handoff=mode)
                o.backward(dout)
                arms["autograd"].append(
                    {"out": o.detach().clone(), **_leaf_grads(leaves, gate)}
                )
                for reentrant in (False, True):
                    leaves = _leaves(inputs, gate)
                    o = checkpoint(
                        lambda *a, _m=mode: _autograd(
                            list(a), cu, gate, checkpoint_handoff=_m
                        ),
                        *leaves,
                        use_reentrant=reentrant,
                    )
                    o.backward(dout)
                    label = "checkpoint_reentrant" if reentrant else "checkpoint"
                    arms[label].append(_leaf_grads(leaves, gate))
        monkeypatch.delenv("CAKE_KDA_CKPT_HANDOFF", raising=False)
        assert all(len(runs) == 3 * 3 for runs in arms.values())
        return arms, inputs

    _judge(record_property, draw, sum(lengths) <= JUDGE_MAX_TOKENS)
    stats = cake_backend._handoff_stats(device.index)
    assert all(stats[name] == 0 for name in cake_backend._HANDOFF_COUNTERS), stats
    assert stats["live_records"] == 0, stats


@pytest.mark.parametrize("gate", GATES)
def test_l2n_checkpointed_forward_retains_no_workspace(
    gate: str, record_property
) -> None:
    """Under activation checkpointing an l2n forward keeps nothing beyond its output (the implicit workspace, a
    per-call one, a caller's), checked on every draw; per workspace the warm-up forward's ``out`` (``<workspace>``)
    and the gradients through the checkpoint (``<workspace>_checkpoint``) are judged."""
    from torch.utils.checkpoint import checkpoint

    device = _l2n_device()
    lengths, H = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    plan = cake_backend.plan(
        S,
        N,
        H,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
        gate=gate,
        l2n="fused",
    )
    shared = cake_backend.CakeKDANondetWorkspace(plan, device, backward=True)
    assert shared.l2n == "fused" and shared.gate == gate

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]

        def implicit(*a):
            return _autograd(list(a), cu, gate)

        def per_call(*a):
            ws = cake_backend.CakeKDANondetWorkspace(plan, device, backward=True)
            return _autograd(list(a), cu, gate, workspace=ws)

        def on_shared(*a):
            return _autograd(list(a), cu, gate, workspace=shared)

        arms = {}
        leaves = _leaves(inputs, gate)
        for fn in (implicit, per_call, on_shared):
            warm = fn(*leaves)
            arms[fn.__name__] = {"out": warm.detach().clone()}
            del warm
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            o = checkpoint(fn, *leaves, use_reentrant=False)
            torch.cuda.synchronize()
            kept = torch.cuda.memory_allocated() - before - o.untyped_storage().nbytes()
            assert kept == 0, (fn.__name__, kept)
            for leaf in leaves:
                leaf.grad = None
            o.backward(dout)
            arms[f"{fn.__name__}_checkpoint"] = {
                name: t.clone() for name, t in _leaf_grads(leaves, gate).items()
            }
            del o
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", SHAPES_L2N[1:3], ids=_shape_id)
def test_l2n_fused_and_fallback_routes_are_judged(
    shape, gate: str, record_property
) -> None:
    """The fused route (the default entry) and the force-fallback knob (an explicit fallback workspace), each judged:
    the forward and the backward from its saved set (``fused`` / ``fallback``) and the replay (``fused_replay`` /
    ``fallback_replay``); every draw's sets are checked against their plans (:func:`_check_set`).  That the fallback
    route launches the deterministic package's fallback records is
    :func:`test_l2n_launch_sequences_are_the_deterministic_l2n`'s (a mode contract)."""
    device = _l2n_device()
    _l2n_device("fallback")
    lengths, H = shape
    S, N = sum(lengths), len(lengths)
    ws = cake_backend.CakeKDANondetWorkspace.for_problem(
        S, N, H, device, gate=gate, l2n="fallback"
    )
    assert ws.l2n == "fallback" and ws.plan.dispatched("qk_pullback", "light")
    p = _plan_of(lengths, H, device, gate)

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        out_a, saved_a = _fwd(inputs, gate)
        out_b, saved_b = _fwd(inputs, gate, workspace=ws)
        _check_set(saved_a, p, lengths, gate)
        _check_set(saved_b, ws.plan, lengths, gate)
        arms = {
            "fused": _step(out_a, _grads(_bwd(inputs, saved_a, gate), gate)),
            "fallback": _step(
                out_b, _grads(_bwd(inputs, saved_b, gate, workspace=ws), gate)
            ),
            "fused_replay": _grads(_bwd(inputs, None, gate), gate),
            "fallback_replay": _grads(_bwd(inputs, None, gate, workspace=ws), gate),
        }
        return arms, inputs

    _judge(record_property, draw, S <= JUDGE_MAX_TOKENS)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("shape", SHAPES_L2N[1:3], ids=_shape_id)
def test_l2n_hook_rows_are_the_reference_rows(
    shape, gate: str, record_property, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test hook's rows (``_l2n_record``) on the light, tape and replay paths: on the first draw the forward's
    rstd / qn / kn and the backward's qn / kn are bitwise the shared bitwise device-order reference's
    (``qk_l2norm``), and the backward takes the path the deterministic package's backward takes (a mode contract:
    the path, not a result); on every draw the tape and replay backwards report their own path, and ``out`` and every
    result with the hook on are judged (``light`` / ``tape`` / ``replay``)."""
    device = _l2n_device()
    lengths, H = shape

    def draw(j, s):
        inputs = _inputs(lengths, H, device, gate, s)
        if j == 0:
            qn_ref, rq_ref = ref_l2n.qk_l2norm(inputs["q"].cpu())
            kn_ref, rk_ref = ref_l2n.qk_l2norm(inputs["k"].cpu())
        arms = {}
        for route in ("light", "tape", "replay"):
            nd_rec: dict = {}
            kw = {"save_mode": "tape"} if route == "tape" else {}
            monkeypatch.setattr(cake_backend, "_l2n_record", nd_rec)
            out, saved = _fwd(inputs, gate, **kw)
            grads = _grads(
                _bwd(inputs, None if route == "replay" else saved, gate), gate
            )
            torch.cuda.synchronize(device)
            monkeypatch.setattr(cake_backend, "_l2n_record", None)
            arms[route] = _step(out, grads)
            if route != "light":
                assert nd_rec["bwd.path"] == route, (route, nd_rec["bwd.path"])
            if j == 0:
                det_rec: dict = {}
                monkeypatch.setattr(det_backend, "_l2n_record", det_rec)
                _, det_saved = _fwd(inputs, gate, det=True, **kw)  # mode contract
                # mode contract: the path the deterministic package's backward takes
                _bwd(inputs, None if route == "replay" else det_saved, gate, det=True)
                torch.cuda.synchronize(device)
                monkeypatch.setattr(det_backend, "_l2n_record", None)
                assert nd_rec["bwd.path"] == det_rec["bwd.path"], route
                rows = {
                    "fwd.qn": qn_ref,
                    "fwd.kn": kn_ref,
                    "fwd.qk_rstd": torch.stack((rq_ref, rk_ref)),
                    "bwd.qn": qn_ref,
                    "bwd.kn": kn_ref,
                }
                for key, want in rows.items():
                    assert _bits_equal(nd_rec[key].cpu(), want), (route, key)
        return arms, inputs

    _judge(record_property, draw, sum(lengths) <= JUDGE_MAX_TOKENS)


def test_l2n_refusals_on_device() -> None:
    """Saved sets and workspaces of the other mode are refused, and a set of the deterministic package by its type
    before the mode; a strided q the rule refuses is never copied."""
    device = _l2n_device()
    lengths, H = (300, 200), 2
    S, N = sum(lengths), len(lengths)
    inputs = _inputs(lengths, H, device)
    i = inputs
    _, saved = _fwd(inputs)
    _, det_saved = _fwd(inputs, det=True)  # mode contract
    for source, refused in (
        (saved, "use_qk_l2norm_in_kernel"),
        (det_saved, ND_REFUSES),
    ):
        with pytest.raises(ValueError, match=refused):
            cake_kda_nondet_train_backward(
                i["q"].contiguous(),
                i["k"].contiguous(),
                i["v"],
                i["g"],
                i["beta"],
                i["cu_seqlens"],
                i["dout"],
                saved=source,
            )
    _, off_saved = cake_kda_nondet_train_forward(
        i["q"], i["k"], i["v"], i["g"], i["beta"], i["cu_seqlens"]
    )
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        _bwd(inputs, off_saved)
    ws_off = cake_backend.CakeKDANondetWorkspace.for_problem(S, N, H, device)
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        _fwd(inputs, workspace=ws_off)
    ws_on = cake_backend.CakeKDANondetWorkspace.for_problem(
        S, N, H, device, l2n="fused"
    )
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        cake_kda_nondet_train_forward(
            i["q"], i["k"], i["v"], i["g"], i["beta"], i["cu_seqlens"], workspace=ws_on
        )
    bad_q = torch.zeros(S, H, 2 * D, dtype=torch.bfloat16, device=device)[:, :, ::2]
    with pytest.raises(ValueError, match="never copied"):
        _fwd(dict(inputs, q=bad_q))
