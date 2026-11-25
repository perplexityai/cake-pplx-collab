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

Host dispatch policy of the Cake DSA indexer top-k programs of the
non-deterministic package.

The generated scan kernel exists in several physical forms (unit geometry,
key-range split, CTA pair, tile-loop unroll, snake unit order, sampled first
threshold on buffer-fitting units), and the host picks one per call from the
call's geometry alone: ``T`` queries, ``Tkv`` keys, ``S`` segments, the
compression ``ratio``, ``top_k`` and the persistent grid (the device's SM
count).  A call whose plan splits the key ranges runs the plain split merge
(``MERGE_KEY``) after its scan; no call runs an ordering pass.  The rules are
those of the deterministic package's policy without its finalize and
rank-merge rules, so a call of either input form picks the scan program,
candidate capacity, check period, split count, CTA pair and ticket plan that
package picks for it.
Every threshold of that choice is a plain number in the registry
record of the architecture (``cake_jit.POLICY[arch]``); this module evaluates
the record.  It depends on nothing but the standard library so that the
generated-program export can run the very same code against the kernel's own
dispatch functions before a record is frozen (a disagreement fails the export,
never the user).  A record may give the wide program precedence over the
three-warpgroup rule from ``wide_first_rule[0]`` mean unit tiles on, or
replace both wide rules by ``wide_wave_rule``, which compares the waves of the
persistent grid the wide unit needs with those of the unit the other rules
leave (``unit_block_q_for``), the
CTA pair serves the unit geometries of ``pair_units`` only and the key-range
split those of ``split_units`` only (the units whose split program the
architecture registers).  ``l6_solo_rule`` may prefer the three-warpgroup unit
without a CTA pair on calls that do not split, and ``l6_solo_unsampled_max_queries``
the unpaired form of a CTA-pair l6 plan on short calls with the sampled first threshold off (``select_program``, which takes
the registered scan keys so that it never picks an unregistered program;
without them the preference stays off).  ``snake_packed_rule`` bounds the
snake unit order of packed calls by their mean unit tiles, sampling mode and
waves of the persistent grid,
and ``snake_l6_fallback_waves`` lets a sampling-off call of few waves that the
l6 preference moves run the plain-order l6 program where the snake-order one
is not registered.  A record may also
replace the sampled rank shift from a ``top_k`` of ``sample_shift_rule[0]`` on
(``sample_shift_for``); an explicit ``sample_shift_permille`` knob still
overrides it.  The candidate-capacity floor may depend on ``top_k`` the same
way: ``cand_cap_floor_rule`` = ``[min top_k, floor]`` replaces
``cand_cap_floor`` from that ``top_k`` on (``cand_cap_floor_for``), so that a
small-``top_k`` call keeps the smaller buffers (half the workspace at
``top_k = 512``) and a large-``top_k`` call the larger floor.

Nothing here reads device memory: the segment boundaries stay on the device
and the dispatch sees only host-known integers.  The chosen program never
changes the selected set -- every form selects the same ids per row; only the
order of a row's entries can differ -- so a wrong
choice costs time, not correctness.

The opt-in FP8 input form (float8_e4m3fn ``q`` / ``k``) has its own scan
programs, keyed ``scan:<kind>:u<unroll>:s<snake>:f1:fp8[:lsf]``, for the narrow,
l6, split and wide kinds, in the plain unit order (``s0``) but for the snake copy
(``s1``) of the sm_107a l6 program.  An FP8 call runs the same decision procedure under the
architecture's FP8 record (``fp8_policy_record``: the ``POLICY`` record with
the fields of the registry's ``POLICY_FP8`` record replaced) and then the
registered FP8 program of the decided kind, or of its ``FP8_KIND_FALLBACK``
kind (``select_fp8_program``).  Every FP8 program of one architecture selects
the same set per row.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Collection, NamedTuple, Optional

# Record fields in canonical order (the export writes them in this order).
POLICY_FIELDS = (
    "tile_keys",  # keys per key tile (capacity and sampling granule)
    "block_q_narrow",  # queries per unit of the narrow program (two math warpgroups, one query pair each)
    "block_q_wide",  # queries per unit of the wide program (M128 x N256 MMA, staged weights)
    "block_q_l6",  # queries per unit of the three-warpgroup program
    "candidate_entry_bytes",  # packed (score bits, key id) entry size
    "candidate_multiplier",  # default capacity multiplier: cand_cap ~ multiplier * top_k
    "candidate_slack",  # minimum headroom above top_k before a buffer can be compacted
    "cand_mult_rule",  # [min mean unit tiles, multiplier] for long-unit calls, or None
    "cand_cap_floor",  # minimum candidate slots per (warpgroup, query), or None
    "cand_cap_floor_rule",  # [min top_k, floor] replacing cand_cap_floor from that top_k on (sampled calls), or None
    "candidate_multiplier_unsampled",  # capacity multiplier when the sampled first threshold is off
    "cand_mult_rule_unsampled",  # its [min mean unit tiles, multiplier] long-unit rule, or None
    "cand_cap_floor_unsampled",  # its minimum candidate slots, or None
    "cand_cap_ceiling",  # most candidate slots the default multiplier requests per (warpgroup, query), or None; lowers the multiplier, the floor and top_k + slack still hold
    "cand_cap_ceiling_unsampled",  # its ceiling when the sampled first threshold is off, or None
    "l6_rule",  # [max mean unit tiles, min top_k] of the three-warpgroup program, or None
    "wide_rule",  # [min wide rounds per CTA, min rounds x mean unit tiles] of the wide program
    "wide_first_rule",  # [min mean unit tiles] from which a wide-eligible call takes the wide program before the three-warpgroup rule, or None (the three-warpgroup rule first)
    "wide_first_k_rule",  # [max top_k, min mean unit tiles] replacing wide_first_rule up to that top_k: a wide-eligible call of at most that top_k takes the wide program before the three-warpgroup rule from that many mean unit tiles on, or None (wide_first_rule on every call)
    "wide_wave_rule",  # [min narrow / wide wave ratio, min l6 / wide wave ratio, min mean unit tiles against l6(, the top_k up to which the tile minima hold, their exponent in top_k above it, the minimum of a packed call, the tiles a one-segment minimum rises per wide wave below the next element, the wide waves from which it holds, the l6 / wide wave ratio W below which the minimum is at least the next element / (W - 1) and on a packed call at least element 10 / (W - 1) ** 2, and from which the minimum of a one-document call of fewer wide waves than element 7 is at most element 11 / (W - 1) ** 2, on two or more wide waves at most the larger of that and element 12, from element 13 wide waves on of that and element 14)] replacing wide_rule and wide_first_rule, or None; see unit_block_q_for
    "desc_walk_kinds",  # [physical scan kinds whose descending-walk twin (key kind "<kind>_desc") a call with the sampled first threshold off runs, the same for a sampled call], or None; see select_program
    "split_max",  # most key-range splits per unit
    "split_min_range_tiles",  # a split's key range is at least this many tiles
    "split_wave_rule",  # [min mean unit tiles, min CTA-efficiency gain] for a two-way split, or None
    "split_wave_scope",  # [unit geometries, max segments, sampled calls only] the two-way split of split_wave_rule is limited to, or None (every call)
    "split_fit_rule",  # [merge chunk entries]: a call the split rule splits takes ceil(mean unit keys / top_k) splits, rounded up to whole merge chunks, where that is more and fits split_max and the grid; or None
    "split_resident_rule",  # [merge chunk entries, fit entries, merge-resident entries, max splits, min splits] replacing split_max and split_fit_rule on a call the split rule splits, or None; see split_rule_for
    "split_units",  # unit geometries ("l6", "narrow", "wide") whose split program the architecture registers; any other unit never splits
    "pair_rule",  # [min mean unit tiles, max mean unit tiles, min top_k] of the CTA-pair program, or None
    "pair_units",  # unit geometries ("l6", "narrow", "wide") whose CTA-pair program the architecture registers; the pair rule applies to these units only
    "l6_solo_rule",  # [max mean unit tiles, min segments] of the unpaired three-warpgroup preference on sampled calls, or None (no preference); see select_program
    "l6_solo_unsampled_max_queries",  # with the sampled first threshold off, a call of at most this many queries planned on the CTA-pair l6 unit runs it unpaired (other units keep their plan), or None (sampling off follows l6_solo_rule); see l6_solo_for
    "wide_pair_snake_only",  # a sampled call pairs the wide unit only when it walks the snake order (else the unpaired wide program)
    "snake_default",  # snake unit order when no rule applies
    "snake_rule",  # [program kinds, min mean tiles, min mean tiles (ratio > 1), min unit-cost spread], or None
    "snake_cap_rule",  # [max segments, max waves of the persistent grid] within which an FP8 call the decision marks snake runs the snake copy of its program, or None (no cap); read only by select_fp8_program
    "snake_packed_rule",  # [program kinds, min mean unit tiles sampled, min mean unit tiles with sampling off or None (never), max waves or None] a call of two or more segments needs for the snake order of snake_rule beyond that many waves of its persistent grid, or None (snake_rule alone); see snake_for
    "snake_l6_fallback_waves",  # max waves of the planned program's persistent grid within which a sampling-off call the l6 preference moves runs the plain-order l6 program where the snake-order one is not registered, or None; see select_program
    "tile_unroll_default",  # tile-loop unroll when no rule applies
    "tile_unroll_factor",  # tile-loop unroll when the rule applies
    "tile_unroll_rule",  # [program kinds, min top_k, yields to the snake order], or None
    "sample_fit_max_mean_tiles",  # sampled first threshold on buffer-fitting units up to this mean tile count
    "sample_fit_kinds",  # physical scan kinds whose sampled-fit program the architecture registers; every other kind runs its plain program
    "sample_tiles_max",  # sample-tile cap of long units (0 = no sampling)
    "sample_tiles_short_units",  # sample-tile cap of short units
    "sample_dispatch_mean_tiles_max",  # mean unit tiles up to which the short-unit cap applies
    "sample_tiles_tiny_units",  # sample-tile cap of tiny units (lever K3b)
    "sample_dispatch_tiny_tiles_max",  # mean unit tiles up to which the tiny-unit cap applies
    "sample_shift_permille",  # conservative shift of the sampled rank
    "sample_class_rule",  # [tiny-unit sample tiles, shift permille] for the wide-unit programs (shift only) and the tiny dispatch class (shift, and tiles where the mean unit holds more), or None; explicit knobs override it
    "sample_long_rule",  # [physical scan kinds, shift permille] of the long-unit calls of those kinds (more than sample_dispatch_mean_tiles_max mean unit tiles), or None; an explicit shift knob overrides it
    "sample_shift_rule",  # [min top_k, shift permille] replacing sample_shift_permille from that top_k on, or None (the default shift on every call); an explicit sample_shift_permille knob overrides both
    "lsf_programs",  # the architecture ships longest-first segment order copies (scan key suffix :lsf) of its scan programs; the host runs the decided program's copy on packed calls whose key segments differ in length, the program itself where the registry has no copy of it
    "lsf_single_rule",  # [min mean unit tiles] from which a one-segment FP8 call of the wide unit runs the :lsf copy of its program, or None (never); read only by select_fp8_program
    "static_scan_kinds",  # physical scan kinds whose programs keep the static unit enumeration on a ticket architecture: they never touch the ticket tail, so the host gives their runners none
    "far_sample",  # 1: the sampled first threshold of the ascending-tile-order programs ranks the far sample tiles only (the tile nearest the diagonal left out); 0: every sample tile
    "check_period_max",  # cap on the default selection-trigger check period (tiles)
    "check_period_knob_max",  # cap on an explicit check period (tiles)
    "check_period_cap_divisor",  # default period = cand_cap // divisor
    "check_period_kind_overrides",  # {program kind: period} overrides, clamped to the exactness bound
)

KINDS = ("narrow", "wide", "l6", "split", "pair_narrow", "pair_wide", "pair_l6")
UNITS = ("narrow", "wide", "l6")
# Names of the compiled scan program kinds (the split program is one per unit geometry).
PHYSICAL_KINDS = tuple(
    sorted(set(KINDS) - {"split"} | {f"split_{unit}" for unit in UNITS})
)
MERGE_KEY = "merge"
# Input forms of a call, selected by its q / k dtype: ``bf16`` (the default) and ``fp8`` (float8_e4m3fn q / k with
# per-key FP32 scales, opt-in).  An FP8 scan program's key carries the form tag before the LSF suffix,
# ``scan:<kind>:u<unroll>:s<snake>:f<fit>:fp8[:lsf]``; the plain merge program serves both forms.
BF16 = "bf16"
FP8 = "fp8"
FORMS = (BF16, FP8)
# The FP8 scan programs exist for the narrow, l6, split and wide kinds, in the plain unit order (sm_107a also registers the
# snake copy of its l6 program, which ``select_fp8_program`` runs for a snake-marked call within ``snake_cap_rule``) and in
# the sampled-fit form.
# An FP8 call runs the decision procedure of ``select_program`` under the FP8 record (``fp8_policy_record``); a decided
# kind without an FP8 program runs the FP8 program of its fallback kind below (unpaired; the same unit geometry where it
# has one, except that a CTA-pair wide unit runs l6).  An unpaired, unsplit wide unit (8 queries) runs the FP8 wide
# program where the architecture registers one (sm_100a, sm_103a and sm_107a); else it, and a CTA-pair wide unit always,
# runs the l6 unit (6 queries), the largest FP8 unit below the wide one: per key tile it reads K once for 6 queries
# rather than for the narrow unit's 4.  A split wide unit (sm_107a only) runs split_narrow: no architecture that splits
# wide units registers an FP8 split_l6 program.
FP8_KIND_FALLBACK = {
    "wide": "l6",
    "pair_l6": "l6",
    "pair_narrow": "narrow",
    "pair_wide": "l6",
    "split_wide": "split_narrow",
    "split_l6": "split_narrow",
}
# Admissible sampled rank shift (permille): the backend's ``sample_shift_permille`` knob bound and the bound of a record's
# ``sample_shift_rule`` shift.  Every value selects the same set; the shift only sets how conservative the sampled first
# threshold is.
SAMPLE_SHIFT_PERMILLE_BOUNDS = (-1000, 10000)
# Global unit tickets (every scan program but the kinds of a record's
# ``static_scan_kinds``, on ``cake_jit.TICKET_TAIL_ARCHES``): the CTAs draw every
# work unit from the low u32 word of the first int64 behind the candidate buffers
# (and behind the split staging), the first of a tail every plan's workspace
# carries: 16 int64 for the ticket word, then the longest-first segment order the
# scan prologue writes (one int64 per segment, up to 1024 segments).
# Every completed scan launch leaves the word at 0; the host (``cake_backend``)
# zeroes the tail once at prepare on a backend-owned workspace and before every
# run on a caller-owned ``workspace_buffer``, which others may write between
# runs -- except where the call always takes the static fast path
# (``ticket_static_path``: the scan never touches the word) or runs a program of
# ``static_scan_kinds``.  A scan program whose registry record lists the
# architecture in ``ticket_arg_arches`` draws from its ``TicketWord`` block
# instead and never draws from this tail (the eager entry hands it the tail's
# first 16 bytes); its :lsf copies (sm_107a's l6 ticket programs) still keep the
# segment order in the tail, which every launch writes before it reads, so the
# host zeroes nothing for them.  Every plan carries the tail, so the size does
# not depend on the program.
TICKET_TAIL_BYTES = 8 * (16 + 1024)


def ticket_units_bound(choice: "ProgramChoice") -> int:
    """An upper bound, from the call's shape alone, on the work units ``n_total`` a ticket scan launch of ``choice``
    enumerates.  The kernel counts ``n_total`` from ``cu_seqlens_q`` on the device: ``ceil(l_s / block_q)`` blocks per
    segment, times ``n_split`` on a split plan, ``ceil(blocks / 2)`` per segment on a CTA pair.  For any segment
    lengths ``l_s >= 0`` with ``sum l_s <= T``, ``sum ceil(l_s / bq) <= floor((T + S (bq - 1)) / bq)`` and
    ``sum ceil(b_s / 2) <= floor((sum b_s + S) / 2)``; the bound reads no tensor, so a caller that rewrites the
    boundaries in place cannot move a call across it."""
    bq = int(choice.block_q)
    blocks = (int(choice.num_queries) + int(choice.num_segments) * (bq - 1)) // bq
    if choice.split:
        return blocks * int(choice.n_split)
    if choice.pair:
        return (blocks + int(choice.num_segments)) // 2
    return blocks


def ticket_static_path(choice: "ProgramChoice") -> bool:
    """True when every launch of ``choice`` takes the ticket scan's static fast path: ``n_total <= G`` (``G`` = the
    launched grid, half of it on a CTA pair), where the kernel gives every CTA / cluster at most one unit by its index
    and never reads or writes the ticket word (no draw, no publish).  Decided by ``ticket_units_bound``."""
    agents = int(choice.grid_ctas) // 2 if choice.pair else int(choice.grid_ctas)
    return ticket_units_bound(choice) <= agents


# sample_class_rule's tiny-class part applies to packed calls of short query segments only
CLASS_RULE_MIN_SEGMENTS = 2
CLASS_RULE_MAX_QUERIES_PER_SEGMENT = 4096
# lever LSF: the longest-first segment order copy of a scan program runs on calls of at least this many segments
LSF_MIN_SEGMENTS = 2


def sample_class_for(
    policy: "DispatchPolicy",
    physical: str,
    num_queries: int,
    num_keys: int,
    num_segments: int,
    ratio: int,
    tiles: int,
    shift: int,
    *,
    tiles_knob: bool,
    shift_knob: bool,
) -> tuple[int, int]:
    """The sampled first threshold ``(tiles, shift)`` of a scan program of the physical kind ``physical`` under the
    record's ``sample_class_rule`` and ``sample_long_rule``, from the call's ``tiles`` / ``shift``: the wide-unit
    programs take the class rule's shift on packed calls (at least ``CLASS_RULE_MIN_SEGMENTS`` segments), the tiny
    dispatch class (packed calls of short query segments and few mean unit tiles) also its tile count where the mean
    unit holds more tiles than that count; a long-unit call
    (more than ``sample_dispatch_mean_tiles_max`` mean unit tiles) of a kind the long rule names takes its shift.  A
    knob the caller set (``tiles_knob`` / ``shift_knob``) stays; without the rules or with sampling off nothing
    changes."""
    class_rule, long_rule = policy.sample_class_rule, policy.sample_long_rule
    T, S = int(num_queries), int(num_segments)
    if tiles <= 0:
        return tiles, shift
    # the long-unit calls of the long rule's kinds shift further (sm_107a)
    if (
        long_rule is not None
        and physical in long_rule[0]
        and policy.mean_unit_tiles(T, num_keys, S, ratio)
        > policy.sample_dispatch_mean_tiles_max
    ):
        if not shift_knob:
            shift = int(long_rule[1])
        # a long-unit call keeps its tile count and never takes the class rule, which is for few mean unit tiles
        return tiles, shift
    if class_rule is None:
        return tiles, shift
    # the wide-unit programs shift only; the tiny dispatch class also samples more tiles
    if "wide" in physical:
        if not shift_knob and S >= CLASS_RULE_MIN_SEGMENTS:
            shift = int(class_rule[1])
    elif (
        not tiles_knob
        and policy.mean_unit_tiles(T, num_keys, S, ratio)
        <= policy.sample_dispatch_tiny_tiles_max
        and S >= CLASS_RULE_MIN_SEGMENTS
        and T <= CLASS_RULE_MAX_QUERIES_PER_SEGMENT * S
    ):
        # no more sample tiles than the mean unit holds: a shorter unit would be sampled whole
        if policy.mean_unit_tiles(T, num_keys, S, ratio) > int(class_rule[0]):
            tiles = int(class_rule[0])
        if not shift_knob:
            shift = int(class_rule[1])
    return tiles, shift


_INT = ("tile_keys", "block_q_narrow", "block_q_wide", "block_q_l6", "candidate_entry_bytes", "candidate_multiplier",
        "candidate_multiplier_unsampled", "candidate_slack", "split_max", "split_min_range_tiles", "tile_unroll_default", "tile_unroll_factor",
        "sample_tiles_max", "sample_tiles_short_units", "sample_tiles_tiny_units", "sample_shift_permille", "far_sample",
        "check_period_max", "check_period_knob_max", "check_period_cap_divisor")  # fmt: skip
_BOOL = (
    "wide_pair_snake_only",
    "snake_default",
    "lsf_programs",
)
_NUMBER = (
    "sample_fit_max_mean_tiles",
    "sample_dispatch_mean_tiles_max",
    "sample_dispatch_tiny_tiles_max",
)
_OPTIONAL_INT = (
    "l6_solo_unsampled_max_queries",
    "snake_l6_fallback_waves",
    "cand_cap_floor",
    "cand_cap_floor_unsampled",
    "cand_cap_ceiling",
    "cand_cap_ceiling_unsampled",
)
_OPTIONAL_RULE = (
    "cand_mult_rule",
    "cand_cap_floor_rule",
    "cand_mult_rule_unsampled",
    "l6_rule",
    "wide_first_rule",
    "wide_first_k_rule",
    "wide_wave_rule",
    "desc_walk_kinds",
    "split_wave_rule",
    "split_wave_scope",
    "split_fit_rule",
    "split_resident_rule",
    "pair_rule",
    "l6_solo_rule",
    "snake_rule",
    "snake_cap_rule",
    "lsf_single_rule",
    "snake_packed_rule",
    "tile_unroll_rule",
    "sample_shift_rule",
    "sample_class_rule",
    "sample_long_rule",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class DispatchPolicy:
    """The host dispatch rules of one architecture, evaluated from the registry record.

    Every method mirrors the kernel's own host function of the same name; the
    export verifies the mirror against the kernel before the record is frozen.
    """

    tile_keys: int
    block_q_narrow: int
    block_q_wide: int
    block_q_l6: int
    candidate_entry_bytes: int
    candidate_multiplier: int
    candidate_slack: int
    cand_mult_rule: Optional[list]
    cand_cap_floor: Optional[int]
    cand_cap_floor_rule: Optional[list]
    candidate_multiplier_unsampled: int
    cand_mult_rule_unsampled: Optional[list]
    cand_cap_floor_unsampled: Optional[int]
    cand_cap_ceiling: Optional[int]
    cand_cap_ceiling_unsampled: Optional[int]
    l6_rule: Optional[list]
    wide_rule: list
    wide_first_rule: Optional[list]
    wide_first_k_rule: Optional[list]
    wide_wave_rule: Optional[list]
    desc_walk_kinds: Optional[list]
    split_max: int
    split_min_range_tiles: int
    split_wave_rule: Optional[list]
    split_wave_scope: Optional[list]
    split_fit_rule: Optional[list]
    split_resident_rule: Optional[list]
    split_units: list
    pair_rule: Optional[list]
    pair_units: list
    l6_solo_rule: Optional[list]
    l6_solo_unsampled_max_queries: Optional[int]
    wide_pair_snake_only: bool
    snake_default: bool
    snake_rule: Optional[list]
    snake_cap_rule: Optional[list]
    snake_packed_rule: Optional[list]
    snake_l6_fallback_waves: Optional[int]
    tile_unroll_default: int
    tile_unroll_factor: int
    tile_unroll_rule: Optional[list]
    sample_fit_max_mean_tiles: float
    sample_fit_kinds: list
    sample_tiles_max: int
    sample_tiles_short_units: int
    sample_dispatch_mean_tiles_max: float
    sample_tiles_tiny_units: int
    sample_dispatch_tiny_tiles_max: float
    sample_shift_permille: int
    sample_shift_rule: Optional[list]
    sample_class_rule: Optional[list]
    sample_long_rule: Optional[list]
    lsf_programs: bool
    lsf_single_rule: Optional[list]
    static_scan_kinds: list
    far_sample: int
    check_period_max: int
    check_period_knob_max: int
    check_period_cap_divisor: int
    check_period_kind_overrides: dict

    @classmethod
    def from_record(cls, raw: Any) -> "DispatchPolicy":
        if not isinstance(raw, dict) or set(raw) != set(POLICY_FIELDS):
            raise ValueError(
                f"dispatch policy record must carry exactly the fields {POLICY_FIELDS}, got {raw!r}"
            )
        for name in _INT:
            if not _is_int(raw[name]):
                raise ValueError(
                    f"policy[{name!r}] must be an integer, got {raw[name]!r}"
                )
        for name in _BOOL:
            if not isinstance(raw[name], bool):
                raise ValueError(f"policy[{name!r}] must be a bool, got {raw[name]!r}")
        for name in _NUMBER:
            if not _is_number(raw[name]):
                raise ValueError(
                    f"policy[{name!r}] must be a number, got {raw[name]!r}"
                )
        for name in _OPTIONAL_INT:
            if raw[name] is not None and not _is_int(raw[name]):
                raise ValueError(
                    f"policy[{name!r}] must be an integer or None, got {raw[name]!r}"
                )
        for name in _OPTIONAL_RULE:
            if raw[name] is not None and not isinstance(raw[name], (list, tuple)):
                raise ValueError(
                    f"policy[{name!r}] must be a rule list or None, got {raw[name]!r}"
                )
        if (
            not isinstance(raw["wide_rule"], (list, tuple))
            or len(raw["wide_rule"]) != 2
        ):
            raise ValueError(
                f"policy['wide_rule'] must be [min rounds, min work], got {raw['wide_rule']!r}"
            )
        first = raw["wide_first_rule"]
        if first is not None and (
            len(first) != 1 or not _is_number(first[0]) or first[0] < 0
        ):
            raise ValueError(
                f"policy['wide_first_rule'] must be [min mean unit tiles] or None, got {first!r}"
            )
        first_k = raw["wide_first_k_rule"]
        if first_k is not None and (
            len(first_k) != 2
            or not _is_int(first_k[0])
            or first_k[0] < 1
            or not _is_number(first_k[1])
            or first_k[1] < 0
        ):
            raise ValueError(
                f"policy['wide_first_k_rule'] must be [max top_k >= 1, min mean unit tiles >= 0] or None, got {first_k!r}"
            )
        waves = raw["wide_wave_rule"]
        if waves is not None and (
            len(waves) not in (3, 15)
            or not all(_is_number(v) and v >= 0 for v in waves)
            or (
                len(waves) == 15
                and not (
                    _is_int(waves[3])
                    and waves[3] >= 1
                    and waves[5] >= waves[2]
                    and _is_int(waves[7])
                    and waves[7] >= 1
                    and waves[8] >= waves[1]
                    and _is_int(waves[13])
                    and waves[13] >= 1
                )
            )
        ):
            raise ValueError(
                f"policy['wide_wave_rule'] must be [min narrow / wide wave ratio, min l6 / wide wave ratio, min mean unit tiles] or that with [the top_k (an int >= 1) up to which the tile minima hold, their exponent in top_k above it, the minimum of a packed call (>= the one-segment minimum), the tiles the one-segment minimum rises per wide wave below the next element, the wide waves (an int >= 1) from which it holds, the l6 / wide wave ratio W (>= element 1) below which the minimum is at least the next element / (W - 1) and on a packed call at least element 10 / (W - 1) ** 2, and from which the minimum of a one-document call of fewer wide waves than element 7 is at most element 11 / (W - 1) ** 2, on two or more wide waves at most the larger of that and element 12, from element 13 (an int >= 1) wide waves on of that and element 14], each >= 0, or None, got {waves!r}"
            )
        desc = raw["desc_walk_kinds"]
        if desc is not None and (
            len(desc) != 2
            or not all(
                isinstance(kinds, (list, tuple))
                and all(isinstance(k, str) and k in PHYSICAL_KINDS for k in kinds)
                and list(kinds) == sorted(set(kinds))
                for kinds in desc
            )
        ):
            raise ValueError(
                f"policy['desc_walk_kinds'] must be [unsampled kinds, sampled kinds] (ascending lists of physical scan kinds {PHYSICAL_KINDS}) or None, got {desc!r}"
            )
        pair_units = raw["pair_units"]
        if (
            not isinstance(pair_units, (list, tuple))
            or not all(isinstance(u, str) and u in UNITS for u in pair_units)
            or list(pair_units) != sorted(set(pair_units))
        ):
            raise ValueError(
                f"policy['pair_units'] must be an ascending list of unit geometries {UNITS}, got {pair_units!r}"
            )
        if raw["pair_rule"] is not None and not pair_units:
            raise ValueError(
                "policy['pair_units'] must name at least one unit geometry when policy['pair_rule'] is set"
            )
        split_units = raw["split_units"]
        if (
            not isinstance(split_units, (list, tuple))
            or not all(isinstance(u, str) and u in UNITS for u in split_units)
            or list(split_units) != sorted(set(split_units))
        ):
            raise ValueError(
                f"policy['split_units'] must be an ascending list of unit geometries {UNITS}, got {split_units!r}"
            )
        scope = raw["split_wave_scope"]
        if scope is not None and (
            len(scope) != 3
            or not isinstance(scope[0], (list, tuple))
            or not scope[0]
            or not all(isinstance(u, str) and u in UNITS for u in scope[0])
            or list(scope[0]) != sorted(set(scope[0]))
            or not _is_int(scope[1])
            or scope[1] < 1
            or not isinstance(scope[2], bool)
        ):
            raise ValueError(
                f"policy['split_wave_scope'] must be [ascending unit geometries {UNITS}, max segments >= 1, sampled only (bool)] or None, got {scope!r}"
            )
        fit_rule = raw["split_fit_rule"]
        if fit_rule is not None and (
            len(fit_rule) != 1 or not _is_int(fit_rule[0]) or fit_rule[0] < 1
        ):
            raise ValueError(
                f"policy['split_fit_rule'] must be [merge chunk entries >= 1] or None, got {fit_rule!r}"
            )
        resident = raw["split_resident_rule"]
        if resident is not None and (
            len(resident) != 5
            or not all(_is_int(v) and v >= 1 for v in resident)
            or not 2 <= resident[4] <= resident[3]
        ):
            raise ValueError(
                f"policy['split_resident_rule'] must be [merge chunk entries, fit entries, merge-resident entries, max splits, "
                f"min splits] (positive integers, 2 <= min splits <= max splits) or None, got {resident!r}"
            )
        solo = raw["l6_solo_rule"]
        if solo is not None and (
            len(solo) != 2
            or not _is_number(solo[0])
            or solo[0] < 0
            or not _is_int(solo[1])
            or solo[1] < 1
        ):
            raise ValueError(
                f"policy['l6_solo_rule'] must be [max mean unit tiles >= 0, min segments >= 1] or None, got {solo!r}"
            )
        solo_queries = raw["l6_solo_unsampled_max_queries"]
        if solo_queries is not None and solo_queries < 1:
            raise ValueError(
                f"policy['l6_solo_unsampled_max_queries'] must be an integer >= 1 or None, got {solo_queries!r}"
            )
        snake_cap = raw["snake_cap_rule"]
        if snake_cap is not None and (
            len(snake_cap) != 2 or not all(_is_int(v) and v >= 1 for v in snake_cap)
        ):
            raise ValueError(
                f"policy['snake_cap_rule'] must be [max segments >= 1, max waves >= 1] or None, got {snake_cap!r}"
            )
        lsf_single = raw["lsf_single_rule"]
        if lsf_single is not None and (
            len(lsf_single) != 1
            or not (_is_number(lsf_single[0]) and lsf_single[0] >= 0)
        ):
            raise ValueError(
                f"policy['lsf_single_rule'] must be [min mean unit tiles >= 0] or None, got {lsf_single!r}"
            )
        packed = raw["snake_packed_rule"]
        if packed is not None and (
            len(packed) != 4
            or not isinstance(packed[0], (list, tuple))
            or not packed[0]
            or not all(
                isinstance(k, str) and k in KINDS and k != "split" for k in packed[0]
            )
            or list(packed[0]) != sorted(set(packed[0]))
            or not _is_number(packed[1])
            or packed[1] < 0
            or not (packed[2] is None or (_is_number(packed[2]) and packed[2] >= 0))
            or not (packed[3] is None or (_is_int(packed[3]) and packed[3] >= 1))
        ):
            raise ValueError(
                f"policy['snake_packed_rule'] must be [ascending program kinds {KINDS} but split, min mean unit tiles sampled >= 0, "
                f"min mean unit tiles with sampling off >= 0 or None, max waves >= 1 or None] or None, got {packed!r}"
            )
        fallback = raw["snake_l6_fallback_waves"]
        if fallback is not None and fallback < 1:
            raise ValueError(
                f"policy['snake_l6_fallback_waves'] must be an integer >= 1 or None, got {fallback!r}"
            )
        shift_lo, shift_hi = SAMPLE_SHIFT_PERMILLE_BOUNDS
        shift_rule = raw["sample_shift_rule"]
        if shift_rule is not None and (
            len(shift_rule) != 2
            or not _is_int(shift_rule[0])
            or shift_rule[0] < 1
            or not _is_int(shift_rule[1])
            or not shift_lo <= shift_rule[1] <= shift_hi
        ):
            raise ValueError(
                f"policy['sample_shift_rule'] must be [min top_k >= 1, shift permille in [{shift_lo}, {shift_hi}]] or None, got {shift_rule!r}"
            )
        class_rule = raw["sample_class_rule"]
        if class_rule is not None and (
            len(class_rule) != 2
            or not _is_int(class_rule[0])
            or class_rule[0] < 1
            or not _is_int(class_rule[1])
            or not shift_lo <= class_rule[1] <= shift_hi
        ):
            raise ValueError(
                f"policy['sample_class_rule'] must be [tiny-unit sample tiles >= 1, shift permille in [{shift_lo}, {shift_hi}]] or None, got {class_rule!r}"
            )
        long_rule = raw["sample_long_rule"]
        if long_rule is not None and (
            len(long_rule) != 2
            or not isinstance(long_rule[0], (list, tuple))
            or not long_rule[0]
            or not all(isinstance(kind, str) for kind in long_rule[0])
            or list(long_rule[0]) != sorted(set(long_rule[0]))
            or not set(long_rule[0]) <= set(PHYSICAL_KINDS)
            or not _is_int(long_rule[1])
            or not shift_lo <= long_rule[1] <= shift_hi
        ):
            raise ValueError(
                f"policy['sample_long_rule'] must be [ascending physical scan kinds {PHYSICAL_KINDS}, shift permille in "
                f"[{shift_lo}, {shift_hi}]] or None, got {long_rule!r}"
            )
        floor_rule = raw["cand_cap_floor_rule"]
        if floor_rule is not None and (
            len(floor_rule) != 2 or not all(_is_int(v) and v >= 1 for v in floor_rule)
        ):
            raise ValueError(
                f"policy['cand_cap_floor_rule'] must be [min top_k >= 1, floor slots >= 1] or None, got {floor_rule!r}"
            )
        if not isinstance(raw["check_period_kind_overrides"], dict):
            raise ValueError("policy['check_period_kind_overrides'] must be a mapping")
        for floor_name, ceiling_name, rule_name in (
            ("cand_cap_floor", "cand_cap_ceiling", "cand_cap_floor_rule"),
            ("cand_cap_floor_unsampled", "cand_cap_ceiling_unsampled", None),
        ):
            ceiling = raw[ceiling_name]
            floors = [raw[floor_name]]
            if rule_name is not None and raw[rule_name] is not None:
                floors.append(raw[rule_name][1])
            if ceiling is not None and (
                ceiling <= 0 or any(f is not None and ceiling < f for f in floors)
            ):
                raise ValueError(
                    f"policy[{ceiling_name!r}] must be a positive slot count at or above policy[{floor_name!r}]"
                    f"{' and the floor of policy[' + repr(rule_name) + ']' if rule_name else ''}, got {ceiling!r}"
                )
        static_kinds = raw["static_scan_kinds"]
        if (
            not isinstance(static_kinds, (list, tuple))
            or not all(isinstance(k, str) and k in PHYSICAL_KINDS for k in static_kinds)
            or list(static_kinds) != sorted(set(static_kinds))
        ):
            raise ValueError(
                f"policy['static_scan_kinds'] must be an ascending list of physical scan kinds {PHYSICAL_KINDS}, got {static_kinds!r}"
            )
        fit_kinds = raw["sample_fit_kinds"]
        if (
            not isinstance(fit_kinds, (list, tuple))
            or not all(isinstance(k, str) and k in PHYSICAL_KINDS for k in fit_kinds)
            or list(fit_kinds) != sorted(set(fit_kinds))
        ):
            raise ValueError(
                f"policy['sample_fit_kinds'] must be an ascending list of physical scan kinds {PHYSICAL_KINDS}, got {fit_kinds!r}"
            )
        for name in ("tile_keys", "block_q_narrow", "block_q_wide", "block_q_l6", "candidate_entry_bytes",
                     "candidate_multiplier", "candidate_multiplier_unsampled", "candidate_slack", "split_max",
                     "split_min_range_tiles",
                     "tile_unroll_default", "tile_unroll_factor", "sample_tiles_short_units", "sample_tiles_tiny_units",
                     "check_period_max", "check_period_knob_max", "check_period_cap_divisor"):  # fmt: skip
            if raw[name] <= 0:
                raise ValueError(
                    f"policy[{name!r}] must be positive, got {raw[name]!r}"
                )
        if raw["sample_tiles_max"] < 0:
            raise ValueError("policy['sample_tiles_max'] must be >= 0")
        if raw["far_sample"] not in (0, 1):
            raise ValueError("policy['far_sample'] must be 0 or 1")
        values = dict(raw)
        for name in _OPTIONAL_RULE:
            if values[name] is not None:
                values[name] = [
                    list(v) if isinstance(v, (list, tuple)) else v for v in values[name]
                ]
        values["wide_rule"] = list(values["wide_rule"])
        values["sample_fit_kinds"] = [str(k) for k in values["sample_fit_kinds"]]
        values["static_scan_kinds"] = [str(k) for k in values["static_scan_kinds"]]
        values["pair_units"] = [str(u) for u in values["pair_units"]]
        values["split_units"] = [str(u) for u in values["split_units"]]
        values["check_period_kind_overrides"] = {
            str(k): int(v) for k, v in values["check_period_kind_overrides"].items()
        }
        return cls(**values)

    def record(self) -> dict:
        """The record in canonical field order."""
        values = asdict(self)
        return {name: values[name] for name in POLICY_FIELDS}

    # ------------------------------------------------------------------ geometry estimates

    def mean_unit_tiles(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> float:
        """Estimated mean key tiles per unit, ``(Tkv - T / (2 ratio)) / S / tile_keys`` (a causal segment's
        units see on average half of its query span less than its key count)."""
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        mean_keys = (int(num_keys) - int(num_queries) / (2 * r)) / segments
        return max(0.0, mean_keys) / self.tile_keys

    def _mean_unit_tiles_floor(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> int:
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        mean_keys = (int(num_keys) - int(num_queries) / (2 * r)) / segments
        return int(max(0.0, mean_keys) // self.tile_keys)

    def offset_tiles(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> float:
        """Estimated mean segment offset in key tiles, ``(Tkv - T / ratio) / S / tile_keys``."""
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        return (
            max(0.0, (int(num_keys) - int(num_queries) / r) / segments) / self.tile_keys
        )

    # ------------------------------------------------------------------ runtime scalars

    def sample_tiles_for(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> int:
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        if mean_tiles <= self.sample_dispatch_tiny_tiles_max:
            return self.sample_tiles_tiny_units
        return (
            self.sample_tiles_short_units
            if mean_tiles <= self.sample_dispatch_mean_tiles_max
            else self.sample_tiles_max
        )

    def sample_shift_for(self, top_k: int) -> int:
        """Conservative shift (permille) of the sampled rank for a call: ``sample_shift_rule[1]`` from the rule's
        ``top_k`` floor on, else ``sample_shift_permille``.  Every value selects the same set -- the shift only sets
        how conservative the sampled first threshold is (a threshold that proves too aggressive is streamed again
        exactly); an unsampled call (``sample_tiles_max=0``) never reads it."""
        rule = self.sample_shift_rule
        if rule is not None and int(top_k) >= int(rule[0]):
            return int(rule[1])
        return self.sample_shift_permille

    def cand_cap_floor_for(self, top_k: int) -> Optional[int]:
        """Candidate-slot floor of a sampled call: ``cand_cap_floor_rule[1]`` from the rule's ``top_k`` floor on, else
        ``cand_cap_floor``.  The larger floor pays for itself on large-``top_k`` calls only (their units fill the buffer
        early; the floor spares compactions and keeps 64-127-tile units off the sampled path); below the rule's ``top_k``
        the smaller floor halves the candidate buffers -- 57 MiB instead of 114 MiB for a ``top_k = 512`` call (the
        ratio-4 pooled selection) on a 152-SM grid with six queries per unit -- selecting the same ids.
        Unsampled calls (``sample_tiles_max=0``) read ``cand_cap_floor_unsampled``.  The shipped ``sm_100a`` record sets
        the rule; ``sm_103a`` (whose pooled rows measured slower with the smaller buffers) and ``sm_107a`` set none."""
        rule = self.cand_cap_floor_rule
        if rule is not None and int(top_k) >= int(rule[0]):
            return int(rule[1])
        return self.cand_cap_floor

    def cand_cap_multiplier_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        top_k: int,
        sampled: bool = True,
    ) -> int:
        """Capacity multiplier of a call: the sampled values by default, the ``*_unsampled`` values
        when the sampled first threshold is off (``sample_tiles_max=0``), whose larger buffers would
        only delay the first selection.  The long-unit rule replaces the multiplier, the floor
        (``cand_cap_floor_for``: keyed by ``top_k`` where the record sets ``cand_cap_floor_rule``) raises it
        and the ceiling lowers it (``cand_cap_ceiling // top_k``, so the buffer stops growing with
        ``top_k`` once it holds the ceiling)."""
        if sampled:
            mult, rule, floor, ceiling = (
                self.candidate_multiplier,
                self.cand_mult_rule,
                self.cand_cap_floor_for(top_k),
                self.cand_cap_ceiling,
            )
        else:
            mult, rule, floor, ceiling = (
                self.candidate_multiplier_unsampled,
                self.cand_mult_rule_unsampled,
                self.cand_cap_floor_unsampled,
                self.cand_cap_ceiling_unsampled,
            )
        if rule is not None:
            if (
                self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
                >= rule[0]
            ):
                mult = int(rule[1])
        if floor is not None and int(top_k) * mult < int(floor):
            mult = -(-int(floor) // int(top_k))
        if ceiling is not None and int(top_k) * mult > int(ceiling):
            mult = max(1, int(ceiling) // int(top_k))
        return mult

    def candidate_capacity(self, top_k: int, multiplier: Optional[int] = None) -> int:
        """Per-(warpgroup, query) candidate slots: ``max(multiplier * top_k, top_k + slack)`` rounded up to whole tiles."""
        mult = self.candidate_multiplier if multiplier is None else int(multiplier)
        if mult < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        cap = max(int(top_k) * mult, int(top_k) + self.candidate_slack)
        return (cap + self.tile_keys - 1) // self.tile_keys * self.tile_keys

    def check_period_for(
        self, top_k: int, cand_cap: int, kind: Optional[str] = None
    ) -> int:
        tile = self.tile_keys
        period = max(
            1,
            min(
                self.check_period_max,
                (int(cand_cap) - int(top_k)) // tile,
                int(cand_cap) // self.check_period_cap_divisor,
            ),
        )
        override = (
            self.check_period_kind_overrides.get(kind) if kind is not None else None
        )
        if override is not None:
            period = max(
                1,
                min(
                    int(override),
                    self.check_period_knob_max,
                    (int(cand_cap) - int(top_k)) // tile,
                    int(cand_cap) // (2 * tile),
                ),
            )
        return period

    def check_period_limit(self, top_k: int, cand_cap: int) -> int:
        """Largest explicit check period that is still exact for the capacity (the kernel's headroom rule)."""
        tile = self.tile_keys
        return max(
            1,
            min(
                self.check_period_knob_max,
                (int(cand_cap) - int(top_k)) // tile,
                int(cand_cap) // (2 * tile),
            ),
        )

    # ------------------------------------------------------------------ program selection

    def unit_name(self, block_q: int) -> str:
        if int(block_q) == self.block_q_l6:
            return "l6"
        if int(block_q) == self.block_q_wide:
            return "wide"
        if int(block_q) == self.block_q_narrow:
            return "narrow"
        raise ValueError(
            f"block_q must be {self.block_q_narrow}, {self.block_q_wide} or {self.block_q_l6}, got {block_q!r}"
        )

    def program_kind(self, block_q: int, split: bool, pair: bool) -> str:
        if split:
            return "split"
        base = self.unit_name(block_q)
        return f"pair_{base}" if pair else base

    def unit_block_q_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        top_k: int,
    ) -> int:
        units_wide = -(-int(num_queries) // self.block_q_wide)
        rounds = units_wide / max(1, int(grid))
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        l6 = self.l6_rule
        three_wg = l6 is not None and mean_tiles <= l6[0] and int(top_k) >= int(l6[1])
        waves = self.wide_wave_rule
        if waves is not None:
            # the unit the other rules leave (three-warpgroup or narrow) against the wide unit, in waves of the
            # persistent grid: the wide unit when the other unit needs at least the record's ratio of the wide unit's
            # waves (against the three-warpgroup unit also from a mean unit tile count on).  A fifteen-element rule
            # raises that count by waves[6] tiles per wide wave below waves[7] (a short call's last waves cost the wide
            # unit more) but on one document to at most waves[11] / (W - 1) ** 2, from two wide waves on to at most the
            # larger of that and waves[12] (from waves[13] wide waves on, of that and waves[14]: there the wide unit
            # measured slower up to more tiles), while the wave ratio W of the three-warpgroup unit to the wide unit is
            # at least waves[8] (the more waves the wide unit saves, the shorter the units that pay for its last wave;
            # one wide wave has no last wave to pay for), to at least waves[5] on a packed call (two or more segments)
            # and to at least waves[9] / (W - 1) while W stays below waves[8] (the fewer waves the wide unit saves, the
            # more tiles its longer units need to pay for them), on a packed call then also to at least waves[10] /
            # (W - 1) ** 2, and multiplies it by (top_k / waves[3]) ** waves[4] above top_k = waves[3] (each unit
            # selects more candidates)
            other = self.block_q_l6 if three_wg else self.block_q_narrow
            ctas = max(1, int(grid))
            waves_wide = -(-units_wide // ctas)
            units_other = -(-int(num_queries) // other)
            waves_other = -(-units_other // ctas)
            if three_wg:
                floor = waves[2]
                if len(waves) > 3:
                    floor += waves[6] * max(0, int(waves[7]) - waves_wide)
                    if (
                        int(num_segments) < 2
                        and waves_wide < int(waves[7])
                        and waves_other > waves_wide
                        and waves_other >= waves[8] * waves_wide
                    ):
                        # 1 / (W - 1)
                        inv = waves_wide / (waves_other - waves_wide)
                        cap = waves[11] * inv * inv
                        if waves_wide > 1:
                            cap = max(
                                cap,
                                waves[14]
                                if waves_wide >= int(waves[13])
                                else waves[12],
                            )
                        floor = min(floor, cap)
                    if int(num_segments) >= 2:
                        floor = max(floor, waves[5])
                    if waves_other < waves[8] * waves_wide:
                        if waves_other > waves_wide:
                            # 1 / (W - 1)
                            inv = waves_wide / (waves_other - waves_wide)
                            floor = max(floor, waves[9] * inv)
                            if int(num_segments) >= 2:
                                floor = max(floor, waves[10] * inv * inv)
                        else:
                            floor = float("inf")
                    floor *= max(1.0, int(top_k) / waves[3]) ** waves[4]
                wide = waves_other >= waves[1] * waves_wide and mean_tiles >= floor
            else:
                wide = waves_other >= waves[0] * waves_wide
            return self.block_q_wide if wide else other
        min_rounds, min_work = self.wide_rule
        wide = rounds >= min_rounds and rounds * mean_tiles >= min_work
        first = self.wide_first_for(top_k)
        if wide and first is not None and mean_tiles >= first:
            return self.block_q_wide
        if three_wg:
            return self.block_q_l6
        return self.block_q_wide if wide else self.block_q_narrow

    def wide_first_for(self, top_k: int) -> Optional[float]:
        """The mean unit tiles from which a wide-eligible call of ``top_k`` takes the wide program before the
        three-warpgroup rule (``unit_block_q_for``): ``wide_first_k_rule[1]`` up to the rule's ``top_k``, else
        ``wide_first_rule[0]``, or None (the three-warpgroup rule first).  The three-warpgroup unit serves ``top_k``
        from ``l6_rule[1]`` on, so the rule moves calls of exactly that ``top_k`` up to the rule's: on ``sm_100a`` and
        ``sm_103a`` the ``top_k = 512`` calls of 128 (``sm_100a``) and 64 (``sm_103a``) mean unit tiles and more, where
        the wide program ran 1-4 % (``sm_103a``, 64-448 mean unit tiles) and 7-10 % (``sm_100a``, 128-255 mean unit
        tiles) faster than the CTA-pair l6 program these calls ran before (16384 queries over 16384-65536 keys, 8 x 2048
        queries over 8 x 17408 keys and an 8-segment packed call of 16231 queries over 268757 keys; same finalize
        program in both arms); ``sm_100a`` measured parity at 64 and 96 mean unit tiles, so its floor stays at 128.
        ``sm_107a`` sets no floor: with one at 0 its ``top_k = 512`` calls of 64-192 mean unit tiles ran the same programs
        as without it (its unpaired-l6 preference and CTA-pair rule decide them after the floor), and the single-CTA wide
        program ran 3.9 % slower than its l6 program at 64 mean unit tiles.  The ``wide_wave_rule`` records (the FP8 form
        on ``sm_100a`` / ``sm_103a``) decide by waves instead and never read it."""
        rule = self.wide_first_k_rule
        if rule is not None and int(top_k) <= int(rule[0]):
            return float(rule[1])
        first = self.wide_first_rule
        return None if first is None else float(first[0])

    def n_split_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        block_q: int,
        sampled: bool = True,
        top_k: Optional[int] = None,
    ) -> int:
        """Key-range splits per unit of the unit ``block_q``: the split rule (``split_rule_for``) for a unit of
        ``split_units``; 1 for any other unit, whose split program the architecture does not register."""
        if self.unit_name(block_q) not in self.split_units:
            return 1
        return self.split_rule_for(
            num_queries, num_keys, num_segments, ratio, grid, block_q, sampled, top_k
        )

    def split_rule_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        block_q: int,
        sampled: bool = True,
        top_k: Optional[int] = None,
    ) -> int:
        """The split rule alone (no registration guard): ``min(grid // units, mean tiles // split_min_range_tiles,
        split_max)``, at least 1, and the two-way wave split of ``split_wave_rule`` where that gives 1 -- inside
        ``split_wave_scope`` only (its unit geometries, calls of at most its segment count, and with its third item the
        sampled calls only: ``sampled`` says whether the call's sampled first threshold is on).

        With ``top_k`` given, a call that rule splits takes the count of ``split_resident_rule`` [chunk, fit, resident,
        max, min] where the record sets it: ``min(grid // units, resident // top_k, max, max(min, n_fit if n_fit x top_k
        <= fit else max))``, at least 2, with ``n_fit = ceil(mean unit keys / top_k)`` rounded up to whole merge chunks
        of ``chunk`` entries.  The merge keeps its ``n x top_k`` staged entries in shared memory up to ``resident``
        entries, so the count stays there; below it more splits shorten each split's scan.  Otherwise
        ``split_fit_rule``."""
        units = max(1, -(-int(num_queries) // int(block_q)))
        mean_tiles = self._mean_unit_tiles_floor(
            num_queries, num_keys, num_segments, ratio
        )
        n = max(
            1,
            min(
                int(grid) // units,
                mean_tiles // self.split_min_range_tiles,
                self.split_max,
            ),
        )
        rule = self.split_wave_rule
        scope = self.split_wave_scope
        if scope is not None and (
            self.unit_name(block_q) not in scope[0]
            or int(num_segments) > int(scope[1])
            or (scope[2] and not sampled)
        ):
            rule = None
        if (
            n == 1
            and rule is not None
            and mean_tiles >= int(rule[0])
            and mean_tiles >= 2 * self.split_min_range_tiles
        ):

            def eff(
                k: int,
            ) -> (
                float
            ):  # share of the CTAs busy over the call's rounds: waves / ceil(waves)
                return (units * k) / (int(grid) * (-(-(units * k) // int(grid))))

            if eff(2) - eff(1) >= float(rule[1]):
                n = 2
        if n == 1 or top_k is None or int(top_k) <= 0:
            return n
        k = int(top_k)
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        mean_keys = max(0.0, (int(num_keys) - int(num_queries) / (2 * r)) / segments)

        def fit_splits(chunk: int) -> int:
            # splits that hold at most top_k keys each, rounded up to whole merge chunks of ``chunk`` entries (more
            # splits in the last chunk cost the merge nothing)
            per_chunk = max(1, int(chunk) // k)
            return -(-max(1, -(-int(mean_keys) // k)) // per_chunk) * per_chunk

        resident = self.split_resident_rule
        if resident is not None:
            chunk, fit_entries, resident_entries, most, least = (
                int(v) for v in resident
            )
            n_fit = fit_splits(chunk)
            want = n_fit if n_fit * k <= fit_entries else most
            n = min(int(grid) // units, resident_entries // k, most, max(least, want))
            return max(2, n)
        fit = self.split_fit_rule
        if fit is not None:
            # split_fit_rule: a split call whose splits can each hold at most top_k keys needs no selection inside a
            # split; the merge reads its staging in chunks of fit[0] entries
            n_fit = fit_splits(int(fit[0]))
            if n < n_fit <= min(int(grid) // units, self.split_max):
                n = n_fit
        return n

    def pair_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        top_k: int,
        block_q: int,
    ) -> bool:
        """The two-CTA multicast program of the chosen unit ``block_q``: the pair serves the unit geometries of
        ``pair_units`` only (the units whose pair program the architecture registers), inside the ``pair_rule``
        mean-tile interval and from its ``top_k`` floor, on a grid of at least two CTAs."""
        rule = self.pair_rule
        if rule is None or int(grid) < 2:
            return False
        if self.unit_name(block_q) not in self.pair_units:
            return False
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        return rule[0] <= mean_tiles <= rule[1] and int(top_k) >= int(rule[2])

    def l6_solo_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        sampled: bool,
        pair_l6: bool = False,
    ) -> bool:
        """The preference for the three-warpgroup unit without a CTA pair.  A call with the sampled first threshold off
        (``sampled`` False): where the record sets ``l6_solo_unsampled_max_queries``, only a call planned on the CTA-pair
        l6 unit (``pair_l6``) of at most that many queries; else every such call where the record sets ``l6_solo_rule``.
        A sampled call: at least ``l6_solo_rule[1]`` segments or at most ``l6_solo_rule[0]`` mean unit tiles.  Never
        without either field.  ``select_program`` applies it only to a call whose plan does not split, whose l6 unit would
        not split either, and where the architecture registers the program it picks."""
        rule = self.l6_solo_rule
        if not sampled:
            cap = self.l6_solo_unsampled_max_queries
            if cap is not None:
                return pair_l6 and int(num_queries) <= int(cap)
            return rule is not None
        if rule is None:
            return False
        return int(num_segments) >= int(rule[1]) or self.mean_unit_tiles(
            num_queries, num_keys, num_segments, ratio
        ) <= float(rule[0])

    def snake_for(
        self,
        kind: str,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        sampled: bool = True,
        grid: Optional[int] = None,
    ) -> bool:
        """The snake unit order of a program kind: where ``snake_rule`` names the kind, from its mean unit tiles (the
        ratio-1 or the ratio > 1 floor) and unit-cost spread on.  On a call of two or more segments whose kind
        ``snake_packed_rule`` names, the order also needs the rule's mean unit tiles for the call's sampling mode
        (``sampled``: the sampled first threshold is on; with it off a None floor keeps the plain order), unless the call
        needs at most ``snake_packed_rule[3]`` waves of a persistent grid of ``grid`` CTAs (``ceil(ceil(T / block_q) /
        slots)``, slots = CTAs, or CTA pairs for a pair kind; never without ``grid``): the snake order balances the last
        waves of one causal document and of calls of few waves, and packed documents of few tiles over more waves run
        faster in the plain order."""
        rule = self.snake_rule
        if rule is None or kind not in rule[0]:
            return self.snake_default
        r = max(1, int(ratio))
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        spread = min(1.0, int(num_queries) / max(1.0, r * float(num_keys)))
        snake = mean_tiles >= (rule[1] if r == 1 else rule[2]) and spread >= rule[3]
        packed = self.snake_packed_rule
        if (
            snake
            and packed is not None
            and int(num_segments) >= 2
            and kind in packed[0]
        ):
            floor = packed[1] if sampled else packed[2]
            if floor is not None and mean_tiles >= float(floor):
                return True
            if packed[3] is None or grid is None:
                return False
            pair = kind.startswith("pair_")
            block_q = getattr(self, f"block_q_{kind[5:] if pair else kind}")
            ctas = max(1, int(grid))
            slots = max(1, (ctas - ctas % 2) // 2) if pair else ctas
            units = -(-int(num_queries) // int(block_q))
            return -(-units // slots) <= int(packed[3])
        return snake

    def snake_l6_fallback_for(
        self, num_queries: int, block_q: int, pair: bool, grid: int, sampled: bool
    ) -> bool:
        """Whether a sampling-off call (``sampled`` False) planned on the unit ``block_q`` (a CTA pair where ``pair``)
        over a persistent grid of ``grid`` CTAs needs at most ``snake_l6_fallback_waves`` waves of that grid,
        ``ceil(ceil(T / block_q) / slots)`` with ``slots`` the grid's CTAs (its CTA pairs for a pair program).  Such a
        call, which ``l6_solo_for`` moves to the unpaired l6 unit, runs that unit's plain-order program where its
        snake-order program is not registered (``select_program``); never where the record leaves the field None."""
        cap = self.snake_l6_fallback_waves
        if cap is None or sampled:
            return False
        ctas = max(1, int(grid))
        slots = max(1, (ctas - ctas % 2) // 2) if pair else ctas
        units = -(-int(num_queries) // max(1, int(block_q)))
        return -(-units // slots) <= int(cap)

    def snake_copy_fits(
        self, num_queries: int, num_segments: int, block_q: int, grid: int
    ) -> bool:
        """Whether a call of ``num_segments`` segments whose ``num_queries`` queries in units of ``block_q`` take a
        persistent grid of ``grid`` CTAs stays within ``snake_cap_rule`` (at most its segments and waves; always where
        the record leaves the rule None).  The snake unit order balances the last waves of one causal document; over
        packed documents, or over many waves, the plain order is faster (the FP8 sm_107a l6 program), so
        ``select_fp8_program`` runs a snake copy only within the rule."""
        rule = self.snake_cap_rule
        if rule is None:
            return True
        units = -(-int(num_queries) // max(1, int(block_q)))
        waves = -(-units // max(1, int(grid)))
        return int(num_segments) <= int(rule[0]) and waves <= int(rule[1])

    def lsf_single_fits(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> bool:
        """Whether a one-segment call reaches ``lsf_single_rule`` (at least its mean unit tiles; never where the record
        leaves the rule None).  On one segment the :lsf copy walks the units of the plain program in the same order and
        selects the same ids; on the FP8 sm_103a wide program it measures faster from 64 mean unit tiles on and slower
        below, so ``select_fp8_program`` runs it on a one-segment wide call only within the rule."""
        rule = self.lsf_single_rule
        return (
            rule is not None
            and int(num_segments) == 1
            and self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
            >= rule[0]
        )

    def tile_unroll_for(self, kind: str, top_k: int, snake: bool, block_q: int) -> int:
        rule = self.tile_unroll_rule
        if rule is None or (bool(rule[2]) and snake):
            return self.tile_unroll_default
        lookup = (
            "split_wide"
            if kind == "split" and int(block_q) == self.block_q_wide
            else kind
        )
        return (
            self.tile_unroll_factor
            if lookup in rule[0] and int(top_k) >= int(rule[1])
            else self.tile_unroll_default
        )

    def desc_walk_for(self, physical_kind: str, sampled: bool) -> bool:
        """The call runs the descending-walk twin of its scan program (key kind ``<physical_kind>_desc``, the same unit,
        split, pair, tile unroll and fit form): ``desc_walk_kinds`` names the physical kind for the call's sampling
        mode (``select_program`` takes the twin only where the registry has it)."""
        rule = self.desc_walk_kinds
        return rule is not None and physical_kind in rule[1 if sampled else 0]

    def sample_fit_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        cand_cap: int,
        physical_kind: Optional[str] = None,
    ) -> bool:
        """Sampled first threshold on buffer-fitting units: the mean unit is at most ``sample_fit_max_mean_tiles`` and
        the mean segment offset leaves units that fit the capacity.  A program kind outside ``sample_fit_kinds`` has no
        sampled-fit program registered and runs its plain program on every call (``physical_kind`` None: no kind gate)."""
        if physical_kind is not None and physical_kind not in self.sample_fit_kinds:
            return False
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        offset_tiles = self.offset_tiles(num_queries, num_keys, num_segments, ratio)
        fit_tiles = int(cand_cap) // self.tile_keys - 1
        return (
            mean_tiles <= self.sample_fit_max_mean_tiles
            and offset_tiles + 1 <= fit_tiles
        )


def physical_kind(kind: str, unit: str) -> str:
    """Name of the physical scan program kind: the split program is one per unit geometry."""
    return f"split_{unit}" if kind == "split" else kind


def scan_key(
    physical: str,
    tile_unroll: int,
    snake: bool,
    sample_fit: bool,
    lsf: bool = False,
    form: str = BF16,
) -> str:
    """``:lsf``: the longest-first (LSF) copy, which hands out the program's tickets longest segment first; ``form``
    ``"fp8"``: the FP8 program, tagged ``:fp8`` before the ``:lsf`` suffix."""
    key = f"scan:{physical}:u{int(tile_unroll)}:s{int(bool(snake))}:f{int(bool(sample_fit))}"
    if form == FP8:
        key += ":fp8"
    elif form != BF16:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    return key + ":lsf" if lsf else key


class ScanKeyFields(NamedTuple):
    """The fields of a scan program key (``scan_key``)."""

    physical: str
    tile_unroll: int
    snake: bool
    sample_fit: bool
    form: str
    lsf: bool


_SCAN_KEY = re.compile(r"scan:([a-z0-9_]+):u([0-9]+):s([01]):f([01])(:fp8)?(:lsf)?")


def scan_key_fields(key: str) -> Optional[ScanKeyFields]:
    """The fields of the scan program key ``key``; None for any other key."""
    match = _SCAN_KEY.fullmatch(key)
    if match is None:
        return None
    physical, unroll, snake, fit, fp8, lsf = match.groups()
    return ScanKeyFields(
        physical, int(unroll), snake == "1", fit == "1", FP8 if fp8 else BF16, bool(lsf)
    )


def scan_key_form(key: str) -> Optional[str]:
    """The input form (``FORMS``) of the scan program key ``key``; None for any other key."""
    fields = scan_key_fields(key)
    return None if fields is None else fields.form


# the fields of a ``POLICY`` record the FP8 form does not take over unless its FP8 record names them: the top_k-keyed
# wide-first floor was measured on the BF16 programs only, so the FP8 plans keep the programs of their own records
FP8_UNSET_FIELDS = {"wide_first_k_rule": None}


def fp8_policy_record(record: dict, fp8_record: dict) -> dict:
    """The FP8 form's dispatch record of an architecture: its record ``record`` (``POLICY[arch]``) with the fields of
    its FP8 record ``fp8_record`` (``POLICY_FP8[arch]``) replaced and the ``FP8_UNSET_FIELDS`` it does not name unset.
    ``fp8_record`` names its own ``static_scan_kinds`` (the FP8 programs never keep the BF16 programs' static kinds) and
    no field outside ``POLICY_FIELDS``."""
    if (
        not isinstance(fp8_record, dict)
        or "static_scan_kinds" not in fp8_record
        or not set(fp8_record) <= set(POLICY_FIELDS)
    ):
        raise ValueError(
            f"an FP8 dispatch record must name static_scan_kinds and only fields of {POLICY_FIELDS}, got {fp8_record!r}"
        )
    return {**record, **FP8_UNSET_FIELDS, **fp8_record}


def fp8_scan_programs(scan_keys: Collection[str]) -> dict:
    """The plain FP8 scan programs of a registry's keys: ``{(physical kind, lsf): {tile unroll: key}}`` over the FP8
    keys in the plain unit order and the sampled-fit form (``s0`` / ``f1``).  A snake copy (``s1``) is not indexed
    here: ``select_fp8_program`` swaps it in for its plain program where the decision marks the call snake."""
    programs: dict = {}
    for key in scan_keys:
        fields = scan_key_fields(key)
        if (
            fields is None
            or fields.form != FP8
            or fields.snake
            or not fields.sample_fit
        ):
            continue
        programs.setdefault((fields.physical, fields.lsf), {})[fields.tile_unroll] = key
    return programs


def stage_slug(key: str) -> str:
    """Stage / template name of a program key (``scan:narrow:u1:s0:f0`` -> ``scan_narrow_u1_s0_f0``)."""
    return key.replace(":", "_")


@dataclass(frozen=True)
class ProgramChoice:
    """Everything the host decides for one call from host-known integers."""

    num_queries: int
    num_keys: int
    num_segments: int
    ratio: int
    top_k: int
    max_seqlen_k: Optional[
        int
    ]  # the call's bound on every key segment (None = not given); read by the longest-first (LSF) rule
    grid: int  # requested persistent grid (SM count or the explicit knob)
    grid_ctas: int  # launched persistent grid (even for the CTA-pair program)
    sample_tiles_max: int
    sample_shift_permille: int
    far_sample: int
    cand_cap_multiplier: int
    cand_cap: int
    first_cap: int
    check_period: int
    block_q: int
    unit: str
    n_split: int
    split: bool
    pair: bool
    kind: str
    physical_kind: str
    snake: bool
    tile_unroll: int
    sample_fit: bool
    lsf: bool  # the scan program's longest-first segment order copy (lever LSF, scan key suffix :lsf)
    scan_key: str
    merge_key: Optional[str]  # MERGE_KEY on a split call, else None
    scan_entries: int  # int64 candidate entries of the persistent buffers
    staging_entries: int  # int64 split-staging entries behind them (0 without a split)
    workspace_bytes: int

    @property
    def keys(self) -> tuple:
        """Program keys in launch order: the scan, then the plain merge of a split call."""
        return (self.scan_key,) + (
            (self.merge_key,) if self.merge_key is not None else ()
        )

    def as_dict(self) -> dict:
        return asdict(self)


def select_program(
    policy: DispatchPolicy,
    *,
    num_queries: int,
    num_keys: int,
    num_segments: int,
    ratio: int,
    top_k: int,
    grid: int,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    scan_keys: Optional[Collection[str]] = None,
) -> ProgramChoice:
    """The host dispatch of one call, in the kernel launcher's own order.

    The keyword knobs change the internal partition and scheduling only
    (every admissible value selects the same ids per row; the
    order of a row's entries may differ); ``None`` takes
    the policy's default (``sample_shift_permille``: ``sample_shift_for`` of
    the call's ``top_k``).  ``far_sample`` (``0`` or ``1``) is the far-sample
    first threshold switch of the ascending-tile-order scan programs (the
    descending-order and wide programs ignore it).  ``max_seqlen_k`` (the call's bound on every key
    segment, optional) lets the longest-first rule (lever LSF) tell a packed call of equal key segments, which runs
    the program without the longest-first order; the selected sets are identical with and without it.  With the
    sampled first threshold off (``sample_tiles_max=0``) the candidate capacity follows the record's ``*_unsampled``
    values.  A call whose plan splits the key ranges runs the plain split merge (``merge_key`` = ``MERGE_KEY``) after
    its scan; no call runs an ordering pass.
    ``scan_keys`` (the scan program keys the architecture registers, ``PROGRAM_KEYS[arch]``; None: no registry, the
    preference stays off) bounds the unpaired-l6 preference (``l6_solo_rule``, ``l6_solo_unsampled_max_queries``): it replaces the unit and pair choice of a call that does not
    split with the unpaired l6 unit only where that unit would not split and its program key is registered.  A record's
    ``wide_pair_snake_only`` (sampled calls) drops the CTA pair of a wide unit that would not walk the snake order where the
    unpaired wide program is in ``scan_keys`` (without a registry it stays off, as the l6 preference); ``split_wave_scope``
    limits the two-way wave split.  A record's ``desc_walk_kinds`` runs the descending-walk twin of the decided program
    (key kind ``<kind>_desc``) where it names the kind for the call's sampling mode and ``scan_keys`` has the twin.
    Where the record sets ``lsf_programs``, a packed call of unequal key segments runs the longest-first copy of the
    decided program (key suffix ``:lsf``) unless ``scan_keys`` has the program but not the copy; every rule that asks
    ``scan_keys`` for a program looks up its key without the suffix, so the copy never changes the plan.
    ``snake_packed_rule`` keeps packed calls of few mean unit tiles over more than a few
    waves in the plain unit order (``snake_for``), and ``snake_l6_fallback_waves`` lets a sampling-off call the l6 preference moves run the plain-order
    l6 program where the snake-order one is not registered, within that many waves of the planned program's persistent
    grid (``snake_l6_fallback_for``).
    """
    T, Tkv, S, r, K = (
        int(num_queries),
        int(num_keys),
        int(num_segments),
        int(ratio),
        int(top_k),
    )
    if max_seqlen_k is not None and (
        int(max_seqlen_k) > Tkv or int(max_seqlen_k) * max(1, S) < Tkv
    ):
        raise ValueError(
            f"max_seqlen_k={int(max_seqlen_k)} is not an upper bound of every key segment: {Tkv} keys in {S} segments"
        )
    grid = int(grid)
    if grid < 1:
        raise ValueError("grid_ctas must be >= 1")
    tiles = (
        policy.sample_tiles_for(T, Tkv, S, r)
        if sample_tiles_max is None
        else int(sample_tiles_max)
    )
    sampled = tiles > 0
    shift = (
        policy.sample_shift_for(K)
        if sample_shift_permille is None
        else int(sample_shift_permille)
    )
    far = policy.far_sample if far_sample is None else int(far_sample)
    if far not in (0, 1):
        raise ValueError(f"far_sample must be 0 or 1, got {far_sample!r}")
    mult = (
        policy.cand_cap_multiplier_for(T, Tkv, S, r, K, sampled=tiles > 0)
        if candidate_multiplier is None
        else int(candidate_multiplier)
    )
    block_q = policy.unit_block_q_for(T, Tkv, S, r, grid, K)
    n_split = policy.n_split_for(
        T, Tkv, S, r, grid, block_q, sampled=tiles > 0, top_k=K
    )
    split = n_split > 1
    pair = (not split) and policy.pair_for(T, Tkv, S, r, grid, K, block_q)
    # lever LSF: the longest-first segment order pays off only where the segments differ, so the copy that
    # carries it runs on packed calls whose key segments are not all equal (one segment, or num_keys == S x the
    # call's max_seqlen_k, runs the program without it; without max_seqlen_k the host cannot tell and runs the copy).
    # The copy never changes the plan: every rule below that asks ``scan_keys`` for a program looks up its key without
    # the suffix, and the decided program takes the suffix last, where the registry has its copy
    lsf = (
        policy.lsf_programs
        and S >= LSF_MIN_SEGMENTS
        and not (max_seqlen_k is not None and Tkv == S * int(max_seqlen_k))
    )
    cap = policy.candidate_capacity(K, mult)
    # wide_pair_snake_only: a sampled call keeps the CTA pair of the wide unit only where the pair walks the snake order;
    # elsewhere it runs the unpaired wide program (same unit, no split) when that program is in ``scan_keys`` (no registry: off)
    if (
        pair
        and tiles > 0
        and policy.wide_pair_snake_only
        and block_q == policy.block_q_wide
        and not policy.snake_for(
            policy.program_kind(block_q, False, True), T, Tkv, S, r, sampled, grid
        )
    ):
        wide_snake = policy.snake_for("wide", T, Tkv, S, r, sampled, grid)
        wide_key = scan_key(
            physical_kind("wide", "wide"),
            policy.tile_unroll_for("wide", K, wide_snake, block_q),
            wide_snake,
            policy.sample_fit_for(T, Tkv, S, r, cap, physical_kind("wide", "wide")),
        )
        if scan_keys is not None and wide_key in scan_keys:
            pair = False
    # l6_solo_rule / l6_solo_unsampled_max_queries: a call that does not split takes the l6 unit without a CTA pair where
    # ``l6_solo_for`` prefers it (sampling off: every call, or with the query bound a CTA-pair l6 plan of at most that many
    # queries; sampled: several segments or short units), the l6 unit would not split either (the split rule
    # alone, before the registration guard of ``n_split_for``) and the program it picks is in ``scan_keys``; else the plan above.
    # snake_l6_fallback_waves: a sampling-off call whose snake-order l6 program is not registered takes the plain-order
    # one instead where the planned program needs at most that many waves of its persistent grid
    solo_plain = False
    if (
        not split
        and (block_q, pair) != (policy.block_q_l6, False)
        and policy.l6_solo_for(
            T,
            Tkv,
            S,
            r,
            sampled=tiles > 0,
            pair_l6=(block_q, pair) == (policy.block_q_l6, True),
        )
        and policy.split_rule_for(T, Tkv, S, r, grid, policy.block_q_l6, tiles > 0) == 1
    ):
        solo_snake = policy.snake_for("l6", T, Tkv, S, r, sampled, grid)
        solo_fit = policy.sample_fit_for(T, Tkv, S, r, cap, "l6")
        solo_key = scan_key(
            "l6",
            policy.tile_unroll_for("l6", K, solo_snake, policy.block_q_l6),
            solo_snake,
            solo_fit,
        )
        if (
            scan_keys is not None
            and solo_key not in scan_keys
            and solo_snake
            and policy.snake_l6_fallback_for(T, block_q, pair, grid, sampled)
        ):
            solo_key = scan_key(
                "l6",
                policy.tile_unroll_for("l6", K, False, policy.block_q_l6),
                False,
                solo_fit,
            )
            solo_plain = True
        if scan_keys is not None and solo_key in scan_keys:
            block_q, pair = policy.block_q_l6, False
        else:
            solo_plain = False
    kind = policy.program_kind(block_q, split, pair)
    unit = policy.unit_name(block_q)
    snake = False if solo_plain else policy.snake_for(kind, T, Tkv, S, r, sampled, grid)
    unroll = policy.tile_unroll_for(kind, K, snake, block_q)
    grid_ctas = grid - grid % 2 if pair else grid
    physical = physical_kind(kind, unit)
    tiles, shift = sample_class_for(
        policy,
        physical,
        T,
        Tkv,
        S,
        r,
        tiles,
        shift,
        tiles_knob=sample_tiles_max is not None,
        shift_knob=sample_shift_permille is not None,
    )
    fit = policy.sample_fit_for(T, Tkv, S, r, cap, physical)
    if check_period is None:
        period = policy.check_period_for(K, cap, kind)
    else:
        period = int(check_period)
        limit = policy.check_period_limit(K, cap)
        if not 1 <= period <= limit:
            raise ValueError(
                f"check_period must be in [1, {limit}] for top_k={K}, cand_cap={cap}, got {check_period!r}"
            )
    scan_entries = grid_ctas * block_q * cap
    staging = T * n_split * K if split else 0
    key = scan_key(physical, unroll, snake, fit)
    # desc_walk_kinds: the descending-walk twin of the decided program, where the record names its kind for the call's
    # sampling mode and the registry has it (no registry: off); the plan, the knobs and the workspace stay
    if scan_keys is not None and policy.desc_walk_for(physical, tiles > 0):
        desc_key = scan_key(f"{physical}_desc", unroll, snake, fit)
        if desc_key in scan_keys:
            key = desc_key
    # lever LSF: the decided program runs its longest-first copy, but where the registry has the program and not its
    # copy it runs in the argument order (sm_107a registers the copies of its two l6 ticket programs only, sm_100a and
    # sm_103a that of every program); without a registry, or for a key the registry has in neither form (which then
    # fails closed by its :lsf name in program_for), the copy stays
    if (
        lsf
        and scan_keys is not None
        and key in scan_keys
        and key + ":lsf" not in scan_keys
    ):
        lsf = False
    if lsf:
        key += ":lsf"
    return ProgramChoice(
        num_queries=T,
        num_keys=Tkv,
        num_segments=S,
        ratio=r,
        top_k=K,
        max_seqlen_k=None if max_seqlen_k is None else int(max_seqlen_k),
        grid=grid,
        grid_ctas=grid_ctas,
        sample_tiles_max=tiles,
        sample_shift_permille=shift,
        far_sample=far,
        cand_cap_multiplier=mult,
        cand_cap=cap,
        first_cap=cap,
        check_period=period,
        block_q=block_q,
        unit=unit,
        n_split=n_split,
        split=split,
        pair=pair,
        kind=kind,
        physical_kind=physical,
        snake=snake,
        tile_unroll=unroll,
        sample_fit=fit,
        lsf=lsf,
        scan_key=key,
        merge_key=MERGE_KEY if split else None,
        scan_entries=scan_entries,
        staging_entries=staging,
        workspace_bytes=(scan_entries + staging) * policy.candidate_entry_bytes
        + TICKET_TAIL_BYTES,
    )


def fp8_program_choice(
    policy: DispatchPolicy,
    choice: ProgramChoice,
    key: str,
    n_split: Optional[int] = None,
) -> ProgramChoice:
    """``choice`` run by the FP8 scan program ``key``: the unit geometry, the split and the tile unroll follow the key,
    an FP8 program has no CTA pair (the launched grid is the requested one), and the capacity, check period and sampling
    knobs of ``choice`` stay; a split key runs the plain split merge.  ``n_split``: the key-range splits of a split key
    (default: the plan's, at least 2); the workspace bound follows the unit geometry and the split."""
    fields = scan_key_fields(key)
    if fields is None or fields.form != FP8:
        raise ValueError(f"{key!r} is not an FP8 scan program key")
    physical = fields.physical
    split = physical.startswith("split_")
    unit = physical[len("split_") :] if split else physical
    if unit not in UNITS:
        raise ValueError(f"no FP8 program runs the scan kind of {key!r}")
    block_q = {
        "narrow": policy.block_q_narrow,
        "wide": policy.block_q_wide,
        "l6": policy.block_q_l6,
    }[unit]
    if not split:
        n = 1
    elif n_split is None:
        n = max(2, int(choice.n_split))
    else:
        n = int(n_split)
        if n < 2:
            raise ValueError(f"a split program needs n_split >= 2, got {n_split!r}")
    grid_ctas = int(choice.grid)
    scan_entries = grid_ctas * block_q * int(choice.cand_cap)
    staging = int(choice.num_queries) * n * int(choice.top_k) if split else 0
    return replace(
        choice,
        grid_ctas=grid_ctas,
        block_q=block_q,
        unit=unit,
        n_split=n,
        split=split,
        pair=False,
        kind="split" if split else unit,
        physical_kind=physical,
        snake=fields.snake,
        tile_unroll=fields.tile_unroll,
        sample_fit=fields.sample_fit,
        lsf=fields.lsf,
        scan_key=key,
        merge_key=MERGE_KEY if split else None,
        scan_entries=scan_entries,
        staging_entries=staging,
        workspace_bytes=(scan_entries + staging) * policy.candidate_entry_bytes
        + TICKET_TAIL_BYTES,
    )


def fallback_knobs(
    policy: DispatchPolicy, choice: ProgramChoice, physical: str, call: dict
) -> ProgramChoice:
    """``choice`` with the sampled first threshold and the check period of the scan kind ``physical`` that runs in place
    of the decided one (``FP8_KIND_FALLBACK``): the record's ``sample_class_rule`` and ``check_period_kind_overrides``
    key on the program kind, so a wide decision's wide-unit shift and check period do not carry over to the l6
    program.  The knobs ``call`` sets stay."""
    T, Tkv, S, r, K = (
        choice.num_queries,
        choice.num_keys,
        choice.num_segments,
        choice.ratio,
        choice.top_k,
    )
    tiles_knob = call.get("sample_tiles_max")
    shift_knob = call.get("sample_shift_permille")
    tiles, shift = sample_class_for(
        policy,
        physical,
        T,
        Tkv,
        S,
        r,
        policy.sample_tiles_for(T, Tkv, S, r)
        if tiles_knob is None
        else int(tiles_knob),
        policy.sample_shift_for(K) if shift_knob is None else int(shift_knob),
        tiles_knob=tiles_knob is not None,
        shift_knob=shift_knob is not None,
    )
    period = choice.check_period
    if call.get("check_period") is None:
        kind = "split" if physical.startswith("split_") else physical
        period = policy.check_period_for(K, choice.cand_cap, kind)
    return replace(
        choice, sample_tiles_max=tiles, sample_shift_permille=shift, check_period=period
    )


def select_fp8_program(
    policy: DispatchPolicy,
    *,
    scan_keys: Collection[str],
    fp8_programs: Optional[dict] = None,
    **call: Any,
) -> ProgramChoice:
    """The host dispatch of one FP8 call: the decision procedure of ``select_program`` (``call``: its keywords) under the
    architecture's FP8 record ``policy`` (``fp8_policy_record``) with the registry's ``scan_keys``, run by the FP8
    program (``fp8_program_choice``) of the decided physical kind -- of its ``FP8_KIND_FALLBACK`` kind, with that
    kind's sampled first threshold and check period (``fallback_knobs``), where no FP8 program of the decided kind is
    registered -- in the decided segment order (``:lsf`` where its FP8 copy is registered; also a one-segment call of
    the wide unit within the record's ``lsf_single_rule``, ``DispatchPolicy.lsf_single_fits``), with the decided tile
    unroll where an FP8 program of it is registered, else the smallest registered one, and in the snake unit order
    where the decision marks it, the registry has the snake copy (``s1``) of that program and the call stays within
    the record's ``snake_cap_rule`` (``DispatchPolicy.snake_copy_fits``).  ``fp8_programs``:
    ``fp8_scan_programs(scan_keys)``, made once by the caller.  Raises ``NotImplementedError`` when no FP8 program
    serves the kind."""
    choice = select_program(policy, scan_keys=scan_keys, **call)
    programs = fp8_scan_programs(scan_keys) if fp8_programs is None else fp8_programs
    physical = choice.physical_kind
    if not any(kind == physical for kind, _lsf in programs):
        physical = FP8_KIND_FALLBACK.get(physical, physical)
    # lsf_single_rule: a one-segment call of the wide unit runs the :lsf copy within the rule (the plan leaves it off)
    lsf = choice.lsf or (
        physical == "wide"
        and (physical, True) in programs
        and policy.lsf_single_fits(
            choice.num_queries, choice.num_keys, choice.num_segments, choice.ratio
        )
    )
    unrolls = programs.get((physical, lsf)) or programs.get((physical, False))
    if not unrolls:
        raise NotImplementedError(
            f"the FP8 dispatch selected the {physical!r} scan kind, which no registered FP8 program serves "
            f"(registered FP8 kinds: {sorted({kind for kind, _lsf in programs})})"
        )
    key = unrolls.get(choice.tile_unroll, unrolls[min(unrolls)])
    if physical != choice.physical_kind:
        choice = fallback_knobs(policy, choice, physical, call)
    plain = fp8_program_choice(policy, choice, key)
    twin = key.replace(":s0:", ":s1:", 1)
    if (
        choice.snake
        and twin in scan_keys
        and policy.snake_copy_fits(
            plain.num_queries, plain.num_segments, plain.block_q, plain.grid_ctas
        )
    ):
        return fp8_program_choice(policy, choice, twin)
    return plain


# ---------------------------------------------------------------------------
# Candidate blocks (``scan_blocks`` programs)
# ---------------------------------------------------------------------------

# A ``scan_blocks`` program is a scan program that also writes each query's candidate blocks: ``CB_SLOTS`` int32 slots
# per row naming blocks of ``CB_BLOCK`` keys (block ``b`` holds the segment-local keys ``8 b .. 8 b + 7``).  The
# unsplit, unpaired BF16 narrow and l6 units carry it (``scan_blocks:<narrow|l6>:u<n>:s0:f<fit>``) and an architecture
# may register twins of the CTA-pair l6 unit (``scan_blocks:pair_l6:u1:s0:f0[:lsf]``, ``CB_TWIN_FORMS``), so a call
# runs the ``scan_blocks`` program of the unit ``CB_KIND_FALLBACK`` maps the decided physical kind to (a wide unit, a
# CTA pair and a key-range split run their unit geometry's unpaired, unsplit program; a split call runs narrow, the
# unit of more work units per query), with ``n_split = 1`` -- unless a twin serves the call
# (``CB_TWIN_RULE``).  ``top_k`` is at most ``CB_MAX_TOP_K``: then every returned id lies in a returned block.
# A call whose longest key segment holds at most ``CB_SLOTS`` blocks (``cb_block_cap`` 0) keeps every block of every
# row, so no block score decides a row: it runs the plain call's plan as dispatched (any unit, pair, split with its
# merge, and order) and the backend writes each row's ``0 .. nb - 1`` and ``-1`` tail on the device before it (stage
# ``CB_IOTA_STAGE``; ``CandidateBlocksChoice.iota``).
CB_ROLE = "scan_blocks"
CB_IOTA_STAGE = "cb_iota"
CB_SLOTS = 2048
CB_BLOCK = 8
CB_MAX_TOP_K = CB_SLOTS - 1
CB_ENTRY_BYTES = 8
CB_KIND_FALLBACK = {
    "narrow": "narrow",
    "l6": "l6",
    "wide": "l6",
    "pair_narrow": "narrow",
    "pair_l6": "l6",
    "pair_wide": "l6",
    "split_narrow": "narrow",
    "split_l6": "narrow",
    "split_wide": "narrow",
}
# The twins: plain scan programs, as ``(physical kind, lsf)``, whose own ``scan_blocks`` copy an architecture may
# register beside the units of ``CB_KIND_FALLBACK``, each in its no-sample-fit form only (``f0``): the CTA-pair l6
# program in both segment orders (sm_103a).  A twin keeps its plain program's query layout: each CTA of a pair scores
# its own ``block_q`` queries in its own query slots, so a twin's block entries are those of its launched grid.  A call
# with block entries (``cb_block_cap`` > 0) inside ``CB_TWIN_RULE`` whose plain plan runs one of them in the plain unit
# order (``s0``) runs its twin where the registry has it (``cb_twin_key``): the plain plan as decided -- unit, CTA
# pair, launched grid (even for the pair), segment order, sampled first threshold, check period, workspace and the rest
# of the call -- with the twin's key.  On sm_103a's record no call of the package tests' dense grid inside
# ``CB_TWIN_RULE`` plans that program, so the calls the rule admits take the wide route below.  A sample-fit plan, a
# plan of any other program, a call outside ``CB_TWIN_RULE`` and a plan whose twin the registry lacks run the
# ``CB_KIND_FALLBACK`` unit as before.
CB_TWIN_FORMS = frozenset({("pair_l6", False), ("pair_l6", True)})
# the units of ``CB_KIND_FALLBACK``, whose programs in the argument segment order ``cb_scan_programs`` indexes
CB_UNITS = ("narrow", "l6")
# The calls a twin serves (``cb_twin_rule_holds``): ``(min top_k, max top_k, min mean unit tiles, max mean unit tiles,
# min queries, sampled calls only)``, a window of ``top_k``, mean unit tiles and query count around the calls the twins
# were timed on.  Each twin was timed at kernel level, as a forced program on the call's geometry against the unpaired
# l6 ``scan_blocks`` program, on GB300 (sm_103a, 152 SMs), with equal bits, at ``top_k`` 512 with the sampled first
# threshold, on three calls whose plain plan runs the wide program: the CTA-pair twin on 16384 queries at the tail of
# 65536 keys (448 mean unit tiles) and of 262144 keys (1984) and on the 8-segment packed call of 16231 queries over
# 268757 keys (254.5), its longest-first copy on that packed call only; each ran faster there.  The tile floor and cap
# are the smallest and largest mean unit tiles of those calls and the query floor their smallest query count (a call of
# fewer queries fills fewer waves of the grid).  The rule extends the timed calls, untimed, to the mean unit tiles
# between them, to more queries (more waves of the same units) and to other segment layouts; no twin was timed at
# another ``top_k``, with the sampled first threshold off, on a plan of the CTA-pair l6 program itself or on another
# device.  Outside the window the CTA-pair twin was timed at 16384 queries only: slower at 96 mean unit tiles and faster
# at 192; a sample-fit CTA-pair program that is not shipped was slower at 64, on a call that keeps every block.
CB_TWIN_RULE = (512, 512, 254.5, 1984.0, 16231, True)
# The CTA-pair twin of a wide plan: a call with block entries inside ``CB_TWIN_RULE`` whose plan runs the unsplit wide
# program (either walk direction) on a grid of at least two CTAs, at a ``top_k`` from the one at which the record's CTA
# pair serves the l6 unit (``pair_units``, ``pair_rule[2]``) up to ``wide_first_k_rule[0]``, with no sample fit for the
# CTA-pair l6 unit and no preference for the l6 unit without a pair (``l6_solo_for``), runs the CTA-pair l6 twin in the
# plan's segment order (``scan_blocks:pair_l6:u1:s0:f0``, ``:lsf`` for a longest-first plan; ``cb_pair_twin_key``)
# where the registry has it, on the CTA-pair l6 unit's geometry (``block_q_l6``, the requested grid made even) with
# that unit's sampled first threshold and check period (``fallback_knobs``), on a call where the rule asks for the
# sampled first threshold only when that unit's threshold is on too; else the ``CB_KIND_FALLBACK`` unit (l6).  On
# sm_103a (``pair_rule[2]`` = ``wide_first_k_rule[0]`` = 512) these are the wide plans of ``top_k`` 512, the three
# timed calls among them.  The plain dispatch plans the CTA-pair l6 unit up to ``pair_rule[1]`` (1024) mean unit tiles
# only; above it the twin runs where the plain dispatch never plans that unit, timed on the call of 1984 mean unit
# tiles alone.  Only sm_103a registers the twins.
_CB_KEY = re.compile(r"scan_blocks:(narrow|l6|pair_l6):u([0-9]+):s0:f([01])(:lsf)?")


def cb_block_cap(num_keys: int, max_seqlen_k: Optional[int] = None) -> int:
    """Block entries per query slot of a ``scan_blocks`` launch: ``ceil(W / 8) - 1`` older blocks of the longest key
    segment ``W`` (``max_seqlen_k`` when given, else ``Tkv``) when ``ceil(W / 8) > CB_SLOTS``, else 0 (every row keeps
    all its blocks and stores none: the call runs the plain plan, ``CandidateBlocksChoice.iota``).  ``max_seqlen_k``
    must bound every key segment of the call: a row with more older blocks than this traps on the device."""
    w = int(num_keys if max_seqlen_k is None else max_seqlen_k)
    blocks = (w + CB_BLOCK - 1) // CB_BLOCK
    return blocks - 1 if blocks > CB_SLOTS else 0


@dataclass(frozen=True)
class CandidateBlocksChoice:
    """The plan of one call that also returns its candidate blocks: ``choice`` runs the ``scan_blocks`` program
    (``choice.scan_key``) with its workspace (``choice.workspace_bytes``: the candidate buffers and the ticket tail);
    ``blk_cap`` block entries per query slot follow it (``blk_bytes`` = ``grid_ctas x block_q x blk_cap x 8``), so
    the call's workspace is ``workspace_bytes`` = ``choice.workspace_bytes + blk_bytes``.  The ``[T, CB_SLOTS]``
    int32 output is the caller's and not part of it.  ``iota`` (``blk_cap`` 0: every row keeps all its blocks):
    ``choice`` is the plain call's plan (``select_program``: its scan and merge programs) with its
    workspace, no block entries, and the backend writes the rows before its launches (``CB_IOTA_STAGE``)."""

    choice: ProgramChoice
    blk_cap: int
    blk_bytes: int
    workspace_bytes: int
    iota: bool = False

    @property
    def keys(self) -> tuple:
        return self.choice.keys

    @property
    def blocks_at(self) -> int:
        """Byte offset of the block entries in the workspace (behind the ticket tail)."""
        return int(self.choice.workspace_bytes)

    def as_dict(self) -> dict:
        return {
            **self.choice.as_dict(),
            "blk_cap": self.blk_cap,
            "blk_bytes": self.blk_bytes,
            "cb_workspace_bytes": self.workspace_bytes,
            "cb_iota": self.iota,
        }


def cb_scan_programs(scan_keys: Collection[str]) -> dict:
    """The ``scan_blocks`` programs of the ``CB_KIND_FALLBACK`` units an architecture registers: ``(kind, sample_fit)
    -> {tile_unroll: key}``, in the argument segment order (the twins of ``CB_TWIN_FORMS`` are looked up by their own
    key, ``cb_twin_key``)."""
    programs: dict = {}
    for key in scan_keys:
        match = _CB_KEY.fullmatch(key)
        if match is not None and match[1] in CB_UNITS and match[4] is None:
            programs.setdefault((match[1], match[3] == "1"), {})[int(match[2])] = key
    return programs


def cb_twin_key(choice: ProgramChoice) -> Optional[str]:
    """The ``scan_blocks`` twin of the plain plan ``choice``'s own scan program (``CB_TWIN_FORMS``): its scan key with
    the ``scan_blocks`` role (``scan:pair_l6:u1:s0:f0:lsf`` -> ``scan_blocks:pair_l6:u1:s0:f0:lsf``); None for any
    other plan (a split, sample-fit, snake-order or descending-walk program, or a program of another form).
    ``select_cb_program`` runs it only on a call inside ``CB_TWIN_RULE``."""
    if (
        choice.split
        or choice.snake
        or choice.sample_fit
        or (choice.physical_kind, bool(choice.lsf)) not in CB_TWIN_FORMS
    ):
        return None
    key = scan_key(choice.physical_kind, choice.tile_unroll, False, False, choice.lsf)
    return CB_ROLE + key[len("scan") :] if key == choice.scan_key else None


def cb_twin_rule_holds(policy: DispatchPolicy, choice: ProgramChoice) -> bool:
    """Whether the plain plan ``choice`` is a call of ``CB_TWIN_RULE``: its ``top_k`` and its mean unit tiles
    (``DispatchPolicy.mean_unit_tiles``) in the rule's ranges, at least the rule's queries and, where the rule asks for
    it, the sampled first threshold on (``sample_tiles_max`` > 0)."""
    min_k, max_k, min_tiles, max_tiles, min_queries, sampled_only = CB_TWIN_RULE
    tiles = policy.mean_unit_tiles(
        choice.num_queries, choice.num_keys, choice.num_segments, choice.ratio
    )
    return (
        int(min_k) <= int(choice.top_k) <= int(max_k)
        and float(min_tiles) <= tiles <= float(max_tiles)
        and int(choice.num_queries) >= int(min_queries)
        and (not sampled_only or int(choice.sample_tiles_max) > 0)
    )


def cb_pair_twin_key(policy: DispatchPolicy, choice: ProgramChoice) -> Optional[str]:
    """The CTA-pair l6 twin of the plain plan ``choice`` when it runs the unsplit wide program on a grid of at least two
    CTAs, its ``top_k`` lies from the one at which the record's CTA pair serves the l6 unit (``pair_units``,
    ``pair_rule[2]``) up to ``wide_first_k_rule[0]``, the CTA-pair l6 unit has no sample fit and the l6 unit without a
    pair is not preferred (``l6_solo_for``):
    ``scan_blocks:pair_l6:u<n>:s0:f0`` at that unit's tile unroll, with ``:lsf`` for a longest-first plan; None for any
    other plan.  ``select_cb_program`` runs it only on a call inside ``CB_TWIN_RULE``."""
    T, Tkv, S, r = (
        choice.num_queries,
        choice.num_keys,
        choice.num_segments,
        choice.ratio,
    )
    rule = policy.pair_rule
    if (
        choice.physical_kind != "wide"
        or choice.split
        or int(choice.grid) < 2
        or rule is None
        or "l6" not in policy.pair_units
        or int(choice.top_k) < int(rule[2])
        or policy.wide_first_k_rule is None
        or int(choice.top_k) > int(policy.wide_first_k_rule[0])
        or policy.sample_fit_for(T, Tkv, S, r, choice.cand_cap, "pair_l6")
        or policy.l6_solo_for(
            T, Tkv, S, r, sampled=int(choice.sample_tiles_max) > 0, pair_l6=True
        )
    ):
        return None
    unroll = policy.tile_unroll_for("pair_l6", choice.top_k, False, policy.block_q_l6)
    return f"{CB_ROLE}:pair_l6:u{int(unroll)}:s0:f0" + (":lsf" if choice.lsf else "")


def select_cb_program(
    policy: DispatchPolicy,
    *,
    scan_keys: Collection[str],
    iota: bool = True,
    **call: Any,
) -> CandidateBlocksChoice:
    """The host dispatch of one call that also returns candidate blocks: the decision procedure of ``select_program``
    (``call``: its keywords), on a call inside ``CB_TWIN_RULE`` run by the twin of the decided program
    (``cb_twin_key``, ``CB_TWIN_FORMS``) where ``scan_keys`` registers it -- the decided plan with the twin's key --,
    else, for a wide plan, by the CTA-pair l6 twin where registered (``cb_pair_twin_key``) on the CTA-pair l6 unit's
    geometry with that unit's knobs; else by the ``scan_blocks`` program of the decided kind's ``CB_KIND_FALLBACK``
    unit,
    unsplit and unpaired on the requested grid, without the snake and longest-first orders, with that unit's sampled
    first threshold and check period where the kind changed (``fallback_knobs``), the sample-fit form of its unit and
    the decided tile unroll where its program is registered, else the smallest registered one.  The block entries
    follow ``cb_block_cap(num_keys, max_seqlen_k)``, per query slot of the launched grid.  When that is 0 (every row
    keeps all its blocks) the plan is
    ``select_program``'s own with ``iota`` set: the plain call's programs, workspace and launches, and the rows written
    by the backend (``iota=False``: the ``scan_blocks`` program writes them, the plan of the package before ``iota``,
    the same rows; measurements only).  Raises ``NotImplementedError`` when ``scan_keys`` registers no
    ``scan_blocks`` program of the unit (an ``iota`` plan: no ``scan_blocks`` program at all)."""
    choice = select_program(policy, scan_keys=scan_keys, **call)
    programs = cb_scan_programs(scan_keys)
    blk_cap = cb_block_cap(choice.num_keys, call.get("max_seqlen_k"))
    if iota and blk_cap == 0:
        if not programs:
            raise NotImplementedError(
                "the candidate-block call needs an architecture with scan_blocks programs: no registered scan_blocks "
                "program serves this registry"
            )
        return CandidateBlocksChoice(
            choice=choice,
            blk_cap=0,
            blk_bytes=0,
            workspace_bytes=int(choice.workspace_bytes),
            iota=True,
        )
    # the twin of the decided program, on a call with block entries inside CB_TWIN_RULE: the decided plan with the
    # twin's key, the block entries of its launched grid (even for the CTA pair) behind its workspace
    in_rule = blk_cap > 0 and cb_twin_rule_holds(policy, choice)
    twin = cb_twin_key(choice) if in_rule else None
    if twin is not None and twin in scan_keys:
        choice = replace(choice, scan_key=twin)
        blk_bytes = (
            int(choice.grid_ctas) * int(choice.block_q) * blk_cap * CB_ENTRY_BYTES
        )
        return CandidateBlocksChoice(
            choice=choice,
            blk_cap=blk_cap,
            blk_bytes=blk_bytes,
            workspace_bytes=int(choice.workspace_bytes) + blk_bytes,
        )
    # the CTA-pair twin of a wide plan inside CB_TWIN_RULE: the CTA-pair l6 unit on the requested grid made even, its
    # sampled first threshold (on where the rule asks for it) and check period, unsplit, in the plan's segment order
    pair_twin = cb_pair_twin_key(policy, choice) if in_rule else None
    knobs = (
        fallback_knobs(policy, choice, "pair_l6", call)
        if pair_twin is not None and pair_twin in scan_keys
        else None
    )
    if knobs is not None and (not CB_TWIN_RULE[5] or int(knobs.sample_tiles_max) > 0):
        choice = knobs
        grid_ctas = int(choice.grid) - int(choice.grid) % 2
        block_q = policy.block_q_l6
        unroll = policy.tile_unroll_for("pair_l6", choice.top_k, False, block_q)
        scan_entries = grid_ctas * block_q * int(choice.cand_cap)
        workspace = scan_entries * policy.candidate_entry_bytes + TICKET_TAIL_BYTES
        choice = replace(
            choice,
            grid_ctas=grid_ctas,
            block_q=block_q,
            unit="l6",
            n_split=1,
            split=False,
            pair=True,
            kind="pair_l6",
            physical_kind="pair_l6",
            snake=False,
            tile_unroll=int(unroll),
            sample_fit=False,
            scan_key=pair_twin,
            merge_key=None,
            scan_entries=scan_entries,
            staging_entries=0,
            workspace_bytes=workspace,
        )
        blk_bytes = grid_ctas * block_q * blk_cap * CB_ENTRY_BYTES
        return CandidateBlocksChoice(
            choice=choice,
            blk_cap=blk_cap,
            blk_bytes=blk_bytes,
            workspace_bytes=int(choice.workspace_bytes) + blk_bytes,
        )
    physical = CB_KIND_FALLBACK[choice.physical_kind]
    if physical != choice.physical_kind:
        choice = fallback_knobs(policy, choice, physical, call)
    T, Tkv, S, r = (
        choice.num_queries,
        choice.num_keys,
        choice.num_segments,
        choice.ratio,
    )
    block_q = policy.block_q_narrow if physical == "narrow" else policy.block_q_l6
    fit = policy.sample_fit_for(T, Tkv, S, r, choice.cand_cap, physical)
    unrolls = programs.get((physical, fit))
    if not unrolls:
        raise NotImplementedError(
            f"the candidate-block dispatch selected the {physical!r} unit (sample_fit={int(fit)}), which no registered "
            f"scan_blocks program serves (registered: {sorted(key for keys in programs.values() for key in keys.values())})"
        )
    unroll = choice.tile_unroll if choice.tile_unroll in unrolls else min(unrolls)
    grid_ctas = int(choice.grid)
    scan_entries = grid_ctas * block_q * int(choice.cand_cap)
    choice = replace(
        choice,
        grid_ctas=grid_ctas,
        block_q=block_q,
        unit=physical,
        n_split=1,
        split=False,
        pair=False,
        kind=physical,
        physical_kind=physical,
        snake=False,
        tile_unroll=unroll,
        sample_fit=fit,
        lsf=False,
        scan_key=unrolls[unroll],
        merge_key=None,
        scan_entries=scan_entries,
        staging_entries=0,
        workspace_bytes=scan_entries * policy.candidate_entry_bytes + TICKET_TAIL_BYTES,
    )
    blk_bytes = grid_ctas * block_q * blk_cap * CB_ENTRY_BYTES
    return CandidateBlocksChoice(
        choice=choice,
        blk_cap=blk_cap,
        blk_bytes=blk_bytes,
        workspace_bytes=int(choice.workspace_bytes) + blk_bytes,
    )


# ---------------------------------------------------------------------------
# Candidate-restricted top-k (the ``cand`` program)
# ---------------------------------------------------------------------------

# A ``cand`` program ranks, per query, only the visible keys of the 8-key blocks its row of ``candidate_blocks`` names
# (any order, ``-1`` anywhere, at most ``CAND_WIDTH_MAX`` slots), so its work per row is bounded by its candidate
# keys.  Persistent CTAs (one per SM) draw units of ``CAND_UNIT_Q`` queries from a self-resetting 16-byte ticket word;
# each CTA keeps a double-buffered list of each unit query's valid blocks (``CAND_LIST_BYTES``) and a candidate buffer
# of ``cand_cap`` packed entries per unit query, compacted every ``check_period`` steps.  The bound depends on the
# grid and ``top_k`` only, never on ``T``, ``Tkv``, the segments or the candidate content.
CAND_ROLE = "cand"
CAND_KEY = "cand"
CAND_WIDTH_MAX = 2048
CAND_MAX_TOP_K = 2048
CAND_UNIT_Q = 4
CAND_LIST_BYTES = 2 * CAND_UNIT_Q * CAND_WIDTH_MAX * 4
CAND_TICKET_BYTES = 16
CAND_CAP_FLOOR = 4096


def cand_capacity(top_k: int) -> int:
    """Candidate entries per unit query of a ``cand`` launch: ``max(CAND_CAP_FLOOR, 4 top_k)``."""
    return max(CAND_CAP_FLOOR, 4 * int(top_k))


def cand_check_period(cand_cap: int, top_k: int) -> int:
    """Steps between two settles of a ``cand`` launch: ``max(1, min(32, (cand_cap - top_k) // 128, cand_cap // 256))``
    (the program needs ``cand_cap >= top_k + 128 check_period``)."""
    return max(1, min(32, (int(cand_cap) - int(top_k)) // 128, int(cand_cap) // 256))


@dataclass(frozen=True)
class CandidateTopKChoice:
    """Everything the host decides for one candidate-restricted call from host-known integers: the ``cand`` program
    on ``grid_ctas`` persistent CTAs with ``cand_cap`` entries per unit query and its ``check_period``; the workspace
    ``[candidate buffers: grid x 4 x cand_cap x 8 B][block lists: grid x CAND_LIST_BYTES][ticket word: 16 B]``.
    The non-deterministic package returns the program's unordered rows as they are."""

    num_queries: int
    num_keys: int
    num_segments: int
    top_k: int
    cand_width: int
    grid_ctas: int
    cand_cap: int
    check_period: int
    cand_bytes: int
    lists_at: int
    lists_bytes: int
    ticket_at: int
    workspace_bytes: int
    cand_key: str = CAND_KEY

    @property
    def keys(self) -> tuple:
        return (self.cand_key,)

    def as_dict(self) -> dict:
        return asdict(self)


def select_cand_program(
    *,
    num_queries: int,
    num_keys: int,
    num_segments: int,
    top_k: int,
    grid: int,
    program_keys: Collection[str],
    cand_width: int = CAND_WIDTH_MAX,
    cand_cap: Optional[int] = None,
    check_period: Optional[int] = None,
) -> CandidateTopKChoice:
    """The host plan of one candidate-restricted call (see :class:`CandidateTopKChoice`): ``cand_cap`` defaults to
    ``cand_capacity(top_k)`` and ``check_period`` to ``cand_check_period``; an explicit pair must keep
    ``cand_cap >= top_k + 128 check_period``.  Raises ``NotImplementedError`` when ``program_keys`` registers
    no ``cand`` program."""
    if CAND_KEY not in program_keys:
        raise NotImplementedError(
            "no candidate-restricted top-k program (key 'cand') is registered for this architecture"
        )
    k = int(top_k)
    cap = cand_capacity(k) if cand_cap is None else int(cand_cap)
    period = cand_check_period(cap, k) if check_period is None else int(check_period)
    if period < 1 or cap < k + 128 * period:
        raise ValueError(
            f"cand_cap {cap} must hold top_k + 128 x check_period = {k} + 128 x {period} entries"
        )
    grid_ctas = int(grid)
    cand_bytes = grid_ctas * CAND_UNIT_Q * cap * 8
    lists_bytes = grid_ctas * CAND_LIST_BYTES
    ticket_at = cand_bytes + lists_bytes
    return CandidateTopKChoice(
        num_queries=int(num_queries),
        num_keys=int(num_keys),
        num_segments=int(num_segments),
        top_k=k,
        cand_width=int(cand_width),
        grid_ctas=grid_ctas,
        cand_cap=cap,
        check_period=period,
        cand_bytes=cand_bytes,
        lists_at=cand_bytes,
        lists_bytes=lists_bytes,
        ticket_at=ticket_at,
        workspace_bytes=ticket_at + CAND_TICKET_BYTES,
    )


__all__ = [
    "BF16",
    "CAND_CAP_FLOOR",
    "CAND_KEY",
    "CAND_LIST_BYTES",
    "CAND_MAX_TOP_K",
    "CAND_ROLE",
    "CAND_TICKET_BYTES",
    "CAND_UNIT_Q",
    "CAND_WIDTH_MAX",
    "CB_BLOCK",
    "CB_ENTRY_BYTES",
    "CB_IOTA_STAGE",
    "CB_KIND_FALLBACK",
    "CB_MAX_TOP_K",
    "CB_ROLE",
    "CB_SLOTS",
    "CB_TWIN_FORMS",
    "CB_TWIN_RULE",
    "CB_UNITS",
    "FORMS",
    "FP8",
    "FP8_KIND_FALLBACK",
    "KINDS",
    "MERGE_KEY",
    "POLICY_FIELDS",
    "SAMPLE_SHIFT_PERMILLE_BOUNDS",
    "UNITS",
    "CandidateBlocksChoice",
    "CandidateTopKChoice",
    "DispatchPolicy",
    "ProgramChoice",
    "ScanKeyFields",
    "cand_capacity",
    "cand_check_period",
    "cb_block_cap",
    "cb_pair_twin_key",
    "cb_scan_programs",
    "cb_twin_key",
    "cb_twin_rule_holds",
    "fallback_knobs",
    "fp8_policy_record",
    "fp8_program_choice",
    "fp8_scan_programs",
    "physical_kind",
    "sample_class_for",
    "scan_key",
    "scan_key_fields",
    "scan_key_form",
    "select_cand_program",
    "select_cb_program",
    "select_fp8_program",
    "select_program",
    "stage_slug",
    "ticket_static_path",
    "ticket_units_bound",
]
