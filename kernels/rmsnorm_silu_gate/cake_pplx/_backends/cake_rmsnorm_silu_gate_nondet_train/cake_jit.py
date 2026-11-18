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

Host plan and loader glue of the Cake SiLU-gated RMSNorm training kernels of
the non-deterministic package (forward and backward; SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its role, its
launch record, its grid rule, the workspace buffers of the backward and the
kernel-parameter -> logical-argument map.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through the package's
JIT shim from the same sources and flags, verifying every source digest before the
first build.

A role is one kernel of a call: ``fwd`` (the forward: ``rstd`` from
``rsqrt.approx`` and the SiLU gate from ``ex2.approx`` and ``rcp.approx``,
every row's arithmetic fixed, so ``y`` is the same bits on every call) and
``bwd_bf16`` / ``bwd_f32`` (the backward: ``dx``, ``dz`` and the weight
gradient in one kernel; each backward program repeats ``dx`` and ``dz`` bit
for bit for the same inputs on the same device, and the programs'
``.f32x2`` FMA chains for a row's sums differ by program and by
architecture, so the last bits of ``dx`` and ``dz`` can differ between
programs (the fused one runs the deterministic package's fused backward's
row arithmetic, with its approximate gate and ``rstd``, and the repeatable
one is that program itself); the tests hold every floating output to the
PyTorch BF16 naive implementation of the same operation; the chunked
programs sum the FP32 partial of each chunk of rows exactly, as integers,
into a weight gradient rounded once to FP32 from the exact sum in integer
arithmetic, then to BF16 for a BF16 one, so every output is the same bits
at any grid and in any order of the CTAs; the repeatable programs of
either weight gradient, the deterministic package's fused backward, add
each thread's FP32 runs of rows into plain FP32 words and each CTA's FP32
column words exactly into fixed-point digits, so every output is the same bits
for the rows and the SM count; the fused and the row-claims
programs of a BF16 weight gradient add each CTA's FP32 column sums, one
plain FP32 word a column, into FP32 replicas with FP32 atomic adds, which the
CTA that arrives last sums, so only their weight gradient may vary in its
last bits from run to run).
``ROUTES[arch][role]`` names the one module a role runs on an architecture;
a module may be delivered for more architectures than route it (every
forward module of this package's own kernels is delivered for every
architecture, so a new cut between them is a host-plan change; the pipelined
forwards are delivered where they are routed: the deterministic package's
forward program on sm_100a and sm_103a, its program of packed rows on
sm_107a; the repeatable backwards and their long programs of both
weight-gradient dtypes, the deterministic package's fused backward
programs, are delivered and routed on every architecture).
:func:`module_for_rows` picks a call's module from the rows, the device's SM
count and the layout of the inputs: the forward of packed rows (``x`` and
``z`` hold row ``n`` at element ``128 n``, fewer than ``PACKED_ROWS`` rows)
by ``FORWARD_BANDS``, of other layouts by ``STRIDED_FORWARD_BANDS``, bands of
rows per SM (the tiny forward, which reads packed rows only, the short
forward, the mid, long and extra-long forwards, the pipelined forwards and
the forward of the call role, each where the architecture's bands name it);
the backward by ``BACKWARD_BANDS``, bands of rows or of rows per SM: its
small-row roles (``BAND_ROLES``: the
short backward ``bwd_bf16_short`` and the exact tiny backward ``_t512``, each
up to a count of rows), its fused role (``_fused``: a BF16 weight gradient,
in a band of rows per SM) and its repeatable role (``_rep``: a BF16
weight gradient past the fused band, and on sm_107a also before it; an FP32
weight gradient past its small-row band but in the band of its call role on
sm_100a and sm_103a; on packed rows from ``LONG_BACKWARD_ROWS_PER_SM``
rows per SM its long role ``_rep_long`` in its place), its call role (the
static backward, where an architecture routes one for the role: an FP32
weight gradient on sm_100a and sm_103a, in a band of rows per SM; a BF16
one on sm_107a, which no band runs), for a BF16 weight gradient its
row-claims roles (``_rc``, ``_rc_mid``) and its mid role (the claimed
backward), which no band runs.
The forward's programs: the tiny forward (16 threads to a row, one 8-row
block per 128-thread CTA, the gate before the sum of squares, packed rows
only; the short forward's arithmetic and bits), the short forward (16
threads to a row, one 16-row block per CTA, the gate before the sum of
squares), the forward (64-thread
CTAs, 8 threads to a row), the mid forward (128-thread CTAs walking 16-row
blocks, thread 0 warming the CTA's next block and the first block of the CTA
a wave later in L2), the long forward (256-thread CTAs walking 32-row
blocks) and the extra-long forward (the long forward whose thread 0 also
warms the CTA's next block in L2); and the pipelined forwards, builds of the
deterministic package's forward kernel file with a gate that does not flush
(its builds with ``split_gate`` 1: the reciprocal of a scaled ``1 + e``, which
does not flush where the deterministic gate gives 0, ``z`` from about -87.34
down, and gives the deterministic bits elsewhere), under this package's names,
routed where they make the training step, two forwards and the backward, faster
than this package's own, and on sm_107a from 281 rows per SM, where with them
the forward alone is about 10 % faster and the step at most about 3.5 %
slower: 128-thread CTAs walking 16-row blocks, each thread
loading its row of the CTA's next block into registers while it computes the
current one, with the mid forward's L2 prefetches and the L2 ``evict_first``
policy on the loads of ``x`` and ``z``, one CTA per block below 8192 rows per
SM (``fwd_pipe`` on sm_100a and sm_103a; on sm_107a ``fwd_pipe_packed``, for
packed rows, with four sum chains, ``rsqrt.approx.ftz``, ``w`` loaded behind
the first block's loads and one 32-byte store per lane).  Each program's
arithmetic is fixed; the programs associate a row's sum of squares differently
(the tiny and the short forward: four 2-term FMA chains of 16 lanes added in
pairs, then folded by XOR shuffles 1, 2, 4, 8; the mid forward, the
extra-long forward but on sm_100a and the sm_107a pipelined forward: four
4-term chains of 8 lanes added in pairs, then folded by 1, 2, 4; the plain
forward and the pipelined forward on sm_100a and sm_103a: two 8-term chains
of 8 lanes (the even and the odd elements) added, then folded by 1, 2, 4; the
long forward, and the extra-long forward on sm_100a: 16-term chains of 8
lanes folded by 1, 2, 4), so ``y``'s bits can differ between the programs, and
so between the bands of one architecture, while each program gives the same
bits for the same inputs on the same device.

The call role of a BF16 weight gradient is the static backward (CTA ``b``
runs the chunks ``b``, ``b + grid``, ... and claims nothing; sm_107a only);
its mid role is the claimed backward (the CTAs claim runs of chunks from the
workspace's claim counter at run time, so a CTA slowed beside another kernel
takes fewer).  Each is a build per architecture, routed, and no band runs
it.  On sm_100a and sm_103a the claimed backward warms the CTA's next block
in L2, thread 0 with one bulk prefetch per tensor where ``dy``, ``x`` and
``z`` hold packed rows (head stride 128 and token stride ``H * 128``; other
layouts: a prefetch per thread).  On sm_107a: the static backward adds its
windows to 4 copies with an occupancy word per copy (``sparse``) and runs
one wave at most, and the claimed backward warms the block two ahead.  The
call role of an FP32 weight gradient is the static backward on sm_100a and
sm_103a (``bwd_f32``: no prefetch, its CTA that arrives last reading each
window by an atomic exchange with 0, the same sums in fewer instructions),
which its band runs past ``F32_REP_LAST`` and up to ``F32_STATIC_LAST``
rows per SM; it routes no module on sm_107a, and no claimed backward of an
FP32 weight gradient ships: elsewhere past its small-row band an FP32
weight gradient runs the repeatable backward (``bwd_f32_rep``).  The
row-claims backwards of a BF16 weight gradient,
on sm_100a and sm_103a (256 threads, 16-row blocks, the CTAs past one wave
claiming blocks from the claim counter; ``_rc_mid``: 5 CTAs per SM, thread
0 warming the next block of packed rows in L2), add each thread's FP32
runs of rows into FP32 words and add the CTA's FP32 column sums into 4
FP32 replicas in ``dw_sums`` with FP32
atomic adds; the CTA that arrives last sums the replicas in FP32 (no FP64).
A small-row
backward (whose BF16 band a call with ``compute_grad_weight=False`` also
takes) is one thread-block cluster of 16 CTAs that sums the weight gradient
on chip in a fixed order and rounds it once, with no workspace: the exact
tiny backward (``<role>_t512``: 4 teams of the 128-thread CTA of the
deterministic backward at ``2e989556`` a CTA, up to 512 rows; the
weight gradient its FP32 terms summed in plain FP32 words in an order of
its own) and, for a BF16 weight gradient, the short
backward (``bwd_bf16_short``: 256-thread CTAs with the row-claims backward's
row arithmetic, each CTA finishing 8 columns from every CTA's FP32 partials
in rank order in FP32, up to 256 rows).  The fused backward of a BF16
weight gradient on every architecture (``bwd_bf16_fused``, build
``{"math": "ax", "min_blocks": 4, "prefetch_share": None, "slots": 8}``)
runs the deterministic package's fused backward's schedule and row
arithmetic: 256-thread CTAs, 16 lanes a row, 16-row blocks, at most 4 CTAs
per SM (with more blocks than CTAs the CTAs claim the blocks in chunks of a
halving table), the next block warmed in L2 up to the architecture's share
of blocks a CTA, the approximate gate and ``rstd``; a row's sums run on FMA
chains of its own, set per architecture, so the last bits of ``dx`` and
``dz`` may differ from that program's.  Its
weight gradient is the row-claims backward's kind: each thread's FP32 runs
of at most 16 rows added into FP32 words, a CTA's FP32 column sums
added into 8 FP32 replicas in ``dw_sums`` and the
CTA that arrives last summing the replicas in FP32, so its last bits may
vary from run to run.
Its kernel parameters, grid rule and workspace buffers are the row-claims
backward's (with 8 replicas, not 4).  The repeatable backward of a BF16
weight gradient on every
architecture (``bwd_bf16_rep``) is a build of the deterministic package's
fused backward kernel file under this package's name (its build
``{"math": "ax", "min_blocks": 4, "out": "bf16", "prefetch_share":
None, "threads": 256}``), the same schedule and row arithmetic as the
fused backward's, each thread's FP32 runs of rows added into plain FP32
words, each CTA's FP32 column words added exactly into fixed-point digits
and the sum rounded once by the CTA that arrives last,
so its ``dx``, ``dz`` and ``dw`` are the deterministic package's bits, the
same for the rows and the SM count.
Its long program (``bwd_bf16_rep_long``, every architecture; build
``{"lookahead": True, "math": "ax", "min_blocks": 4, "out": "bf16",
"prefetch_share": None, "threads": 256}``) is the same build in the
look-ahead loop: a block's rows load while the previous block's row
arithmetic runs, with the same FP32 operations in the same order, so its
outputs are the repeatable backward's bits at every row count and grid.  It
runs in the repeatable backward's place on packed rows (``dy``, ``x`` and
``z`` with head stride 128 and token stride ``H * 128``) from
``LONG_BACKWARD_ROWS_PER_SM`` rows per SM, where it is faster.
The repeatable backward of an FP32 weight gradient (``bwd_f32_rep``, every
architecture; build ``{"math": "ax", "min_blocks": 4, "out": "f32",
"prefetch_share": None, "threads": 256}``) and its long program
(``bwd_f32_rep_long``, every architecture; the same build with
``"lookahead": True``) are the same kernel file's FP32 builds under this
package's names, with the same schedule, grid and workspace: each thread's
FP32 runs of at most 16 rows added into plain FP32 words of its chunk,
each CTA's FP32 column words added exactly into fixed-point digits, the sum
rounded once to FP32, so their ``dx``, ``dz`` and ``dw`` are the
deterministic package's FP32 backward's bits, the same for the rows and the
SM count (the chunks follow the grid).  In either repeatable backward a
column whose terms are finite but whose FP32 run, fold or chunk word
overflows gets an infinity (or a NaN) where the exact sum stays finite.
A BF16 weight gradient runs the fused backward only where the
band-selection measurement found its varying weight gradient making the
checkpointed training step (2 x forward + backward) about 3 % or more
faster than with the repeatable backward (near a band's edges the gain can
be smaller), and the repeatable backward past that; an FP32 weight gradient
runs the repeatable backward past its small-row band (512 rows), but on
sm_100a and sm_103a the static backward past one and up to two of the
repeatable backward's waves of 64 rows per SM, where the band-selection
measurement found the static backward saving about 2 to 7 % (B200) and
5 to 8 % (GB300) of the checkpointed training step.
``BACKWARD_BANDS`` gives each backward its rows on each architecture.
A backward that computes no weight gradient (``compute_grad_weight=False``)
takes the BF16 bands and runs, in place of its band's module, a program
built without the weight gradient where the architecture routes one
(``NODW_ROLES``, :func:`nodw_module`): the band's program with all of its
weight-gradient work removed at build time and the same kernel parameters
(the weight-gradient and workspace ones unread), so ``dx`` and ``dz`` are
the bits of the band's program at every grid.  A small-row band runs its
one-cluster program built so (``<band>_nodw``: no weight-gradient sums and
no cluster synchronization).  On sm_100a and sm_103a the fused band runs
its strided program (``<band>_nodw_strided``: CTA ``b`` runs the 16-row
blocks ``b``, ``b + grid``, ... of 256-thread CTAs with no claim, no ticket
and no workspace, packed rows read at their linear offsets) up to
``NODW_STRIDED_ROWS_PER_SM`` rows per SM and its claimed program
(``<band>_nodw``: the band's claims, ticket, grid and workspace) past it,
and the repeatable band its claimed program; on sm_107a, which routes no
claimed program, the fused and the repeatable band run their strided
programs at every row count, the repeatable band's on packed rows in its
128-thread build (``<band>_nodw_strided_packed``: 8-row blocks, a row to
each 16 lanes, as in the 256-thread programs) and on other layouts in
256-thread CTAs of 16-row blocks.  The repeatable band's long role runs its
long program built so (``bwd_bf16_rep_nodw_long``), or, where none is
routed, the long role with ``weight_grad`` 0.  A band whose architecture
routes no such program runs its own module with ``weight_grad`` 0, the
same bits.

Grids (:func:`launch_grid`): the forward and the short forward launch one
CTA per block of ``rows_per_cta`` rows; the extra-long, long and mid forwards
the fewest whole waves of ``ctas_per_sm * SMs`` CTAs whose CTAs walk at most
``max_blocks_per_cta`` blocks of ``rows_per_block`` rows each, never more
CTAs than blocks (CTA ``b`` walks the blocks ``b``, ``b + grid``, ...), and
so do the pipelined forwards from ``persistent_from_rows_per_sm`` rows per SM
on, one CTA per block below it; a
small-row backward one cluster of ``cluster`` CTAs (``launch.cluster``)
whatever the rows, up to its rule's ``max_rows``.  The static and the
claimed backward cut the rows into blocks of ``rows_per_block`` rows and the
blocks into the chunks of the deterministic backward at ``2e989556``
(:func:`backward_chunks`): ``blocks_per_chunk = max(1, ceil(blocks /
chunk_limit))`` consecutive blocks a chunk, ``C = ceil(blocks /
blocks_per_chunk)`` chunks, and launch :func:`backward_ctas` ``G = min(C,
ctas_per_sm * SMs)`` CTAs; the host passes ``blocks_per_chunk`` to the
kernel.  A row-claims or a fused backward launches ``G = min(ceil(rows /
rows_per_cta), ctas_per_sm * SMs)`` CTAs, the repeatable backwards and their
long programs ``G =
min(ceil(rows / rows_per_block), ctas_per_sm * SMs)`` (:func:`backward_ctas`;
the CTAs claim the blocks).  A strided program without the weight gradient
launches the same ``G = min(ceil(rows / rows_per_block), ctas_per_sm *
SMs)`` of its own rule (16-row blocks and 4 CTAs per SM, the 128-thread
program of the repeatable band's packed rows 8-row blocks and 8 CTAs per
SM), CTA ``b`` walking the blocks ``b``, ``b + G``, ... with no claim.
Every backward is correct at any grid, every
chunked or small-row one gives the same bits at any grid and the
repeatable ones the same bits for the rows and the SM count; the cuts between
them are a matter of speed and of the varying BF16 weight gradient of the
row-claims and the fused programs.  No module is cooperative.  A kernel
resident on another stream delays a call roughly in proportion to the SM
slots it holds, and one that holds resources a kernel needs on every SM can
still delay its start; a persistent forward CTA or a static backward CTA
whose slot such a kernel holds runs all of its blocks once it starts.

The workspace of the static and the claimed backwards is three buffers
(:func:`workspace_layout`, :func:`workspace_bytes`): ``dw_win``, the 64-bit
integer windows of the exact sum (``slots`` copies of 7 windows of 128
columns, 4 or 5 slots; with ``sparse`` the last copy holds the occupancy
words), ``dw_spec``, 128 FP32 class words for infinite and NaN partials,
and ``dw_done``, two 32-bit words (the arrival ticket and the claim counter);
that of a row-claims or a fused backward is ``dw_sums``, its FP32
replicas (one FP32 sum for each of 128 columns, 4 replicas for a
row-claims backward, 8 for the fused one), and ``dw_done`` (2056 and 4104
bytes); that of a repeatable
backward and its long program (either weight gradient) is ``acc``, its
fixed-point digits (2304
u64), ``flags``, a 32-bit flag word a column, and ``done``, the arrival
ticket and the claim counter (18952 bytes).  All of it is zero when a
launch starts and zero again when it ends, so a
zero-filled workspace stays reusable for any later call on its stream, and
one of the largest module's size serves every backward module of an
architecture.  Every
launch of those backwards takes the ticket, and so does a claimed program
without the weight gradient (``<band>_nodw`` and the long one, on its band
program's workspace layout; the kernel leaves the claim counter and the
ticket words zero), so each needs the workspace; the small-row backwards,
their programs without the weight gradient and the strided programs take
none (:func:`workspace_bytes` is 0), and the largest layout does not change.

The program name ``cake_rmsnorm_silu_gate_train_<20 hex>`` is the first twenty
hex digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
resolved and sorted>, "module": <dotted kernel module>}`` of the kernel the
program was rendered from; each record's comment names its build.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    # The forward: 64 threads, 8 lanes per row (8-row blocks), 16-byte
    # accesses, the gate computed before the sum of squares, approximate FP32
    # intrinsics, the split gate (build {"early": true, "hints": false,
    # "lanes": 8, "rows_per_thread": 1, "split_gate": 1, "threads": 64,
    # "width": 128}).
    "cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd",
    },
    # The extra-long forward: the long forward whose thread 0 also warms the
    # CTA's next row block of packed rows in L2 with one bulk prefetch a
    # tensor (build {"early": false, "lanes": 8, "min_blocks": 3,
    # "prefetch_ahead": 1, "prefetch_wave": 0, "split_gate": 1, "threads":
    # 256}); whole waves of at most 6 blocks per CTA; FORWARD_BANDS gives it
    # its rows.
    "cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 3,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 6,
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_xlong",
    },
    # The long forward: 256 threads, 8 lanes per row (32-row blocks), 32-byte
    # accesses, at most 3 CTAs per SM, each loading its next block while it
    # computes the current one (build {"early": false, "lanes": 8,
    # "min_blocks": 3, "prefetch_ahead": 0, "prefetch_wave": 0, "split_gate":
    # 1, "threads": 256}); whole waves of at most 6 blocks per CTA;
    # FORWARD_BANDS gives it its rows (sm_100a).
    "cake_rmsnorm_silu_gate_train_0dc2a8df1f94d878ccbb": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 3,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 6,
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_0dc2a8df1f94d878ccbb",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_0dc2a8df1f94d878ccbb",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_0dc2a8df1f94d878ccbb",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_long",
    },
    # The mid forward: 128 threads, 8 lanes per row (16-row blocks), at most 6
    # CTAs per SM, thread 0 warming the CTA's next block and the first block
    # of the CTA one wave later in L2 (build {"early": false, "lanes": 8,
    # "min_blocks": 6, "prefetch_ahead": 1, "prefetch_wave": 1, "split_gate":
    # 1, "threads": 128}); whole waves of at most 2 blocks per CTA;
    # FORWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_mid",
    },
    # The short forward: 256 threads, 16 lanes per row (16-row blocks), one
    # block per CTA, the gate before the sum of squares (build {"early": true,
    # "lanes": 16, "min_blocks": 4, "prefetch_ahead": 0, "prefetch_wave": 0,
    # "split_gate": 1, "threads": 256}); FORWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_7b542c846911aca66512": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_7b542c846911aca66512",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_7b542c846911aca66512",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_7b542c846911aca66512",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_short",
    },
    # The tiny forward: 128 threads, 16 lanes per row (8-row blocks), one
    # block per CTA, packed rows only (row n at element 128 n), the short
    # forward's bits (build {"split_gate": 1, "threads": 128}); FORWARD_BANDS
    # gives it its rows.
    "cake_rmsnorm_silu_gate_train_242b29d118997473fc3e": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_242b29d118997473fc3e",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_242b29d118997473fc3e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_242b29d118997473fc3e",
        "logical": {
            "eps": "eps",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "y": "y",
            "z": "z",
        },
        "role": "fwd_tiny",
    },
    # The pipelined forward: the deterministic package's forward program with
    # a gate that does not flush (split_gate 1, under this package's name): 128
    # threads, 8 lanes per row (16-row blocks), at most 6 CTAs per SM, each
    # thread loading its row of the CTA's next block into registers while it
    # computes the current one, thread 0 warming the CTA's next block and the
    # first block of the CTA one wave later in L2 (build {"min_blocks": 6,
    # "packed_rows": 0, "prefetch_ahead": 1, "prefetch_wave": 1, "rsqrt_ftz":
    # 0, "split_gate": 1, "store_width": 128, "sum_pairs": 0, "threads": 128,
    # "w_late": 0});
    # one CTA per block below 8192 rows per SM, whole waves of at most 2
    # blocks per CTA from there; FORWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": 8192,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_pipe",
    },
    # The sm_107a pipelined forward of packed rows (x and z hold row n at
    # element 128 n, fewer than PACKED_ROWS rows): the deterministic package's
    # sm_107a forward of packed rows with a gate that does not flush, the
    # schedule above with the sum of squares as four FMA chains,
    # rsqrt.approx.ftz, w loaded behind the first block's loads, each row's
    # offset taken from its index alone and one 32-byte store per lane (build
    # {"min_blocks": 6, "packed_rows": 1, "prefetch_ahead": 1, "prefetch_wave":
    # 1, "rsqrt_ftz": 1, "split_gate": 1, "store_width": 256, "sum_pairs": 1,
    # "threads": 128, "w_late": 1}); FORWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_9cb071b2dfe75474aaa7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": 8192,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_9cb071b2dfe75474aaa7",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_9cb071b2dfe75474aaa7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_9cb071b2dfe75474aaa7",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_pipe_packed",
    },
    # The static backward into a BF16 weight gradient on sm_107a: the static
    # schedule with the windows in 4 copies and an occupancy word per copy
    # (sparse: a CTA adds only the windows its columns use, the last CTA reads
    # only the occupied ones; one row block a CTA of its grid of 8 CTAs per
    # SM); routed, but no BACKWARD_BANDS band runs it (build
    # {"bulk": 0, "chunks_per_step": 0, "min_blocks": 8, "out": "bf16",
    # "prefetch_ahead": 0, "schedule": "static", "slots": 5, "sparse": 1}).
    "cake_rmsnorm_silu_gate_train_6d7cf79fb3d679d07410": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_6d7cf79fb3d679d07410",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_6d7cf79fb3d679d07410",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_6d7cf79fb3d679d07410",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16",
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 4480,
        },
    },
    # The claimed backward into a BF16 weight gradient on sm_100a and sm_103a:
    # the static backward's rows and sums, its CTAs claiming runs of chunks
    # from the claim counter, thread 0 warming the CTA's next block in L2 with
    # one bulk prefetch per tensor where dy, x and z hold packed rows (head
    # stride 128, token stride H * 128; other layouts: a prefetch per thread);
    # routed, but no BACKWARD_BANDS band runs it (build {"bulk": 1,
    # "chunks_per_step": 0, "min_blocks": 8, "out": "bf16", "prefetch_ahead":
    # 1, "schedule": "claimed", "slots": 4, "sparse": 0}).
    "cake_rmsnorm_silu_gate_train_2a3bbf332a501a11ca31": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_2a3bbf332a501a11ca31",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_2a3bbf332a501a11ca31",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_2a3bbf332a501a11ca31",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_mid",
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The claimed backward into a BF16 weight gradient on sm_107a: the static
    # backward's rows and sums, its CTAs claiming runs of chunks from the claim
    # counter, thread 0 warming the rows of the block two ahead in L2;
    # routed, but no BACKWARD_BANDS band runs it (build {"bulk": 0,
    # "chunks_per_step": 0, "min_blocks": 8, "out": "bf16", "prefetch_ahead":
    # 2, "schedule": "claimed", "slots": 4, "sparse": 0}).
    "cake_rmsnorm_silu_gate_train_51408d05160a7056aad8": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_51408d05160a7056aad8",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_51408d05160a7056aad8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_51408d05160a7056aad8",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_mid",
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The row-claims backward into a BF16 weight gradient: 256 threads, 16
    # lanes per row (16-row blocks), at most 4 CTAs per SM that claim the
    # blocks from the claim counter past one wave, each thread's FP32 runs of
    # rows added into FP32 words, the CTA's FP32 column sums added into 4
    # FP32 replicas in dw_sums with FP32 atomic adds
    # and the CTA that arrives last summing the replicas in FP32 (no FP64), so
    # the weight gradient's last bits may vary from run to run (dx and dz do
    # not);
    # routed, but no BACKWARD_BANDS band runs it (build {"lanes": 16, "math":
    # "x2", "out": "bf16", "prefetch": "rows", "rows_per_thread": 1, "slots":
    # 4, "threads": 256, "width": 128}).
    "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "row_claims",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12544,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rc",
        "workspace": {
            "dw_done": 2,
            "dw_sums": 512,
        },
    },
    # The row-claims mid backward into a BF16 weight gradient on sm_100a and
    # sm_103a: the row-claims backward with 5 CTAs per SM, thread 0 warming the
    # next block of packed dy, x and z in L2 with one bulk prefetch per tensor,
    # the weight gradient the row-claims backward's 4 FP32 replicas, so its
    # last bits may vary from run to run; routed, but no BACKWARD_BANDS band
    # runs it (build {"lanes": 16, "math": "x2", "out": "bf16", "prefetch":
    # "bulk", "rows_per_thread": 1, "slots": 4, "threads": 256, "width":
    # 128}).
    "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 5,
            "kind": "row_claims",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12544,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rc_mid",
        "workspace": {
            "dw_done": 2,
            "dw_sums": 512,
        },
    },
    # The short backward into a BF16 weight gradient on every architecture:
    # one cluster of 16 CTAs of 256 threads (CTA r runs the 16-row blocks r,
    # r + 16, ...), the row-claims backward's row arithmetic (16 lanes per row,
    # the gate on packed FP32 pairs), CTA r of the cluster finishing the
    # weight-gradient columns 8r to 8r + 7 from every CTA's FP32 partials in
    # rank order in FP32 (a fixed order: the same bits on every
    # call), no
    # workspace (build {"cluster": 16, "out": "bf16", "threads": 256}), up to
    # its max_rows; BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 256,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6144,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_short",
    },
    # The exact small-row backward into a BF16 weight gradient (a call with
    # compute_grad_weight=False runs bwd_bf16_t512_nodw in its band):
    # one cluster of 16 CTAs of 4 teams of the 128-thread CTA of the
    # deterministic backward at 2e989556 (team q of CTA r runs the 8-row
    # block r + 16q), each term and partial of the weight gradient one FP32
    # word, CTA r summing the columns 8r to 8r + 7 in plain FP32 in an order
    # of its own (the same bits on every call, not the deterministic
    # backward's), no
    # workspace (build {"out": "bf16", "teams": 4}), up to 512 rows;
    # BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 512,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 19584,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_t512",
    },
    # The static backward into a FP32 weight gradient on sm_100a and sm_103a:
    # 128 threads, 16 lanes per row (8-row blocks), the deterministic
    # backward's chunks, CTA b running the chunks b, b + grid, ... with no
    # claim and no prefetch, the chunk partials summed exactly in integer
    # windows, the CTA that arrives last reading each window by an atomic
    # exchange with 0 (the same sums in fewer instructions), the weight
    # gradient rounded once to FP32; BACKWARD_BANDS gives it its rows, past 64
    # and up to 128 rows per SM (F32_REP_LAST, F32_STATIC_LAST; at most 2 row
    # blocks a CTA of its grid of 8 CTAs per SM) (build {"bulk": 0,
    # "chunks_per_step": 0, "finish": "exch", "min_blocks": 8, "out": "f32",
    # "prefetch_ahead": 0, "schedule": "static", "slots": 4, "sparse": 0}).
    "cake_rmsnorm_silu_gate_train_421e73578399336342e3": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_421e73578399336342e3",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_421e73578399336342e3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_421e73578399336342e3",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32",
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The exact small-row backward into a FP32 weight gradient:
    # one cluster of 16 CTAs of 4 teams of the 128-thread CTA of the
    # deterministic backward at 2e989556 (team q of CTA r runs the 8-row
    # block r + 16q), each term and partial of the weight gradient one FP32
    # word, CTA r summing the columns 8r to 8r + 7 in plain FP32 in an order
    # of its own (the same bits on every call, not the deterministic
    # backward's), no
    # workspace (build {"out": "f32", "teams": 4}), up to 512 rows;
    # BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 512,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 19584,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32_t512",
    },
    # The fused backward into a BF16 weight gradient: 256 threads, 16 lanes
    # per row (16-row blocks), at most 4 CTAs per SM claiming the blocks in
    # chunks of the deterministic package's fused backward, its approximate
    # gate and rstd (dx and dz are that program's bits on sm_100a; elsewhere
    # a row's sums take other FMA chains), each thread's FP32
    # runs of rows added into FP32 words, the CTA's FP32 column sums added
    # into 8 FP32 replicas in dw_sums with FP32
    # atomic adds, summed in FP32 by the CTA that arrives last (no FP64), so
    # the weight gradient's last bits may vary from run to run; the row-claims
    # backward's kernel parameters, grid rule and workspace buffers;
    # BACKWARD_BANDS gives it its rows (build {"math": "ax", "min_blocks": 4,
    # "prefetch_share": None, "slots": 8}).
    "cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "row_claims",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12544,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_fused",
        "workspace": {
            "dw_done": 2,
            "dw_sums": 1024,
        },
    },
    # The repeatable backward into a BF16 weight gradient: the deterministic
    # package's fused backward program (its build {"math": "ax",
    # "min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads":
    # 256}): 16 lanes per row (16-row blocks), at most 4 CTAs per SM claiming
    # the blocks, the weight gradient summed exactly in fixed point in the
    # workspace (``acc``, ``flags``, ``done``) and rounded once, so every
    # output repeats bit for bit; BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rep",
        "workspace": {
            "acc": 2304,
            "done": 2,
            "flags": 128,
        },
    },
    # The repeatable backward's long program into a BF16 weight gradient: the
    # repeatable backward in the look-ahead loop (the deterministic package's
    # long program, its build {"lookahead": True, "math": "ax", "min_blocks":
    # 4, "out": "bf16", "prefetch_share": None, "threads": 256}): a block's
    # rows load while the previous block's row arithmetic runs, with the same
    # FP32 operations in the same order, so dx, dz and dw are the repeatable
    # backward's bits at every row count and grid; it runs in the repeatable
    # backward's place on packed rows from LONG_BACKWARD_ROWS_PER_SM rows per
    # SM.
    "cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rep_long",
        "workspace": {
            "acc": 2304,
            "done": 2,
            "flags": 128,
        },
    },
    # The repeatable backward into an FP32 weight gradient: the deterministic
    # package's fused backward program (its build {"math": "ax",
    # "min_blocks": 4, "out": "f32", "prefetch_share": None, "threads":
    # 256}): 16 lanes per row (16-row blocks), at most 4 CTAs per SM claiming
    # the blocks, each thread's FP32 runs of at most 16 rows added into FP32
    # words of its chunk, each CTA's FP32 column words added exactly into
    # fixed-point digits in the workspace (``acc``, ``flags``, ``done``) and
    # the sum rounded once to FP32, so every output repeats bit for bit for
    # the rows and the SM count (its chunks follow the grid); BACKWARD_BANDS
    # gives it its rows.
    "cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32_rep",
        "workspace": {
            "acc": 2304,
            "done": 2,
            "flags": 128,
        },
    },
    # The repeatable backward's long program into an FP32 weight gradient:
    # the repeatable backward in the look-ahead loop (the deterministic
    # package's long program, its build {"lookahead": True, "math": "ax",
    # "min_blocks": 4, "out": "f32", "prefetch_share": None, "threads": 256}):
    # a block's rows load while the previous block's row arithmetic runs,
    # with the same FP32 operations in the same order, so dx, dz and dw are
    # the repeatable backward's bits at every row count and grid; it runs in
    # the repeatable backward's place on packed rows from
    # LONG_BACKWARD_ROWS_PER_SM rows per SM.
    "cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32_rep_long",
        "workspace": {
            "acc": 2304,
            "done": 2,
            "flags": 128,
        },
    },
    # The short backward without the weight gradient: the short backward
    # built without the weight gradient (no column sums and no cluster
    # synchronization; one cluster of 16 CTAs of 256 threads, CTA r running
    # the 16-row blocks r, r + 16, ...), so dx and dz are the short
    # backward's bits; no workspace; a call with compute_grad_weight=False
    # runs it in the short backward's band (build {"cluster": 16, "dw":
    # False, "out": "bf16", "threads": 256}).
    "cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 256,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_short_nodw",
    },
    # The exact tiny backward without the weight gradient: the exact tiny
    # backward of a BF16 weight gradient built without the weight gradient
    # (no column sums and no cluster synchronization; one cluster of 16 CTAs
    # of 4 teams, team q of CTA r running the 8-row block r + 16q), so dx and
    # dz are the exact tiny backward's bits; no workspace; a call with
    # compute_grad_weight=False runs it in that backward's band (build
    # {"dw": False, "out": "bf16", "teams": 4}).
    "cake_rmsnorm_silu_gate_train_a374cf1736d501ae6d3f": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 512,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_a374cf1736d501ae6d3f",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_a374cf1736d501ae6d3f",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_a374cf1736d501ae6d3f",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_t512_nodw",
    },
    # The fused backward without the weight gradient, strided: the fused
    # backward built without the weight gradient, CTA b running the 16-row
    # blocks b, b + grid, ... with no claim, no ticket and no workspace word
    # (packed rows read at their linear offsets; a CTA of one block takes its
    # block count and prefetch test without an integer division),
    # so dx and dz are the fused backward's bits at any grid; the fused
    # backward's kernel parameters (the weight-gradient and workspace ones
    # unread) and CTA count; a call with compute_grad_weight=False runs it in
    # the fused band up to NODW_STRIDED_ROWS_PER_SM rows per SM, on sm_107a at
    # every row count of the band (build {"dw": False, "math": "ax",
    # "min_blocks": 4, "prefetch_share": None, "slots": 8, "strided": True}).
    "cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "strided_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_fused_nodw_strided",
    },
    # The fused backward without the weight gradient, claimed: the fused
    # backward built without the weight gradient, with its claims, its
    # arrival ticket and its claim counter (its grid rule and workspace; the
    # kernel leaves the claim counter and the ticket words zero), so dx and
    # dz are the fused backward's bits; a call with compute_grad_weight=False
    # runs it in the fused band past NODW_STRIDED_ROWS_PER_SM rows per SM on
    # sm_100a and sm_103a (build {"dw": False, "math": "ax", "min_blocks": 4,
    # "prefetch_share": None, "slots": 8}).
    "cake_rmsnorm_silu_gate_train_20a7c07ccc0866350d49": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "row_claims",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_20a7c07ccc0866350d49",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_20a7c07ccc0866350d49",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_20a7c07ccc0866350d49",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_fused_nodw",
        "workspace": {
            "dw_done": 2,
            "dw_sums": 1024,
        },
    },
    # The repeatable backward without the weight gradient, strided: the
    # deterministic package's BF16 backward program built without the weight
    # gradient (its build plus "dw": False and "strided": True), CTA b
    # running the 16-row blocks b, b + grid, ... with no claim, no ticket and
    # no workspace word (a CTA of one block takes its block count and
    # prefetch test without an integer division), so dx and dz are the
    # repeatable backward's bits at any grid; the repeatable backward's
    # kernel parameters (the weight-gradient and workspace ones unread) and
    # CTA count; a call with compute_grad_weight=False runs it on rows that
    # are not packed (packed rows run bwd_bf16_rep_nodw_strided_packed) in a
    # repeatable band on sm_107a at every row count (build {"dw": False,
    # "math": "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None,
    # "strided": True, "threads": 256}).
    "cake_rmsnorm_silu_gate_train_cfe88bc164eeaa58821d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "strided_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_cfe88bc164eeaa58821d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_cfe88bc164eeaa58821d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_cfe88bc164eeaa58821d",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rep_nodw_strided",
    },
    # The repeatable backward without the weight gradient, strided, for
    # packed rows: the deterministic package's BF16 backward program built
    # without the weight gradient, strided, in 128-thread CTAs (its build
    # with "threads": 128, "min_blocks": 8, "dw": False and "strided":
    # True), a row to each 16 lanes as in the repeatable backward, CTA b
    # running the 8-row blocks b, b + grid, ... with no claim, no ticket and
    # no workspace word (packed rows read at their linear offsets; a CTA of
    # one block takes its block count and prefetch test without an integer
    # division), so dx and dz are the repeatable backward's bits at any grid;
    # the repeatable backward's kernel parameters (the weight-gradient and
    # workspace ones unread) and its own grid, min(ceil(rows / 8), 8 * SMs)
    # CTAs; a call with compute_grad_weight=False runs it in the strided
    # program's place on packed rows (dy, x and z of head stride 128 and
    # token stride heads * 128) in a repeatable band on sm_107a at every row
    # count but the long program's (build {"dw": False, "math": "ax",
    # "min_blocks": 8, "out": "bf16", "prefetch_share": None, "strided":
    # True, "threads": 128}).
    "cake_rmsnorm_silu_gate_train_07de4b4c9a438a52ff79": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 8,
            "kind": "strided_row_blocks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_07de4b4c9a438a52ff79",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_07de4b4c9a438a52ff79",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_07de4b4c9a438a52ff79",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rep_nodw_strided_packed",
    },
    # The repeatable backward without the weight gradient, claimed: the
    # deterministic package's BF16 backward program built without the weight
    # gradient (its build plus "dw": False), with its claims, its arrival
    # ticket and its claim counter (its grid rule and workspace; the kernel
    # leaves the claim counter and the ticket words zero), so dx and dz are
    # the repeatable backward's bits; a call with compute_grad_weight=False
    # runs it in a repeatable band on sm_100a and sm_103a at every row count
    # but the long program's (build {"dw": False, "math": "ax", "min_blocks":
    # 4, "out": "bf16", "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_train_cefc51faec88b7212f3c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_cefc51faec88b7212f3c",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_cefc51faec88b7212f3c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_cefc51faec88b7212f3c",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rep_nodw",
        "workspace": {
            "acc": 2304,
            "done": 2,
            "flags": 128,
        },
    },
    # The repeatable backward's long program without the weight gradient: the
    # deterministic package's long program built without the weight gradient
    # (its build plus "dw": False: the look-ahead loop, claims, ticket, grid
    # rule and workspace), so dx and dz are the repeatable backward's bits; a
    # call with compute_grad_weight=False runs it where the long role
    # bwd_bf16_rep_long runs (build {"dw": False, "lookahead": True, "math":
    # "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None,
    # "threads": 256}).
    "cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_rep_nodw_long",
        "workspace": {
            "acc": 2304,
            "done": 2,
            "flags": 128,
        },
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830",
        "fwd_long": "cake_rmsnorm_silu_gate_train_0dc2a8df1f94d878ccbb",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3",
        "fwd_short": "cake_rmsnorm_silu_gate_train_7b542c846911aca66512",
        "fwd_tiny": "cake_rmsnorm_silu_gate_train_242b29d118997473fc3e",
        "fwd_pipe": "cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_2a3bbf332a501a11ca31",
        "bwd_bf16_rc": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "bwd_bf16_rc_mid": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "bwd_bf16_short": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "bwd_bf16_t512": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "bwd_bf16_fused": "cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0",
        "bwd_bf16_rep": "cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_rep_long": "cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992",
        "bwd_bf16_short_nodw": "cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b",
        "bwd_bf16_t512_nodw": "cake_rmsnorm_silu_gate_train_a374cf1736d501ae6d3f",
        "bwd_bf16_fused_nodw": "cake_rmsnorm_silu_gate_train_20a7c07ccc0866350d49",
        "bwd_bf16_fused_nodw_strided": "cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605",
        "bwd_bf16_rep_nodw": "cake_rmsnorm_silu_gate_train_cefc51faec88b7212f3c",
        "bwd_bf16_rep_nodw_long": "cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_421e73578399336342e3",
        "bwd_f32_t512": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "bwd_f32_rep": "cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736",
        "bwd_f32_rep_long": "cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3",
        "fwd_short": "cake_rmsnorm_silu_gate_train_7b542c846911aca66512",
        "fwd_tiny": "cake_rmsnorm_silu_gate_train_242b29d118997473fc3e",
        "fwd_pipe": "cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_2a3bbf332a501a11ca31",
        "bwd_bf16_rc": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "bwd_bf16_rc_mid": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "bwd_bf16_short": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "bwd_bf16_fused": "cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0",
        "bwd_bf16_rep": "cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_rep_long": "cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992",
        "bwd_bf16_short_nodw": "cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b",
        "bwd_bf16_fused_nodw": "cake_rmsnorm_silu_gate_train_20a7c07ccc0866350d49",
        "bwd_bf16_fused_nodw_strided": "cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605",
        "bwd_bf16_rep_nodw": "cake_rmsnorm_silu_gate_train_cefc51faec88b7212f3c",
        "bwd_bf16_rep_nodw_long": "cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_421e73578399336342e3",
        "bwd_f32_t512": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "bwd_f32_rep": "cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736",
        "bwd_f32_rep_long": "cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_train_786ee0cda6a87f643bf8",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_33ef19a28735b3e470b3",
        "fwd_short": "cake_rmsnorm_silu_gate_train_7b542c846911aca66512",
        "fwd_tiny": "cake_rmsnorm_silu_gate_train_242b29d118997473fc3e",
        "fwd_pipe_packed": "cake_rmsnorm_silu_gate_train_9cb071b2dfe75474aaa7",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_6d7cf79fb3d679d07410",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_51408d05160a7056aad8",
        "bwd_bf16_short": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "bwd_bf16_t512": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "bwd_bf16_fused": "cake_rmsnorm_silu_gate_train_2ce27f38a52585400ca0",
        "bwd_bf16_rep": "cake_rmsnorm_silu_gate_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_rep_long": "cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992",
        "bwd_bf16_short_nodw": "cake_rmsnorm_silu_gate_train_44693c0bd2ecbeaa207b",
        "bwd_bf16_t512_nodw": "cake_rmsnorm_silu_gate_train_a374cf1736d501ae6d3f",
        "bwd_bf16_fused_nodw_strided": "cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605",
        "bwd_bf16_rep_nodw_long": "cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_rep_nodw_strided": "cake_rmsnorm_silu_gate_train_cfe88bc164eeaa58821d",
        "bwd_bf16_rep_nodw_strided_packed": "cake_rmsnorm_silu_gate_train_07de4b4c9a438a52ff79",
        "bwd_f32_t512": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "bwd_f32_rep": "cake_rmsnorm_silu_gate_train_ae2073f31d0923e61736",
        "bwd_f32_rep_long": "cake_rmsnorm_silu_gate_train_be861a132bbc76cb6433",
    },
}

# The forward module of each architecture by the rows per SM of a call of
# packed rows: ``(last, role)`` bands in increasing ``last`` (``None``: no
# end); a call of ``rows`` rows on a device of ``sm_count`` SMs runs the
# module of the first band's routed ``role`` with ``rows <= last *
# sm_count``.  The cuts follow the checkpointed training step at each rows
# per SM (two forwards and the backward, of a BF16 and of an FP32 weight
# gradient; CUPTI, cold L2) on every architecture: the pipelined forwards
# load ``x`` and ``z`` with the L2 ``evict_first`` policy, so where the rows
# fit in L2 the step's later kernels read them from memory again and this
# package's own forwards make the step faster, though the pipelined forward
# alone is faster.  On sm_100a each band runs the program of the fastest
# step.  On sm_103a the mid forward up to 240 rows per SM and the extra-long
# forward up to 600 make the step 4 to 10 % faster than the pipelined
# forward, which alone is at most 4.5 % faster; past 600 the pipelined
# forward runs, but the mid forward from 6001 to 8192 rows per SM, where the
# pipelined forward still launches a CTA per block.  One cut is not the
# step's: on sm_107a the pipelined forward runs past 280 rows per SM, where
# it is about 10 % faster alone than the extra-long forward and makes the
# step at most about 3.5 % slower.  The tiny forward ends where beside a
# kernel holding 16 SMs it still ends no later than the deterministic
# package's forward: 36 rows per SM on sm_100a and sm_103a (on sm_100a,
# from 37 rows per SM beside 16 CTAs of 1024 threads with the max-shared
# carveout, it can end about 2 microseconds later, and the pipelined forward,
# which is that forward, runs from there up to 85 rows per SM), and 65 on
# sm_107a.  Host plan only: a new cut needs no new program.
FORWARD_BANDS: dict[str, tuple[tuple[int | None, str], ...]] = {
    "sm_100a": (
        (36, "fwd_tiny"),
        (85, "fwd_pipe"),
        (160, "fwd_mid"),
        (400, "fwd_long"),
        (520, "fwd_xlong"),
        (6500, "fwd_pipe"),
        (None, "fwd_mid"),
    ),
    "sm_103a": (
        (36, "fwd_tiny"),
        (240, "fwd_mid"),
        (600, "fwd_xlong"),
        (6000, "fwd_pipe"),
        (8192, "fwd_mid"),
        (None, "fwd_pipe"),
    ),
    "sm_107a": (
        (65, "fwd_tiny"),
        (175, "fwd_mid"),
        (280, "fwd_xlong"),
        (None, "fwd_pipe_packed"),
    ),
}

# The forward bands of a call whose ``x`` or ``z`` rows are not packed (the
# tiny forward reads packed rows only), or of packed rows from
# ``PACKED_ROWS`` on, as ``FORWARD_BANDS``.  The cuts follow the
# checkpointed training step as ``FORWARD_BANDS`` do, except the short
# forward's end: 12 rows per SM on sm_100a and sm_103a, where its grid of
# one 16-row block a CTA has at most as many CTAs as the SMs less 16 (from 64
# SMs), and 59 on sm_107a, where it fits one wave of 4 CTAs per SM on all SMs
# but 16 (from 205 SMs), so beside a kernel holding 16 SMs it ends no later
# than the plain forward.  The plain forward (the call role ``fwd``: one
# 64-thread CTA per 8-row block) runs past the persistent forwards' bands,
# from 701 rows per SM on sm_100a and sm_103a and from 481 on sm_107a: where
# measured there (from 1000 rows per SM on sm_100a and sm_103a, from 520 on
# sm_107a) it is the fastest forward alone and makes the step 1 to 6 %
# faster.  No band of this table runs a pipelined forward, which makes the
# step slower at every rows per SM measured (on sm_103a 3 to 7 % slower than
# the mid forward from 13 to 100 rows per SM).
STRIDED_FORWARD_BANDS: dict[str, tuple[tuple[int | None, str], ...]] = {
    "sm_100a": (
        (12, "fwd_short"),
        (140, "fwd_mid"),
        (400, "fwd_long"),
        (700, "fwd_xlong"),
        (None, "fwd"),
    ),
    "sm_103a": (
        (12, "fwd_short"),
        (250, "fwd_mid"),
        (700, "fwd_xlong"),
        (None, "fwd"),
    ),
    "sm_107a": (
        (59, "fwd_short"),
        (480, "fwd_mid"),
        (None, "fwd"),
    ),
}

# The backward programs of each backward role on each architecture, by the
# rows of a call: ``(last, role)`` bands in order; a call of ``rows`` rows on
# a device of ``sm_count`` SMs runs the module of the first band's routed
# ``role`` whose ``last`` holds: ``(ROWS, n)`` while ``rows <= n`` (a count
# of rows: the small-row backwards, one cluster each, whose grid does not
# grow with the rows; ``n`` at most the module's ``max_rows``), a number
# while ``rows <= last * sm_count`` (rows per SM, as in ``FORWARD_BANDS``),
# and ``None`` (the last band) always; a routed backward role in no band
# runs for no row count.  The cuts are this table only (host plan: a new
# cut needs no new program).  An FP32 weight gradient runs, past its
# small-row band (512 rows), the repeatable backward (``_rep``: the
# deterministic package's FP32 backward program, at most 4 CTAs per SM
# claiming 16-row blocks, so a wave is 64 rows per SM), but on sm_100a and
# sm_103a the static backward (the call role: 8 CTAs per SM, CTA ``b``
# running the chunks ``b``, ``b + grid``, ... of 8-row blocks) past
# ``F32_REP_LAST`` and up to ``F32_STATIC_LAST`` rows per SM: past one wave
# and up to two the repeatable backward's CTAs take a second round of
# blocks, the static backward splits those rows more evenly, and the
# band-selection measurement found it saving about 2-7 % (B200) and 5-8 %
# (GB300) of the checkpointed training step (both cuts on the repeatable
# backward's waves of 64 rows per SM); elsewhere, where it measured, it
# found the repeatable backward faster in the step or tying it, and faster
# in the backward alone.  No claimed FP32 backward ships, nor a static one
# on sm_107a.  A BF16 weight gradient runs, past
# its small-row bands, the fused backward (``_fused``, whose CTAs claim their
# blocks) and the repeatable backward (``_rep``, the same schedule with the
# deterministic package's exact weight-gradient sum): the fused backward's
# weight gradient may vary in its last bits from run to run (and so may the
# row-claims backwards', ``_rc``, ``_rc_mid``, which no band runs), so it
# runs only where the band-selection measurement found the checkpointed
# training step about 3 % or more faster than with the repeatable backward
# (near a band's edges the gain can be smaller); every other band repeats
# its bits.  On sm_100a and sm_103a the fused backward runs up to 1000 rows per
# SM and the repeatable backward past it.  On sm_107a the fused backward
# saved about 3 % or more of the step only past 64 and up to 256 rows per SM (both
# cuts on the repeatable backward's waves of 64 rows per SM), so the
# repeatable backward runs up to ``SM107A_BF16_REP_LAST`` rows per SM, the
# fused backward up to ``SM107A_BF16_FUSED_LAST`` and the repeatable backward
# again past it.
ROWS = "rows"
# The sm_107a BF16 cuts in rows per SM: the end of the first repeatable band
# and the end of the fused band.
SM107A_BF16_REP_LAST = 64
SM107A_BF16_FUSED_LAST = 256
# The sm_100a and sm_103a FP32 cuts in rows per SM: the end of the first
# repeatable band and the end of the static band.
F32_REP_LAST = 64
F32_STATIC_LAST = 128
BACKWARD_BANDS: dict[
    str, dict[str, tuple[tuple[tuple[str, int] | float | None, str], ...]]
] = {
    "sm_100a": {
        "bwd_bf16": (
            ((ROWS, 256), "bwd_bf16_short"),
            ((ROWS, 512), "bwd_bf16_t512"),
            (1000, "bwd_bf16_fused"),
            (None, "bwd_bf16_rep"),
        ),
        "bwd_f32": (
            ((ROWS, 512), "bwd_f32_t512"),
            (F32_REP_LAST, "bwd_f32_rep"),
            (F32_STATIC_LAST, "bwd_f32"),
            (None, "bwd_f32_rep"),
        ),
    },
    "sm_103a": {
        "bwd_bf16": (
            ((ROWS, 256), "bwd_bf16_short"),
            (1000, "bwd_bf16_fused"),
            (None, "bwd_bf16_rep"),
        ),
        "bwd_f32": (
            ((ROWS, 512), "bwd_f32_t512"),
            (F32_REP_LAST, "bwd_f32_rep"),
            (F32_STATIC_LAST, "bwd_f32"),
            (None, "bwd_f32_rep"),
        ),
    },
    "sm_107a": {
        "bwd_bf16": (
            ((ROWS, 256), "bwd_bf16_short"),
            ((ROWS, 512), "bwd_bf16_t512"),
            (SM107A_BF16_REP_LAST, "bwd_bf16_rep"),
            (SM107A_BF16_FUSED_LAST, "bwd_bf16_fused"),
            (None, "bwd_bf16_rep"),
        ),
        "bwd_f32": (
            ((ROWS, 512), "bwd_f32_t512"),
            (None, "bwd_f32_rep"),
        ),
    },
}

# A band role's long role (the band's name + ``LONG``) runs in its place on
# packed rows (``dy``, ``x`` and ``z`` with head stride 128 and token stride
# ``H * 128``, the kernel's own test) from this many rows per SM, by
# architecture and band role: the repeatable backward's long program (its
# look-ahead loop, the same bits) is faster there.  The cuts are the
# deterministic package's for the same pairs of programs (its BF16 and its
# FP32 backward), the same for both weight-gradient dtypes: 2560 rows per SM
# on sm_100a and sm_103a and 1536 on sm_107a; of a role with two repeatable
# bands only the last reaches it (of a BF16 weight gradient on sm_107a past
# ``SM107A_BF16_FUSED_LAST``, of an FP32 one on sm_100a and sm_103a past
# ``F32_STATIC_LAST``).  Other layouts and fewer rows run the band's module.
# Host plan only, as the bands.
LONG = "_long"
LONG_BACKWARD_ROWS_PER_SM: dict[str, dict[str, int]] = {
    "sm_100a": {"bwd_bf16_rep": 2560, "bwd_f32_rep": 2560},
    "sm_103a": {"bwd_bf16_rep": 2560, "bwd_f32_rep": 2560},
    "sm_107a": {"bwd_bf16_rep": 1536, "bwd_f32_rep": 1536},
}

# The suffixes of the roles of a backward without the weight gradient
# (:func:`nodw_module`): a band role's program built without the weight
# gradient is ``band + NODW`` (claimed, or one cluster for a small-row band),
# its strided program ``band + NODW + STRIDED`` (on packed rows its
# 128-thread build ``band + NODW + STRIDED + PACKED`` where routed: the
# repeatable band's on sm_107a) and its long role's ``band + NODW + LONG``.
# A call with ``compute_grad_weight=False`` takes the BF16 bands and, past
# the long cut on packed rows, runs the band's long program without the
# weight gradient (or, where none is routed, the band's long role with
# ``weight_grad`` 0); else the strided program up to this many
# rows per SM, by architecture and band role; else the claimed program; else
# the strided program; else the band's module with ``weight_grad`` 0.  A band
# has a cut where its architecture routes both its strided and its claimed
# program: the fused band on sm_100a and sm_103a, the cuts measured per
# architecture (CUDA Graph replays, cold L2), up to which the strided program
# is the faster.  The repeatable band routes only its claimed program there
# (its strided program is slower in the band's rows), and sm_107a routes no
# claimed program (its strided programs are as fast or faster at every row
# count measured), so neither has a cut.  A strided program launches
# ``min(blocks, ctas_per_sm * SMs)`` CTAs of its own rule (no claim and no
# ticket): ``min(ceil(rows / 16), 4 * SMs)`` 256-thread CTAs over 16-row
# blocks (in the fused band the fused backward's grid), the 128-thread
# program of the repeatable band's packed rows ``min(ceil(rows / 8), 8 *
# SMs)`` CTAs over 8-row blocks: measured on sm_107a, the 128-thread build is
# the faster on packed rows and the 256-thread one on other layouts once a
# 128-thread CTA walks two or more blocks.  Host plan only, as the bands.
NODW = "_nodw"
STRIDED = "_strided"
PACKED = "_packed"
NODW_STRIDED_ROWS_PER_SM: dict[str, dict[str, int]] = {
    "sm_100a": {"bwd_bf16_fused": 512},
    "sm_103a": {"bwd_bf16_fused": 448},
    "sm_107a": {},
}


ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The kernels of a call.  The forward runs on every architecture, its other
# roles (``fwd_tiny``, ``fwd_short``, ``fwd_mid``, ``fwd_long``,
# ``fwd_xlong``) in the rows of their ``FORWARD_BANDS`` bands; a backward
# role (its own module, the static backward, where an architecture routes
# one: ``bwd_bf16`` on sm_107a, which no band runs; ``bwd_f32`` on sm_100a
# and sm_103a, in a band of rows per SM), its small-row roles
# (``BAND_ROLES``), its fused and repeatable roles
# (``<role>_fused``, ``<role>_rep``: ``bwd_bf16_fused``, ``bwd_bf16_rep``
# and ``bwd_f32_rep``, every architecture) and the repeatable roles' long
# roles (``bwd_bf16_rep_long`` and ``bwd_f32_rep_long``, every
# architecture, on packed rows from ``LONG_BACKWARD_ROWS_PER_SM`` rows per
# SM) in the rows of their ``BACKWARD_BANDS`` bands, which cover every row
# count; for a BF16 weight gradient its row-claims roles (``<role>_rc``,
# ``<role>_rc_mid``; sm_100a and sm_103a) and its mid role (``<role>_mid``,
# the claimed backward) are routed, but no band runs them; the programs
# without the weight gradient (``NODW_ROLES``) run in their bands' places
# for calls with ``compute_grad_weight=False`` (:func:`nodw_module`).
ROLES = ("fwd", "bwd_bf16", "bwd_f32")
# The backward's small-row roles, which ``BACKWARD_BANDS`` routes by the rows.
BAND_ROLES = (
    "bwd_bf16_short",
    "bwd_bf16_t512",
    "bwd_f32_t512",
)
# The roles of the programs built without the weight gradient, which a call
# with ``compute_grad_weight=False`` runs in its band's place where routed
# (:func:`nodw_module`): the small-row backwards', the fused backward's and
# the repeatable backward's (claimed and strided; the repeatable one's also
# strided for packed rows), and the repeatable backward's long role's.
NODW_ROLES = (
    "bwd_bf16_short_nodw",
    "bwd_bf16_t512_nodw",
    "bwd_bf16_fused_nodw",
    "bwd_bf16_fused_nodw_strided",
    "bwd_bf16_rep_nodw",
    "bwd_bf16_rep_nodw_strided",
    "bwd_bf16_rep_nodw_strided_packed",
    "bwd_bf16_rep_nodw_long",
)
GRID_KINDS = (
    "row_blocks",
    "persistent_row_blocks",
    "chunks",
    "row_claims",
    "claimed_row_blocks",
    "cluster",
    "strided_row_blocks",
)
# The backward's workspace buffers in their order in one workspace, each with
# the bytes of one element: the u64 windows, the FP32 replicas (a
# backward whose record names ``dw_sums``: the row-claims and the fused
# backwards), the
# FP32 class words and the u32 arrival ticket and claim counter; for a
# repeatable backward (the deterministic package's fused backward program,
# either weight gradient) its u64 fixed-point digits, its u32 flag word a
# column and its u32 arrival ticket and claim counter.  A module's workspace
# takes the buffers its record counts, in this order.
WORKSPACE_BUFFERS: tuple[tuple[str, int], ...] = (
    ("dw_win", 8),
    ("dw_sums", 4),
    ("dw_spec", 4),
    ("dw_done", 4),
    ("acc", 8),
    ("flags", 4),
    ("done", 4),
)
# Each workspace buffer starts at a multiple of this many bytes.
WORKSPACE_ALIGNMENT = 16
# Elements of one normalized row.
HEAD_DIM = 128
# The persistent forwards read rows with 32-byte vector accesses (the forward
# and the backward with 16-byte ones): a tensor is read in place when its
# storage offset and its token and head strides are multiples of this many
# BF16 elements.
ALIGNMENT_ELEMENTS = 16
# The kernels index rows with 32-bit integers.
INT32_MAX = 2**31 - 1
# A forward of packed rows (the tiny forward, the pipelined forward of packed
# rows) takes the offset of an element of row ``n`` as ``n * 128 + column`` in
# 32-bit arithmetic: ``FORWARD_BANDS`` serves fewer rows than this (2**31 //
# HEAD_DIM), a call of more takes ``STRIDED_FORWARD_BANDS``.
PACKED_ROWS = 16777216


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_module(arch: str, role: str) -> str:
    """The module of ``role`` on ``arch`` (``ValueError`` when none is exported)."""

    name = ROUTES.get(arch, {}).get(role)
    if name is None:
        raise ValueError(
            f"no exported Cake SiLU-gated RMSNorm {role!r} module on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return name


def routes_every_call(arch: str) -> bool:
    """Whether ``arch`` routes a module for every call of every role: every
    band of the forward's ``FORWARD_BANDS`` and ``STRIDED_FORWARD_BANDS`` and
    of each backward role's ``BACKWARD_BANDS`` names a role ``ROUTES[arch]``
    routes, a role whose bands do not end with a ``None`` band routes its
    own module (:func:`module_for_rows` falls back to it past the bands), and
    every band role ``LONG_BACKWARD_ROWS_PER_SM[arch]`` names routes its long
    role."""

    routes = ROUTES.get(arch, {})
    tables: list[tuple[str, tuple[tuple[object, str], ...]]] = [
        ("fwd", FORWARD_BANDS.get(arch, ()))
    ]
    tables.append(("fwd", STRIDED_FORWARD_BANDS.get(arch, ())))
    for role in ROLES:
        if role != "fwd":
            tables.append((role, BACKWARD_BANDS.get(arch, {}).get(role, ())))
    for role, bands in tables:
        if any(band not in routes for _, band in bands):
            return False
        if all(last is not None for last, _ in bands) and role not in routes:
            return False
    return all(
        band + LONG in routes for band in LONG_BACKWARD_ROWS_PER_SM.get(arch, {})
    )


def backward_chunks(name: str, rows: int) -> tuple[int, int]:
    """``(blocks_per_chunk, chunks)`` of the static or claimed backward module
    ``name`` for ``rows`` rows: blocks of ``rows_per_block`` rows,
    ``blocks_per_chunk = max(1, ceil(blocks / chunk_limit))`` consecutive
    blocks a chunk (the chunks of the deterministic backward at ``2e989556``)
    and ``ceil(blocks / blocks_per_chunk)`` chunks."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] != "chunks":
        raise ValueError(f"module {name} does not cut its rows into chunks")
    blocks = -(-int(rows) // int(rule["rows_per_block"]))
    per_chunk = max(1, -(-blocks // int(rule["chunk_limit"])))
    return per_chunk, -(-blocks // per_chunk)


def module_for_rows(
    arch: str, role: str, rows: int, sm_count: int, *, packed: bool = True
) -> str:
    """The module that runs ``role`` (one of ``ROLES``) for ``rows`` rows on
    ``arch`` with ``sm_count`` SMs: for the forward the module of the first of
    its ``FORWARD_BANDS[arch]`` bands (``STRIDED_FORWARD_BANDS[arch]`` when
    ``packed`` is false: ``x`` or ``z`` does not hold row ``n`` at element
    ``128 n``, or the call has ``PACKED_ROWS`` rows or more) with ``rows <=
    last * sm_count`` (or no ``last``); for a
    backward role the module of the first of its ``BACKWARD_BANDS[arch]``
    bands whose ``last`` holds: ``(ROWS, n)`` with ``rows <= n``, a number
    with ``rows <= last * sm_count``, or ``None``, and in its place the
    band's long role (``band + LONG``, the same bits) when ``packed`` holds
    (``dy``, ``x`` and ``z`` hold packed rows: the caller's test) and the
    rows reach the band's ``LONG_BACKWARD_ROWS_PER_SM[arch]`` cut, ``rows >=
    cut * sm_count``; else :func:`route_module`.  ``packed`` defaults to
    true, the forward's packed rows: a backward caller passes its own test."""

    if role == "fwd":
        packed = packed and int(rows) < PACKED_ROWS
        table = FORWARD_BANDS if packed else STRIDED_FORWARD_BANDS
        for last, band in table.get(arch, ()):
            if last is None or int(rows) <= int(last) * int(sm_count):
                return route_module(arch, band)
        return route_module(arch, role)
    for bound, band in BACKWARD_BANDS.get(arch, {}).get(role, ()):
        name = route_module(arch, band)
        if isinstance(bound, tuple):
            runs = int(rows) <= int(bound[1])
        else:
            runs = bound is None or int(rows) <= bound * int(sm_count)
        if runs:
            cut = LONG_BACKWARD_ROWS_PER_SM.get(arch, {}).get(band)
            if packed and cut is not None and int(rows) >= cut * int(sm_count):
                return route_module(arch, band + LONG)
            return name
    return route_module(arch, role)


def nodw_module(
    arch: str, name: str, rows: int, sm_count: int, *, packed: bool = False
) -> str:
    """The module of a backward without the weight gradient for ``rows``
    rows on ``arch`` with ``sm_count`` SMs, given ``name``, the module
    :func:`module_for_rows` names for the call (the BF16 bands' role
    ``bwd_bf16``) and its band role ``band`` (the record's role), and
    ``packed``, whether ``dy``, ``x`` and ``z`` hold packed rows (the
    caller's test, as for :func:`module_for_rows`): for a long role (packed
    rows past its cut) the band's long program without the weight gradient
    (``band + NODW + LONG``, ``band`` the long role's band) where ``arch``
    routes one, else ``name`` itself (run with ``weight_grad`` 0); otherwise
    the strided program (``band + NODW + STRIDED``; for packed rows ``band +
    NODW + STRIDED + PACKED`` in its place where ``arch`` routes that too)
    where ``arch`` routes one and ``rows <=
    NODW_STRIDED_ROWS_PER_SM[arch][band] * sm_count``, else the band's
    program without the weight gradient (``band + NODW``: claimed, or one
    cluster for a small-row band) where routed, else the strided program
    where routed (a band without a claimed program has no cut: its strided
    program runs every row count), else ``name``.  Every one gives the
    ``dx`` and ``dz`` bits of ``name``."""

    routes = ROUTES.get(arch, {})
    band = MODULES[name]["role"]
    if band.endswith(LONG):
        return routes.get(band[: -len(LONG)] + NODW + LONG, name)
    strided = routes.get(band + NODW + STRIDED)
    if packed and strided is not None:
        strided = routes.get(band + NODW + STRIDED + PACKED, strided)
    cut = NODW_STRIDED_ROWS_PER_SM.get(arch, {}).get(band)
    if strided is not None and cut is not None and int(rows) <= cut * int(sm_count):
        return strided
    own = routes.get(band + NODW)
    if own is not None:
        return own
    return name if strided is None else strided


def backward_ctas(name: str, rows: int, sm_count: int) -> int:
    """CTAs of the backward module ``name`` for ``rows`` rows on a device with
    ``sm_count`` SMs: the static and the claimed backward ``min(C,
    ctas_per_sm * sm_count)`` for ``C`` chunks (:func:`backward_chunks`), so
    every chunk has a CTA while ``C`` fits one wave; a row-claims backward
    ``min(B, ctas_per_sm * sm_count)`` for ``B`` blocks of ``rows_per_cta``
    rows (its CTAs claim the blocks at run time), and so does a fused
    backward, the repeatable backward the same for blocks of
    ``rows_per_block`` rows, and so does a strided program without the
    weight gradient (its CTAs walk their blocks); a small-row backward its
    cluster's ``cluster`` CTAs whatever the rows."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] == "cluster":
        return int(rule["cluster"])
    if rule["kind"] == "row_claims":
        units = -(-int(rows) // int(rule["rows_per_cta"]))
    elif rule["kind"] in ("claimed_row_blocks", "strided_row_blocks"):
        units = -(-int(rows) // int(rule["rows_per_block"]))
    elif rule["kind"] == "chunks":
        units = backward_chunks(name, rows)[1]
    else:
        raise ValueError(f"module {name} is not a backward module")
    return min(units, max(1, int(rule["ctas_per_sm"]) * int(sm_count)))


def workspace_layout(name: str) -> tuple[tuple[str, int, int], ...]:
    """``(buffer, byte offset, elements)`` of each workspace buffer of the
    backward module ``name`` in ``WORKSPACE_BUFFERS`` order, each starting at
    a multiple of ``WORKSPACE_ALIGNMENT`` bytes; empty for a small-row
    backward, which sums the weight gradient on chip and takes no workspace,
    and for a strided or small-row program without the weight gradient,
    which reads none."""

    counts = MODULES[name].get("workspace")
    if not counts:
        return ()
    layout = []
    offset = 0
    for buffer, size in WORKSPACE_BUFFERS:
        if buffer not in counts:
            continue
        offset = -(-offset // WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT
        layout.append((buffer, offset, int(counts[buffer])))
        offset += int(counts[buffer]) * size
    return tuple(layout)


def workspace_bytes(name: str) -> int:
    """Bytes of the workspace of the backward module ``name``: its buffers
    (:func:`workspace_layout`) end to end; 0 for a small-row backward and
    for a strided or small-row program without the weight gradient."""

    sizes = dict(WORKSPACE_BUFFERS)
    return max(
        (
            offset + elements * sizes[buffer]
            for buffer, offset, elements in workspace_layout(name)
        ),
        default=0,
    )


def launch_grid(
    record: dict[str, Any], *, rows: int, sm_count: int, ctas: int = 0
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device with
    ``sm_count`` SMs (``ctas``: the backward's CTA count,
    :func:`backward_ctas`).  The long and mid forwards take the fewest whole
    waves of ``ctas_per_sm * sm_count`` CTAs whose CTAs walk at most
    ``max_blocks_per_cta`` blocks of ``rows_per_block`` rows each, and at most
    one CTA per block; a rule with ``persistent_from_rows_per_sm`` (the
    pipelined forwards) takes one CTA per block below that many rows per SM.
    A small-row backward takes one cluster of ``cluster`` CTAs; every other
    backward (a strided program without the weight gradient among them)
    ``ctas``."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "row_blocks":
        return (-(-int(rows) // int(rule["rows_per_cta"])), 1, 1)
    if kind == "persistent_row_blocks":
        blocks = -(-int(rows) // int(rule["rows_per_block"]))
        start = rule.get("persistent_from_rows_per_sm")
        if start is not None and int(rows) < int(start) * int(sm_count):
            return (blocks, 1, 1)
        wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
        waves = max(1, -(-blocks // (wave * int(rule["max_blocks_per_cta"]))))
        return (min(blocks, waves * wave), 1, 1)
    if kind in ("chunks", "row_claims", "claimed_row_blocks", "strided_row_blocks"):
        return (int(ctas), 1, 1)
    if kind == "cluster":
        return (int(rule["cluster"]), 1, 1)
    raise ValueError(
        f"unknown Cake SiLU-gated RMSNorm grid rule: {kind!r} (known: {GRID_KINDS})"
    )


@functools.cache
def device_sm_count(device_index: int) -> int:
    """SM count of one CUDA device, read once per device (no device work)."""

    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            "Cake SiLU-gated RMSNorm target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake SiLU-gated RMSNorm module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else the JIT
    build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up)."""

    return cake_build.jit_specs(arch)


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps logical names to tensors and Python scalars; the module
    record maps its kernel parameter names onto that vocabulary.  The launch
    grid is the module's :func:`launch_grid` on the device's SM count (the
    backward reads ``ctas`` from ``values``), and the kernel runs on the
    current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(
        record,
        rows=rows,
        sm_count=device_sm_count(device_index),
        ctas=int(values.get("ctas", 0)),
    )
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ALIGNMENT_ELEMENTS",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BACKWARD_BANDS",
    "BAND_ROLES",
    "FORWARD_BANDS",
    "GRID_KINDS",
    "HEAD_DIM",
    "INT32_MAX",
    "LONG",
    "LONG_BACKWARD_ROWS_PER_SM",
    "MODULES",
    "NODW",
    "NODW_ROLES",
    "NODW_STRIDED_ROWS_PER_SM",
    "PACKED",
    "PACKED_ROWS",
    "ROLES",
    "ROUTES",
    "STRIDED",
    "STRIDED_FORWARD_BANDS",
    "WORKSPACE_ALIGNMENT",
    "WORKSPACE_BUFFERS",
    "arch_for_capability",
    "backward_chunks",
    "backward_ctas",
    "device_arch",
    "device_sm_count",
    "launch_grid",
    "library_name",
    "load",
    "module_for_rows",
    "nodw_module",
    "route_module",
    "run_module",
    "specs_for_arch",
    "workspace_bytes",
    "workspace_layout",
]
