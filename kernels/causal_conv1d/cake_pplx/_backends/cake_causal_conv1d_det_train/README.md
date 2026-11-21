# Cake deterministic packed causal conv1d + SiLU training backend (SM100 / SM103 / SM107)

Backend package of the deterministic (`det`) packed width-4 depthwise causal
convolution + SiLU training routes: the single-rank route
(`cake_pplx.cake_causal_conv1d_det_train`) and the context-parallel (KCP)
route (`cake_pplx.cake_causal_conv1d_det_train_kcp`, module `kcp` of this
package).  The facades are the stable APIs; this package is their backend.

## Single-rank route

The stable entry points live in `cake_pplx/cake_causal_conv1d_det_train.py`; that module imports this package on first use, and nothing selects it automatically (check `is_cake_causal_conv1d_det_train_supported(device)` and call it).

### Entry points (`cake_pplx/cake_causal_conv1d_det_train.py`)

```python
y = cake_causal_conv1d_det_train_forward(x, weight, cu_seqlens, bias=None, out=None, workspace=None)
dx, dweight = cake_causal_conv1d_det_train_backward(x, weight, cu_seqlens, dy)                 # no bias
dx, dweight, dbias = cake_causal_conv1d_det_train_backward(x, weight, cu_seqlens, dy, bias=bias)
dx, dweight = cake_causal_conv1d_det_train_backward(x, weight, cu_seqlens, dy,                 # opt-in pair sums
                                                    silu_grad_preactivation="fp32", dweight_sum="fp32_pair")
y = cake_causal_conv1d_det(x, weight, cu_seqlens, bias,                                        # autograd
                           silu_grad_preactivation="bf16", bwd_grid_mult=1, dweight_sum="fp32")
fws = cake_causal_conv1d_det_train_forward_workspace(S, C, device)
ws = cake_causal_conv1d_det_train_backward_workspace(S, C, device, bwd_grid_mult=1)
is_cake_causal_conv1d_det_train_supported(device)
```

`x` is one packed BF16 `[S, C]` (or `[1, S, C]`) activation, and `C` must be a multiple of 8. `cu_seqlens` holds the document boundaries as int32 `[num_docs + 1]` prefix sums on the device. Each document starts with zero history. The taps `weight[C, 4]` and the optional `bias[C]` can each be FP32 or BF16, independently of each other. Arithmetic is FP32, and the bias is added last, in the same order as flash-linear-attention. `y` and `dx` are BF16. `dweight` and `dbias` are FP32 or BF16: they take the dtype of their parameter unless you pass preallocated outputs.

The kernels access the operands as vectors, so each address (`data_ptr()`) must be aligned: `x`, `dy`, `out` and `dx` to 16 bytes, FP32 taps to 16 and BF16 taps to 8, a BF16 bias to 16, and an FP32 bias to 32 (16 for a standalone backward call). A misaligned tensor raises `ValueError` before anything launches. A fresh tensor or a `.clone()` is aligned, and so is a view that starts at a whole row (`x[1:]`, `weight[1:]`).

- **Forward:** one ordinary launch. Co-resident CTAs take token chunks from int32 counters in the forward workspace. The workspace must be zero before the first launch, and every launch leaves it zero.
- **Forward at small S:** from version 0.2.5, inside the token range of a row of `cake_jit.FALLBACK_ROWS` or `cake_jit.DIRECT_ROWS` (C 576-768, by architecture and route), the forward runs one of two programs that take no counters. Each row gives its own token range. From version 0.2.6, every architecture and route has a row at each C 576-768 (0.2.5 had none for sm_100a `fwd_bias` at C 640, sm_103a `fwd_bias` at C 576 and 712, and sm_103a `fwd` at C 768). For most C the first range starts at S 1. The sm_103a `fwd_bias` rows at C 720-768 start at S 13824 (C 736: 14336) and have only a static range. Outside a row's range, and where an architecture, route and C have no row, the forward runs as in 0.2.4, with counters in the workspace. The static program runs on the route's tile plan. The direct program runs on its own channel tiles (`cake_jit.DIRECT_PLANS`) with at most one CTA per token block. Inside a row's range the call zeroes and uses no forward workspace, eager or captured, and a passed `workspace` is checked and not read. The forward workspace size is unchanged. Every forward program writes the same `y` bits.
- **Backward:** two ordinary (non-cooperative) launches on one stream. A persistent main program writes `dx` and one FP32 partial row of `dweight` (and `dbias`) per CTA. A reduce program then sums those rows in a fixed order. No floating-point atomics are used.
- **Autograd:** saves only `x`, `weight`, `cu_seqlens` and `bias`. The backward recomputes the preactivation.
- **Libraries:** load prebuilt from `aot/` when present; otherwise they are JIT-built from `build.json`. Once built, the calls can be captured in a CUDA Graph. Build every library for the arch before capture (see `cake_jit.specs_for_arch`).
- The host never reads `cu_seqlens`, so the calls do not sync with the host. `S == 0` returns empty outputs without launching.

### Determinism, per output

Below, "same call" means the same inputs, device (architecture and SM count), package version and keyword settings. The keyword settings are the taps, bias and output dtypes, `silu_grad_preactivation`, `bwd_grid_mult` and `dweight_sum`. The package's GPU tests repeat calls and compare the bits on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).

| Output | Repeats bit for bit when | What changes its bits | Accuracy vs exact FP64 |
| --- | --- | --- | --- |
| `y` | same call, whichever CTA computes a chunk | Not `bwd_grid_mult`. May change between package versions | one BF16 rounding of the FP32 result |
| `dx` | same call | Not `bwd_grid_mult` or the backward tile plan. May change between package versions | within one BF16 ulp under both `silu_grad_preactivation` policies |
| `dweight` | same call, including `bwd_grid_mult` and `dweight_sum` | Summation order follows the backward tile plan, the device's SM count and `bwd_grid_mult`, so different settings or SM counts differ in the last bits. `dweight_sum="fp32_pair"` changes the last bits. Package version | FP32 output: about 3e-5 rel-L2 with `"bf16"`; with `"fp32"` the rounded preactivation no longer contributes. By default (`dweight_sum="fp32"`) every sum is one plain FP32 word; `"fp32_pair"` makes it an FP32 (hi, lo) pair. BF16 output: limited by BF16 storage |
| `dbias` | same as `dweight` | same as `dweight` | same as `dweight` |

- With `silu_grad_preactivation="bf16"` (the default), SiLU' is evaluated at the BF16-rounded preactivation when `weight` is FP32, the same boundary flash-linear-attention uses, and at the unrounded FP32 preactivation when `weight` is BF16. With `"fp32"`, it is evaluated at the unrounded FP32 preactivation. The `"fp32"` program is slower. By default (`dweight_sum="fp32"`), every `dweight` and `dbias` sum is one plain FP32 word under both policies: each thread sums its tokens in FP32, the `"fp32"` programs fold those sums into shared memory every few token stages, and the reduce program adds the FP32 partial rows in a fixed order. The default has no compensated (two-word) sums. In version 0.2.0 this changes the `"fp32"` routes' `dweight` and `dbias` bits from 0.1.0. The opt-in `dweight_sum="fp32_pair"` (with `"fp32"` only) runs the pair programs: the fold is Kahan-compensated, the reduce program adds the partial rows as FP32 (hi, lo) pairs in the same fixed order, and the result is rounded once. The pair programs keep `dx`, change the last bits of `dweight` and `dbias`, and repeat bit for bit for the same call. No program uses FP64.
- Single-tile rows run narrower channel tiles. Every backward route runs 32-channel-vector tiles at C 224 and C 520-1016, except that from a per-architecture token floor (S 65536 on sm_100a, 24576 on sm_103a, 32768 on sm_107a) the default backward runs C 520-615 (C 520-687 on sm_107a) on its SM-wide program, one whole channel tile per CTA; below the floor these rows keep the 32-vector tiles. Every forward route runs 32-vector tiles at C 576-768. From a per-architecture token floor, C 576 runs whole 18-vector tiles and C 640 runs whole 40-vector tiles; below the floor these rows keep the 32-vector tiles. Inside the token ranges of `cake_jit.DIRECT_ROWS`, the direct program runs its own `cake_jit.DIRECT_PLANS` tiles instead. From version 0.2.4, every forward route runs C 520-568 on narrower tiles from a token floor per architecture and channel count (S 12288 to 62585, `cake_jit.NARROW_ROW_FLOORS`): C 520, 528, 544, 552 and 560 run whole tiles of 13, 11, 17, 23 and 14 vectors, and C 536 and 568 run 32-vector tiles. Below the floor these rows keep one tile. At C 520-568 the forward workspace grows from 2 to 6-12 int32 words, so a forward workspace sized by an earlier version can raise `ValueError` there. The backward tile plan sets the `dweight` and `dbias` summation order. The forward tile plan never changes `y` bits. At those backward widths, `dweight` and `dbias` bits differ from versions before the 32-vector backward tiles, and in the SM-wide band from the floor `dweight` bits differ from versions before 0.1.0. `y` and `dx` keep their bits, and the accuracy bounds are unchanged.
- Because the backward recomputes the preactivation, a checkpointed (recomputed) forward gives the same `y` bits. The backward from it gives the same gradient bits.

### Keyword controls

There are no environment variables. Every setting is a keyword argument.

| Keyword | Values | Default | Effect |
| --- | --- | --- | --- |
| `silu_grad_preactivation` | `"bf16"`, `"fp32"` | `"bf16"` | Sets where the backward evaluates SiLU'. `"fp32"` is more accurate and slower. |
| `bwd_grid_mult` | `1` to `4` | `1` | Splits the backward work more finely. Changes the last bits of `dweight` / `dbias`, never `y` / `dx`. Size the backward workspace with the same or a larger value. |
| `dweight_sum` | `"fp32"`, `"fp32_pair"` | `"fp32"` | The sum form of `dweight` / `dbias`. `"fp32"` sums in plain FP32 words. `"fp32_pair"` is an opt-in: it runs the FP32 (hi, lo) pair programs. They keep `dx` and the workspace and change the last bits of `dweight` / `dbias`. It needs `silu_grad_preactivation="fp32"`, else `ValueError`. Another string raises `ValueError`; a non-string raises `TypeError`. |
| `bias` | `None`, or `[C]` FP32 / BF16 | `None` | Adds the bias. The backward then also returns `dbias`. |
| `out` | BF16 tensor | `None` | Preallocated forward output `y`. |
| `dx`, `dweight`, `dbias` | tensors | `None` | Preallocated backward outputs. `dweight` / `dbias` can be FP32 or BF16. |
| `workspace` | from `cake_causal_conv1d_det_train_forward_workspace` / `cake_causal_conv1d_det_train_backward_workspace` | `None` | Workspace you allocate yourself for the forward or backward. |

### Sharing the GPU with other streams

The backward's main program is persistent, so a kernel that stays resident on another stream slows it down. A repro timed the backward alone and next to a resident kernel on another stream, using CUDA events over 18 shapes each. With `bwd_grid_mult=1`, the median ratio of time next to the side kernel over time alone was 1.43-1.45 on GB300 (sm_103a, 152 SMs) and 1.42-1.48 on R200 (sm_107a, 212 SMs). That range covers three backward routes (default, bias, `"fp32"` SiLU') and two side kernels (one-thread and NCCL-like). A finer split costs some time when the backward runs alone but cuts the slowdown. The table below shows medians for the NCCL-like side kernel on the default route:

| `bwd_grid_mult` | GB300 next-to / alone | GB300 alone (us) | R200 next-to / alone | R200 alone (us) |
| --- | --- | --- | --- | --- |
| 1 | 1.45 | 142.0 | 1.42 | 94.1 |
| 2 | 1.20 | 153.0 | 1.17 | 105.6 |
| 3 | 1.13 | 162.2 | 1.10 | 114.5 |
| 4 | 1.09 | 168.8 | 1.06 | 122.8 |

The default stays 1 because it is fastest alone. Set 2-4 when the backward usually overlaps collectives, and size the backward workspace with the same or a larger `bwd_grid_mult`.

## Context-parallel (KCP) route

Every rank convolves its fragments of a packed batch.  The first three tokens
of a fragment that continues a document read the three document tokens before
it (history rows) from the ranks that hold them; the backward sends the
derivative of those first tokens back to the predecessor fragments.  Tokens
never move otherwise: there is no activation concatenation and no
full-sequence gather.  The route keeps the package's FP32-tap contract: FP32
taps, no bias, FP32 arithmetic, one BF16 rounding per output, and the SiLU
derivative taken at the FP32 preactivation.

### Entry points

```python
from cake_pplx.cake_causal_conv1d_det_train_kcp import (
    cake_causal_conv1d_det_kcp,          # autograd, one tensor
    cake_causal_conv1d_det_kcp_multi,    # autograd, n tensors of one layer (q / k / v), one exchange per direction
    CakeCausalConv1dDetKcpRecomputeHandle,
)

y = cake_causal_conv1d_det_kcp(x_local, weight, ctx)                     # cooperative=None: False
q, k, v = cake_causal_conv1d_det_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)
```

| operand | shape | dtype | notes |
|---|---|---|---|
| `x_local`, `y`, `dx` | `[T_r, C]` or `[1, T_r, C]` | BF16 | contiguous, 16-byte aligned; the rank's fragments back to back in the order of `ctx.cu_seqlens_local`; `C` a multiple of 8; `T_r` may be 0 |
| `weight`, `dweight` | `[C, 4]` | FP32 | 16-byte aligned; `weight[c, 3]` multiplies the current token; `dweight` is the rank-local partial |
| `hist` | `[n_hist_rows, C]` | BF16 | the forward's history pool, passed back to the backward unchanged; contiguous, 16-byte aligned |
| `ctx` | | | the rank's KCP context, built by the calling family's context builder (for example `cake_kda_det_train_kcp_build_context(..., conv_kernel_size=4)`): fragment table, halo plan, device tables, process group |

A misaligned operand raises `ValueError` before anything launches; a fresh
tensor or a `.clone()` is aligned.

Every rank of `ctx.group` calls the entry once per layer, in the same order,
with the same tensor count and channel counts; the ranks are not compared (a
mismatch hangs the exchange or corrupts the history rows).  The module `kcp`
also holds the explicit calls: `cake_causal_conv1d_det_train_kcp_forward`
(returns `y` and the history pool `hist`), `cake_causal_conv1d_det_train_kcp_backward`
(returns `dx` and the `dweight` partial) and their `_multi_` forms, with the
keyword `graph="replay"` (keyed CUDA-graph capture and replay, bitwise the
eager result; call `cake_causal_conv1d_det_train_kcp_graph_cache_clear()`
before `torch.distributed.destroy_process_group()`).  Every switch is a keyword
argument; the backend reads no environment variable.  The KCP route has no
`dweight_sum` keyword: it offers no FP32 (hi, lo) pair opt-in, and its
`dweight` is summed in plain FP32 words.

### Launch forms (`cooperative`)

The keyword selects the launch form of the backward's main program; the
forward and the backward's head program are the same in both forms.  The
forward's `kcp_fwd` program is also an ordinary launch one SM short of the
co-resident CTA count (`kcp.FWD_RESERVE_SMS`) whenever that grid holds every
channel tile of `C` (else it takes the full grid).  Beside one small kernel of
another stream that occupies a single SM, all CTAs of the shorter grid can then
start at once; a kernel on more SMs still makes some of them wait for a later
wave.  The single-rank main program of the two-launch forward (below) launches
its record's full co-resident grid.  `y` does not depend on its grid.

| `cooperative` | program | behavior |
|---|---|---|
| `False` (default; `None` = `kcp.COOPERATIVE_DEFAULT`) | `kcp_bwd_split` + its `reduce` program | an ordinary launch one SM short of the co-resident CTA count writes per-CTA partial rows; a programmatically dependent reduce program on the same stream sums them in a fixed order.  It starts beside a small kernel resident on another stream (for example a communication kernel); more such kernels delay some CTAs, never the whole launch |
| `True` | `kcp_bwd` | one cooperative launch whose grid barrier precedes the fixed-order `dweight` reduction.  It starts only when all its CTAs fit at once, so it waits for every kernel of another stream that occupies an SM |

Every rank of a group passes the same `cooperative` value.  The form changes
no collective, so a mismatch does not hang, but the ranks' `dweight` partials
would then mix two reduction orders.

With `cooperative=False` at more than one rank the backward overlaps its halo
exchange: the `kcp_bwd_split` program and its reduce program run while the
exchange is in flight, as if no tail row had arrived, and after the wait the
`kcp_bwd_halo` program writes `dx` of the last `min(3, len)` rows of every
fragment with a tail row again, recomputing their `dpre` with the main
program's instructions and folding the received rows with its FMAs in its
order.  `dx` and `dweight` keep the bits of the waiting launch; the exchange
and its wait are unchanged, and one more program runs when the rank receives
tail rows.  On sm_103a and sm_107a the multi entry overlaps only its first
tensor this way and runs the other tensors' programs after the wait, so it
also adds one program per call; on sm_100a, where this measured no gain, the
multi entry waits first (`MULTI_BWD_OVERLAP_ARCHS`).  `cooperative=True` waits
before its launch.

### Two-launch forward

The forward can run as two programs: a main program that reads no received
history row, then the `kcp_fwd_halo` program, which writes `y` of the first
`min(3, len)` rows of every fragment with a history row again from the
received pool with the `kcp_fwd` program's instructions, so `y` keeps the bits
of the one-launch forward.  The second program runs only when the rank
receives history rows.

- Main program: on the architectures of `kcp.FWD_MAIN_ARCHS`, the single-rank
  `fwd` route program (the program the single-rank forward runs for the call's
  `C` and token count) on the rank's fragments, each a document of its own,
  with its plan, grid and token-chunk counters (none for the static and the
  direct program of a small-S row); elsewhere the `kcp_fwd` program.
- Order: on the architectures of `kcp.FWD_OVERLAP_ARCHS`, with an exchange in
  flight (more than one rank), the main program runs before the exchange's
  wait and the `kcp_fwd_halo` program after it; a `kcp_fwd` main program then
  runs with every `hist_row_off` entry `-1`.  Otherwise the forward runs after
  the wait: the single-rank main program and the `kcp_fwd_halo` program, or,
  with the `kcp_fwd` program as main, the one-launch forward (the `kcp_fwd`
  program with the context's table, no second program).
- Multi entry: its own sets, `kcp.MULTI_FWD_MAIN_ARCHS` and
  `kcp.MULTI_FWD_OVERLAP_ARCHS`.  It overlaps its first tensor only; the other
  tensors run after the wait.

The sets come from measurements of each entry against the one-launch forward
(their comment in `kcp.py`).  On every architecture the single call runs the
single-rank main program after the wait; so does the multi entry on sm_100a
and sm_103a, and on sm_107a the multi entry overlaps its first tensor.  The
exchange, its wait and the collectives are the same in every form.

### Halo exchange

One `all_to_all_single` per direction and call:

- Forward: the history rows (up to three per continuing fragment) travel to
  the ranks of the fragments that read them, into the history pool `hist`
  (`[n_hist_rows, C]` BF16).  `hist` is saved for the backward; the backward
  does not exchange it again.
- Backward: the head program writes the FP32 derivative of each continuing
  fragment's first tokens (the halo adjoints; a token feeds at most three
  predecessor fragments) into the send layout; the exchange delivers them to
  the predecessors as `dpre_tail` (`[n_tail_rows, C]` FP32), and the main
  program (in the overlapped backward, the tail-row program after the wait)
  adds them to the last rows' `dx` before the one BF16 rounding.

The multi entry exchanges the rows of its `n` tensors side by side in one
collective per direction and copies each tensor's rows into its own pool after
the wait; each tensor then runs the single call's programs (in the overlapped
backward of sm_103a and sm_107a only the first tensor takes the overlapped
form; the others run the waiting form, whose `dx` and `dweight` are bitwise the
same), so its outputs and gradients are bitwise those of one single call per
tensor with the same `cooperative`.  With one rank no collective is issued.

### Recompute handle

`recompute_handle=CakeCausalConv1dDetKcpRecomputeHandle()` makes the checkpoint
recompute of a call issue no collective: the original forward exchanges and
stores `hist` in the handle (key `"conv"`; `n_hist_rows x C x 2` bytes retained
per tensor), and torch's recompute of that forward inside the backward binds it
and launches the forward's programs on it, with no gather, no collective and no
wait.  The backward still exchanges once.  One handle serves one call (single
or multi) of one micro-step and is created outside the outermost checkpoint
region; every rank passes a handle for the same call or none.  A handle is
refused with `graph="replay"` and inside a CUDA-graph capture.

### Determinism

"Repeats bit for bit" means: the same bits on every call with identical inputs,
the same device type with the same SM count, the same configuration (shapes and
channel counts; graph replay and the eager path give the same bits), the same
CP layout (the context's partition) and the same `cooperative` value on every
rank.  Bits across CP layouts or SM counts are not guaranteed.
`torch.use_deterministic_algorithms(True)` raises no alert on either form.

| output | repeats bit for bit | the two `cooperative` forms | notes |
|---|---|---|---|
| `y` | yes | bitwise equal | within one BF16 ulp of an FP64 oracle; this revision's `y` is also bitwise the single-rank `y` of the same documents (the forward runs the single-rank forward's instructions with history rows), a property of this revision, not a guarantee |
| `hist` (history pool, the saved set's tensor) | yes | bitwise equal | copies of `x_local` rows of other fragments |
| halo adjoints (`dpre` rows sent, `dpre_tail` received) | yes | bitwise equal | written by the head program, which is the same in both forms |
| `dx` | yes | bitwise equal | within one BF16 ulp of an FP64 oracle; not bitwise the single-rank `dx`, which differentiates the BF16-rounded preactivation |
| `dweight` (rank-local FP32 partial) | yes, within one form | bits differ | a fixed-order sum of the rank's partial rows in plain FP32 words; the two forms sum different partial rows |
| checkpoint replay | yes | | with a recompute handle `y` and the saved set `(x_local, weight, hist)` are bitwise those of the exchanging recompute, and the gradients bitwise those of the saved-set path (no checkpoint) |

The package tests judge the L2 and max error of `y`, `dx` and `dweight` against
an FP64 oracle of the unpartitioned documents, relative to a PyTorch BF16
implementation of the same operation, over 32 random draws.

The package issues no reduction collective for `dweight`: each rank returns
its partial, and the sum over the ranks is the framework's (for example the
data-parallel or FSDP reduction of the weight gradient).  Whether that sum
repeats bit for bit depends on the framework's reduction; a reduction in a
fixed rank order keeps it.
