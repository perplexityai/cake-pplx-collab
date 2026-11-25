# Deterministic ragged BF16 MoE grouped GEMM (Cake backend)

This package is the backend of the module
`cake_pplx.cake_moe_grouped_gemm_det_train`. The module's functions are stable
APIs: the one-shot functions
`cake_moe_grouped_gemm_det_train_forward`, `cake_moe_grouped_gemm_det_train_dgrad`
and `cake_moe_grouped_gemm_det_train_wgrad`, the prepared launches
`cake_moe_grouped_gemm_det_train_prepare_forward` / `_prepare_dgrad` /
`_prepare_wgrad`, the autograd entry `cake_moe_grouped_gemm_det` and the support
query `is_cake_moe_grouped_gemm_det_train_supported`. The forward also has a
`grouped_mm_bf16`-shaped handoff, `cake_backend.grouped_det_mm_bf16_cake`.
There is no automatic backend selection.

Three operations over `E` groups of rows, described by an int32 **device**
tensor `offs[E]` of cumulative end offsets (`offs[-1] <= sum_m`; groups may be
imbalanced, empty, one row long, or any size that is not a tile multiple) or
an `m_indptr[E + 1]` (`torch._grouped_mm` / `grouped_mm_bf16` form, whose first
entry may be above 0: device slices, below; `cake_moe_grouped_gemm_det_train_wgrad` needs
`num_groups=E` to tell the two forms apart). The kernels read the offsets on the device; the host never
synchronises on them.

The slice lower bound is `0` for `e == 0` and `offs[e-1]` for `e > 0`.
For `m_indptr` input, `offs` in these formulas means `m_indptr[1:]` and the
lower bound of group 0 is `m_indptr[0]` (device slices, below).

| helper | computation | inputs | output |
| --- | --- | --- | --- |
| `cake_moe_grouped_gemm_det_train_forward(x, w, offs, out=None)` | `Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T` | `x [sum_m, K]`, `w [E, N, K]` bf16 | `Y [sum_m, N]` bf16 |
| `cake_moe_grouped_gemm_det_train_dgrad(g, w, offs, out=None)` | `dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]` (W read in place) | `g [sum_m, N]`, `w [E, N, K]` bf16 | `dX [sum_m, K]` bf16 |
| `cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | `dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]` (with `accumulate=True`: `out[e] += dW[e]`) | `g [sum_m, N]`, `x [sum_m, K]` bf16 | `dW [E, N, K]` bf16 or fp32 |

Alignment: `fwd` needs `N % 256 == 0` and `K % 64 == 0`; `dgrad` needs
`K % 256 == 0` and `N % 64 == 0`; `wgrad` needs `N % 256 == 0` and `K` a
multiple of its k tile (256, or 512 when selected, see below). Outputs may be
row-padded (any leading strides, unit stride along the last dimension); the
padding is never written.

Numerics: fp32 tensor-core accumulation with one bf16 rounding at the store
(the fp32 weight-gradient variant stores the accumulator); bitwise
deterministic (no atomics; the split-K tail sums fp32 partials in a fixed
order); `dW[e] == 0` exactly for empty groups.

## Device slices (`m_indptr[0] > 0`)

An `m_indptr[E + 1]` may start above 0. The call then covers the rows
`[m_indptr[0], m_indptr[E])` of the whole `x` / `g`, in absolute row numbers:
for example the experts `e0 .. e1 - 1` of a larger table, one chunk of experts
at a time, without copying rows or reading the table on the host.

```python
# m_indptr: [E_all + 1] int32 on the device; x: [R, K], g: [R, N]: all routed rows
chunk = m_indptr[e0 : e1 + 1]  # a view; chunk[0] may be above 0
y = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
cake_moe_grouped_gemm_det_train_forward(x, w[e0:e1], chunk, out=y)  # writes rows chunk[0] .. chunk[-1] - 1 of y
dw = cake_moe_grouped_gemm_det_train_wgrad(g, x, chunk, num_groups=e1 - e0)
```

- Rows outside `[m_indptr[0], m_indptr[E])` do not affect the results and are
  not written: the forward and the data gradient leave those rows of `out` as
  they are, so slices of disjoint rows may write one `out`, also from different
  streams. The entries must be nondecreasing row numbers of the operands, at
  most `sum_m`.
- The weight gradient reads the `[E + 1]` form only with `num_groups=E`;
  without it, `E + 1` entries are `E + 1` end offsets.
- The kernels read `m_indptr[0]` on the device at every launch: a prepared
  launch and a CUDA graph that captured it follow new boundaries, the first
  entry included.
- Programs: the `m_indptr` form binds the lead twins of the programs (the
  routes `<route>_lead` of `cake_jit`, programs `<program>_lead` with
  `twin_of`); the `[E]` form
  binds the programs themselves (`launch.bound_routes`, `launch.bound_programs`
  name them). A process JIT-compiles a twin at its first call where the package
  ships no prebuilt build of it.
- Cost: the twins read `m_indptr[0]` once per role, and some launches are
  slightly slower (most a weight gradient with a split tail). When
  `m_indptr[0]` is 0, pass `m_indptr[1:]` (the `[E]` form): it binds the
  original programs at no extra cost and gives the same bits.
- Bits: the plan (k tile, stage, grid, tail split) comes from the shapes, and
  `sum_m` is `x.shape[0]`, every row, not the rows of the slice. A slice gives
  the bits of the host-offset call over its rows (`x[m_indptr[0]:m_indptr[E]]`
  with `offs = m_indptr[1:] - m_indptr[0]`) when the two calls plan the same.
  When they plan differently (for example the rows per group cross
  `k512_min_rows_per_group`), it gives the bits of that host-offset call with
  its rows padded by rows of no group to the same `sum_m`, which plans the
  same. Both results repeat bit for bit and meet the same accuracy bar.
- A slice with no row and `accumulate=True` (`m_indptr[0] == m_indptr[E]`)
  still launches, as the host does not read the table. Its add of exact zeros turns a `-0.0` in `out` into `+0.0`. The values are
  equal. The host-offset call with no row launches nothing.
- Autograd: `cake_moe_grouped_gemm_det(x, w, chunk)` writes only the slice's rows of
  `Y` and of the gradient of `x`; the other rows of these new tensors are
  uninitialized (`torch.empty`). Summing the `x` gradients of several slices
  of one shared `x` is therefore not supported: a sliced layout calls the
  op-level functions with caller-owned outputs.

## Reproducibility and accuracy

Every output repeats bit for bit run to run: the same inputs on the same
device give the same bits at every launch, with other streams running, after
new group boundaries are written into `offs`, in CUDA graph replays and in a
fresh process. No output depends on which cluster computed which tile.

| output | dtype | what is fixed | bit for bit run to run |
| --- | --- | --- | --- |
| `Y` (`fwd`) | bf16 | the k steps of every element, in order | yes |
| `dX` (`dgrad`) | bf16 | the k steps of every element, in order | yes |
| `dW` (`wgrad`) | bf16, fp32 | the k steps in order; a split tail's partials summed by the tail reduce in a fixed order | yes |
| `out += dW`, fused (`accumulate=True`, fp32 `out`) | fp32 | the fp32 `dW` above, then one fp32 add per element of `out` | yes; the same bits as the composed route |
| `out += dW`, composed (`accumulate=True`) | bf16, fp32 | the `dW` above in `out`'s dtype, then `out.add_` | yes |

Accuracy has no tolerance. The tests judge every floating output against a naive
BF16 implementation (`tests/cake/test_helpers/cake_naive.py`, the family adapter
`tests/cake/test_helpers/cake_moe_judge.py`): torch's matmul of the same BF16
operands per group, rounded once to BF16 (widened exactly for an fp32 output).
The oracle is the float64 product of the BF16 operands of each group (an empty
group gives 0); an accumulated `out` has the oracle `out0 + n dW`. Over 32
independent random inputs per case, the output's error against the oracle in
the L2 and in the max norm (absolute, over the finite oracle elements) must
have a median and a largest value at most 1.005 times the naive tier's, and the
error must be 0 on every input where the tier's is 0. Where an op's whole
oracle exceeds 2^37 multiply-adds (the MoE projection rows), the judge takes
sampled rows with all their columns: the first and last row of every group (of
`N` for `dW`) and one row in every 64; the whole output still has to be finite
where the oracle is. The pytest summary prints one `TIER` line per judged
output with both errors.
`test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output`
judges a sample of every group (the first and last 16 rows by 512 columns of
`Y` and `dX`, a 64 x 256 block of each `dW[e]`) of the E4 (112576 balanced and
136064 skewed rows), E8, E32 and E32-with-hot-groups rows (131072 rows), on both
projections (gate/up `N` 4096, `K` 6144; down `N` 6144, `K` 2048).
On each board the fused route gives the composed route's bits.

Known gaps: on B200, GB200, GB300 and VR200 some tests miss the naive tier on a
BF16 output, in the max norm only (their L2 error meets it). They are BF16
rounding near ties: the kernel's FP32 sum lands on the other side of a BF16
rounding midpoint a little more often than torch's. `tests/known_gaps.py` lists
them per GPU model, each output with two caps, one for the ratio of its median
max-norm error over the inputs to the tier's and one for that of its largest:
the largest value of that ratio the validation runs measured on that model,
rounded up to four decimals, or the tier's own 1.005 where the runs met it.
The suites mark each one a strict xfail on that model only; a ratio above its
cap, and any other failure of such a test, still fails.

## Mode rules

The module is the mode: this package and `cake_pplx.cake_moe_grouped_gemm_det_train`
are deterministic only, and every public name carries `det`.

| call | result |
| --- | --- |
| any entry point of this package or of `cake_pplx.cake_moe_grouped_gemm_det_train` | the deterministic programs: fixed reduction order, bitwise reproducible run to run |
| a `deterministic` argument (`True` or `False`), e.g. `cake_moe_grouped_gemm_det(x, w, offs, deterministic=False)` | `TypeError`: no deterministic entry takes one |
| a fourth argument of `CakeMoEGroupedGemmDetFunction.apply(x, w, offs)` other than `None` (e.g. a stale `True` / `False` mode; the fifth is `fuse_wgrad_accumulation`) | `TypeError` |
| `accumulate=True` / `fuse_wgrad_accumulation=True` | the deterministic weight gradient added into `out` / `w.main_grad`, bitwise reproducible run to run (Weight-gradient accumulation, below) |
| `torch.use_deterministic_algorithms(True)` (also `warn_only=True`) | no change: no alert, no warning |
| `cake_backend.grouped_det_mm_bf16_cake` (the `grouped_mm_bf16` handoff) | the deterministic forward (the forward programs of both modes give the same bits) |

The non-deterministic mode is the separate module
`cake_pplx.cake_moe_grouped_gemm_nondet_train` (backend
`cake_pplx._backends.cake_moe_grouped_gemm_nondet_train`); its entries
have the same shapes, carry `nondet` and never route here.

## Usage

```python
import torch
from cake_pplx.cake_moe_grouped_gemm_det_train import (
    cake_moe_grouped_gemm_det,
    cake_moe_grouped_gemm_det_train_dgrad,
    cake_moe_grouped_gemm_det_train_forward,
    cake_moe_grouped_gemm_det_train_prepare_wgrad,
    cake_moe_grouped_gemm_det_train_wgrad,
)

E, N, K = 4, 4096, 6144
sizes = [28128, 28128, 28128, 28192]
sum_m = sum(sizes)
offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
x = torch.randn(sum_m, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
g = torch.randn(sum_m, N, dtype=torch.bfloat16, device="cuda")

y = cake_moe_grouped_gemm_det_train_forward(x, w, offs)  # one-shot: prepares and launches
dx = cake_moe_grouped_gemm_det_train_dgrad(g, w, offs)
dw = cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out_dtype=torch.float32)
y = cake_moe_grouped_gemm_det(x, w, offs)  # autograd wrapper: dX and dW in backward

# Prepare once, launch many (CUDA-graph capturable; new offs / values may be
# written into the bound tensors between launches):
dw_launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(g, x, offs, out_dtype=torch.float32)
dw = dw_launch.launch()
print(dw_launch.record_name, dw_launch.plan)
```

The entries are imported from `cake_pplx.cake_moe_grouped_gemm_det_train`: each
one hands off to the plain function of the same name in this package
(`cake_moe_grouped_gemm_det_train_forward`, `_dgrad` and `_wgrad` in the package
root, `cake_moe_grouped_gemm_det_train_prepare_*` and `CakeMoEGroupedGemmDetLaunch`
in `cake_backend`, `cake_moe_grouped_gemm_det` and `CakeMoEGroupedGemmDetFunction`
in `cake_autograd`). Importing the module loads none of this package. A build that
vendors the package calls those plain functions directly.

`cake_moe_grouped_gemm_det_train_prepare_*` validates, allocates the output (when not given) and
the weight-gradient workspaces, resolves the route and binds the generated
programs, and returns a `CakeMoEGroupedGemmDetLaunch`. `launch()` passes the operands'
TMA descriptors by value (the binding encodes them on the host from the bound
tensors' metadata at every launch) and performs no host synchronisation. It
allocates only the stream workspaces (scheduler and wave buffers; see the
`cake_backend` module docstring): the first launch on a CUDA stream allocates
that stream's buffers and later launches on it allocate nothing, while a launch
during a CUDA-graph capture, or for a device other than the current one, takes
a set of its own (from the graph's pool under capture). Prepare a new launch
when a shape, dtype or tensor binding changes; the one-shot helpers prepare on
every call.

## grouped_mm_bf16 handoff and autograd

The forward projection also has a handoff with the call shape of
`grouped_mm_bf16(a, b, m_indptr, out=None, out_dtype=torch.bfloat16, tactic=-1)`:
`cake_backend.grouped_det_mm_bf16_cake` (bfloat16 outputs only, no tactic
index, `N % 256 == 0`, `K % 64 == 0`; rows of `a` before `m_indptr[0]` and
past `m_indptr[-1]` are left untouched). It rejects the options the backend does not take and emits no
warning.

`cake_moe_grouped_gemm_det(x, w, offs)` is a `torch.autograd.Function`
wrapper (`CakeMoEGroupedGemmDetFunction`): `backward` produces `x.grad` with the `dgrad`
program and `w.grad` with the `wgrad` program in the dtype of `w`. The
programs always reduce in a fixed order and there is no `deterministic`
argument; `cake_moe_grouped_gemm_det_train_wgrad(..., out_dtype=torch.float32)`
gives an fp32 weight gradient, and `fuse_wgrad_accumulation=True` adds it
into an fp32 `w.main_grad` (next section). `CakeMoEGroupedGemmDetFunction` calls the three programs as
plain functions; no torch custom op is registered.
The wrapper launches the forward GEMM after `CakeMoEGroupedGemmDetFunction.apply` returns,
that is after autograd saved `x`, `w` and `offs`, as for a built-in op. Under
`torch.utils.checkpoint(..., use_reentrant=False)` with early stop on (the
default), the recomputation therefore stops before this GEMM when these are
the last saves of the innermost checkpointed region; a later save in that
region (of its output or of any other tensor) makes it run, with the same
bits. A direct `CakeMoEGroupedGemmDetFunction.apply(x, w, offs)` launches inside
`forward` and is always recomputed, as is the wrapper under `torch.jit.trace`
and `torch.compile`.

## Weight-gradient accumulation

`cake_moe_grouped_gemm_det_train_wgrad(..., out=out, accumulate=True)` (also
`cake_moe_grouped_gemm_det_train_prepare_wgrad(..., accumulate=True)` and
`cake_pplx.cake_moe_grouped_gemm_det_train.cake_moe_grouped_gemm_det_train_wgrad`) adds the deterministic weight
gradient into the caller's `out` instead of overwriting it:
`out[e] += G[e].T @ X[e]`, Megatron's `gradient_accumulation_fusion` into an
fp32 or bf16 `main_grad`.
`cake_moe_grouped_gemm_det(..., fuse_wgrad_accumulation=True)` does the same in its
backward into `w.main_grad` (below). Both switches are keyword-only and off by
default; with the defaults the routes, programs, plans and bits are unchanged,
and `plan["accumulate"]` is `"off"`.

The host plans the launch exactly as the overwrite of `out.dtype` (route, k
tile, stage, grid, tail split, raster, wave mode); `plan["accumulate"]` records
how it adds, and every other plan key is the overwrite's:

| `plan["accumulate"]` | when | what runs |
| --- | --- | --- |
| `"fused"` | the 512-wide k tile; the route serves the accumulate build of the selected stage (`stage + "_acc"`) on the architecture; when the tail splits (`plan["tail_splits"] > 1`), the route of that tail (the 256-wide k tile's for `main_halves_k256t`) serves `tail_reduce_acc`; every such build's library can load (below); `out` is 32-byte aligned (base, row pitch, expert pitch) | the overwrite's stages with `_acc` (`stages == (stage + "_acc",)`, or `(stage + "_acc", "tail_reduce_acc")` for a split tail), bound to `out`: the main kernel adds each whole tile into `out` once and writes a split tile's partials as the overwrite does; the tail reduce sums those partials in the overwrite's fixed order and adds the sum into `out` once; no scratch |
| `"composed"` | otherwise; `plan["accumulate_reason"]` is the first condition that fails: `"tile_k"` (the 256-wide k tile), `"stage"` (no accumulate build of the stage for the dtype and architecture: every bf16 `out`, and the `main` / `main_wave` forms), `"tail"` (a split tail whose route serves no `tail_reduce_acc`), `"min_cuda"` (a build cannot load) or `"alignment"`; the last two warn once per process | the overwrite's stages into a contiguous scratch of `out.dtype`, then `out.add_(scratch)` on the same stream, outside autograd |

A build can load when its prebuilt library is under `aot/` or the JIT shim's
toolkit reaches its `build.json` `min_cuda` (13.4 for the main stages'
accumulate builds on `sm_100a` and `sm_103a`, which use the PTX ISA 9.4
`red.global.add.noftz.f32`; 13.5 for every `sm_107a` build), with
`nvcc` under the CUDA path and `CAKE_PPLX_DISABLE_JIT` unset;
`cake_backend.acc_library_loadable` applies that rule without building or
loading anything, and
`generated_program_available(device, "wgrad", out_dtype, accumulate=True)` tells
whether some call into an `out` of that dtype can take the fused route.

Numerical contract:

- `out` gets AccumulateGrad's bits of the deterministic weight gradient in its
  dtype. fp32 `out`: `RN_fp32(out + dW32)`, one fp32 add (round to nearest,
  subnormals kept) of the fp32 overwrite of the same call. bf16 `out`:
  `RN_bf16(out + dW_bf16)` with the bf16 overwrite, the bits autograd gives
  when it accumulates `w.grad`.
- The fused and the composed route give the same bits: the fused kernels add
  the bits of the fp32 overwrite, each element of `out` exactly once.
- Bitwise reproducible run to run, as the overwrite: every element is reduced
  in a fixed order (the split tail by the ordered tail reduce) and added once.
  `torch.use_deterministic_algorithms(True)` changes nothing (no alert, no
  warning).
- Accuracy: the deterministic `dW` plus one rounding in `out`'s dtype; the
  tests judge `out` against `out0 + n dW` in float64 beside the naive BF16
  tier (above).
- An empty group adds exact zeros, with the semantics of `out.add_`: its values
  are kept, `-0.0` becomes `+0.0` and a signaling NaN is quieted. `sum_m == 0`
  launches nothing and leaves `out` unchanged.

Checks (`ValueError`, before anything is allocated or bound): `out` given,
`out_dtype` None or `out.dtype`, bfloat16 or float32, `[E, N, K]` with unit
stride along `K`, rows and experts that do not overlap (row stride at least
`K`, expert stride at least `N` row strides: an expanded `out` is refused), no
memory shared with `g`, `x` or `offs`, and `accumulate` a bool. Row and expert
padding is allowed and never read or written.

Prepared launches and CUDA graphs: `launch()` and every replay of a graph that
captured it add again (a prepared accumulate launch is not idempotent); a
launch bumps `out._version` once at `launch()` and at capture, not at a graph
replay (the composed route through `out.add_`, the fused route through
`torch.autograd.graph.increment_version`; an inference tensor gets no bump).
The composed scratch is owned by the prepared launch, as its descriptor slots
are, and is allocated on the stream current at the preparation: keep the
launch alive while a graph that captured it may replay (the graph does not keep
the scratch alive) and, after a launch on another stream, until that launch
completes; do not have one prepared accumulate launch in flight on two streams
at once. A launch prepared inside a captured backward allocates the scratch
from, and returns it to, the graph's pool within that backward. As for any in-place
operation, two launches into the same `out` on different streams must be
ordered by the caller.

`cake_moe_grouped_gemm_det(x, w, offs, fuse_wgrad_accumulation=True)` follows
Megatron and Transformer Engine, as the non-deterministic wrapper does:
`w.main_grad` must be an fp32 or bf16 tensor of `w`'s shape on its device with
unit stride along the last dimension and rows and experts that do not overlap
(row padding allowed). It is checked at the forward call when autograd records
it (`w` requires grad, grad mode is on, not a checkpoint recompute), so not
under `torch.no_grad()`, and again at the backward, which reads `w.main_grad`
then, so a re-pointed buffer is honoured, and which also refuses a `main_grad`
that shares memory with the operands.
The backward adds the deterministic `dW` into it with the accumulate mode and
returns for `w` a fresh `torch.empty` of `w`'s shape and dtype when `w` has the
attribute `grad_added_to_main_grad` (it is set to `True`; `torch.zeros` when
`w.zero_out_wgrad` is true), else `None` (`w.grad` stays untouched and no
gradient hook of `w` fires). `x.grad` is unchanged, bitwise. Every backward
adds: `retain_graph` twice adds twice, and a `torch.autograd.grad(...,
inputs=...)` that excludes `w` still adds into `w.main_grad`, so split
backward-input / backward-weight schedules are not supported. A `w` that does
not require grad is never added to. The flag raises `NotImplementedError` under
`torch.jit.trace`, `torch.compile` and in a `create_graph` backward; under a
reentrant checkpoint `w` arrives detached without `main_grad` (`ValueError`;
use `use_reentrant=False`). A non-reentrant checkpoint's early stop still skips
the recomputed forward GEMM. A direct `CakeMoEGroupedGemmDetFunction.apply(x, w, offs, None,
True)` takes the flag as its fifth argument (the fourth is an internal slot)
and checks `main_grad` at the backward only.

Cost (`S = E * N * K` elements):

- Fused route: no scratch and no extra launch; the kernels read and write each
  element of `out` once (`8 * S` bytes for fp32) in place of the overwrite's
  store (`4 * S` bytes), and the backward allocates no buffer of the weight
  gradient's size but the `grad_added_to_main_grad` placeholder.
- Composed route, memory: the scratch takes `4 * S` bytes for an fp32 `out`
  (`2 * S` for bf16) for the life of the prepared launch (in the autograd
  wrapper, the weight-gradient launch of one backward). Gate/up projection (`N` 4096, `K`
  6144): 384 MiB at `E` 4, 768 MiB at 8, 1.5 GiB at 16, 3 GiB at 32. Down
  projection (`N` 6144, `K` 2048): 192 MiB, 384 MiB, 768 MiB, 1.5 GiB. With an
  fp32 `main_grad` the backward's peak is `2 * S` bytes above the default
  flag's (a bf16 `w.grad` of `2 * S` bytes that Megatron then adds into
  `main_grad`); with a bf16 `main_grad` the scratch is `2 * S` bytes and the
  peak equals the default flag's. The `grad_added_to_main_grad` placeholder
  (`2 * S` bytes) is allocated after the scratch is released.
- Composed route, traffic: the programs write the scratch and the add reads it
  back, `8 * S` bytes beyond the add's read and write of `out`; the time is
  that of the weight gradient followed by `main_grad.add_`.
- Under `torch.use_deterministic_algorithms(True)` with
  `torch.utils.deterministic.fill_uninitialized_memory` on (the default),
  torch fills the scratch and the `torch.empty` placeholder with NaN when it
  allocates them, one more write of each.

```python
from cake_pplx.cake_moe_grouped_gemm_det_train import (
    cake_moe_grouped_gemm_det,
    cake_moe_grouped_gemm_det_train_prepare_wgrad,
    cake_moe_grouped_gemm_det_train_wgrad,
)

# Megatron-style accumulation into an fp32 main_grad, deterministic
w = torch.nn.Parameter(torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda"))
w.main_grad = torch.zeros(E, N, K, dtype=torch.float32, device="cuda")
# (Megatron's DDP also sets w.grad_added_to_main_grad; the backward then returns a
# placeholder gradient for its hooks.  Without the attribute w.grad stays None.)
for x, g in micro_batches:
    y = cake_moe_grouped_gemm_det(x, w, offs, fuse_wgrad_accumulation=True)
    y.backward(g)  # w.main_grad[e] += G[e].T @ X[e]; x.grad as usual

# Low level: add into any fp32 / bf16 buffer
cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out=main_grad, accumulate=True)
launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(g, x, offs, out=main_grad, accumulate=True)
launch.launch()  # main_grad += dW; launch.plan["accumulate"]: "fused" or "composed"
```

## Host plan

Every decision is taken from host-known scalars (shapes, SM count,
architecture), never from `offs`, so one prepared launch serves any group
distribution. `cake_backend.py` reproduces the production planner from the
constants delivered with the generated programs (`HOST_PLAN_CONSTANTS`):

- Grid: one 2-CTA cluster per SM pair, capped by the tile count
  (`persistent_grid`); for `fwd` / `dgrad` the tile count bound lets every
  group add one partial 256-row tile (`max_cluster_tiles_upper_bound`).
- `fwd` / `dgrad` output tile (`fwd_dgrad_tile_n`): 512 columns
  (`wide_block_n`) when the output columns (`N` for `fwd`, `K` for `dgrad`)
  are a multiple of 512 and the architecture is listed in `wide_fwd_archs`
  (`fwd`) or `wide_dgrad_archs` (`dgrad`), 256 (`block_n`) otherwise. The
  512-column programs are the routes' `main_wide` stage; every output element
  accumulates its k steps in the same order in both tiles, so both give the
  same bits. The `sm_107a` `dgrad` program of that stage keeps each G tile in
  the tensor core's A collector for both 256-column halves of the tile.
- `dgrad` L2 evict-first twin (`dgrad_evict_first`): on the architectures
  listed in `dgrad_evict_first_archs`, when the groups average at least the
  architecture's `dgrad_evict_first_min_rows` rows (`sum_m // E`; 4096 on
  `sm_100a`, 16384 on `sm_107a`) and `dX` has at least
  `dgrad_evict_first_min_k` columns (`K`, the gate/up projection), the wide
  tile is the route's `main_wide_ef` stage: the same program with G loaded and
  `dX` stored under the L2 evict-first policy, so a group's weights stay
  resident in L2 while its row blocks stream past them (the domain-scheduled
  program on `sm_100a`; on `sm_107a` the A-collector program, which also
  splits two head and two tail k steps of an item around the release of the
  accumulator's two halves instead of four and three, so fewer of an item's k
  steps pay the split). A data gradient whose reduction `N` (`w.shape[1]`) is
  at most the architecture's `dgrad_evict_first_skip_max_n` (1024 on
  `sm_100a`, as in the gate/up projection of a model with a 1024-wide expert
  intermediate) keeps `main_wide`: there the plain wide tile was faster or level
  (timed at `N` 64 to 1024). Every element still accumulates its k steps in
  order, so its output is bitwise that of `main_wide`.
- `fwd` / `dgrad` 3 / 3 head / tail twin (`fwd_wide_ts33` /
  `dgrad_wide_ts33`): where the evict-first twin does not apply, the wide tile
  is the route's `main_wide_ts33` stage, the same program with three head and
  three tail k steps of an item split around the release of the accumulator's
  two halves instead of four and three (the A-collector program on
  `sm_103a`). `dgrad` takes it on the architectures listed in
  `dgrad_wide_ts33_archs` when the groups average at least the architecture's
  `dgrad_wide_ts33_min_rows` rows (`sum_m // E`; from
  `dgrad_wide_ts33_short_n_min_rows` rows, 1024 on `sm_103a`, when the
  reduction `N` is from the architecture's `dgrad_wide_ts33_short_n_min` to
  its `dgrad_wide_ts33_short_n`, 512 to 1024 on `sm_103a`) and `dX` has at
  least `dgrad_wide_ts33_min_k` columns (`K`); `fwd` on those listed in
  `fwd_wide_ts33_archs` when the reduction `K` is at most
  `fwd_wide_ts33_max_k` and the groups average at most
  `fwd_wide_ts33_max_rows` rows (the down projection over many experts), or
  when `K` is from the architecture's `fwd_wide_ts33_short_k_min` to its
  `fwd_wide_ts33_short_k` (512 to 1024 on `sm_103a`) at any rows per group.
  On `sm_103a` the twin was faster or level at every timed shape with 8 to 16
  k steps per item (a reduction of 512 to 1024) and slower with 4 (a
  reduction of 256), so a shorter reduction keeps `dgrad_wide_ts33_min_rows`
  and the `fwd_wide_ts33_max_k` / `fwd_wide_ts33_max_rows` rule. Every element
  still accumulates its k steps in order, so its output is bitwise that of
  `main_wide`.
- `fwd` last-wave-halves evict-first twin (`fwd_lwh_ef`): on the architectures
  listed in `fwd_lwh_ef_archs`, when the reduction `K` (the columns of `x`) is
  at most `fwd_lwh_ef_max_k`, as in the down projection, the wide tile is the
  route's `main_wide_ef` stage: the A-collector program that releases the last
  wave of tiles in two halves, stores Y under the L2 evict-first policy and
  splits two head and two tail k steps of an item around the release of the
  accumulator's two halves instead of four and three. Its output is bitwise
  that of `main_wide`.
- `fwd` X / Y evict-first twin (`fwd_evict_first`): on the architectures
  listed in `fwd_evict_first_archs`, when the reduction `K` is at least
  `fwd_evict_first_min_k`, as in the gate/up projection, the wide tile is the
  route's `main_wide_ef` stage: the domain-scheduled program with X loaded and
  Y stored under the L2 evict-first policy, so the group's weights stay
  resident in L2 while its rows stream past them. A shorter reduction (the
  down projection) keeps `main_wide`. The cache policy changes no arithmetic:
  its output is bitwise that of `main_wide`.
- `fwd` / `dgrad` tile order on `sm_100a` / `sm_103a`: the column tiles of
  every 256-row block are split between two claim queues, one per locality
  domain of the device, in proportion to the clusters that claim each queue
  first (`c0 = clusters // 2`, `c1 = clusters - c0`, `clusters = grid_x // 2`);
  a cluster finds its domain on the device and claims from the other queue
  when its own is empty, so every tile is computed exactly once whatever the
  split. The `sm_107a` programs stride the tiles over the clusters in the same
  row-block-major order and ignore the queues.
- Probed queue split (`plan["queue_split"]`): a cluster's first queue is the
  one whose 4 KB probe chunk of `sched` (at the first 8 KB boundary 128 bytes
  past it, or the next chunk) is homed in its own die, and which die homes a
  chunk depends on the physical page, so the even split misplaces tiles on a
  device whose dies have unequal SM pairs (36 / 40 on GB300, 35 / 39 on B200).
  On the architectures listed in `domain_split_archs`, a launch of an op in
  `domain_split_ops` (the data gradient; the forward keeps the even split)
  whose group weight has at most `domain_split_max_weight_elems` elements
  (`N * K`, the down projection) passes the SM pairs nearer each chunk of its
  own `sched` as `c0` / `c1` (`"probed"`; otherwise `"even"`). The first such
  launch on a kept stream workspace enqueues a probe on the stream (five
  rounds of one CTA per SM timing an add-zero atomic on a line of each chunk,
  majority per SM pair) and copies the counts to pinned host memory behind an
  event; later launches pass them once the event has completed, and nothing
  waits for it. Until then, under CUDA graph capture, for another device,
  without a CUDA compiler (the probe kernel is built once per compute
  capability with `tvm_ffi.cpp.load_inline` at preparation) or when the probe
  does not see every SM and both dies, the launch passes the even split. The
  output bits do not depend on the split.
  `stream_queue_split(device, stream=None)` returns the split a stream's
  probed launches pass, or `None` while it is the even split.
- Weight-gradient k tile (`select_wgrad_tile`): 512 when `K % 512 == 0`, the
  architecture is listed in `k512_archs` and the average rows per group reach
  the architecture's `k512_min_rows_per_group`; 256 otherwise. The two tiles
  are separate registered programs; the fp32 output is the compile-line
  specialization `-DCAKE_WGRAD_OUT=32` of the bf16 program.
- Weight-gradient tail: the tiles of the last partial wave are split into
  `tail_splits` k-step chunks over all clusters (`wgrad_tail_tiles`,
  `wgrad_tail_plan`, a closed-form cost model of rounds, partial traffic and one
  launch); when `tail_splits > 1` an ordered fp32 tail-reduce kernel
  (`wgrad_reduce_grid` CTAs) sums the chunks, otherwise only the main kernel
  runs. The tile raster inside a group follows the per-wave operand footprint
  (`wgrad_raster_rows`).
- Weight-gradient wave sync: when `K` reaches `wave_sync_min_k` (the
  gate-up-shaped rows), `wave_sync_archs` lists the architecture for the
  selected k tile and the route has a `main_wave` stage on it, the main kernel
  is the wave-synchronized build, launched with the architecture's
  `wave_sync_mode` as `wave_mode`; otherwise the plain build (`wave_mode = 0`).
  Both builds give the same output bits for a given route.
- Weight-gradient accumulator release: with the 512-wide k tile on the
  architectures listed in `k512_release_archs`, the main kernel is the build
  that hands the tensor-memory accumulator to the epilogue in two halves
  (stage `main_halves`, or `main_wave_halves` for the wave-synchronized build),
  so the next tile's MMAs start on the first half while the epilogue drains the
  second. Its output is bitwise that of the build that drains the whole
  accumulator at once. The `main_halves` program also keeps each A tile in the
  tensor core's A collector; on `sm_100a` and `sm_103a` it also walks the tiles
  in an empty-aware order that the kernel plans itself (no host input). On
  `sm_100a` and `sm_103a` the `main_wave_halves` program keeps the A tile in
  the collector and synchronizes the waves every eight waves. None of these
  changes the output bits.
- Weight-gradient queue walk (`wgrad_queue_walk`): on `sm_100a` and `sm_103a`
  the 512-wide k tile routes also serve `main_halves_queue` (and
  `main_halves_queue_acc` on the fp32 route), the A-collector half-release
  build whose tiles before the split tail are claimed from one global counter
  in tile order, so the tiles in flight are consecutive and tiles that read
  the same `g` or `x` strip run at nearly the same time. The host binds it in
  place of the stage it selects (`main_halves` or `main_wave_halves`) where a
  rule of `HOST_PLAN_CONSTANTS["queue_walk"]` names the call: the device's
  architecture and SM count, the selected stage, the output (`"bfloat16"` /
  `"float32"`, or `"accumulate"` for a fused accumulate) and a range of group
  counts. The call keeps the selected stage's plan; its split tail is that
  stage's, summed by the same ordered tail reduce, and every tile keeps its
  k-step order, so the output bits are the selected stage's. The rules cover
  the 148-SM `sm_100a` and 152-SM `sm_103a` parts, where the queue walk was
  timed against the stage it replaces and no row was confirmed slower across
  several GPUs, except one disclosed row (the 8-group gate/up weight gradient
  with a composed bfloat16 accumulate, lead route, eager: slower on one of
  three 148-SM `sm_100a` GPUs, faster on the other two); on the 148-SM
  `sm_100a` part the `main_halves` stage keeps it for the fused accumulate
  only (a bfloat16 row there was slower), and the `main_wave_halves` stage
  keeps it up to 16 groups (a 32-group bfloat16 row there was slower). An
  `m_indptr` row table binds the queue stage of the lead route (its `_lead`
  twin, with `m_begin`), as every other stage.
- Weight-gradient k256 tail (`wgrad_k256_tail`): on the architectures listed
  in `k256_tail_archs`, when `K` (the columns of `dW[e]` and of `x`) is at
  most `k256_tail_max_k` and the groups average at least `k256_tail_min_rows`
  rows (`sum_m // E`), as in the down projection over few groups, the main
  kernel is the route's `main_halves_k256t` stage instead of `main_halves`:
  the A-collector half-release kernel runs only the full waves of the 512-wide
  tile, then the tiles left as the 256-column half tiles of the 256-wide
  tiling, walked n-major (`raster_rows` 0). The last partial wave of that
  tiling is split as the 256-wide k tile splits its tail (`tail_tile_k` 256 in
  `wgrad_tail_tiles` / `wgrad_tail_plan`): each half tile runs as
  `tail_splits` k-step chunks whose fp32 partials, 256 x 256 slots
  `j * tail_splits + s`, the tail-reduce stage of the 256-wide tile's route
  sums (`wgrad_reduce_grid` CTAs). Every element accumulates its rows in the
  order of the 256-wide k tile, so where the two plans split the same tiles
  the same way the output bits are that tile's.
- Wide outputs (`row_chunks`): the `fwd` / `dgrad` programs address `out`
  with 32-bit element offsets, so on the architectures of `wide_rows_archs`
  (`sm_100a`, `sm_103a`, `sm_107a`) a call whose output span passes
  `int32_elems` (`sum_m * out.stride(0) > 2^31 - 1`) launches the stage
  selected on the whole call once per row chunk: `n = ceil(sum_m / R)` chunks
  of `ceil(sum_m / n)` consecutive rows, `R = int32_elems // out.stride(0)`
  (`plan["launches"]` is `n`, `plan["row_chunks"]` the `(start, rows)` pairs
  and `plan["grids"]` each chunk's grid; `stages` repeats the stage). At
  655,360 rows of 4096 columns that is 2 launches of 327,680 rows. Each chunk
  binds its rows of the row operand and of `out`, `sum_m` = its rows, and a
  row table that `launch()` computes on the device from the caller's `offs` /
  `m_indptr` at every launch: the table less the chunk's first row, clamped
  to the chunk's rows (one clamp per chunk and one subtraction from the second
  chunk on). There is no host read, so the launch stays CUDA-graph capturable
  and valid for any `offs` written later. Every element is the same ordered
  k-step sum of its row in any chunk, so the output bits are those of one
  launch. On an architecture outside `wide_rows_archs` such a call raises
  `ValueError`. A call within the span takes one launch, bound as before.
- Workspaces (allocated at preparation, retained by the launch object): the
  per-CTA TMA descriptor slots of the weight gradient (a ring of nine
  descriptor pairs per CTA: a group switch edits the next pair and warms the
  one after it, never a pair an in-flight load may still read) and its fp32
  tail partials, and the row tables of a wide `fwd` / `dgrad` (one int32
  buffer of the caller's table size per chunk: `n * (E + 1) * 4` bytes or
  less; the forward and data gradient retain nothing else). One prepared
  wide launch must not be in flight on two streams at once, must be kept
  alive while a CUDA graph that captured it may replay and, after a launch on
  another stream, must be kept alive until that launch completes (the
  buffers are allocated on the stream current at the preparation). The operand
  descriptors are launch arguments; there is no descriptor workspace and no
  descriptor upload kernel.
- Stream workspaces (kept per device and CUDA stream, reused by every launch
  on that stream): the `fwd` / `dgrad` scheduler words (`sched`,
  `sched_words` u32 words whose first three rest at `sched_rest`), the
  wave-sync counters (`wave_sync`, `wave_sync_words` u32 words resting at
  zero) and the queue walk's tile counters (`queue_sched`, the argument the
  queue programs name `sched`: `queue_sched_words` u32 words resting at zero;
  the last cluster of a launch to retire sets them back to zero). The first launch on a stream allocates a buffer at its rest state;
  every kernel leaves it at its rest state, so launches on one stream share it
  and launches on two streams never do. A launch under CUDA graph capture, or
  for a device that is not the current one, takes buffers of its own, set to
  their rest state at the launch. `stream_workspace(kind, device, stream=None)`
  returns the buffer kept for a stream, or `None` before a launch on it needed
  one.

`CakeMoEGroupedGemmDetLaunch.plan` exposes `grid`, `tile_k`, `tail_splits`,
`raster_rows`, `launches` and, for `fwd` / `dgrad`, `tile_n`, `clusters`, `c0`,
`c1` (the even split) and `queue_split` (and `row_chunks` and `grids` for a wide
output), for `wgrad` `tail_tile_k` (the k width of the split tail tiles),
`wave_mode`, `accumulate` (`"off"`, `"fused"` or `"composed"`) and, on the
composed route, `accumulate_reason`, for inspection and tests;
`CakeMoEGroupedGemmDetLaunch.stages` names the bound stages in launch order,
`CakeMoEGroupedGemmDetLaunch.add_from` is the scratch of a composed accumulate
launch (else `None`) and `CakeMoEGroupedGemmDetLaunch.bump_version` is set on a
fused one.

## Build targets and layout

| feature | sm_100a | sm_103a | sm_107a |
| --- | --- | --- | --- |
| `wgrad` k512 half-release stages | `main_halves` (planned walk), `main_wave_halves` (wave period 8), `main_halves_queue` (queue walk) | `main_halves` (planned walk), `main_wave_halves` (wave period 8), `main_halves_queue` (queue walk) | `main_halves`, `main_halves_k256t` |
| `fwd` / `dgrad` L2 evict-first twin launched (`main_wide_ef`) | `fwd` (`K >= 6144`); `dgrad` (`K >= 4096`, `N > 1024`, `sum_m // E >= 4096`) | `fwd` (`K >= 6144`) | `fwd` (`K <= 2048`); `dgrad` (`K >= 4096`, `sum_m // E >= 16384`) |
| `wgrad` accumulate, fused (`_acc` builds, fp32 `out` only) | `main_halves`, `main_wave_halves`, `main_halves_queue`, `tail_reduce` | `main_halves`, `main_wave_halves`, `main_halves_queue`, `tail_reduce` | `main_halves`, `main_halves_k256t`, `tail_reduce` |
| `wgrad` accumulate, composed | bf16 `out`; k256 tile; `main` / `main_wave`; misaligned `out`; `_acc` library not loadable | bf16 `out`; k256 tile; `main` / `main_wave`; misaligned `out`; `_acc` library not loadable | bf16 `out`; k256 tile; `main`; misaligned `out`; `_acc` library not loadable |
| CUDA toolkit floor (`min_cuda`) | 12.8 | 12.9 | 13.5 |
| CUDA toolkit floor of the fused main stages (`min_cuda`) | 13.4 | 13.4 | 13.5 |

The programs are exact-architecture tcgen05 payloads built for the exact
targets `sm_100a`, `sm_103a` and `sm_107a` (`build.json` `arch_flags`).
`cake_jit.PROGRAMS` lists thirty-two libraries, one per kernel and build,
each a kernel and a host-binding translation unit under
`csrc/cake_moe_grouped_det_gemm/` shared by the architectures it is built for (an
architecture-dependent line is folded under `__CUDA_ARCH__`):

- `fwd` and `dgrad` in six builds each: the domain-scheduled build for
  `sm_100a` and `sm_103a` and the static-schedule build for `sm_107a`, each
  with the 256-column output tile (the routes' `main` stage) and with the
  512-column one (`main_wide`), and the evict-first twin of each 512-column
  build (`main_wide_ef`; the domain-scheduled `fwd` twin, which loads X and
  stores Y evict-first, for `sm_100a` and `sm_103a`, the domain-scheduled
  `dgrad` twin for `sm_100a`); plus the 3 / 3 head / tail twin of the
  domain-scheduled A-collector 512-column build for `fwd` and `dgrad` on
  `sm_103a` (`main_wide_ts33`). A route spells the two builds of a stage as
  two stages of one name on disjoint architectures, and
  `cake_jit.stage_binding(name, stage, arch)` returns the one built for the
  architecture.
- the wave-synchronized weight-gradient main kernel with a 256- or 512-wide
  k tile, for `sm_100a` and `sm_103a` (the routes' `main_wave` stage; the
  host launches it on no other architecture).
- the 512-wide k tile's A-collector main kernel that releases the
  accumulator in two halves (`main_halves`): with the planned tile walk for
  `sm_100a` and `sm_103a`, without it for `sm_107a`; for `sm_107a` its twin
  that runs the tiles left after the full waves as 256-column half tiles
  (`main_halves_k256t`); the wave-synchronized A-collector kernel that
  releases it in two halves (`main_wave_halves`), for `sm_100a` and `sm_103a`;
  the A-collector half-release kernel on the queue walk
  (`main_halves_queue`), for `sm_100a` and `sm_103a`.
- the 512-wide k tile's main kernel for `sm_107a`, a second `main` stage
  beside the one built for `sm_100a` and `sm_103a`.
- the fp32 accumulate builds (`_acc`): the twins of `main_halves` (one for
  `sm_100a` and `sm_103a`, one for `sm_107a`), `main_wave_halves` (`sm_100a`,
  `sm_103a`), `main_halves_queue` (`sm_100a`, `sm_103a`) and
  `main_halves_k256t` (`sm_107a`) on the fp32 512-wide k tile
  route, which add the weight gradient into `C` with one fp32 add per element
  (their `build.json` entries need CUDA 13.4 on `sm_100a` and `sm_103a`,
  `min_cuda`), and the twin of the tail-reduce that adds the ordered sum of a
  split tail's partials into `C` (`tail_reduce_acc`, on the fp32 512- and
  256-wide k tile routes, for all three architectures).
- the weight-gradient main kernel with a 256- or 512-wide k tile (the routes'
  `main` stage): the 512-wide one for `sm_100a` and `sm_103a`, the 256-wide
  one in two builds, one for `sm_100a` and `sm_103a` and one for `sm_107a`
  that also reads the rows of `g` and `x` through `A_rows` / `B_rows` at
  their row strides, which the host passes as `lda` / `ldb`.
- the weight-gradient tail-reduce, one kernel for both k tiles.

The host launches a stage only where the constants above list the
architecture, so a library may be built for an architecture its stage is not
launched on.

A specialization is a compile-line definition on its `build.json` entries:
the tail-reduce k tile (`-DCAKE_WGRAD_TILE=512`; 256 is the default) and the
fp32 weight-gradient output (`-DCAKE_WGRAD_OUT=32`; the bf16 output is the
default). `cake_jit.ROUTES` maps each public route (operation, output dtype,
k tile) to its stage programs and specializations. `build.json` (schema `cake-build/1`, written by the
export) is the only record of what to compile: one entry per build
`<library>[<specialization tag>]_<arch>` -- the tag is `_wgrad_tile256` /
`_wgrad_tile512` for the specialized tail-reduce builds, `_wgrad_out32` for the
fp32 weight-gradient builds (composed on the fp32 tail-reduce:
`_wgrad_out32_wgrad_tile256` / `_wgrad_out32_wgrad_tile512`), empty otherwise
-- with its translation units, digests, nvcc flags (the `-D` defines
included), per-architecture closure digest and `aot/<name>.so` path.
`cake_jit.load_module(program, arch, specializations)` loads that entry
through the shared `cake_build.py` loader: the prebuilt library when `aot/`
holds it, else the build of the package's JIT shim `_cake_pplx_jit.py` (the
cache name carries the exact target and the closure digest; the target comes
only from the entry's `-gencode` flags). The bindings include the package's
`csrc/cake_ffi_checks.h`.

Every kernel translation unit is self-contained. Programs whose bindings
differ only in the kernel they launch, its block size and its dynamic shared
memory share one binding, which a member's entries compile under
`-DCAKE_KERNEL_SYMBOL`, `-DCAKE_PROGRAM_SHIM` and, where they differ from the
owner's, `-DCAKE_BINDING_THREADS` / `-DCAKE_BINDING_SMEM`.

Measured on B200 (`sm_100a`), GB300 (`sm_103a`) and R200 (`sm_107a`); the
performance summary is in the delivery pull requests.

## Standalone import

The package imports only itself, `torch` and `tvm_ffi`, so a copy of it loads
under another top-level name. The loader's JIT fallback in `cake_build.py`
imports the package's JIT shim `_cake_pplx_jit.py` inside `try/except
ImportError`, reached after import when a library is not prebuilt under
`aot/`. The API logging belongs to the entry points
of `cake_pplx.cake_moe_grouped_gemm_det_train`, so a direct call of the
package's functions emits no log and no warning.

## Tests and benchmark

```bash
pytest tests/experimental/test_cake_moe_grouped_gemm_det_train.py tests/experimental/test_cake_moe_grouped_gemm_det_train_api.py tests/experimental/test_cake_moe_grouped_gemm_det_train_names.py tests/experimental/test_cake_moe_grouped_gemm_det_train_standalone.py tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py tests/experimental/test_cake_moe_grouped_gemm_det_train_acc.py tests/experimental/test_cake_moe_grouped_gemm_det_train_device_slice.py tests/experimental/test_cake_moe_grouped_gemm_det_train_wide_rows.py
python benchmarks/bench_cake_moe_grouped_gemm_det_train.py --help
```

`test_cake_moe_grouped_gemm_det_train.py` (the backend suite, which imports the package
through `tests.cake.test_helpers.cake_package`) judges every output against the
naive BF16 tier (above; including an empty group, a one-row group, padded outputs,
determinism and the fp32 output) and skips when no supported device or no
registered program is present. `test_cake_moe_grouped_gemm_det_train_api.py` covers the
entry points of `cake_pplx.cake_moe_grouped_gemm_det_train` (stable APIs with the
package functions' signatures and docstrings, no warning, the support query, the
deferred import), the `backend="cake"` route of `grouped_mm_bf16` and the mode
rules above (the `TypeError` of a `deterministic` argument, no change under
torch's deterministic mode). `test_cake_moe_grouped_gemm_det_train_names.py`
(CPU) checks the names: every public name carries `det`, no callable takes a
`deterministic` argument, and every removed path and name fails at import.
`test_cake_moe_grouped_gemm_det_train_standalone.py` copies the package under another
name into a CPU subprocess and checks that it imports and raises its host-side
errors without any warning, and that its only guarded import is the JIT-shim
fallback in `cake_build.py`.
`test_cake_moe_grouped_gemm_det_train_determinism.py` repeats every route (forward,
data gradient, weight gradient in bf16 and fp32 at both k tiles with the tail
reduce, the probed split of the data gradient) on fixed inputs and zeroed
accumulators and requires bitwise-identical outputs, also across fresh
processes.
`test_cake_moe_grouped_gemm_det_train_acc.py` covers the weight-gradient accumulation:
on the CPU the shipped accumulate builds (registry, README feature matrix,
`build.json`, the loader's `min_cuda` rule), the plans (equal to the
overwrite's, bound to `out` on the fused route and to the scratch on the
composed one), every reason of the route choice, the checks on both routes,
the launch's add or version bump, the facade's handoff and the autograd flag's
Megatron contract with CPU stand-ins for the preparations; on a GPU every
weight-gradient form with and without a split tail, fp32 and bf16, bitwise
against `out0.add_(dW)` and run to run, the fused route bitwise against the
composed one (split tails, the planned walk with empty groups last, the wave
halves, the k256 tail, `-0.0`, 32 groups), each judged against the naive BF16
tier, CUDA graph replays, streams, layouts, and the autograd flag with the
generated programs.
`test_cake_moe_grouped_gemm_det_train_device_slice.py` covers the device
slices (above): on the CPU the lead routes against their routes, the lead
builds' CUDA floor, the row table's views and the lead binding over every
architecture; on a GPU each slice against the `[E]` call over the same rows
moved to row 0 (bitwise) and the `[E]` call over the slice's rows alone
(bitwise where it plans the same), every forced stage, edges with the rows
outside poisoned, padded rows, graph replay over device-copied routings, two
streams, accumulation, the autograd entry and the bits of the judged `[E]`
cases.
`test_cake_moe_grouped_gemm_det_train_wide_rows.py` covers the wide
outputs (host plan, above): on the CPU the chunk arithmetic, the chunks' row
tables against the rows of each group in each chunk, a call within the span
bound as one launch, a wide call's chunks, rows and row tables and the
refusals (an architecture outside `wide_rows_archs`, a row pitch past the
span); on a GPU forced chunks of
every served `fwd` / `dgrad` stage bitwise against one launch (an empty chunk,
a first group past the first chunk, both row tables), the issue's 655,360
rows over 64 and 32 skewed groups and the boundary row counts 524,287,
524,288, 524,289 and 1,048,576 against the naive BF16 tier with windows of
rows (each chunk boundary, row 524,288) bitwise against independent calls
within the span, an `m_indptr` device slice into a slice of a larger `out`,
changing device routings eager and from one CUDA graph with no host
synchronization and no new program load, a checkpointed MoE block bitwise
against the block without checkpoint, a weight gradient whose operands pass
2^31 elements against the naive BF16 tier, and repeats. These GPU tests skip
on a device with too little free memory (up to 72 GiB for the block).
`test_cake_moe_grouped_gemm_det_train_acceptance.py` repeats every output of
the reproducibility table five times into a zeroed and five times into a
filled output under concurrent copy streams, over changing offsets, and
judges the outputs of the accuracy sample above against the naive BF16 tier. The accumulation is timed by the accumulation rows of
`benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py` (arm `det_mgX`, beside the
deterministic baselines `det_acc` and `det_mgB` and the non-deterministic arms).
The benchmark `benchmarks/bench_cake_moe_grouped_gemm_det_train.py` times the prepared
launches against `torch._grouped_mm` and a per-group cuBLAS loop on the MoE projection rows of the
design (E = 4 ragged distributions, E = 8 / 16 / 32 balanced, fp32 weight
gradients); `--wide-rows` runs the wide-output rows instead (the down
projection's forward and the gate/up projection's data gradient, `[rows, 4096]`
at 327,680 and 655,360 rows over 64 and 32 balanced or skewed groups).

## Limitations

- bf16 inputs only; `E`, `sum_m`, `N` and `K` must fit 32-bit integers. On
  `sm_100a`, `sm_103a` and `sm_107a`, a forward or data-gradient output whose
  span passes 2^31 - 1 elements runs the selected program once per row chunk
  whose span fits 32-bit element addressing (`plan["row_chunks"]`); the chunk
  row tables are computed on the device. The weight-gradient output span must
  fit 32-bit addressing.
- The weight-gradient accumulation is fused for an fp32 `out` on the
  half-release forms of the 512-wide k tile only; every other call composes,
  with a scratch of the weight gradient's size (above).
- The alignment rules above; no transposed operand layouts other than the
  documented in-place `W` read of `dgrad`.
- Compute capability 10.0, 10.3 or 10.7 only; other devices raise.
- The autograd entry does not support summing the `x` gradients of several
  device slices of one shared `x` (device slices, above).
