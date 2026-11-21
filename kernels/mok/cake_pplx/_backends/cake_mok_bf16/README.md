# Cake MoK training backend (`cake_mok_bf16`)

This backend provides the MoE layer only, not a full-model training result.
The explicit `cake_pplx.cake_mok_det_train.prepare_mok_bf16` API returns a Mixture of Kittens
(MoK) functional adapter for expert-parallel MoE training. It covers
scheduling, forward, context-only recompute and backward, including input
gradients, router-score gradients, and all six expert-weight gradients.
Shared-expert weight gradients remain source-rank local.

- **Routed experts** run in BF16, or natively in **MXFP8** (MoK recipe:
  caller-prequantized weights, activations and gradients quantized inside the
  fused kernels). The shared expert always runs in BF16.
- **Clamped SwiGLU** (`swiglu_limit=L`): `silu(min(gate, L)) * clamp(up, -L, L)`
  for routed and shared experts, with the matching backward masks. The default
  `swiglu_limit=None` is plain SwiGLU.
- **Context-only recompute** (`recompute_forward_context`) for activation
  checkpointing: it rebuilds the backward context without down projections,
  combine or the output epilogue.
- **FP32 weight-gradient accumulation** (`fp32_wgrad=True`): backward adds
  every weight-gradient contribution into caller-owned FP32 accumulators.
- **Unequal per-rank source counts**, including zero, as runtime inputs.
- **Router weight before the down projection**
  (`router_weight_position="before_down"`, SM100a and SM103a): each route's
  FP32 router weight `w` scales the SwiGLU activation `a` before the down
  projection, `h = BF16(w * a)`, and the routed outputs are summed unweighted;
  top-k 6 over 384 routed experts (`(4, 96)`, `(8, 48)`, `(16, 24)`,
  `(32, 12)`, `(64, 6)`) and top-k 2 on the diagnostic layouts, with clamped
  SwiGLU and FP32 weight-gradient accumulation (see Contract).

The implementation uses standalone generated CUDA sources, loaded prebuilt
from `aot/` when the package ships them and otherwise built by the package's
TVM-FFI JIT shim from `build.json`. It preserves the original launch boundaries: three scheduler
kernels, metadata gathering and peer barriers, one fused forward (or
recompute), one fused backward, separate output/input-gradient epilogues, and
empty-expert gradient zeroing. Dispatch, GEMMs, SwiGLU, quantization, combine
and backward ring replay stay inside the fused kernels. Framework copies,
resets and the router-gradient clone remain part of a complete iteration.

## Use

The package includes its caller-owned workspace and metadata layer; no separate
MoK installation or extension is needed. PyTorch with symmetric-memory
multicast support is required. Minimum CUDA toolkit (nvcc) per target, the
`min_cuda` of every library in `build.json`: 12.8 for SM100a, 12.9 for SM103a
and 13.5 for SM107a (the SM100a and SM103a libraries were checked by compiling
each with 12.8, 12.9, 13.0 and 13.1: each compiles from its floor up, and 12.8
rejects `compute_103a`).
Distributed ranks must share one peer-accessible NVLink domain (peers may span
hosts of one NVL72 domain).

```python
import math

import torch

from cake_pplx.cake_mok_det_train import (
    create_mok_bf16_workspace,
    prepare_mok_bf16,
    quantize_mok_mxfp8_weights,
)

config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=n, hidden_size=6144, topk=8,
)
backend = prepare_mok_bf16(ep_size=16, local_experts=16, topk=8)
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=16)
y, context = backend.forward(config, workspace, schedule, x, scores, *weights)
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)

# Native MXFP8 routed experts: prequantize after every weight update.
fwd_routed, bwd_routed = quantize_mok_mxfp8_weights(gate_w, up_w, down_w)
backend.prepare(mxfp8=True)  # compile before CUDA Graph capture
y, context = backend.forward(config, workspace, schedule, x, scores,
                             *shared, *fwd_routed)
# Activation checkpointing: drop the context, rebuild it before backward.
del context
context = backend.recompute_forward_context(
    config, workspace, schedule, x, shared[0], shared[1], *fwd_routed[:2])
grads = backend.backward(config, workspace, schedule, context, dy, x, scores,
                         *shared, *bwd_routed)

# BF16 activation checkpointing without storing the context at all.
backend.prepare(save_context=False)  # compile before CUDA Graph capture
y, _ = backend.forward(config, workspace, schedule, x, scores, *weights,
                       save_context=False)
context = backend.recompute_forward_context(
    config, workspace, schedule, x, *weights[:2], *weights[3:5])
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)

# Router weight before the down projection: 384 experts at EP64 (6 local
# experts), top-6, on a workspace of its own at the worst-case schedule
# capacity (see Contract). n: this rank's source rows; ids: int64 [n, 6]
# expert IDs; router_w: FP32 [n, 6] router weights; weights: BF16, 6 routed
# experts. Every rank must pass the same configuration, so the multiplier
# comes from max_rows, a bound on every rank's n that all ranks agree on (for
# example the training configuration's tokens per rank), reserved as
# source_capacity.
ep, local, k = 64, 6, 6
worst = min(k, local) * max_rows * ep + local * 255  # routed rows, padded
m = math.ceil(worst / (max_rows * k)) / ep
config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=n, hidden_size=5120, topk=k,
    source_capacity=max_rows, minibatch_size=4096, macrobatch_size=262144,
    schedule_capacity_multiplier=m,
)
backend = prepare_mok_bf16(ep_size=ep, local_experts=local, topk=k, clamped_swiglu=True,
                           fp32_wgrad=True, router_weight_position="before_down")
backend.prepare(10.0, save_context=False)  # compile before CUDA Graph capture
backend.prepare_recompute(10.0)
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=local)
accumulators = [torch.zeros(w.shape, device=device) for w in weights]  # FP32
y, _ = backend.forward(config, workspace, schedule, x, router_w, *weights,
                       swiglu_limit=10.0, save_context=False)
context = backend.recompute_forward_context(
    config, workspace, schedule, x, *weights[:2], *weights[3:5], swiglu_limit=10.0,
    router_weights=router_w)
grads = backend.backward(config, workspace, schedule, context, dy, x, router_w, *weights,
                         swiglu_limit=10.0, weight_grad_accumulators=accumulators)
```

`weights` contains shared gate/up/down followed by routed gate/up/down; gate
and up are `[I, H]` (shared) or `[E_local, I, H]` (routed) and down is
`[H, I]` or `[E_local, H, I]`. `grads` contains input, router scores, routed
gate/up/down, then shared gate/up/down gradients. MXFP8 routed weights follow
MoK's functional API: forward and recompute take `(data, scales)` pairs;
backward takes `(data, scales, data_t, scales_t)` for gate and up and the
transposed `(data_t, scales_t)` pair for down. `quantize_mok_mxfp8_weights`
produces both sets (E4M3 data, UE8M0 scales for 32-element blocks along the
reduction dimension, tensor-core scale tiles `[E * rows / 128, cols / 128, 32,
16]`) with PyTorch operations; this package has no Triton dependency. MXFP8
weight gradients are BF16 (or the FP32 accumulators).

`swiglu_limit` is a per-call runtime value; plain and clamped SwiGLU are
separate kernel builds. `prepare_mok_bf16(clamped_swiglu=..., fp32_wgrad=...)`
compiles one BF16 variant immediately; call `prepare(swiglu_limit, mxfp8)` and
`prepare_recompute(swiglu_limit, mxfp8)` for every other variant before CUDA
Graph capture. With `fp32_wgrad=True`, pass six FP32 tensors (shared gate, up,
down, routed gate, up, down) as `weight_grad_accumulators`; backward adds into
them and returns them.

Prepare and warm up before CUDA Graph capture. All ranks must execute
matching collective calls. Keep workspace allocations and peer mappings alive
while the adapter or a captured graph can use them. Use a separate workspace
and adapter for concurrent executions. Pass the matching context and schedule
to backward; retain both while any captured graph uses them.

With `router_weight_position="before_down"` the workspace is unchanged and
`scores` are the router weights `w` (FP32 `[tokens, topk]`, each `>= 0`; a zero
weight is allowed). `prepare_mok_bf16` refuses, before it loads a kernel, a
layout or top-k the mode does not run, `clamped_swiglu=False`,
`fp32_wgrad=False` and an SM107a device; `forward`, `recompute_forward_context`,
`backward`, `prepare` and `prepare_recompute` refuse `swiglu_limit=None` and
MXFP8 routed weights; `recompute_forward_context` requires `router_weights`
(the forward's). A checkpointing caller wraps forward, recompute and backward
in its own `torch.autograd.Function`; the package ships no autograd wrapper.

Calling the API is the opt-in. There is no automatic dispatch or AOT
registration.

## Context, memory and recompute

A forward context holds the shared-expert activations and the routed
activations of the first macrobatch, which stays resident in the forward ring
(MXFP8 keeps the routed x and activation transposed in MXFP8 and gate/up in
E4M3). Backward replays later macrobatches inside the fused kernel. No routed
expert output is saved: the router-score gradient is fused into SwiGLU
backward as `dot(d_hidden, hidden) / score`, as in native MoK. Scores must be
positive; a zero score yields a zero score gradient. With
`router_weight_position="before_down"` the router-weight gradient is
`dot(dh, a)` (`dh` the gradient of the down projection's input `h`, `a` the
SwiGLU activation), with no division: it is defined at a zero weight, whose
route has zero output and zero input gradient. The weights are gathered at
dispatch into an FP32 row ring, and `d_a = w * dh` before the SwiGLU backward. A
larger macrobatch ring uses more memory and replays less in backward.

A context serves one `backward`. When the routed rows exceed one macrobatch,
`backward` replays the later macrobatches into the context's routed ring
tensors (gate, up and hidden; with three or more macrobatches also the routed
x), in either router-weight position. Do not read a context after its
`backward`; call `recompute_forward_context` again before another `backward`
of the same step.

`backward` returns `dx`, `dscores` and the six weight gradients. The fused
kernel's routed and shared scratch is released as soon as the launch is
enqueued, before the output epilogue and the score-gradient copy allocate,
so a training step's peak allocation is inside the fused backward launch for
both precisions, except when the intermediate size is below a third of the
hidden size and a rank's source rows exceed twice the macrobatch.

`recompute_forward_context(config, workspace, schedule, x, shared_gate,
shared_up, routed_gate, routed_up, swiglu_limit=None, *, router_weights=None)`
dispatches and recomputes the gate/up GEMMs and SwiGLU of exactly that context.
Pass the same schedule, inputs, weights and limit as the checkpointed forward
(with `router_weight_position="before_down"` also its router weights, which
the routed hidden `BF16(w * a)` of the context reads); the returned context is
bitwise identical to the forward's and interchangeable with it.

### Context-free forward

With BF16 routed experts, `forward(..., save_context=False)` runs the
context-free forward of an activation-checkpointed step and returns
`(y, None)`. Its fused tiles store hidden, which the down projection reads,
and skip the shared and routed gate/up stores that only backward reads; it
also allocates no gate/up tensors. `y` is bitwise identical to the context
forward's `y`. Before `backward`, rebuild the context with
`recompute_forward_context` (same schedule, inputs, weights and limit);
backward then returns the same bits as after a context forward. Compile it
with `prepare(swiglu_limit, save_context=False)` before CUDA Graph capture.
The MXFP8 forward always saves its context and rejects `save_context=False`.
The kernels are the `forward_noctx` and `forward_clamped_noctx` roles; they
take the forward's arguments and launcher.

Local work follows the real source rows: shared-expert GEMMs, their weight
gradients and both epilogues read only the active prefix rounded up to 256
rows; peers read only routed rows. Reserved source or schedule capacity adds
memory, not work.

## Targets

`sources.json` lists every kernel role with the targets it is generated
for, the argument names of its entry and its architecture-free `library`
stem (`cake_mok_<role>`); `build.json` (schema `cake-build/1`) records the
sources, digests and nvcc flags of every `<library>_<arch>` build
(`cake_mok_<role>_<arch>`, for example `cake_mok_forward_sm_100a`), and the
shared `cake_build.py` loader loads it (prebuilt from `aot/`, else JIT).
`jit.variant(role, capability)` returns the record selected by the device
compute capability and `jit.library_name(role, capability)` the name of its
`build.json` entry, a plain string like the other Cake loaders. The bindings include the
package's `csrc/cake_ffi_checks.h` and no other project header. The fused bindings
(`csrc/sm_100a/*_binding.cu`), the BF16 fused kernels (`csrc/sm_100a/*.cu`,
which select the UMMA shared-memory descriptor address mask from
`__CUDA_ARCH__`: 15 bits on SM107a, 14 bits otherwise) and the scheduler,
communication and epilogue sources are shared by SM100a, SM103a and SM107a.
Launchers of the same shape share one TU: the plain fused launcher also
serves the `_clamped` kernel of its role (and the forward launcher the
`_noctx` kernels), and `csrc/cake_mok_count_binding.cu`,
`cake_mok_rows_binding.cu`, `cake_mok_pad_binding.cu`,
`cake_mok_epilogue_2_binding.cu` and `cake_mok_epilogue_8_binding.cu` serve
their kernel families; the entry's `nvcc_flags` name the kernel with
`-DCAKE_MOK_KERNEL=<symbol>` (and the count/rows launch's dynamic shared
memory with `-DCAKE_MOK_DYN_SMEM=<bytes>`).
The MXFP8 fused kernels have target-specific schedules: SM100a/SM103a (512 tensor-memory
columns) use 256 x 256 routed tiles with a six-stage operand ring, and SM107a
(576 columns) uses 256 x 512 tiles that compute and save gate and up in one
tile, with the K=64 block-scaled MMA. The adapter selects the variant from the
device compute capability (10.0, 10.3 or 10.7) and loads its build for that
exact target; on the JIT path SM107a requires CUDA 13.5 and every generated
source is checksum-verified before compilation.

The router-weight-before-down kernels (`forward_clamped_wbd`,
`forward_clamped_noctx_wbd`, `recompute_clamped_wbd`,
`backward_clamped_fp32_wbd`), the top-k 6 unweighted epilogue
(`epilogue_backward_6`; top-k 6 has no weighted forward epilogue) and the
count / rows kernels of the 384-expert layouts are generated for SM100a and
SM103a only. The forward and recompute take two arguments more than their
clamped roles (`weight_peers`, the uint64 table of the ranks' router-weight
buffers, and `row_weights`, the per-call FP32 `[macro_size]` ring); the
backward takes the clamped FP32 backward's arguments. `prepare_mok_bf16` with
`router_weight_position="before_down"` refuses an SM107a device before it loads
a kernel.

The locality-domain (`_domain`) builds of the MoK kernels, which split work
over the two dies of an SM107a GPU, are not shipped: this package has no such
library, and no entry point selects one.

## Unequal source inputs

`create_mok_bf16_workspace` collectively negotiates capacity from each rank's
actual `num_local_tokens`. Supply optional `source_capacity` with the same
value on every rank to reserve room for future inputs. The returned workspace
contains common physical `storage`; schedules and outputs use logical source
lengths. Invalid expert IDs mask padding, which contributes no routed work.
Empty source ranks still participate in every collective; their local shared
expert weight gradients are zero.

To change counts within capacity, create a new schedule and forward context.
Recapture CUDA Graphs when tensor shapes or addresses change. Same-shape value
updates can replay directly. Earlier graphs can still replay while their
inputs and workspace remain alive. Run graphs sharing a workspace serially
and in the same order across ranks. Growth beyond capacity requires all ranks
to recreate the workspace.

`build_schedule` derives the expert layout once per schedule
(`MoKSchedule.layout`, int32 `[3 * local experts + schedule_capacity / 256]`:
padded counts, row offsets, real rows and the expert of each 256-row block) and
every fused launch of that schedule reads it; a schedule built without it
(`layout=None`) makes each launch derive its own. The layout is a function of
`tokens_per_expert` and `peer_rank`, whose real rows come first within each
expert's padded rows; a schedule copied with `dataclasses.replace` that changes
`peer_rank`, `tokens_per_expert` or the capacity must pass `layout=None`. The
launchers memoise their launch-invariant geometry per distinct launch
arguments (integers and, for MXFP8, the device); the memo holds no tensors
and does not change the launch sequence.

`build_schedule` validates its launches once per workspace and adapter and
keeps them with the workspace's launch plans: the gather, barrier, count, pad,
rows and layout kernels with their workspace arguments and an int32 route
buffer that each call overwrites before the gather (in stream order, as the
workspace's shared buffers already require). Each schedule is one int32
allocation (`peer_token`, `peer_rank`, the counters and the layout) whose
`-1` sentinel and zero counters are written by one copy. The kernels, their
order and their argument values are those of the unplanned launches, which
the first `build_schedule` of a workspace runs when it happens during CUDA
Graph capture. The plan holds normal tensors also when the first
`build_schedule` runs under `torch.inference_mode()`, so later calls may run
in or outside inference mode.

## Repeatability

Within the scope below, every output repeats bit for bit:

| Output | Repeats bit for bit |
| --- | --- |
| `forward`: `y` | yes; identical with `save_context=False` |
| `forward`: context | yes |
| `recompute_forward_context`: context | yes, identical to the forward's |
| `backward`: `dx` | yes |
| `backward`: `dscores` | yes |
| `backward`: six weight gradients | yes |

The context is `x_routed`, `gate_shared`, `gate_routed`, `up_shared`,
`up_routed`, `hidden_shared` and `hidden_routed` (MXFP8 routed fields as data
and scales); the recomputed context is bitwise identical to the forward's (see
Context, memory and recompute). The weight gradients are shared and routed
gate/up/down, or the six FP32 accumulators with `fp32_wgrad=True`. This holds
for BF16 and MXFP8 routed experts, for plain and clamped SwiGLU and for both
router-weight positions.

Scope: repeated calls with the same inputs (source tokens `x`, expert IDs,
scores, per-rank source counts, `dy`), the same weights (and the same MXFP8
prequantized weights), the same `MoKConfig` values other than the
communication-SM split (minibatch and macrobatch sizes, schedule capacity), the
same precision, `swiglu_limit`, `fp32_wgrad` and `router_weight_position`, the
same EP group (size and rank order), the same package build and the same GPU model (compute capability
and SM count). Results may differ bitwise across GPU architectures, EP sizes,
minibatch or macrobatch sizes, precisions, package versions and CUDA
toolkits.

No output bit depends on:

- the communication-SM split: a fixed pair, the `"auto"` rows or the
  imbalance-aware switch, including ranks running different splits in one
  step;
- other kernels resident on other streams that occupy SMs while MoK runs;
- CUDA Graph replay versus eager launches;
- on SM107a MXFP8, the choice of the two-macro-batch forward program (see
  Contract).

Every reduction runs in a fixed order: the top-k combine adds each token's
slots in slot order, each routed weight-gradient tile has one writer over its
full reduction, router-score gradient partials are summed in a fixed order and
later macrobatches are added in macrobatch order. Kernels use atomics only on
integers, to count rows and claim work, never to add floating-point values.

The previous Cake MoK release (2026-11-09) was checked on GB300 (SM103a) by
repeating the checkpointed training step (schedule, forward, context recompute,
backward) five times and comparing every output bitwise on every rank: H 6144,
I 2048, 256 experts, top 8 at EP4/8/16/64 and H 4096, I 2048, 288 experts,
top 8, clamped SwiGLU limit 10 at EP8/32; 16,384 tokens per rank, FP32 accumulators,
BF16 and MXFP8; uniform, ragged (1 to 32,767 tokens, one empty rank) and
1.5x-hot routing; with a resident side-stream kernel holding up to 16 SMs,
across communication-SM splits and under CUDA Graph replay.

## Contract

- Scheduler layouts `(EP, local experts)`: `(1, 4)` and `(4, 4)` diagnostic
  settings; 256 routed experts at EP 4/8/16/32/64 (`(4, 64)`, `(8, 32)`,
  `(16, 16)`, `(32, 8)`, `(64, 4)`); 288 routed experts at EP8
  and EP32 (`(8, 36)`, `(32, 9)`). Top-k is 2 or 8.
- Router-weight positions (`router_weight_position`): the table below. A
  combination outside it raises before any kernel is loaded or launched:
  `TypeError` for a position that is not a `str`, an `ep_size`,
  `local_experts` or `topk` that is not an `int` (a `bool` included) and a
  `clamped_swiglu` or `fp32_wgrad` that is not a `bool`, in both positions;
  `ValueError` for every other value.

  | Position | Layouts (EP, local experts) | Top-k | SwiGLU | Weight gradients | Routed weights | Targets |
  | --- | --- | --- | --- | --- | --- | --- |
  | `"after_down"` (default) | the layouts above | 2 or 8 | plain or clamped | BF16 or FP32 | BF16 or MXFP8 | SM100a, SM103a, SM107a |
  | `"before_down"` | `(4, 96)`, `(8, 48)`, `(16, 24)`, `(32, 12)`, `(64, 6)` | 6 | clamped | FP32 | BF16 | SM100a, SM103a |
  | `"before_down"` | `(1, 4)`, `(4, 4)` (diagnostic) | 2 | clamped | FP32 | BF16 | SM100a, SM103a |

- BF16 inputs, shared weights and outputs; FP32 router scores (positive with
  `"after_down"`, nonnegative with `"before_down"`); contiguous int64 expert
  IDs. Each token selects distinct, valid expert IDs. Scores are already
  normalized/scaled by the caller.
- Logical source token counts may differ across ranks and may be zero. Hidden
  and intermediate widths are positive multiples of 256; MXFP8 also requires a
  hidden width that is a multiple of 512. Below CUDA 13.1 the MXFP8 forward
  and recompute roles convert to E4M3 through FP32
  (`cvt.rn.satfinite.e4m3x2.f32`) instead of `cvt.rn.satfinite.e4m3x2.bf16x2`,
  which older ptxas rejects. Every sm_107a (Rubin) library needs CUDA 13.5.
  Common physical source capacity is
  aligned for tiles/metadata and is at least 512 rows. Communication-SM counts
  are positive and even, leaving compute SMs available.
- Imbalance-aware split (`MoKConfig.imbalance_comm_sms` / `imbalance_threshold`;
  factory default `imbalance_comm_sms="auto"`, `None` keeps the fixed split): every
  `build_schedule` counts the gathered expert IDs per owning rank on device
  inside the scheduler's count kernel (the same pass that counts this rank's
  experts; no extra launch) and copies the counts into pinned host memory
  behind a CUDA event (no host synchronization; skipped during graph capture).
  The gathered IDs and the counts are identical on every rank; each rank
  applies its own latest completed observation, so with back-to-back
  steps a launch may use different splits on different ranks for a step - the
  split is a per-rank launch parameter and the protocol does not require
  agreement. While the hottest rank's load is at least `imbalance_threshold`
  x the mean load (facade default `"auto"`: the device's value in
  `IMBALANCE_DEVICE_THRESHOLDS`, else the precision's value in
  `IMBALANCE_THRESHOLDS`, BF16 4, MXFP8 2.7), the later launches use the
  (forward, backward) imbalance split; balanced steps are unchanged. Measured
  rows (`IMBALANCE_COMM_SMS`, keyed by compute capability and SM count, so B200
  at 148 SMs only - GB200 is also (10, 0) at 152 SMs and keeps the fixed split,
  where the B200 BF16 row measured slower under skew at EP16-64), from EP8
  complete training steps, two rounds on one node, step time against the fixed
  split of this package's programs. B200 BF16 24 / 28 against 20 / 24: hottest
  rank at 5.6x / 7.4x the mean routed load -1.0 / -1.7 % and -1.2 / -1.3 %;
  hottest rank at 3.0x (one rank's experts at twice the routing probability)
  +0.4 / +0.7 % (full step), +0.5 / +0.8 % (checkpoint step) and -0.1 ... +1.6 %
  with ragged source rows (8 cells, 7 slower): no gain, within or just above the
  ~0.8 % spread of identical arms, so the fixed split is kept there; the BF16
  threshold sits between the measured 3.0x and 5.6x (the crossover in between is
  not measured; with 4 ranks the ratio reaches 4 only when one rank carries
  every routed row). B200 MXFP8 44 / 36 against 40 / 32: hottest rank at 3.0x
  -0.8 / -1.2 % (full step) and -1.1 / -1.0 % (checkpoint step), at 5.6x
  -2.6 / -2.9 %; balanced steps +0.6 / +0.3 % (full) and +0.9 / +0.8 %
  (checkpoint) under an unconditional 44 / 36, which is why only imbalanced
  steps switch; with 4 ranks at 2.4x +0.8 % (one screening round). The MXFP8
  threshold sits between the measured 2.4x and 3.0x. Rubin (212 SMs) MXFP8
  72 / 60 over its fixed 60 / 60 from a device threshold of 1.5
  (`IMBALANCE_DEVICE_THRESHOLDS`, which replaces the precision's threshold for
  that device), two-node EP8, 27116 tokens per rank, configurations rotated on
  one workspace, four rounds, hottest-rank medians, MoE training-update proxy:
  hottest rank at 1.51x the mean routed load forward -6.9 % / update -3.8 %,
  2.05x -5.7 / -2.8 %, 3.05x -10.1 / -5.1 %, 4.66x -11.1 / -5.5 %, 6.47x
  -10.6 / -5.5 %; balanced steps do not switch, where an unconditional 72 / 60
  forward is 4.1 % slower. No row elsewhere.
  `MoKConfig.comm_split_log=True` (default `False`; `dataclasses.replace(config,
  comm_split_log=True)` on the returned config) prints each split change per
  rank to stderr; the `CAKE_MOK_COMM_SPLIT_LOG` environment variable that used
  to select this no longer exists.
- Two-macro-batch forward program (sm_107a MXFP8 only): the
  `forward_mxfp8[_clamped]_multimacro` build streams macro batch 0's
  unaliased rows to the experts while macro batch 1 combines, instead of
  after it. Each rank chooses it from the same lagged observation as the
  split, when its padded schedule rows (`num_tokens`) filled exactly two
  macro batches; otherwise, before the first observation and at a schedule
  capacity of at most one macro batch the default build runs (and no
  observation is queued for this alone). Both builds give bitwise the same
  outputs, context and gradients for any schedule, so ranks may run different
  builds in one step and a stale choice costs time only. `comm_split_log`
  also prints each change of the choice.
- Fixed split per device and group size: `fwd_num_comm_sms` / `bwd_num_comm_sms`
  default to `"auto"` in `create_mok_bf16_workspace` (package and
  `cake_mok_det_train` facade): the measured `COMM_SMS` row (compute capability, SM
  count, precision, EP group size), else the factory 40 / 40. Integers fix the
  split; `"auto"` for only one of the two is an error. With MXFP8 routed experts
  pass `precision="mxfp8"` for the MXFP8 row (`precision` only selects rows and
  thresholds). The split changes time, never bits (see Repeatability). GB300
  (152 SMs): BF16 20 / 24 at EP4/8/16/32/64; MXFP8 44 / 36 at EP8/32/64, none
  at EP4/16 (no gain measured). Checkpointed training steps captured in CUDA
  Graphs, 40 / 40 and the row interleaved in one process on the same nodes,
  slowest-rank median of 12 steps, 16,384 tokens per rank, FP32 accumulators:

  | EP | shape | precision | routing | 40 / 40 (ms) | row (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- |
  | 4 | H 6144 / 256 experts | BF16 | uniform | 31.95 | 29.83 | -6.6 % |
  | 8 | H 6144 / 256 experts | BF16 | uniform | 30.89 | 28.50 | -7.7 % |
  | 8 | H 4096 / 288 experts | BF16 | uniform | 21.17 | 19.31 | -8.8 % |
  | 16 | H 6144 / 256 experts | BF16 | uniform | 30.81 | 28.19 | -8.5 % |
  | 32 | H 4096 / 288 experts | BF16 | uniform | 21.11 | 18.84 | -10.7 % |
  | 32 | H 4096 / 288 experts | BF16 | ragged | 26.83 | 24.25 | -9.6 % |
  | 32 | H 4096 / 288 experts | BF16 | 1.5x hot rank | 29.51 | 26.52 | -10.2 % |
  | 64 | H 6144 / 256 experts | BF16 | uniform | 30.91 | 30.12 | -2.6 % |
  | 64 | H 6144 / 256 experts | BF16 | ragged | 39.66 | 38.29 | -3.5 % |
  | 64 | H 6144 / 256 experts | BF16 | 1.5x hot rank | 42.52 | 39.70 | -6.6 % |
  | 8 | H 6144 / 256 experts | MXFP8 | uniform | 21.38 | 21.46 | +0.4 % |
  | 8 | H 4096 / 288 experts | MXFP8 | uniform | 15.60 | 15.54 | -0.4 % |
  | 32 | H 4096 / 288 experts | MXFP8 | uniform | 15.35 | 14.80 | -3.6 % |
  | 32 | H 4096 / 288 experts | MXFP8 | ragged | 20.42 | 20.13 | -1.4 % |
  | 32 | H 4096 / 288 experts | MXFP8 | 1.5x hot rank | 20.77 | 20.40 | -1.8 % |
  | 64 | H 6144 / 256 experts | MXFP8 | uniform | 25.71 | 25.25 | -1.8 % |
  | 64 | H 6144 / 256 experts | MXFP8 | ragged | 36.58 | 35.99 | -1.6 % |
  | 64 | H 6144 / 256 experts | MXFP8 | 1.5x hot rank | 32.67 | 32.89 | +0.7 % |

  The MXFP8 EP8 row first measured -3.3 ... -1.8 % against 40 / 32 with the
  hottest rank at 3.0x-7.4x the mean routed load; on balanced steps it is
  within -0.4 ... +0.4 % of 40 / 40 (five comparisons, EP8, both shapes).
  Rubin (212 SMs) MXFP8 EP8 60 / 60 against 40 / 40, balanced routing,
  hottest-rank medians of four rounds, forward / backward at 8192 / 27116 /
  32768 tokens per rank: -7.4 / -4.6, -13.3 / -8.7, -18.1 / -2.4 % (round
  spread at most 1.2 %); Rubin BF16 had no consistent gain at EP8.
- Mini-batches are multiples of 256; macro-batches are multiples of the
  mini-batch. Schedule capacity must hold every padded route; overflow traps.
  The capacity is `T * topk * max(2, ceil(EP * m))` rows for
  `schedule_capacity_multiplier` `m` and the workspace's physical source rows
  `T` (the largest rank's source rows, or `source_capacity`, rounded up to the
  256-row tile, at least 512); every routing fits when
  `m >= ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP` (`E` local
  experts), about `(EP + 1) / EP`. The bound decreases as `T` grows, so a
  multiplier computed from any `T` up to the physical rows also holds; every
  rank must pass the same `m` (the ranks' configurations are compared at
  workspace creation), so compute it from a row bound all ranks agree on, such
  as `source_capacity`. No upper bound is enforced. The 384-expert layouts at this bound (`(64, 6)` and `(8, 48)` at
  16,384 rows, macrobatch 262,144, minibatch 4,096, every route on one rank)
  run on one GPU with emulated peers: finite outputs, the same bytes run to
  run, routed x ring rows past 2^31 bytes checked exactly, `y`, `dx` and the
  router-weight gradients equal to a 32,768-row macrobatch run, and the routed
  weight gradients of a step where only tokens that share no expert keep their
  router weights equal to that run and exactly EP times one BF16 product per
  element. Multi-GPU runs at these layouts are a separate test.
- Routed weight gradients are summed across macrobatches in BF16 (or in the
  FP32 accumulators). MXFP8 adds quantization error to the routed outputs and
  gradients; the shared expert path is unchanged.

Optimizer steps, router-network backpropagation and shared-gradient
all-reduce are outside this adapter.

## Validation

Tests (`kernels/mok/tests/`, run against the installed distribution as the
repository's `ARCHITECTURE.md` describes; GPU tests skip without an SM100a,
SM103a or SM107a device):

- `tests/experimental/test_mok_bf16.py`: the fused BF16 forward/backward
  (empty experts, multiple ring lengths, variable widths, CUDA Graph replay,
  changed and zero scores), clamped SwiGLU, FP32 accumulation, 288-expert local
  expert counts, recompute versus saved context, the context-free forward,
  every scheduler layout, the counter slab, the launch-geometry memo and the
  epilogues, with the bitwise contracts above as exact checks.
- `tests/experimental/test_mok_mxfp8.py`: the weight quantizer (PyTorch path
  against the recipe reference, byte for byte, NaN / infinity / zero blocks
  included) and the fused MXFP8 forward, forward+backward and recompute.
- `tests/experimental/test_mok_accuracy.py`: accuracy of the fused BF16 and
  MXFP8 outputs against an FP64 oracle, judged against a PyTorch naive
  implementation of the same operation (`tests/cake_naive.py`).
- `tests/experimental/test_mok_distributed.py`: EP1 and EP4 runs (the
  complete training step eager and under CUDA Graph replay, unequal and empty
  ranks, adapter guards, training features), each judged for accuracy over the
  whole group. A test starts its own processes, one per GPU, when one process
  sees enough GPUs of one NVLink domain; run it with the `SLURM_*` task
  variables unset.
- `tests/experimental/test_cake_mok_det_train_api.py`: the
  `cake_pplx.cake_mok_det_train` facade (signatures, the package names it
  serves, argument forwarding and the refused router-weight combinations).
- `router_weight_position="before_down"`: in `test_mok_bf16.py` the launchers'
  arguments against `sources.json` and the refusals before device work (no
  GPU), the refusal on an SM107a device, and on SM100a / SM103a the fused
  training at top-2 and top-6 (6 to 96 local experts; another
  communication-SM split gives the same bytes), zero weights, recompute and
  the context-free forward, the exact limit, the 384-expert scheduler layouts,
  the scheduler and the adapter at the worst-case schedule capacity, and the
  adapter's top-6 step over changing source counts on one workspace (1031,
  768, 257 and 513 rows: padded and unpadded staging; the bytes of a fresh
  workspace whatever the rows past the count hold); in `test_mok_accuracy.py`
  the same cases, the requester's input distribution at H 5120 / I 2304, the
  adapter's top-6 step at every 384-expert layout and over those changing
  counts at EP64 and EP8 (one GPU, emulated peers) judged against the naive
  tier; in `test_mok_distributed.py` the adapter at EP1 over changing source
  counts, with another communication-SM split and a test-local autograd
  function;
  `tests/test_mok_naive_wbd.py` (CPU) the oracle and naive forms.
  `tests/experimental/report_mok_wbd.py` (a measurement report, not a test)
  prints the issue's stage-rounded reference errors on those inputs.

The CUDA schedules derive from Cursor Research's Apache-2.0 Mixture of Kittens
implementation at revision `caeb2963f855c7ad53bb50c8bbf211086405cb98`. Copyright and modification notices are
retained in the generated source files.
