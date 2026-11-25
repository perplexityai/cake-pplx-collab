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
  (`router_weight_position="before_down"`, SM100a, SM103a and SM107a): each
  route's FP32 router weight `w` scales the SwiGLU activation `a` before the
  down projection, `h = BF16(w * a)`, and the routed outputs are summed
  unweighted; top-k 6 over 384 routed experts (`(4, 96)`, `(8, 48)`,
  `(16, 24)`, `(32, 12)`, `(64, 6)`) and top-k 2 on the diagnostic layouts,
  with clamped SwiGLU and FP32 weight-gradient accumulation (see Contract).
- **Explicit modes** (`router_weight_gradient`, `shared_expert`, `activation`;
  defaults `"quotient"`, `"unscaled"`, `"swiglu"` keep every earlier bit):
  the direct router-weight gradient (`"direct"`: `dot(dh, h)` from the
  unscaled `dy`, defined at a zero weight) and the per-token shared-expert
  scale (`"scaled"`: `y = s * S(x) + sum_k w_k E_k(x)` with its gradient
  `d_s`), run by the 512-routed-expert group at EP8 `(8, 64)`, top-k 10, on
  SM100a and SM103a; SiTU-GLU routed experts (`activation="situ_glu"`,
  `beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb)` with per-call
  `situ_beta`, `situ_lb`) without a shared expert (`shared_expert="none"`), run
  with the direct router-weight gradient by the 896-routed-expert group at EP64
  `(64, 14)`, top-k 16, on SM100a and SM103a (see Contract, "Explicit modes").

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

# 512 routed experts at EP8 (64 local experts), top-10, with the direct
# router-weight gradient and a per-token FP32 shared-expert scale s
# (shape [n]): y = s * S(x) + sum_k w_k E_k(x). Plain SwiGLU, FP32 weight
# gradients, BF16 routed weights; a workspace of its own at the worst-case
# schedule capacity, as above.
ep, local, k = 8, 64, 10
worst = min(k, local) * max_rows * ep + local * 255
m = math.ceil(worst / (max_rows * k)) / ep
config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=n, hidden_size=4096, topk=k,
    source_capacity=max_rows, schedule_capacity_multiplier=m,
)
accumulators = [torch.zeros(w.shape, device=device) for w in weights]  # FP32
backend = prepare_mok_bf16(ep_size=ep, local_experts=local, topk=k, fp32_wgrad=True,
                           router_weight_gradient="direct", shared_expert="scaled")
backend.prepare(save_context=False)  # compile before CUDA Graph capture
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=local)
y, _ = backend.forward(config, workspace, schedule, x, router_w, *weights,
                       save_context=False, shared_scale=s)
context = backend.recompute_forward_context(
    config, workspace, schedule, x, *weights[:2], *weights[3:5])
dx, d_w, *weight_grads, d_s = backend.backward(
    config, workspace, schedule, context, dy, x, router_w, *weights,
    weight_grad_accumulators=accumulators, shared_scale=s)

# 896 routed experts at EP64 (14 local experts), top-16, SiTU-GLU without a
# shared expert: None for the three shared weights, situ_beta / situ_lb on
# every call, three FP32 accumulators (routed gate, up, down). A token routes
# at most 14 of its 16 routes to one rank: min(k, local) in the bound.
ep, local, k = 64, 14, 16
worst = min(k, local) * max_rows * ep + local * 255
m = math.ceil(worst / (max_rows * k)) / ep  # 57 / 64 at 16,384 rows
config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=n, hidden_size=3584, topk=k,
    source_capacity=max_rows, schedule_capacity_multiplier=m,
)
routed = (routed_gate, routed_up, routed_down)  # [14, I, H], [14, I, H], [14, H, I]
accumulators = [torch.zeros(w.shape, device=device) for w in routed]  # FP32
backend = prepare_mok_bf16(ep_size=ep, local_experts=local, topk=k, fp32_wgrad=True,
                           router_weight_gradient="direct", shared_expert="none",
                           activation="situ_glu")
backend.prepare(save_context=False)
backend.prepare_recompute()
situ = dict(situ_beta=beta, situ_lb=lb)  # Python floats
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=local)
y, _ = backend.forward(config, workspace, schedule, x, router_w, None, None, None,
                       *routed, save_context=False, **situ)
context = backend.recompute_forward_context(
    config, workspace, schedule, x, None, None, routed[0], routed[1], **situ)
dx, d_w, *routed_grads, _, _, _ = backend.backward(
    config, workspace, schedule, context, dy, x, router_w, None, None, None, *routed,
    weight_grad_accumulators=accumulators, **situ)
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
down, routed gate, up, down) as `weight_grad_accumulators`, or with
`shared_expert="none"` the three routed ones; backward adds into them and
returns them. `situ_beta` and `situ_lb` (`activation="situ_glu"`) are per-call
runtime values like `swiglu_limit`: a captured CUDA graph replays the values of
its capture.

Prepare and warm up before CUDA Graph capture. All ranks must execute
matching collective calls. Keep workspace allocations and peer mappings alive
while the adapter or a captured graph can use them. Use a separate workspace
and adapter for concurrent executions. Pass the matching context and schedule
to backward; retain both while any captured graph uses them.

With `router_weight_position="before_down"` the workspace is unchanged and
`scores` are the router weights `w` (FP32 `[tokens, topk]`, each `>= 0`; a zero
weight is allowed). `prepare_mok_bf16` refuses, before it loads a kernel, a
layout or top-k the mode does not run, `clamped_swiglu=False` and
`fp32_wgrad=False`; `forward`, `recompute_forward_context`, `backward`,
`prepare` and `prepare_recompute` refuse `swiglu_limit=None` and MXFP8 routed
weights; `recompute_forward_context` requires `router_weights`
(the forward's). A checkpointing caller wraps forward, recompute and backward
in its own `torch.autograd.Function`; the package ships no autograd wrapper.

Calling the API is the opt-in. There is no automatic dispatch or AOT
registration.

## Context, memory and recompute

A forward context holds the shared-expert activations and the routed
activations of the first macrobatch, which stays resident in the forward ring
(MXFP8 keeps the routed x and activation transposed in MXFP8 and gate/up in
E4M3). Backward replays later macrobatches inside the fused kernel. No routed
expert output is saved: with the default
`router_weight_gradient="quotient"` the router-score gradient is fused into
SwiGLU backward as `dot(d_hidden, hidden) / score`, as in native MoK. Scores
must be positive there; a zero score yields a zero score gradient. With
`router_weight_gradient="direct"` (the 512-expert top-10 and the 896-expert
top-16 groups) the
router-weight gradient is `dot(dh, h)` with `dh` from the unscaled `dy`, with
no division: router weights are `>= 0` and a zero weight has its gradient. With
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
count / rows kernels of the 384-expert layouts are generated for SM100a,
SM103a and SM107a. The forward and recompute take two arguments more than their
clamped roles (`weight_peers`, the uint64 table of the ranks' router-weight
buffers, and `row_weights`, the per-call FP32 `[macro_size]` ring); the
backward takes the clamped FP32 backward's arguments.

The 512-expert top-10 group's kernels (`MODE_GROUPS["scaled_top10"]`) are
generated for SM100a and SM103a only: the direct-gradient backward
`backward_fp32_direct` (the arguments of `backward_fp32` and `dy_scaled_ptr`,
the second per-call `[macro_size, hidden]` BF16 ring that holds the scaled
`dy` rows the down-projection weight gradient reads), the scaled top-10
epilogues `epilogue_forward_scaled_10` and `epilogue_backward_scaled_10` (the
arguments of the top-8 ones and `shared_scale`, FP32 `[tokens]`), the two
shared-scale helpers `shared_scale_rows` and `shared_scale_dot`, and the count /
rows kernels of `(8, 64)`. The adapter refuses the group on any other target
before it loads a kernel.

The 896-expert top-16 group's kernels (`MODE_GROUPS["situ_top16"]`) are
generated for SM100a and SM103a only: the SiTU-GLU forward, context-free
forward and recompute without a shared expert (`forward_situ_noshared`,
`forward_noctx_situ_noshared`, `recompute_situ_noshared`: the arguments of
their SwiGLU roles without the shared maps and `local_tokens`, with the FP32
`situ_beta` and `situ_lb` in the place of `swiglu_limit`), the direct-gradient
SiTU-GLU backward `backward_fp32_direct_situ_noshared` (the arguments of
`backward_fp32_direct` without the shared maps and `local_tokens`, the same
replacement), the top-16 epilogues without a shared row
`epilogue_forward_none_16` and `epilogue_backward_none_16` (the arguments of
the top-8 ones without `shared`), and the count / rows kernels of `(64, 14)`.
These launches have no shared row, shared task or shared weight-gradient
task. The adapter refuses the group on any other target before it loads a
kernel.

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

A padded source prefix (a source count that is not a multiple of 256) stages
each source buffer (`x`, the router weights, `dy`) with one `torch.cat` of the
caller's rows and the workspace's constant padding rows (BF16 zeros and FP32
ones, 256 rows each, kept with the launch plans): the bytes of the copy and the
fill it replaces, in one launch. A source that autograd records, or one that
overlaps the workspace, takes the copy and the fill. Every call still writes
the padding rows of the workspace's source, combine and gradient buffers; it
never relies on their earlier contents.

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

- the communication-SM split: a fixed pair, the `"auto"` rows, the
  imbalance-aware switch or the source split, including ranks running
  different splits in one step;
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
  | `"before_down"` | `(4, 96)`, `(8, 48)`, `(16, 24)`, `(32, 12)`, `(64, 6)` | 6 | clamped | FP32 | BF16 | SM100a, SM103a, SM107a |
  | `"before_down"` | `(1, 4)`, `(4, 4)` (diagnostic) | 2 | clamped | FP32 | BF16 | SM100a, SM103a, SM107a |

- Explicit modes (`prepare_mok_bf16` keywords; the first value of each table is
  the default and keeps every earlier bit). Their types are checked with the
  other types (`TypeError`), their values after them (`ValueError`):

  | Keyword | Values | Meaning |
  | --- | --- | --- |
  | `router_weight_gradient` (`ROUTER_WEIGHT_GRADIENTS`) | `"quotient"`, `"direct"` | `"after_down"` router-weight gradient. `"quotient"`: `dh = BF16(BF16(w * dy) Wd)`, `d_w = dot(dh, h) / w`, zero at a zero weight. `"direct"`: `dh = BF16(dy Wd)` from the unscaled `dy`, activation gradient `w * dh` in FP32, `d_w = dot(dh, h)` (FP32, fixed order), defined at a zero weight; the down-projection weight gradient reads `BF16(w * dy)` in both, bit for bit the same |
  | `shared_expert` (`SHARED_EXPERT_MODES`) | `"unscaled"`, `"scaled"`, `"none"` | `"scaled"`: `forward` and `backward` take `shared_scale` (FP32 `[tokens]`, the forward's on the backward), `y = BF16(s * BF16(S(x)) + sum_k w_k o_k)` in FP32; `dx` adds `s * dx_shared` (`dx_shared` from the unscaled `dy`); the shared down-projection weight gradient reads `BF16(s * dy)`, the shared gate/up weight gradients `BF16(s * x)`; `backward` returns `d_shared_scale = dot(dh_shared, hidden_shared)` (FP32, fixed order; `dh_shared = BF16(dy Wd_shared)`) as a ninth item, defined at `s = 0`. `s = 1` gives the unscaled values bit for bit. `"none"`: no shared expert, `y = BF16(sum_k w_k o_k)` in FP32 from zero; every call takes `None` for the shared weights, the forward context's shared fields are `None`, and `backward` takes and returns the three routed FP32 accumulators and returns `None` for the three shared weight gradients |
  | `activation` (`ACTIVATIONS`) | `"swiglu"`, `"clamped_swiglu"`, `"situ_glu"` | `"clamped_swiglu"` is `clamped_swiglu=True`; `"situ_glu"`: `beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb)`, `situ_beta` and `situ_lb` (finite positive Python floats, rounded to the FP32 kernel values; `2.88539 / beta`, `2.88539 / lb` and `beta * lb` finite in FP32, else `ValueError`) on every `forward`, `recompute_forward_context` and `backward`, `swiglu_limit=None` |

  A configuration that uses a layout, the top-k or a mode of a kernel group
  (`MODE_GROUPS`) runs exactly that group's configuration:

  | Group | Layouts (EP, local experts) | Top-k | Position | Activation | Weight gradients | `router_weight_gradient` | `shared_expert` | Routed weights | Targets |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `"scaled_top10"` | `(8, 64)` (512 routed experts) | 10 | `"after_down"` | plain SwiGLU | FP32 | `"direct"` | `"scaled"` | BF16 | SM100a, SM103a |
  | `"situ_top16"` | `(64, 14)` (896 routed experts) | 16 | `"after_down"` | SiTU-GLU | FP32 | `"direct"` | `"none"` | BF16 | SM100a, SM103a |

  Recompute (`recompute_forward_context`) is unchanged by the router-weight
  gradient and the shared-expert scale; the SiTU-GLU group recomputes its
  SiTU-GLU context without a shared expert. The scaled group's backward holds, beyond the unscaled backward, the scaled `dy` ring
  (`macro_size x hidden` BF16) and the two shared operands `BF16(s * dy)` and
  `BF16(s * x)` (`source rows x hidden` BF16 each) until its launch ends, and
  `dh_shared` until its row dots are taken.

- BF16 inputs, shared weights and outputs; FP32 router scores (positive with
  `"after_down"` and the default `"quotient"` gradient; nonnegative with
  `"before_down"` and with `"direct"`, where a zero weight has a defined
  gradient); contiguous int64 expert IDs. Each token selects distinct, valid
  expert IDs. Scores are already normalized/scaled by the caller.
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
  rows (`IMBALANCE_COMM_SMS`, keyed by compute capability, SM count and the
  workspace's `precision`, so B200 at 148 SMs - GB200 is also (10, 0) at 152 SMs
  and keeps the fixed split, where the B200 BF16 row measured slower under skew
  at EP16-64), from EP8
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
  forward is 4.1 % slower. No row elsewhere. A CUDA Graph keeps the split in
  force at its capture, and layers that share one workspace share its latest
  observation.
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
  count, routed precision, EP group size), else the factory 40 / 40. The routed
  precision is that of each launch's routed weights (BF16 or MXFP8), not the
  workspace's `precision`: one workspace gives BF16 and MXFP8 launches the row
  of their own precision (`MoKConfig.bf16_comm_sms` / `mxfp8_comm_sms`; the
  workspace's `precision` selects the imbalance row and threshold; on such a
  config a `dataclasses.replace` of `fwd_num_comm_sms` / `bwd_num_comm_sms`
  alone changes no launch: create the workspace with integers, or replace
  `bf16_comm_sms` / `mxfp8_comm_sms` too). Integers set the fixed split of every
  launch of both routed precisions (on a device with an imbalance row the
  imbalance split still replaces it under skew unless
  `imbalance_comm_sms=None`); `"auto"` for only one of the two is an error. The
  split changes time, never bits (see Repeatability). GB300
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

  B300 (148 SMs) BF16 EP8: 24 / 40, the forward count only (the backward keeps
  40), a fixed split with no imbalance row: every launch with BF16 routed
  weights runs 24 / 40 whatever the routing of earlier launches, so CUDA Graph
  replay and layers that share one workspace run the split of an eager
  single-layer step. It was chosen from the fixed forward counts 20, 24 and 28
  (one B300 node, EP8, six rounds interleaved with 40 / 40 and a second 40 / 40
  arm, order reversed every second round, record window = timed repetitions 1-20
  after 3 warm-up repetitions, slowest rank's median, median over rounds), each
  measured with the router weight before the down projection (H 5120 / 384
  experts, top 6) balanced at 16,384 tokens per rank and with the hottest rank
  at 1.5x (4,096 and 16,384 tokens), 3.0x and 5.6x (16,384); H 6144 / 256
  experts (top 8) balanced and at 2.0x, 3.0x and 5.6x; H 4096 / 288 experts (top
  8) at 2.0x. A count qualified when none of its forward or checkpointed-step
  rows was slower than 40 / 40 beyond the noise band and at least one balanced
  or 1.5x forward of the router weight before the down projection was faster;
  qualifying: 24 / 40 (geometric mean 0.9694), 28 / 40 (geometric mean 0.9771).
  The lowest geometric mean of those three forward ratios wins; a tie within
  0.5 % goes to fewer rows slower in every round, then the larger count. Not
  qualifying: 20 / 40 had 3 rows slower than 40 / 40 beyond noise (largest: H
  4096 / 288 experts, top 8, hottest rank 2.0x, 16,384 tokens per rank, forward,
  record window, 12.612 -> 12.973 ms (+2.9 %)). The earlier row, 16 / 40 with an
  imbalance switch to 40 / 40 from a hottest rank at 1.8x decided from the
  workspace's latest completed observation, measured slower than 40 / 40 under
  CUDA Graph replay (4 record-window rows; largest: H 6144 / 256 experts, top 8
  (schedule and forward without context), forward captured in a CUDA Graph on a
  balanced step and replayed with the hottest rank at 2.0x, 19.132 -> 21.701 ms
  (+13.4 %)); with one workspace shared by layers of different skew (5
  record-window rows; largest: H 6144 / 256 experts, top 8 (schedule and forward
  without context), forward of the 3.0x layer of a balanced and a 3.0x layer
  sharing one workspace, 27.257 -> 30.803 ms (+13.0 %)); and with
  imbalance_comm_sms=None (5 record-window rows; largest: H 6144 / 256 experts,
  top 8 (schedule and forward without context), forward of the 3.0x layer of a
  balanced and a 3.0x layer sharing one workspace, 27.257 -> 31.178 ms
  (+14.4 %)). B200 (148 SMs) BF16 EP8 keeps 40 / 40: 16 / 40 measured faster at
  1.5-1.8x and slower from 2.0x (H 6144 / 256 experts, top 8). GB200 (152 SMs)
  BF16 EP4 keeps 40 / 40: 16 / 40 measured slower on its 16-expert top-8
  forward. GB300 BF16 EP4 keeps 20 / 24 (16 / 24 and 12 / 24 measured no gain).
  The row applies to launches with BF16 routed weights; MXFP8 launches take the
  MXFP8 key, which has no row on B300 (40 / 40), whatever precision the
  workspace was created with.

  Table: this package against the previous version 0.2.4 (main, 40 / 40 on B300)
  on one B300 node, EP8, both versions loading GPU programs with the same SASS
  (compared per program before timing), BF16 routed experts, one process per
  version and configuration, the two versions and a second process of the
  previous version interleaved over six rounds with the order reversed in every
  second round, eager (CUDA Graph where stated: captured once on a balanced
  step, then replayed), no CPU binding, FP32 accumulators; the slowest rank's
  median of timed repetitions 1-20 after 3 warm-up repetitions, median over the
  rounds; the table prints those medians and change = printed package
  ms / printed main ms - 1; band = the noise band of the identical arms and of
  the round spread; 'faster' or 'slower' = beyond the band in every round and on
  the GPU-paired interval, 'noise' = inside the band, 'unresolved' = beyond the
  band but not in every round; steady = the change of the medians of timed
  repetitions 41-60 of 60. Forward = schedule and forward without context for
  the router weight before the down projection (H 5120 / I 2304, clamped
  SwiGLU); checkpointed step = schedule, forward, context recompute and
  backward. Every configuration of the table with per-rank output digests (all
  except the CUDA Graph row) gives the same bits at 0.2.4 and with this package,
  and every line repeats its outputs bitwise.

  | device | EP | configuration | routing | tokens per rank | forward main (ms) | forward, this package (ms) | change | checkpointed step main (ms) | this package (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | B300 | 8 | before down, 384 experts, top 6 | uniform | 4,096 | 2.169 | 2.081 | -4.1 % (band 7.2 %, noise; steady -4.8 %) | 9.043 | 8.929 | -1.3 % (band 1.1 %, faster; steady -0.7 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 1.5x hot rank | 4,096 | 2.734 | 2.607 | -4.6 % (band 4.6 %, faster; steady -5.2 %) | 11.447 | 11.234 | -1.9 % (band 2.0 %, faster; steady -0.9 %) |
  | B300 | 8 | before down, 384 experts, top 6 | ragged | 2,048-6,143 | 1.961 | 1.958 | -0.2 % (band 9.7 %, noise; steady +0.5 %) | 8.466 | 8.498 | +0.4 % (band 2.9 %, noise; steady -0.8 %) |
  | B300 | 8 | before down, 384 experts, top 6 | uniform | 16,384 | 7.730 | 7.509 | -2.9 % (band 1.8 %, faster; steady -2.6 %) | 28.863 | 28.583 | -1.0 % (band 1.0 %, faster; steady -0.8 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 1.5x hot rank | 16,384 | 10.435 | 10.208 | -2.2 % (band 1.7 %, faster; steady -1.9 %) | 38.433 | 38.507 | +0.2 % (band 1.4 %, noise; steady -0.0 %) |
  | B300 | 8 | before down, 384 experts, top 6 | ragged | 8,192-24,575 | 6.849 | 6.755 | -1.4 % (band 4.0 %, noise; steady -3.6 %) | 25.859 | 25.528 | -1.3 % (band 0.9 %, faster; steady -1.1 %) |
  | B300 | 8 | before down, 384 experts, top 6, CUDA Graph | uniform | 16,384 | 7.731 | 7.451 | -3.6 % (band 2.2 %, faster; steady -4.1 %) | 28.599 | 28.208 | -1.4 % (band 1.5 %, noise; steady -1.0 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 3.0x hot rank | 4,096 | 5.045 | 4.904 | -2.8 % (band 6.8 %, noise; steady -1.9 %) | 19.037 | 18.734 | -1.6 % (band 4.5 %, noise; steady -1.9 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 3.0x hot rank | 16,384 | 19.887 | 19.294 | -3.0 % (band 2.9 %, noise; steady +0.0 %) | 73.842 | 73.368 | -0.6 % (band 1.9 %, noise; steady -0.9 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 5.6x hot rank | 16,384 | 36.150 | 35.232 | -2.5 % (band 1.2 %, faster; steady -2.5 %) | 133.998 | 133.479 | -0.4 % (band 0.6 %, noise; steady -0.8 %) |

  Table: GB300 (152 SMs), EP4 on one GB300 node, the launches whose split this
  version changes there, against the previous version 0.2.4 with the same method
  as the B300 table (GPU programs with the same SASS in both versions): MXFP8
  routed weights on a workspace created with precision="bf16" (20 / 24 ->
  40 / 40) and BF16 routed weights on a workspace created with precision="mxfp8"
  (40 / 40 -> 20 / 24); forward = the forward of H 6144 / I 2048, 256 experts,
  top 8. Both configurations give the same bits at 0.2.4 and with this package,
  and every line repeats its outputs bitwise.

  | device | EP | configuration | routing | tokens per rank | forward main (ms) | forward, this package (ms) | change | checkpointed step main (ms) | this package (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | GB300 | 4 | H 6144 / 256 experts, top 8, MXFP8 weights, workspace precision bf16 | uniform | 16,384 | 8.383 | 6.548 | -21.9 % (band 2.2 %, faster; steady -22.8 %) | 26.759 | 22.894 | -14.4 % (band 0.7 %, faster; steady -14.1 %) |
  | GB300 | 4 | H 6144 / 256 experts, top 8, workspace precision mxfp8 | uniform | 16,384 | 8.794 | 8.236 | -6.3 % (band 1.4 %, faster; steady -7.6 %) | 31.641 | 30.435 | -3.8 % (band 0.9 %, faster; steady -3.1 %) |

  Table: the measurement that chose the row, this package against version 0.2.2
  (40 / 40 on B300; the B300 GPU programs of version 0.2.3 compile to the same
  SASS; version 0.2.4 changed the schedule's host path after these rows), on one
  B300 node, EP8, one process per arm and configuration, the two arms and an
  identical second main arm interleaved over six rounds with the order reversed
  in every second round, eager (CUDA Graph where stated: captured once on a
  balanced step, then replayed), no CPU binding, BF16 routed experts unless
  stated, FP32 accumulators; the slowest rank's median of timed repetitions 1-20
  after 3 warm-up repetitions, median over the rounds; the table prints those
  medians and change = printed package ms / printed main ms - 1; band = the
  noise band of the identical arms and of the round spread; 'faster' or 'slower'
  = beyond the band in every round and on the GPU-paired interval, 'noise' =
  inside the band, 'unresolved' = beyond the band but not in every round; steady
  = the change of the medians of timed repetitions 41-60 of 60. Forward =
  schedule and forward without context for the router weight before the down
  projection (H 5120 / I 2304, clamped SwiGLU), the forward otherwise (H
  6144 / I 2048 plain, H 4096 / I 2048 clamped); checkpointed step = schedule,
  forward, context recompute and backward. Every configuration of the table with
  per-rank output digests (all except the CUDA Graph row) gives the same bits at
  main and with this row, and every line repeats its outputs bitwise.

  | device | EP | configuration | routing | tokens per rank | forward main (ms) | forward, this package (ms) | change | checkpointed step main (ms) | this package (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | B300 | 8 | before down, 384 experts, top 6 | uniform | 4,096 | 2.205 | 2.130 | -3.4 % (band 10.3 %, noise; steady -4.5 %) | 9.070 | 8.964 | -1.2 % (band 2.1 %, noise; steady -1.1 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 1.5x hot rank | 4,096 | 2.817 | 2.714 | -3.7 % (band 5.7 %, noise; steady -2.7 %) | 11.621 | 11.530 | -0.8 % (band 3.8 %, noise; steady -1.1 %) |
  | B300 | 8 | before down, 384 experts, top 6 | ragged | 2,048-6,143 | 2.024 | 1.993 | -1.5 % (band 10.0 %, noise; steady +0.8 %) | 8.458 | 8.463 | +0.1 % (band 1.9 %, noise; steady +1.4 %) |
  | B300 | 8 | before down, 384 experts, top 6 | uniform | 16,384 | 7.751 | 7.556 | -2.5 % (band 2.3 %, faster; steady -4.8 %) | 29.035 | 28.720 | -1.1 % (band 1.3 %, noise; steady -0.7 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 1.5x hot rank | 16,384 | 10.636 | 10.352 | -2.7 % (band 2.6 %, noise; steady -3.0 %) | 38.499 | 38.637 | +0.4 % (band 1.1 %, noise; steady -0.3 %) |
  | B300 | 8 | before down, 384 experts, top 6 | ragged | 8,192-24,575 | 6.861 | 6.767 | -1.4 % (band 5.3 %, noise; steady -1.8 %) | 25.796 | 25.445 | -1.4 % (band 1.2 %, faster; steady -0.9 %) |
  | B300 | 8 | before down, 384 experts, top 6, CUDA Graph | uniform | 16,384 | 7.568 | 7.370 | -2.6 % (band 5.4 %, noise; steady -3.7 %) | 28.643 | 28.375 | -0.9 % (band 1.2 %, noise; steady -0.8 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 3.0x hot rank | 4,096 | 5.177 | 4.892 | -5.5 % (band 8.7 %, noise; steady -2.8 %) | 19.471 | 19.366 | -0.5 % (band 3.7 %, noise; steady -0.9 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 3.0x hot rank | 16,384 | 20.035 | 19.545 | -2.4 % (band 3.4 %, noise; steady -2.1 %) | 75.551 | 74.539 | -1.3 % (band 1.3 %, noise; steady -1.2 %) |
  | B300 | 8 | before down, 384 experts, top 6 | 5.6x hot rank | 16,384 | 36.770 | 35.689 | -2.9 % (band 0.7 %, faster; steady -2.9 %) | 136.300 | 135.010 | -0.9 % (band 0.6 %, faster; steady -0.9 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8 | uniform | 16,384 | 10.300 | 10.009 | -2.8 % (band 2.6 %, faster; steady -3.3 %) | 38.151 | 37.249 | -2.4 % (band 1.4 %, faster; steady -1.6 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8 | 1.7x hot rank | 16,384 | 16.152 | 15.434 | -4.4 % (band 1.1 %, faster; steady -3.3 %) | 58.500 | 57.490 | -1.7 % (band 1.4 %, faster; steady -2.2 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8 | 1.8x hot rank | 16,384 | 16.987 | 16.310 | -4.0 % (band 0.9 %, faster; steady -3.4 %) | 60.668 | 60.228 | -0.7 % (band 0.9 %, noise; steady -0.5 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8 | 2.0x hot rank | 16,384 | 19.069 | 18.223 | -4.4 % (band 2.1 %, faster; steady -3.5 %) | 69.353 | 68.761 | -0.9 % (band 1.4 %, noise; steady -0.4 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8 | 3.0x hot rank | 16,384 | 27.669 | 26.754 | -3.3 % (band 1.5 %, faster; steady -3.6 %) | 99.806 | 99.470 | -0.3 % (band 0.9 %, noise; steady -0.8 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8 | 5.6x hot rank | 16,384 | 50.213 | 48.369 | -3.7 % (band 0.6 %, faster; steady -3.9 %) | 184.414 | 182.364 | -1.1 % (band 0.6 %, faster; steady -1.0 %) |
  | B300 | 8 | H 6144 / 256 experts, top 2 | uniform | 16,384 | 3.450 | 3.350 | -2.9 % (band 5.5 %, noise; steady -4.8 %) | 13.710 | 13.568 | -1.0 % (band 1.7 %, noise; steady -1.3 %) |
  | B300 | 8 | H 6144 / 256 experts, top 2 | 2.0x hot rank | 16,384 | 5.550 | 5.401 | -2.7 % (band 6.6 %, noise; steady -3.7 %) | 20.797 | 20.550 | -1.2 % (band 1.1 %, faster; steady -1.2 %) |
  | B300 | 8 | H 6144 / 256 experts, top 2 | 3.0x hot rank | 16,384 | 7.750 | 7.428 | -4.2 % (band 4.5 %, noise; steady -3.2 %) | 28.365 | 27.930 | -1.5 % (band 0.9 %, faster; steady -1.2 %) |
  | B300 | 8 | H 4096 / 288 experts, top 8 | uniform | 16,384 | 7.039 | 6.640 | -5.7 % (band 2.7 %, faster; steady -4.9 %) | 25.690 | 25.335 | -1.4 % (band 1.1 %, faster; steady -1.3 %) |
  | B300 | 8 | H 4096 / 288 experts, top 8 | 1.7x hot rank | 16,384 | 10.827 | 10.261 | -5.2 % (band 2.2 %, faster; steady -4.0 %) | 39.063 | 38.918 | -0.4 % (band 1.7 %, noise; steady +0.0 %) |
  | B300 | 8 | H 4096 / 288 experts, top 8 | 2.0x hot rank | 16,384 | 12.570 | 12.007 | -4.5 % (band 2.0 %, faster; steady -4.4 %) | 47.177 | 46.641 | -1.1 % (band 0.8 %, faster; steady -1.3 %) |
  | B300 | 8 | H 4096 / 288 experts, top 8 | 3.0x hot rank | 16,384 | 18.405 | 17.296 | -6.0 % (band 2.1 %, faster; steady -4.1 %) | 68.063 | 67.568 | -0.7 % (band 0.6 %, noise; steady -0.8 %) |
  | B300 | 8 | H 4096 / 288 experts, top 2 | uniform | 16,384 | 2.284 | 2.172 | -4.9 % (band 5.7 %, noise; steady -5.4 %) | 9.169 | 9.125 | -0.5 % (band 3.5 %, noise; steady -0.7 %) |
  | B300 | 8 | H 4096 / 288 experts, top 2 | 2.0x hot rank | 16,384 | 3.622 | 3.590 | -0.9 % (band 11.8 %, noise; steady -3.6 %) | 14.551 | 14.237 | -2.2 % (band 1.3 %, faster; steady -1.3 %) |
  | B300 | 8 | H 4096 / 288 experts, top 2 | 3.0x hot rank | 16,384 | 5.211 | 4.936 | -5.3 % (band 4.9 %, faster; steady -3.9 %) | 19.201 | 19.010 | -1.0 % (band 1.9 %, noise; steady +0.2 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8, workspace precision mxfp8 | uniform | 16,384 | 10.311 | 9.947 | -3.5 % (band 2.8 %, faster; steady -3.6 %) | 38.206 | 37.260 | -2.5 % (band 1.7 %, faster; steady -2.1 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8, workspace precision mxfp8 | 2.0x hot rank | 16,384 | 18.966 | 18.143 | -4.3 % (band 2.8 %, faster; steady -3.4 %) | 69.278 | 68.779 | -0.7 % (band 0.7 %, noise; steady -0.5 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8, MXFP8 weights, workspace precision bf16 | uniform | 16,384 | 7.451 | 7.447 | -0.1 % (band 2.8 %, noise; steady -0.8 %) | 26.378 | 26.349 | -0.1 % (band 1.1 %, noise; steady -0.4 %) |
  | B300 | 8 | H 6144 / 256 experts, top 8, MXFP8 weights, workspace precision bf16 | 2.0x hot rank | 16,384 | 13.684 | 13.698 | +0.1 % (band 0.7 %, noise; steady +0.4 %) | 47.841 | 47.842 | +0.0 % (band 1.0 %, noise; steady -0.2 %) |

  Not measured: EP sizes other than 8 on B300; hottest-rank ratios other than
  those in the tables; BF16 weight-gradient accumulators; complete training
  steps of a model; GB300 launches whose split this version changes: BF16 routed
  weights on a workspace created with `precision="mxfp8"` at EP8, EP32 and EP64:
  44 / 36 -> 20 / 24; BF16 routed weights on a workspace created with
  `precision="mxfp8"` at EP16: 40 / 40 -> 20 / 24; MXFP8 routed weights on a
  workspace created with `precision="bf16"` at EP8, EP32 and EP64: 20 / 24 ->
  44 / 36; MXFP8 routed weights on a workspace created with `precision="bf16"`
  at EP16: 20 / 24 -> 40 / 40 (one GB300 node has four GPUs); Rubin launches
  whose split this version changes: BF16 routed weights on a workspace created
  with `precision="mxfp8"` at EP8: 60 / 60 -> 40 / 40 (with
  `imbalance_comm_sms=None`, and with the default imbalance row while the
  hottest rank's load is below its threshold; above it both versions take the
  imbalance split); MXFP8 routed weights on a workspace created with
  `precision="bf16"` at EP8: 40 / 40 -> 60 / 60 (two Rubin nodes needed).

- Mini-batches are multiples of 256; macro-batches are multiples of the
  mini-batch. Schedule capacity must hold every padded route; overflow traps.
  The capacity is `T * topk * max(2, ceil(EP * m))` rows for
  `schedule_capacity_multiplier` `m` and the workspace's physical source rows
  `T` (the largest rank's source rows, or `source_capacity`, rounded up to the
  256-row tile, at least 512); every routing fits when
  `m >= ceil((min(topk, E) * T * EP + E * 255) / (T * topk)) / EP` (`E` local
  experts), about `(EP + 1) / EP`. At the 896-expert layout `(64, 14)` and
  top-k 16 a token routes at most 14 routes to one rank: `m = 57 / 64` from
  4,096 to 32,768 rows (14,942,208 schedule rows at 16,384). The bound decreases as `T` grows, so a
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
- Source split of the router-weight-before-down BF16 forward:
  `MoKConfig.source_split_comm_sms` (default `None`: off) and
  `source_split_threshold` (default 1.5); `create_mok_bf16_workspace` (package
  and `cake_mok_det_train` facade) defaults `source_split_comm_sms` to `"auto"`:
  the measured `SOURCE_SPLIT_COMM_SMS` row (compute capability, SM count, EP
  group size), else `None`. With the split on, `build_schedule` also returns the
  owner-rank route totals (`MoKSchedule.owner_routes`, int32 `[ep_size]`, the
  same on every rank), and at every launch the context and context-free forward
  kernels of the router weight before the down projection decide on the device:
  a rank whose source rows (its 256-row shared tiles less one) exceed
  `source_split_threshold` times the ranks' mean source rows (route total / (EP
  x top-k); the threshold passes in 1/256 steps) runs `source_split_comm_sms`
  communication SMs instead of the forward count in force, and every other rank
  keeps that count. The shared expert of a source-rich rank grows with its
  source rows while its routed rows match the other ranks', so the SMs move from
  its communication program to its compute tiles. The count only selects which
  CTAs run the communication tasks: every output, context and gradient is
  bitwise that of the split off, the ranks need not agree, and CUDA Graph
  replays and layers that share a workspace decide from their own schedule. A
  count that is odd, below 2 or not below the forward count in force keeps the
  forward count. Row: B200 (148 SMs) EP8 16, GB300 (152 SMs) EP4 16, GB200 (152
  SMs) EP4 16. Measured without a row (no configuration with ragged source rows
  faster than the previous version in the record window; the split stays off
  there): B300 (148 SMs) EP8. No row elsewhere (other devices and group sizes
  were not measured).

  Table: this package against version 0.2.4 (the split absent; measured before
  0.2.5 merged), each device at its shipped setting: the split on at its row,
  off on a device without a row; one node per device, both versions loading GPU
  programs with the same SASS except the two forward kernels of the router
  weight before the down projection (compared per program before timing), BF16
  routed experts, H 5120 / I 2304, 384 experts, top 6, clamped SwiGLU, FP32
  accumulators; one process per version and configuration, this package with the
  split on and with it off and a second process of the previous version
  interleaved over six rounds (B300 ragged 4k: twelve) with the order reversed
  in every second round, eager (CUDA Graph where stated), no CPU binding; the
  slowest rank's median of timed repetitions 1-20 after 3 warm-up repetitions,
  median over the rounds; the table prints those medians and change = printed
  package ms / printed 0.2.4 ms - 1; band = the largest of the deviation of the
  identical-version pair, its round-interval half width and the round spreads of
  both versions; 'faster' = the median of the per-round ratios below 1 minus the
  band, every round's ratio below 1 and the 95 % interval of the GPU-paired
  ratio below 1; 'slower' = the same above 1; 'noise' = the median ratio inside
  the band; 'unresolved' = beyond the band without the other two conditions;
  steady = the change of the medians of timed repetitions 41-60 of 60. Forward =
  schedule and forward without context; forward with context = the forward that
  saves the context, on a schedule built before the timed repetitions;
  checkpointed step = schedule, forward, context recompute and backward. Source
  rows per rank of the ragged configurations: EP8 16k 21,419 / 14,428 / 16,181 /
  11,502 / 10,086 / 21,331 / 9,762 / 0; EP8 4k 2,987 / 4,188 / 5,941 / 5,358 /
  3,942 / 2,899 / 3,618 / 0; EP4 16k 21,419 / 14,428 / 16,181 / 0; EP4 4k 2,987
  / 4,188 / 5,941 / 0. Every eager configuration gives the same bits at 0.2.4,
  with the split on and with it off, and every line repeats its outputs bitwise.

  | device | EP | configuration | routing | tokens per rank | forward 0.2.4 (ms) | forward, this arm (ms) | change | forward with context 0.2.4 (ms) | this arm (ms) | change | checkpointed step 0.2.4 (ms) | this arm (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | B300 | 8 | before down, 384 experts, top 6, split off (shipped) | ragged | as listed (4k) | 1.931 | 2.027 | +5.0 % (band 12.1 %, noise; steady +0.5 %) | 2.022 | 2.047 | +1.2 % (band 8.1 %, noise; steady +0.9 %) | 8.627 | 8.636 | +0.1 % (band 4.6 %, noise; steady +0.5 %) |
  | B300 | 8 | before down, 384 experts, top 6, split off (shipped) | uniform | 16,384 | 7.737 | 7.763 | +0.3 % (band 1.3 %, noise; steady -0.1 %) | 8.020 | 8.014 | -0.1 % (band 1.3 %, noise; steady -0.3 %) | 28.805 | 28.785 | -0.1 % (band 1.1 %, noise; steady +0.2 %) |
  | B300 | 8 | before down, 384 experts, top 6, split off (shipped) | 1.5x hot rank | 16,384 | 10.758 | 10.680 | -0.7 % (band 1.7 %, noise; steady -0.1 %) | 11.165 | 11.187 | +0.2 % (band 1.1 %, noise; steady +0.2 %) | 38.801 | 38.904 | +0.3 % (band 1.3 %, noise; steady +0.8 %) |
  | B300 | 8 | before down, 384 experts, top 6, split off (shipped) | ragged | as listed (16k) | 6.935 | 6.921 | -0.2 % (band 9.1 %, noise; steady -1.2 %) | 7.199 | 7.110 | -1.2 % (band 3.8 %, noise; steady +1.3 %) | 25.866 | 25.848 | -0.1 % (band 2.0 %, noise; steady +0.4 %) |
  | B200 | 8 | before down, 384 experts, top 6, split on (shipped) | ragged | as listed (4k) | 2.140 | 2.120 | -0.9 % (band 1.0 %, noise; steady -1.4 %) | 2.187 | 2.175 | -0.5 % (band 6.9 %, noise; steady +0.7 %) | 9.100 | 9.117 | +0.2 % (band 2.7 %, noise; steady +0.7 %) |
  | B200 | 8 | before down, 384 experts, top 6, split on (shipped) | uniform | 16,384 | 8.166 | 8.093 | -0.9 % (band 3.5 %, noise; steady +0.6 %) | 8.552 | 8.512 | -0.5 % (band 1.6 %, noise; steady +0.8 %) | 30.139 | 30.197 | +0.2 % (band 1.5 %, noise; steady -0.1 %) |
  | B200 | 8 | before down, 384 experts, top 6, split on (shipped) | 1.5x hot rank | 16,384 | 11.140 | 11.096 | -0.4 % (band 1.7 %, noise; steady -0.2 %) | 11.570 | 11.620 | +0.4 % (band 1.9 %, noise; steady +0.3 %) | 40.369 | 40.337 | -0.1 % (band 1.7 %, noise; steady -0.7 %) |
  | B200 | 8 | before down, 384 experts, top 6, split on (shipped) | ragged | as listed (16k) | 7.391 | 6.870 | -7.0 % (band 6.3 %, faster; steady -4.9 %) | 7.722 | 7.413 | -4.0 % (band 6.4 %, noise; steady -3.9 %) | 26.950 | 26.736 | -0.8 % (band 1.2 %, noise; steady -1.3 %) |
  | B200 | 8 | before down, 384 experts, top 6, CUDA Graph (captured on these sources, replayed), split on (shipped) | ragged | as listed (16k) | 7.127 | 6.695 | -6.1 % (band 6.9 %, faster; steady -6.2 %) | - | - | - | 26.402 | 26.232 | -0.6 % (band 1.5 %, noise; steady -0.9 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split on (shipped) | uniform | 4,096 | 2.381 | 2.412 | +1.3 % (band 6.0 %, noise; steady +1.6 %) | 2.312 | 2.333 | +0.9 % (band 4.4 %, noise; steady +1.0 %) | 11.243 | 11.259 | +0.1 % (band 1.2 %, noise; steady +0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split on (shipped) | 1.5x hot rank | 4,096 | 2.752 | 2.779 | +1.0 % (band 4.5 %, noise; steady +0.8 %) | 2.753 | 2.747 | -0.2 % (band 2.2 %, noise; steady +0.7 %) | 12.336 | 12.355 | +0.2 % (band 1.1 %, noise; steady -0.0 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split on (shipped) | ragged | as listed (4k) | 2.147 | 2.157 | +0.5 % (band 4.9 %, noise; steady +0.8 %) | 2.043 | 2.056 | +0.6 % (band 4.8 %, noise; steady +0.8 %) | 10.492 | 10.516 | +0.2 % (band 2.2 %, noise; steady +0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split on (shipped) | uniform | 16,384 | 6.358 | 6.376 | +0.3 % (band 1.3 %, noise; steady +0.9 %) | 6.433 | 6.416 | -0.3 % (band 1.4 %, noise; steady -0.0 %) | 24.333 | 24.324 | -0.0 % (band 0.7 %, noise; steady +0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split on (shipped) | 1.5x hot rank | 16,384 | 8.535 | 8.511 | -0.3 % (band 2.5 %, noise; steady +0.0 %) | 9.021 | 9.057 | +0.4 % (band 2.6 %, noise; steady +0.6 %) | 32.243 | 32.282 | +0.1 % (band 1.5 %, noise; steady +0.2 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split on (shipped) | ragged | as listed (16k) | 5.906 | 5.836 | -1.2 % (band 1.3 %, faster; steady -0.7 %) | 5.937 | 5.889 | -0.8 % (band 1.3 %, noise; steady -0.8 %) | 22.511 | 22.505 | -0.0 % (band 1.0 %, noise; steady -0.2 %) |
  | GB300 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on these sources, replayed), split on (shipped) | ragged | as listed (16k) | 5.665 | 5.711 | +0.8 % (band 3.4 %, noise; steady -0.2 %) | - | - | - | 22.251 | 22.262 | +0.0 % (band 1.0 %, noise; steady -0.4 %) |
  | GB300 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on a uniform step, replayed), split on (shipped) | uniform capture, 2.0x hot rank replay | 16,384 | 10.940 | 10.929 | -0.1 % (band 0.6 %, noise; steady -0.5 %) | - | - | - | 40.690 | 40.643 | -0.1 % (band 1.1 %, noise; steady +0.0 %) |
  | GB300 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on a uniform step, replayed), split on (shipped) | uniform capture, 3.0x hot rank replay | 16,384 | 15.806 | 15.782 | -0.2 % (band 1.1 %, noise; steady -0.1 %) | - | - | - | 58.831 | 58.925 | +0.2 % (band 1.5 %, noise; steady -0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on a uniform step, replayed), split on (shipped) | uniform capture, uniform replay | 16,384 | 6.244 | 6.194 | -0.8 % (band 2.8 %, noise; steady -0.2 %) | - | - | - | 23.854 | 23.870 | +0.1 % (band 0.8 %, noise; steady +0.2 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split on (shipped) | uniform | 4,096 | 2.643 | 2.623 | -0.8 % (band 2.2 %, noise; steady -0.1 %) | 2.792 | 2.832 | +1.4 % (band 3.1 %, noise; steady -1.1 %) | 12.086 | 12.099 | +0.1 % (band 1.1 %, noise; steady -0.1 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split on (shipped) | 1.5x hot rank | 4,096 | 3.100 | 3.104 | +0.1 % (band 10.0 %, noise; steady -0.1 %) | 3.369 | 3.387 | +0.5 % (band 2.3 %, noise; steady +0.4 %) | 13.893 | 13.891 | -0.0 % (band 1.5 %, noise; steady +0.8 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split on (shipped) | ragged | as listed (4k) | 2.219 | 2.242 | +1.0 % (band 1.3 %, noise; steady -0.3 %) | 2.264 | 2.237 | -1.2 % (band 2.4 %, noise; steady -1.4 %) | 10.839 | 10.810 | -0.3 % (band 1.0 %, noise; steady -0.2 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split on (shipped) | uniform | 16,384 | 8.155 | 8.126 | -0.4 % (band 3.0 %, noise; steady +1.1 %) | 8.768 | 8.809 | +0.5 % (band 2.4 %, noise; steady -0.2 %) | 30.484 | 30.574 | +0.3 % (band 0.8 %, noise; steady +0.1 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split on (shipped) | 1.5x hot rank | 16,384 | 10.685 | 10.750 | +0.6 % (band 2.3 %, noise; steady -1.4 %) | 11.211 | 11.260 | +0.4 % (band 1.1 %, noise; steady +0.9 %) | 39.717 | 39.684 | -0.1 % (band 1.8 %, noise; steady +0.1 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split on (shipped) | ragged | as listed (16k) | 7.521 | 7.245 | -3.7 % (band 3.3 %, faster; steady -2.2 %) | 8.118 | 7.812 | -3.8 % (band 3.5 %, faster; steady -2.5 %) | 27.740 | 27.469 | -1.0 % (band 1.7 %, noise; steady -0.8 %) |
  | GB200 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on these sources, replayed), split on (shipped) | ragged | as listed (16k) | 7.236 | 7.075 | -2.2 % (band 4.1 %, noise; steady -3.7 %) | - | - | - | 26.927 | 26.884 | -0.2 % (band 1.1 %, noise; steady -0.7 %) |
  | GB200 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on a uniform step, replayed), split on (shipped) | uniform capture, 2.0x hot rank replay | 16,384 | 14.151 | 14.162 | +0.1 % (band 2.1 %, noise; steady -0.0 %) | - | - | - | 50.332 | 50.409 | +0.2 % (band 0.8 %, noise; steady -0.1 %) |
  | GB200 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on a uniform step, replayed), split on (shipped) | uniform capture, 3.0x hot rank replay | 16,384 | 19.778 | 19.759 | -0.1 % (band 1.3 %, noise; steady -0.2 %) | - | - | - | 72.849 | 73.127 | +0.4 % (band 1.9 %, noise; steady +0.5 %) |
  | GB200 | 4 | before down, 384 experts, top 6, CUDA Graph (captured on a uniform step, replayed), split on (shipped) | uniform capture, uniform replay | 16,384 | 8.048 | 7.980 | -0.8 % (band 2.3 %, noise; steady +0.4 %) | - | - | - | 29.625 | 29.685 | +0.2 % (band 1.7 %, noise; steady +0.1 %) |

  Table: the same devices at the other setting (not shipped), same method:

  | device | EP | configuration | routing | tokens per rank | forward 0.2.4 (ms) | forward, this arm (ms) | change | forward with context 0.2.4 (ms) | this arm (ms) | change | checkpointed step 0.2.4 (ms) | this arm (ms) | change |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | B300 | 8 | before down, 384 experts, top 6, split on (not shipped) | ragged | as listed (4k) | 1.931 | 2.085 | +8.0 % (band 11.8 %, noise; steady +0.4 %) | 2.022 | 2.022 | +0.0 % (band 7.4 %, noise; steady -0.6 %) | 8.627 | 8.580 | -0.5 % (band 4.1 %, noise; steady -0.2 %) |
  | B300 | 8 | before down, 384 experts, top 6, split on (not shipped) | uniform | 16,384 | 7.737 | 7.707 | -0.4 % (band 1.3 %, noise; steady -1.1 %) | 8.020 | 8.066 | +0.6 % (band 1.8 %, noise; steady +0.3 %) | 28.805 | 28.820 | +0.1 % (band 1.1 %, noise; steady +0.4 %) |
  | B300 | 8 | before down, 384 experts, top 6, split on (not shipped) | 1.5x hot rank | 16,384 | 10.758 | 10.641 | -1.1 % (band 2.1 %, noise; steady +0.0 %) | 11.165 | 11.192 | +0.2 % (band 1.6 %, noise; steady -0.5 %) | 38.801 | 38.979 | +0.5 % (band 1.1 %, noise; steady +0.2 %) |
  | B300 | 8 | before down, 384 experts, top 6, split on (not shipped) | ragged | as listed (16k) | 6.935 | 6.606 | -4.7 % (band 4.4 %, noise; steady -4.9 %) | 7.199 | 6.978 | -3.1 % (band 3.1 %, noise; steady -4.8 %) | 25.866 | 25.535 | -1.3 % (band 1.2 %, noise; steady -0.9 %) |
  | B300 | 8 | before down, 384 experts, top 6, CUDA Graph (captured on these sources, replayed), split on (not shipped) | ragged | as listed (16k) | 6.779 | 6.497 | -4.2 % (band 6.0 %, noise; steady -5.5 %) | - | - | - | 25.364 | 25.205 | -0.6 % (band 2.5 %, noise; steady -1.2 %) |
  | B200 | 8 | before down, 384 experts, top 6, split off (not shipped) | ragged | as listed (4k) | 2.140 | 2.142 | +0.1 % (band 6.8 %, noise; steady +0.7 %) | 2.187 | 2.227 | +1.8 % (band 7.8 %, noise; steady +1.9 %) | 9.100 | 9.152 | +0.6 % (band 2.2 %, noise; steady +0.1 %) |
  | B200 | 8 | before down, 384 experts, top 6, split off (not shipped) | uniform | 16,384 | 8.166 | 8.102 | -0.8 % (band 3.5 %, noise; steady -0.3 %) | 8.552 | 8.570 | +0.2 % (band 3.7 %, noise; steady +0.5 %) | 30.139 | 30.292 | +0.5 % (band 1.0 %, noise; steady -0.2 %) |
  | B200 | 8 | before down, 384 experts, top 6, split off (not shipped) | 1.5x hot rank | 16,384 | 11.140 | 11.091 | -0.4 % (band 1.4 %, noise; steady -0.3 %) | 11.570 | 11.624 | +0.5 % (band 1.9 %, noise; steady +0.3 %) | 40.369 | 40.584 | +0.5 % (band 1.7 %, noise; steady -0.8 %) |
  | B200 | 8 | before down, 384 experts, top 6, split off (not shipped) | ragged | as listed (16k) | 7.391 | 7.361 | -0.4 % (band 3.6 %, noise; steady -0.1 %) | 7.722 | 7.724 | +0.0 % (band 3.4 %, noise; steady -0.5 %) | 26.950 | 27.029 | +0.3 % (band 1.5 %, noise; steady +0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split off (not shipped) | uniform | 4,096 | 2.381 | 2.365 | -0.7 % (band 4.3 %, noise; steady -0.6 %) | 2.312 | 2.303 | -0.4 % (band 4.5 %, noise; steady +0.0 %) | 11.243 | 11.212 | -0.3 % (band 2.1 %, noise; steady -0.3 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split off (not shipped) | 1.5x hot rank | 4,096 | 2.752 | 2.778 | +0.9 % (band 4.5 %, noise; steady +0.9 %) | 2.753 | 2.750 | -0.1 % (band 2.1 %, noise; steady +0.9 %) | 12.336 | 12.369 | +0.3 % (band 1.1 %, noise; steady +0.2 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split off (not shipped) | ragged | as listed (4k) | 2.147 | 2.125 | -1.0 % (band 4.9 %, noise; steady -0.6 %) | 2.043 | 2.032 | -0.5 % (band 4.8 %, noise; steady -0.3 %) | 10.492 | 10.467 | -0.2 % (band 2.2 %, noise; steady -0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split off (not shipped) | uniform | 16,384 | 6.358 | 6.380 | +0.3 % (band 2.6 %, noise; steady +0.4 %) | 6.433 | 6.446 | +0.2 % (band 1.3 %, noise; steady -0.3 %) | 24.333 | 24.289 | -0.2 % (band 1.3 %, noise; steady +0.1 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split off (not shipped) | 1.5x hot rank | 16,384 | 8.535 | 8.543 | +0.1 % (band 1.2 %, noise; steady +0.3 %) | 9.021 | 9.070 | +0.5 % (band 2.6 %, noise; steady +0.5 %) | 32.243 | 32.311 | +0.2 % (band 0.7 %, noise; steady -0.2 %) |
  | GB300 | 4 | before down, 384 experts, top 6, split off (not shipped) | ragged | as listed (16k) | 5.906 | 5.926 | +0.3 % (band 2.8 %, noise; steady +0.6 %) | 5.937 | 5.958 | +0.4 % (band 1.0 %, noise; steady +0.4 %) | 22.511 | 22.604 | +0.4 % (band 1.1 %, noise; steady +0.1 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split off (not shipped) | uniform | 4,096 | 2.643 | 2.654 | +0.4 % (band 2.8 %, noise; steady +0.5 %) | 2.792 | 2.815 | +0.8 % (band 3.1 %, noise; steady -1.0 %) | 12.086 | 12.091 | +0.0 % (band 1.3 %, noise; steady -0.3 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split off (not shipped) | 1.5x hot rank | 4,096 | 3.100 | 3.085 | -0.5 % (band 6.4 %, noise; steady -0.4 %) | 3.369 | 3.378 | +0.3 % (band 1.8 %, noise; steady +0.9 %) | 13.893 | 13.880 | -0.1 % (band 1.5 %, noise; steady +0.9 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split off (not shipped) | ragged | as listed (4k) | 2.219 | 2.232 | +0.6 % (band 2.1 %, noise; steady -0.1 %) | 2.264 | 2.290 | +1.1 % (band 4.5 %, noise; steady -0.2 %) | 10.839 | 10.858 | +0.2 % (band 1.0 %, noise; steady +0.2 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split off (not shipped) | uniform | 16,384 | 8.155 | 8.110 | -0.6 % (band 2.6 %, noise; steady +0.2 %) | 8.768 | 8.810 | +0.5 % (band 1.7 %, noise; steady +0.2 %) | 30.484 | 30.620 | +0.4 % (band 1.2 %, noise; steady +0.1 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split off (not shipped) | 1.5x hot rank | 16,384 | 10.685 | 10.736 | +0.5 % (band 1.5 %, noise; steady -0.1 %) | 11.211 | 11.220 | +0.1 % (band 1.2 %, noise; steady -0.0 %) | 39.717 | 39.697 | -0.1 % (band 1.9 %, noise; steady +0.3 %) |
  | GB200 | 4 | before down, 384 experts, top 6, split off (not shipped) | ragged | as listed (16k) | 7.521 | 7.459 | -0.8 % (band 3.3 %, noise; steady +1.0 %) | 8.118 | 8.045 | -0.9 % (band 3.5 %, noise; steady +0.7 %) | 27.740 | 27.807 | +0.2 % (band 1.7 %, noise; steady -0.2 %) |

  After 0.2.5 (its fixed B300 BF16 EP8 split, forward / backward 24 / 40): this
  package (the split off there, no row) against 0.2.5 on B300, same method and
  configurations: record window 0 faster, 0 slower, 12 within noise of 12;
  steady window 0 faster, 0 slower, 12 within noise of 12. 2 of these rows are
  above 0.2.5 in every round and their GPU-paired 95 % interval lies entirely
  above 1, so each is a small slowdown that is consistent across rounds and
  GPUs, inside the band (verdict noise): forward with context, ragged 16k,
  record window +0.57 % (band 3.15 %; GPU-paired 95 % interval 1.0025..1.0082);
  forward, uniform 16,384, record window +1.54 % (band 1.80 %; GPU-paired 95 %
  interval 1.0011..1.0205). With the split forced on (not shipped): record
  window 0 faster, 0 slower, 14 within noise of 14; steady window 0 faster, 0
  slower, 14 within noise of 14.

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
  arguments against `sources.json`, the refusals before device work and the
  targets (no GPU), and on SM100a, SM103a and SM107a the fused training at
  top-2 and top-6 (6 to 96 local experts; another
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
- The 896-expert top-16 group (SiTU-GLU, `shared_expert="none"`): in
  `test_mok_bf16.py` the registry records and the launchers' arguments, the
  refused configurations and calls before device work, the capacity bound at
  `(64, 14)` (no GPU), and on SM100a and SM103a the top-16 epilogues without a
  shared row (the exact backward sum, the top-8 bytes on padded slots, CUDA
  Graph replay), the scheduler at the bound under the two-rank skew (14 routes
  of every token on one rank, 2 on the next, distinct experts), and the
  adapter's step at `(64, 14)` emulated on one GPU with every route on rank 0
  (two routes of each token repeating an expert): the context and checkpointed
  steps bit for bit, run to run and under CUDA Graph replay, the SiTU-GLU
  scalars as run-time values, no host synchronization, changing source counts
  and `torch.utils.checkpoint` without reentrancy; in `test_mok_accuracy.py`
  the forward epilogue and that step on each SiTU-GLU data kind (normal, wide,
  saturated `|gate| <= 190` and `|up| <= 950`, deep negative gates, small,
  exact-zero gates, exact-zero up values) judged against the naive tier
  (`y`, `dx`, the router-weight gradients, also at zero weights, and the
  routed weight gradients). The deep-negative kind keeps the gates near the
  FP32 normal range of `sigmoid(gate)` (about -88.5 to -80.5): accuracy in the
  FP32 subnormal range of `sigmoid` is not required. Multi-GPU runs at
  `(64, 14)` are a separate test.

The CUDA schedules derive from Cursor Research's Apache-2.0 Mixture of Kittens
implementation at revision `caeb2963f855c7ad53bb50c8bbf211086405cb98`. Copyright and modification notices are
retained in the generated source files.
