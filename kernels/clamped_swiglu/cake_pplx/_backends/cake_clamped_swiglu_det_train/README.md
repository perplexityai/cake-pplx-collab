# Deterministic routed-expert clamped SwiGLU training backend (SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_clamped_swiglu_det_train`: the clamped SwiGLU of a routed MoE expert in training, forward
and backward, intermediate width `I = 2304`.  The route weight multiplies the FP32 activation before the one BF16
rounding of `h`, and the route-weight gradient uses the unweighted activation, so a zero weight keeps its gradient.

The facade `cake_pplx.cake_clamped_swiglu_det_train` is a stable API.  This backend package is experimental, and its
internal names may change.  The facade imports the backend on first use.  The supported architectures are SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200).  Another device raises `RuntimeError`.  The package loads its
libraries prebuilt from `aot/` when a builder compiled them.  Otherwise the package JIT-builds them from the sources
and flags that `build.json` records.  The package reads no environment variable.

## Usage

```python
import torch
import torch.utils.checkpoint
from cake_pplx.cake_clamped_swiglu_det_train import cake_clamped_swiglu_det

# gate_up: BF16 [R, 4608] (gate columns [0, 2304), up [2304, 4608)), the output of the gate/up projection
# w: FP32 [R] route weights (>= 0; zeros allowed)
h = cake_clamped_swiglu_det(gate_up, weight=w, limit=10.0)  # BF16 [R, 2304], the down projection's input
h.backward(dh)  # gate_up.grad: BF16 [R, 4608]; w.grad: FP32 [R]

# two tensors, and the shared expert (no route weight)
h = cake_clamped_swiglu_det(gate, up, w, limit=10.0)
h_shared = cake_clamped_swiglu_det(gate_up_shared, limit=10.0)

# under activation checkpointing: only the inputs are kept
h = torch.utils.checkpoint.checkpoint(cake_clamped_swiglu_det, gate_up, None, w, limit=10.0, use_reentrant=False)
```

The steps without autograd, `cake_clamped_swiglu_det_train_forward` and `cake_clamped_swiglu_det_train_backward`,
take the same operands and optional preallocated outputs (`out=`, `grad_gate_up_or_gate=`, `grad_up=`,
`grad_weight=`), for a caller that fuses the activation into its own `torch.autograd.Function`.  `limit` is a
required keyword everywhere.

## Inputs and outputs

| operand | shape | dtype | notes |
|---|---|---|---|
| `gate`, `up` | `[R, 2304]` each | BF16 | or one `[R, 4608]` tensor (first argument, `up=None`); any row stride |
| `weight` | `[R]` | FP32 | route weight, `>= 0`; `None` for an unweighted expert (`w = 1`) |
| `limit` | Python number | | `L`, finite and positive, rounded to FP32; a launch argument |
| `h` | `[R, 2304]` | BF16 | new and contiguous, or the caller's `out` |
| `dh` | `[R, 2304]` | BF16 | upstream gradient |
| `d_gate`, `d_up` | `[R, 2304]` each | BF16 | or one `[R, 4608]` gradient for a merged input |
| `d_weight` | `[R]` | FP32 | `None` without `weight` |

- Every operand is on one CUDA device.  `R` is any count from 0 up to `2**31 - 257`.
- The kernels read every BF16 tensor through its row stride in elements: a row stride that is a multiple of 8, a unit
  last stride and a 16-byte aligned base are read in place (the halves of a merged tensor, row slices of a wider
  tensor).  Another input layout is copied once; a non-contiguous `weight` too.
- A caller's output buffer must already have such a layout, with rows that do not overlap (row stride at least the
  row width); otherwise it is refused (`ValueError`).  An output (`out`, `grad_gate_up_or_gate`, `grad_up`,
  `grad_weight`) must not overlap any input or another output; this is not checked.
- A wrong type, or a `limit` that is not a Python number, raises `TypeError` (a tensor `limit` would need a host
  read).  A wrong device, dtype or shape, a non-finite or non-positive `limit`, or an unfit output raises
  `ValueError`.  An unsupported device raises `RuntimeError`, also with `R = 0`.
- `R = 0` returns empty outputs and launches nothing.

## Programs

Two programs, one for `fwd` and one for `bwd` (`cake_jit.MODULES`, keyed by program name; `cake_jit.PROGRAM_OF` maps
a stage to its program), each delivered for `sm_100a`, `sm_103a` and `sm_107a` and compiled with the exact `-gencode`
flag of each (`build.json`).  Each is one kernel: 32 lanes own a row, each lane 9 loads of 8 BF16 values of the gate
and as many of the up (and of `dh` in the backward), at columns `(p * 32 + lane) * 8`, so every load of a warp is
contiguous.  The forward's CTA of 256 threads owns 8 consecutive rows, and the backward's CTA of 128 threads owns 4;
the grid is `ceil(R / rows a CTA)` CTAs (`cake_jit.grid`, from the launch block and `cake_jit.BUILD_OF`).  A row past
`R` loads the last row and stores nothing.  The backward splits a lane's 9 loads into 3 runs of 3, so it holds fewer
registers and more warps fit on an SM.  No program uses shared
memory, a workspace, a cooperative launch or programmatic dependent launch.  The row count, the limit, the weight flag
and the strides are launch arguments, so one library per program and architecture serves every call.

## Numerics

- `g' = min(g, L)` and `u' = clamp(u, -L, L)` from the BF16 values widened to FP32 (exact); the masks compare the raw
  inputs with the FP32 `L`, inclusively.
- `s = 1 / (1 + 2^(-g' log2 e))`: the approximate base-2 exponential (`ex2.approx.ftz`; `1 + 2^t` absorbs a flushed
  subnormal `2^t`) and the approximate reciprocal plus one round-to-nearest Newton step, which is the IEEE `rcp.rn` of
  every `1 <= d < 2^126`.  A warp with a clamped gate at or below -87 in a run of its loads computes that run with a
  branch-free exact reciprocal: `RN(1 / d)` for every `d`, subnormal results kept, +0 for an infinite `d`.
- `silu = g' s`, `a = silu u'`, `h = bf16(w a)`: one rounding.
- `d_gate = bf16(((dh w) u') (silu (1 - s) + s) [g <= L])` (one FMA for `silu (1 - s) + s`) and
  `d_up = bf16((dh w) silu [-L <= u <= L])`, each mask a factor of 1 or 0, each output rounded once.
- `d_weight = sum over c of dh a` in FP32 from the unweighted `a`: each lane keeps two FMA chains (its even and its
  odd columns) in load order and adds them, then XOR shuffles sum the row's lanes.
- Every product and FMA rounds to nearest without flushing subnormals.  Nothing is FP64.  A NaN input gives a NaN where
  the float64 operation has one (the clamps propagate NaN).

## Determinism

Every output is elementwise or a sum within one row in one fixed order, so every call gives the same bits for the same
inputs on the same architecture, whatever the row count, the layout (merged, two tensors, strided), the stream or a CUDA
graph replay.  No call alerts under `torch.use_deterministic_algorithms(True)`, and no entry point takes a
`deterministic` keyword.  The family has no non-deterministic module: there is no reduction across rows or CTAs to
reorder.

## Workspace and retained state

- Workspace: none.  An unweighted backward allocates the `R` FP32 weight-gradient values the kernel writes and drops
  them.  An unweighted launch passes the device's one-element FP32 buffer, set to 1, as the weight.  The kernels read
  its one element and ignore it.  The buffer is kept for the process; the device's first unweighted launch fills it
  once, one more launch on the same stream.
- Retained state of the autograd function: its inputs only (`gate` and `up` or the merged tensor, and `weight`),
  references with no copy: `R * 4608 * 2 + R * 4` bytes of tensors the caller already holds.  No activation is saved;
  the backward recomputes it.

## Limitations

- The intermediate width is 2304 only (a compile-time width of the kernels).
- No double backward (`create_graph`); not tested under `torch.compile` or `torch.jit.trace`.  No MXFP8 or FP4 output.
- The library builds on first use; before capturing a CUDA graph, run one eager call (or build the libraries with
  `specs_for_arch`).
