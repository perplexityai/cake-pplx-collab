# Deterministic SiTU-GLU training backend (SM100 / SM103)

Backend of `cake_pplx.cake_situ_glu_det_train`: the SiTU-GLU activation of an MLP in training, forward and backward,
at any width `I` that is a positive multiple of 256 (the widths 3072, 6144 and 33,792 among them).  The math is FP32
on the BF16 inputs, with one BF16 rounding of each output.  The activation is unweighted.

The facade `cake_pplx.cake_situ_glu_det_train` is a stable API.  This backend package is experimental, and its
internal names may change.  The facade imports the backend on first use.  The supported architectures are SM100
(B200 / GB200) and SM103 (B300 / GB300).  Another device raises `RuntimeError`.  The package loads its libraries
prebuilt from `aot/` when a builder compiled them.  Otherwise the package JIT-builds them from the sources and flags
that `build.json` records.  The package reads no environment variable.

## Usage

```python
import torch
import torch.utils.checkpoint
from cake_pplx.cake_situ_glu_det_train import cake_situ_glu_det

# gate_up: BF16 [R, 2I] (gate columns [0, I), up [I, 2I)), the output of the gate/up projection
h = cake_situ_glu_det(gate_up)  # BF16 [R, I], the down projection's input; beta = 4.0, lb = 25.0
h.backward(dh)  # gate_up.grad: BF16 [R, 2I]

# two tensors, and other scalars
h = cake_situ_glu_det(gate, up, beta=3.0, lb=40.0)

# under activation checkpointing: only the inputs are kept
h = torch.utils.checkpoint.checkpoint(cake_situ_glu_det, gate_up, use_reentrant=False)
```

The steps without autograd, `cake_situ_glu_det_train_forward` and `cake_situ_glu_det_train_backward`, take the same
operands and optional preallocated outputs (`out=`, `grad_gate_up_or_gate=`, `grad_up=`), for a caller that fuses the
activation into its own `torch.autograd.Function`.  `beta` and `lb` are keywords everywhere, with the defaults 4.0
and 25.0.

## Inputs and outputs

| operand | shape | dtype | notes |
|---|---|---|---|
| `gate`, `up` | `[R, I]` each | BF16 | or one `[R, 2I]` tensor (first argument, `up=None`); any row stride |
| `beta`, `lb` | Python numbers | | finite and positive, rounded to FP32; launch arguments |
| `h` | `[R, I]` | BF16 | new and contiguous, or the caller's `out` |
| `dh` | `[R, I]` | BF16 | upstream gradient |
| `d_gate`, `d_up` | `[R, I]` each | BF16 | or one `[R, 2I]` gradient for a merged input |

- Every operand is on one CUDA device.  `R` is any count from 0 to `2**30`.  `I` is a positive multiple of 256.
- The kernels read every BF16 tensor through its row stride in elements: a row stride that is a multiple of 8, a unit
  last stride and a 16-byte aligned base are read in place (the halves of a merged tensor, row slices of a wider
  tensor).  Another input layout is copied once.
- A caller's output buffer must already have such a layout, with rows that do not overlap (row stride at least the
  row width); otherwise it is refused (`ValueError`).  An output (`out`, `grad_gate_up_or_gate`, `grad_up`) must not
  overlap any input or another output; this is not checked.
- A wrong type, or a `beta` or `lb` that is not a Python number, raises `TypeError` (a tensor would need a host read).
  A wrong device, dtype or shape, a width that is not a positive multiple of 256, a non-finite or non-positive `beta`
  or `lb`, or an unfit output raises `ValueError`.  An unsupported device raises `RuntimeError`, also with `R = 0`.
  Every argument is checked before an input is copied or an output allocated.
- `R = 0` returns empty outputs and launches nothing.

## Programs

Two programs, one for `fwd` and one for `bwd` (`cake_jit.MODULES`, keyed by program name; `cake_jit.PROGRAM_OF` maps a
stage to its program), each delivered for `sm_100a` and `sm_103a` and compiled with the exact `-gencode` flag of each
(`build.json`).  Each is one kernel.  A row is `I / 256` segments of 256 columns, and a warp owns a segment: lane `l`
loads 8 BF16 values at column `256 s + 8 l`, so each load of a warp is contiguous.  A CTA of 128 threads (four warps)
owns a row, and its warps take the row's segments in turn: three segments per warp and pass in the forward, two in
the backward.  Each warp issues the loads of its segments before it computes and stores them; a segment past the
row's last one reloads the last segment and stores nothing.  The grid is one CTA per row (`cake_jit.grid`), and the
CTAs stride over the rows, so any smaller grid would also be correct.  Row offsets are 64-bit.  No program uses shared
memory, a workspace, a cooperative launch or programmatic dependent launch.  The row count, the width, `beta`, `lb`
and the strides are launch arguments, so one library per program and architecture serves every call.

## Numerics

- The BF16 values are widened to FP32 (exact).  With `E = 2^(-2 |g| log2(e) / beta)`, `P = 2^(-|g| log2(e))` and
  `W = 2^(-2 |u| log2(e) / lb)`, all in `[0, 1]` so that no exponential overflows: `tanh(|g| / beta) = (1 - E) /
  (1 + E)`, `sigmoid(g) = 1 / (1 + P)` for `g >= 0` and `P / (1 + P)` for `g < 0`, and `1 - tanh^2 = 4 E / (1 + E)^2`
  (no cancellation); the up takes the same forms with `W`.
- Below `|g| / beta = 1/16` (`|u| / lb` for the up) the tanh is the odd polynomial `z - z^3 / 3 + 2 z^5 / 15`, whose
  next term is under `4e-9` of it.  So `g = 0` or `u = 0` gives `h = 0` exactly.
- `beta = 4` takes `P = E E`; any other `beta` takes `P` from its own exponential.  The choice is the same for every
  element of a call.
- The exponentials are the approximate base-2 exponential.  The forward flushes a subnormal `E` or `W`, which only
  enters `1 + E` and `1 + W` there, and keeps a subnormal `P`.  The backward keeps every subnormal, so a saturated
  gate or up keeps a nonzero gradient where FP32 holds one.
- One reciprocal per element, of `(1 + E)(1 + P)(1 + W)` in `[1, 8]`; the backward multiplies it back out into the
  three factors' reciprocals.  A negative gate's sigmoid numerator `P` is the last factor of each product, so a
  subnormal result rounds once.
- `h`, `d_gate` and `d_up` are each rounded once to BF16.  Every product and FMA rounds to nearest without flushing
  subnormals; pairs of elements use the packed FP32 instructions, which round each half as the scalar instruction
  does.  Nothing is FP64.

## Determinism

Every output is elementwise, so every call gives the same bits for the same inputs on the same architecture, whatever
the row count, the width's place in a wider tensor, the layout (merged, two tensors, row slices), the stream or a CUDA
graph replay.  No call alerts under `torch.use_deterministic_algorithms(True)`, and no entry point takes a
`deterministic` keyword.  The family has no non-deterministic module: there is no reduction across columns, rows or
CTAs to reorder.  It has no KCP module: rows are independent, with no sequence dimension.

## Workspace and retained state

- Workspace: none.
- Retained state of the autograd function: its inputs only (`gate` and `up` or the merged tensor), references with no
  copy: `R * 2I * 2` bytes of tensors the caller already holds.  No activation is saved; the backward recomputes it.

## Limitations

- `sm_107a` (Rubin) is not delivered yet.
- No double backward (`create_graph`); not tested under `torch.compile` or `torch.jit.trace`.  No FP8 or FP4 output.
- The library builds on first use; before capturing a CUDA graph, run one eager call (or build the libraries with
  `specs_for_arch`).
