# Deterministic activation quantize-dequantize backend (SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_act_qdq_det_train`: the forward of a block-scaled quantize-dequantize (QDQ) of BF16
activations back to BF16.  Each group of `G` consecutive elements of the last dimension is quantized to a
block-scaled format with one scale, then dequantized.  The result has the shape and dtype of the input.  The FP32
scale of each group is an optional second output.

The facade `cake_pplx.cake_act_qdq_det_train` is a stable API.  This backend package is experimental, and its
internal names may change.  The facade imports the backend on first use.  The supported architectures are SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200); another device raises `RuntimeError`.  The package loads
its libraries prebuilt from `aot/` when a builder compiled them; otherwise it JIT-builds them from the sources and
flags that `build.json` records.  The package reads no environment variable.

## Usage

```python
import torch
from cake_pplx.cake_act_qdq_det_train import (
    cake_act_qdq_det_train_forward,
    is_cake_act_qdq_det_train_supported,
)

x = torch.randn(16384, 32, 128, device="cuda", dtype=torch.bfloat16)
if is_cake_act_qdq_det_train_supported(x.device):
    y = cake_act_qdq_det_train_forward(x, format="mxfp4_e8m0_g32")            # new BF16 tensor
    y, s = cake_act_qdq_det_train_forward(x, format="mxfp4_e8m0_g32", return_scales=True)
    cake_act_qdq_det_train_forward(x, format="mxfp4_e8m0_g32", out=x)         # in place: returns x
```

`cake_act_qdq_det_train_forward_workspace_bytes(shape, format=...)` returns 0: a call needs no workspace and keeps
no state.  `is_cake_act_qdq_det_train_supported(device)` tells whether the device has kernels.  The facade also
exposes `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`, `FORMATS` and `GROUP_SIZES`.

## Inputs and outputs

| Argument | Rule |
|---|---|
| `x` | BF16 CUDA tensor `[..., D]`, at least one dimension, contiguous, 16-byte aligned, `D` a multiple of `G`, at most `MAX_ELEMENTS` (`8 * (2^31 - 256)`) elements |
| `format` | `"fp4_e4m3_g16"` (`G = 16`), `"mxfp4_e8m0_g32"` (`G = 32`) or `"mxfp8_e8m0_g32"` (`G = 32`) |
| `out` | `None` (a new tensor), `x` itself (in place), or a contiguous, 16-byte aligned BF16 tensor of `x`'s shape on `x`'s device that does not overlap `x` |
| `scales` | `None` or a contiguous FP32 tensor `[..., D / G]` on `x`'s device that does not overlap `x` or `out` |
| `return_scales` | `False` (default) or `True` |

The call returns `y`, or `(y, scales)` when `return_scales` is true or a `scales` buffer is given.  `y` is `out`
when `out` is given, so an in-place call returns `x`.  A new `scales` tensor is allocated when the scales are
requested without a buffer.  Outputs carry no autograd history.  A caller buffer the kernel writes (`out`, `x` in
place, `scales`) must not require grad while grad mode is on, and the call advances its version counter as a
PyTorch in-place op does.

A wrong dtype raises `TypeError`.  An unsupported device raises `RuntimeError`.  Any other violation raises
`ValueError` before any launch.  A tensor with no elements (for example 0 rows) launches nothing and returns empty
outputs of the right shapes.

## Programs

The package has six programs: one per format, each with and without the scale output.  Each is built for `sm_100a`,
`sm_103a` and `sm_107a`, with the exact `-gencode` flag of the architecture, so `build.json` lists 18 libraries.  A
call runs the one program of its route (`cake_jit.selected_module(arch, format, scale_out=...)`).

A program sees the flat input as `n_vecs = numel / 8` vectors of 8 BF16 values (16 bytes).  A thread loads one
vector, and the `G / 8` adjacent threads of one group combine their maxima with warp shuffles.  A CTA has 256
threads and takes 256 vectors per trip of a grid-stride loop that is uniform across the CTA, so every shuffle has its
whole warp and any grid of at least one CTA is correct.  The host launches `min(ceil(n_vecs / 256), 8 * SM count)`
CTAs; the SM count is read once per device.  Neither the last dimension, nor the rank, nor the row count enters the
kernel, so a new shape never rebuilds a library.  Build the libraries of the architecture before a CUDA graph
capture (one eager call does it).

## Numerics

For each group of `G` elements, with `amax` the largest magnitude in the group:

| format | element | `G` | scale `s` | `amax` floor | `Q` |
|---|---|---|---|---|---|
| `fp4_e4m3_g16` | E2M1 | 16 | `e4m3_rne_satfinite(max(amax, 6 * 2^-9) / 6)` | `6 * 2^-9` | 6 |
| `mxfp4_e8m0_g32` | E2M1 | 32 | `2^ceil(log2(max(amax, 6 * 2^-126) / 6))` | `6 * 2^-126` | 6 |
| `mxfp8_e8m0_g32` | E4M3 | 32 | `2^ceil(log2(max(amax, fp32(1e-4)) / 448))` | `fp32(1e-4)` | 448 |

Then `u = clamp(x / s, -Q, Q)` with the exact quotient, `q` is the element format's round-to-nearest-even value of
`u` with the sign of zero kept, and `y = bf16_rne(q * s)`.  The E4M3 scale saturates at 448.  The outputs equal this
rule evaluated in FP64, bit for bit, including the sign of zero.  The kernels use no FP64; every step below is exact
in FP32:

- **E8M0 scale.**  Write `amax = m * 2^e` with `m` in [1, 2), and `Q = mq * 2^c` (`6 = 1.5 * 2^2`,
  `448 = 1.75 * 2^8`).  Then `ceil(log2(amax / Q)) = e - c + [m > mq]`, read from the IEEE fields of `amax`.
  `ceil(log2(.))` is monotone, so the floor becomes a lower bound on the scale exponent: `2^-126` for MXFP4 and
  `2^-22` for MXFP8.
- **E8M0 quotient.**  `s` is a power of two, so `x * 2^-k` and the IEEE quotient `x / s` are the same correctly
  rounded number.  The kernels keep subnormals (no flush to zero), so the product equals the quotient bit for bit.
- **E4M3 scale.**  `max(amax, 6 * 2^-9) / 6` is an IEEE round-to-nearest division, and the conversion to E4M3
  rounds to nearest even and saturates at 448.  `amax` has 8 significant bits, so `amax / 3` is either exact or a
  repeating binary fraction far from every 5-bit value: the FP32 quotient is an E4M3 rounding midpoint only when the
  exact quotient is one.
- **E4M3-scaled quotient.**  Each `x / s` is an IEEE round-to-nearest division (not a multiply by a reciprocal).  The
  exact quotient of an 8-bit `x` by a 4-bit `s` is at least `2^-9` (relative) away from every E2M1 midpoint that it
  does not equal, so rounding it to FP32 first never moves it across a midpoint.
- **Clamp.**  The saturating element conversion is the rule's clamp: a `|u|` above `Q` converts to `Q`.
- **Dequantize.**  `q` has at most 2 (E2M1) or 4 (E4M3) significant bits and `s` at most 4, so `q * s` in FP32 and
  its BF16 rounding are exact, down to the MXFP4 subnormal `2^-127`.  Above the BF16 range the product is `±inf`, as
  the rule's own BF16 rounding gives: an element with `|x| >= 248 * 2^120` (MXFP8) or `|x| >= 224 * 2^120` (MXFP4)
  becomes `±inf`, although `x` is finite.

The programs are compiled with IEEE division and square root and without flush to zero (`--prec-div=true`,
`--prec-sqrt=true`, `--ftz=false`, no `--use_fast_math`), as `build.json` records.

## Determinism

This is the `det` package of the op, and the family has no other package.  For identical inputs every output
repeats bit for bit, on every supported architecture, for any grid and in place or not:

| output | identical inputs |
|---|---|
| `y` (BF16 values) | repeats bit for bit |
| `scales` (FP32 scales) | repeats bit for bit |

Each group is quantized by its own `G / 8` threads, and the group maximum is exact in any order.  The kernels use no
atomics and sum nothing across threads, CTAs or calls.  No entry takes a `deterministic` keyword, and no call alerts
under `torch.use_deterministic_algorithms(True)`.  A non-deterministic package is not planned: no step of this op has
an order to relax.  There is no KCP (context-parallel) package either: the op works on each token alone.

`tests/experimental/test_cake_act_qdq_det_train.py` compares every output bit for bit (signed zeros included) with
the FP64 rule on edge rows (format levels and midpoints, ties, signed zeros, floors, saturation, the infinities), on
every finite BF16 value under every scale each format reaches, and on random inputs of the model shapes, in place and
not.  It also checks five repeated calls, CUDA graph replay, a
checkpointed recompute, steady-state calls without host synchronization, and no rebuild on a new shape.

## Limitations

- BF16 input and output only, with FP32 scales.  The quantized codes are not returned.
- Contiguous input only; the groups run along the last dimension, and `D` must be a multiple of `G`.
- Non-finite inputs (NaN, ±Inf) are outside the rule; their outputs are unspecified.
- Forward only: no backward and no autograd entry.  A straight-through estimator is the caller's choice.
- One call takes at most `8 * (2^31 - 256)` elements.
