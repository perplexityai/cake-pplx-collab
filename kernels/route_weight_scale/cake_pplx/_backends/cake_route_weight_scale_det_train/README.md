# Cake route-weight row scaling (BF16 x FP32 route weight, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_route_weight_scale_det_train`: the route-weight row
scaling of a routed MoE expert in training.  After the down projection, each
received token copy (row `r`) is scaled by its FP32 route weight `w[r]` before
the combine.  `x`, `dy`, `y` and `dx` are BF16 `[rows, H]` with `H` 4096 or
3584; `w` and `dw` are FP32 `[rows]`.  `rows` and the row strides are run-time
values: no shape list, no recompilation, no device-to-host read, no
synchronization.

The facade `cake_pplx.cake_route_weight_scale_det_train` is the API; this
backend package is experimental (no compatibility guarantee for its internal
names).  SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError`.  Each library
is loaded prebuilt from `aot/` when a builder compiled it and JIT-built from the
sources and flags `build.json` records otherwise.  Only the JIT shim
(`_cake_pplx_jit.py`) reads environment variables, and only to build or find a
library: `CUDA_HOME` / `CUDA_PATH` (else `nvcc` on `PATH`), `CAKE_PPLX_NVCC`,
`CXX`, `CAKE_PPLX_CACHE_DIR`, `CAKE_PPLX_DISABLE_JIT` and
`CAKE_PPLX_JIT_WORKERS`.  The rest of the package reads none.

## Opt-in

```python
from cake_pplx.cake_route_weight_scale_det_train import (
    cake_route_weight_scale_det,
    cake_route_weight_scale_det_train_backward,
    cake_route_weight_scale_det_train_forward,
)

y = cake_route_weight_scale_det(x, w)  # autograd: x.grad BF16 [rows, H], w.grad FP32 [rows]

# steps without autograd; out, dx and dw may be row slices of preallocated buffers
y = cake_route_weight_scale_det_train_forward(x, w, out=y_buffer[:rows])
dx, dw = cake_route_weight_scale_det_train_backward(x, dy, w, dx=dx_buffer[:rows], dw=dw_buffer[:rows])
```

## Contract

| Item | Value |
|---|---|
| Forward | `y = bf16(w[:, None] * x)`: the FP32 product rounded once; bit for bit `(w[:, None] * x.float()).bfloat16()` |
| Backward | `dx = bf16(w[:, None] * dy)`, bit for bit `(w[:, None] * dy.float()).bfloat16()`; `dw = sum over c of dy * x`, FP32 `[rows]` |
| BF16 operands | `[rows, H]`, `H` in `WIDTHS` (4096, 3584); unit column stride, row stride a multiple of 8 elements, 16-byte aligned base; an output's row stride at least `H` |
| `w`, `dw` | FP32 `[rows]`, contiguous |
| `rows` | 0 to `2**29 - 1`; 64-bit row offsets, so 1,048,576 rows at `H = 4096` (more than 2^31 elements) are served |
| Workspace | 0 bytes: the forward allocates only `y`, the backward only `dx` and `dw` (none when passed in) |
| Retained state | The autograd function saves `x` and `w` only |
| Empty input | `rows = 0`: empty outputs, no launch, no library load |
| Programs | Eight records (`cake_jit.MODULES`): `fwd_h4096`, `fwd_h4096_t32`, `fwd_h3584`, `fwd_h4096_flat`, `fwd_h3584_flat`, `bwd_h4096`, `bwd_h3584`, `bwd_h3584_warp`, each built for the three architectures |
| Minimum CUDA | The `min_cuda` of each library in `build.json`; an older local toolkit raises `RuntimeError` when the library is built |

A violation of the operand rules is a `ValueError` (a non-tensor operand a
`TypeError`) before any allocation or launch.  The entry points copy nothing;
the autograd backward copies an upstream gradient the kernels cannot read (an
expanded gradient, say) once.

## Kernel

One call launches one kernel on the current torch stream of the operands'
device, a plain stream-ordered launch: nothing cooperative, no PDL, no dynamic
shared memory.  A row's columns are spread over 32 lanes: lane `l` makes
`H / 256` loads of 8 elements at columns `(p * 32 + l) * 8`, so each load of a
warp is contiguous.  The flat forward instead reads the tensors as one
row-major index of `rows * H` elements, torch.compile's elementwise schedule.
Five kinds of program:

| Records | CTA | Work a CTA (`rows_per_cta` or `elements_per_cta`) | Taken for |
|---|---|---|---|
| `fwd_h4096`, `fwd_h3584` | 128 threads, one warp a row | 4 consecutive rows | `fwd_h3584`: the forward at `H = 3584` below the band of `fwd_h3584_flat`; no band takes `fwd_h4096` (the tests' reference) |
| `fwd_h4096_t32` | 32 threads | 1 row | the forward at `H = 4096` below the band of `fwd_h4096_flat` |
| `fwd_h4096_flat`, `fwd_h3584_flat` | 128 threads, two loads of 8 elements a thread | 2048 consecutive elements | the forward in its band of `cake_jit.ROUTES` (many rows) |
| `bwd_h4096`, `bwd_h3584` | 4 warps (128 threads) | 1 row | the backward (at `H = 3584` outside the band of `bwd_h3584_warp`), on a grid of at most 8 CTAs per SM while a call's rows fit in L2 (`cake_jit.ROUTES`) |
| `bwd_h3584_warp` | 128 threads, one warp a row | 4 consecutive rows | the backward at `H = 3584` in its band of `cake_jit.ROUTES` |

`cake_jit.route(stage, H, rows)` picks the record and the grid cap.  The
forward multiplies each pair of elements by the row's weight in FP32 and
rounds the pair to BF16; `fwd_h4096_t32` and the flat programs run the
per-element instructions of `fwd_h<H>`, so they store the same bits.  A flat
program finds each vector's row and column by a division by `H`, in 32-bit
arithmetic when every element index and offset of the call fits, else in
64-bit; both loops make the same loads and stores.  The backward loads `dy` beside
`x` and stores `dx` and `dw` in one pass: it never forms a `[rows, H]` FP32
product.  Each of its four warps sums one run of each lane's loads
(`H = 4096`: 4, 4, 4, 4 loads; `H = 3584`: 4, 4, 4, 2) as a subtree of the
`dw` tree (see Numerics) and stores the subtree's root, two FP32 words a
lane, to static shared memory (1 KiB a CTA).  After a CTA barrier, warp 0
joins the four roots pairwise in the same order, adds the even and odd halves
and makes the five XOR shuffles across its 32 lanes.  So `dw` is summed in
the order of Numerics, bit for bit.  In `bwd_h3584_warp` each lane sums its
14 loads in the same order and the warp makes the five XOR shuffles, so it
stores the same `dx` and `dw` bits as `bwd_h3584`.  Each stage and width
gives the same bits at every row count.

Every program strides over its blocks with a 32-bit block counter.  A grid of
`g` CTAs covers `nblocks = ceil(rows / rows_per_cta)` blocks (a flat program:
`ceil(rows * H / elements_per_cta)`) only while `nblocks + g <= 2**31`, so the
host launches
`min(nblocks, 2**31 - nblocks)` CTAs (`cake_jit.grid`): `nblocks` up to
`2**30` blocks.  Below the row bound `2**29` a flat program has fewer than
`2**30` blocks.  In a band of `cake_jit.ROUTES` with a cap, the host launches
at most that many CTAs per SM of the device: a persistent grid, each CTA
striding over several rows, with the same bits.  Every row offset is 64-bit.

## Numerics

- `y` and `dx`: each element is the FP32 product of the BF16 value and the
  FP32 weight, rounded once to BF16 (round to nearest, no flush).
- `dw`: the product of two BF16 values is exact in FP32, so only the
  additions round.  They form a balanced binary tree in one fixed order: two
  products of a load's even (and odd) columns joined by one FMA, the two pairs
  added, the loads of a lane joined pairwise, the even and odd halves added,
  and five XOR shuffles across 32 lanes.  Every element's path has the fewest
  roundings a pairwise sum allows.  `dw` is one FP32 word per row, stored
  once; it does not divide by `w`, so a zero-weight row keeps its sum.
- Nothing is FP64.  The tests compare `y` and `dx` bit for bit with the
  PyTorch FP32 product rounded to BF16, and `dw` against an FP64 reference
  of the same inputs beside the FP32 row sum `(dy.float() * x.float()).sum(1)`.

## Determinism

Nothing sums across rows, and the one sum, `dw` within a row, has one fixed
order, so the outputs repeat bit for bit for the same inputs on the same
architecture, whatever the row count, the layout, the stream or a CUDA graph.
This is the family's only package; the module name carries `det`, and no entry
takes a `deterministic` keyword.  There is no `nondet` package and no KCP form:
no step of the op has an order to relax, and the op works on each row alone.
The call never reads torch's deterministic-algorithms setting and never alerts.

## Limitations

- `H` 4096 and 3584 only.
- BF16 `x` / `dy` / `y` / `dx` and FP32 `w` / `dw` only.
- Operands of another layout are refused, not copied.
- No build for any architecture other than SM100, SM103 and SM107.
