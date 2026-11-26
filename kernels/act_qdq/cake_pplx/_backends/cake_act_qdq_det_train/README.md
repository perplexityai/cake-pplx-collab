# Deterministic activation quantize-dequantize backend (SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_act_qdq_det_train`: the forward of a block-scaled quantize-dequantize (QDQ) of BF16
activations back to BF16.  Each group of `G` consecutive elements of the last dimension is quantized to a
block-scaled format with one scale, then dequantized.  The result has the shape and dtype of the input.  The FP32
scale of each group is an optional second output.

The facade `cake_pplx.cake_act_qdq_det_train` is a stable API.  This backend package is experimental, and its
internal names may change.  The facade imports the backend on first use.  The supported architectures are SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200); another device raises `RuntimeError`.  The package loads
its libraries prebuilt from `aot/` when a builder compiled them; otherwise it JIT-builds them from the sources and
flags that `build.json` records.  Only the JIT shim (`_cake_pplx_jit.py`) reads environment variables, and only to
build or find a library: `CUDA_HOME` / `CUDA_PATH` (else `nvcc` on `PATH`), `CAKE_PPLX_NVCC`, `CXX`,
`CAKE_PPLX_CACHE_DIR`, `CAKE_PPLX_DISABLE_JIT` and `CAKE_PPLX_JIT_WORKERS`.  The rest of the package reads none.

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

`cake_act_qdq_det_train_forward_workspace_bytes(shape, format=...)` returns 0: a call takes no caller workspace.
`is_cake_act_qdq_det_train_supported(device)` tells whether the device has kernels.
`cake_act_qdq_det_train_domain_empty` and `cake_act_qdq_det_train_domain_plan` serve the `locality="domain"` route of
an R200 ("Locality domains" below).  The facade also exposes `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES`, `FORMATS` and
`GROUP_SIZES`.

## Inputs and outputs

| Argument | Rule |
|---|---|
| `x` | BF16 CUDA tensor `[..., D]`, at least one dimension, contiguous, 16-byte aligned, `D` a multiple of `G`, at most `MAX_ELEMENTS` (`8 * (2^31 - 256)`) elements |
| `format` | `"fp4_e4m3_g16"` (`G = 16`), `"mxfp4_e8m0_g32"` (`G = 32`) or `"mxfp8_e8m0_g32"` (`G = 32`) |
| `out` | `None` (a new tensor), `x` itself (in place), or a contiguous, 16-byte aligned BF16 tensor of `x`'s shape on `x`'s device that does not overlap `x` |
| `scales` | `None` or a contiguous FP32 tensor `[..., D / G]` on `x`'s device that does not overlap `x` or `out` |
| `return_scales` | `False` (default) or `True` |
| `locality` | `"off"` (default) or `"domain"`; checked before every other argument (`TypeError` for a non-string, `ValueError` for another string) |

The call returns `y`, or `(y, scales)` when `return_scales` is true or a `scales` buffer is given.  `y` is `out`
when `out` is given, so an in-place call returns `x`.  A new `scales` tensor is allocated when the scales are
requested without a buffer.  Outputs carry no autograd history.  A caller buffer the kernel writes (`out`, `x` in
place, `scales`) must not require grad while grad mode is on, and the call advances its version counter as a
PyTorch in-place op does.

A wrong dtype raises `TypeError`.  An unsupported device raises `RuntimeError`.  Any other violation raises
`ValueError` before any launch.  A tensor with no elements (for example 0 rows) launches nothing and returns empty
outputs of the right shapes.

## Programs

The package has twenty-three programs: per format, with and without the scale output, a main program, a small one and
an interleaved-domain one; a wide program for MXFP8 without the scale output; and per MXFP4 and MXFP8 format, with and
without the scale output, a large program.  The wide and the large programs are built for `sm_100a` and `sm_103a`, the
interleaved-domain programs for `sm_107a` ("Locality domains" below), every other program for `sm_100a`, `sm_103a` and
`sm_107a`, each with the exact `-gencode` flag of the architecture, so `build.json` lists 52 libraries.  With
`locality="off"`, a call runs one program of its route (`cake_jit.selected_module(arch, format, scale_out=..., n_vecs=...,
in_place=...)`): the small, the wide or the large program when `n_vecs` lies in its `vectors` range for the call's kind,
and the main program otherwise.  The small program's range out of place is 65,537 to 524,288 on `sm_100a` and `sm_103a`
(to 131,072 for MXFP8 without the scale output) and 1 to 262,144 on `sm_107a`; in place it is 1 to 262,144 on `sm_107a`,
and `sm_100a` and `sm_103a` have none, since the small program measured no faster there in place.  The wide program's
range starts above the main and small programs' sizes: out of place from 131,073 vectors to 2,147,483,136 (2^31 - 512,
since it forms `n_vecs + 511`; the main program runs above that), and in place from 262,145 to 33,554,431, below the
large program's range.  The large program's range holds the largest sizes: in place from 4,194,304 vectors (from
33,554,432 for MXFP8 without the scale output) and out of place from 8,388,608 for MXFP8 with the scale output only, all
to 2,147,481,600 (2^31 - 2048, since it forms `n_vecs + 2047`; the main program runs above that).

A program sees the flat input as `n_vecs = numel / 8` vectors of 8 BF16 values (16 bytes), and the `G / 8` adjacent
threads of one group combine their maxima with warp shuffles.  A CTA has 64 threads and takes a tile of 128 vectors (256
on `sm_107a`) per trip of a grid-stride loop that is uniform across the CTA, so every shuffle has its whole warp and any
grid of at least one CTA is correct.  Each thread loads its 2 vectors (4 on `sm_107a`) before it computes or stores any
of them, which keeps about 64 KiB of loads in flight per SM.  The host launches one CTA per tile once the tiles pass 2.5
waves of the CTAs the SMs hold (32 per SM, 16 on `sm_107a`), and otherwise that many CTAs, at most one per tile; the SM
count is read once per device.  The small program is the same kernel with 256-thread CTAs that take 2 vectors per thread
on every architecture (tiles of 512 vectors, 8 CTAs per SM, 4 on `sm_107a`).  Its sizes fill at most about one wave,
where each CTA start costs time (the GPU starts about 1,850 CTAs per microsecond) and its larger tiles need fewer CTAs:
at 524,288 vectors on `sm_100a` and `sm_103a` the main program launches 4,096 CTAs and the small one 1,024.  The wide
program is the same kernel with 128-thread CTAs that take 4 vectors per thread (tiles of 512 vectors), launched one CTA
per tile at every size, and its launch bounds hold it to 48 registers per thread, so 10 CTAs fit on an SM.  It measured
up to 5 % faster than the main program in its range on B200 and GB300, and up to 9 % slower below it.  The large program
is the same kernel with 1024-thread CTAs that take 2 vectors per thread (tiles of 2,048 vectors), launched one CTA per
tile at every size, and its launch bounds hold it to 32 registers per thread, so 2 CTAs fit on an SM.  On B200 and
GB300, under three states of L2 before the call (full of dirty lines, full of clean lines, and holding `x` just
written), it measured faster than the program it replaces in every state inside its ranges: by 0.2 to 3.3 % from
8,388,608 vectors on, and at 4,194,304 vectors by up to 7 % with a clean L2 and up to 20 % with `x` in L2.  Below its
ranges its tiles of 2,048 vectors leave SMs idle, and it measured up to 16 % slower; in place at 16,777,216 vectors it
measured 1.2 % slower than the wide program with a clean L2.  Out of place it measured from 0.5 % slower to 0.7 % faster
for MXFP4 and no faster for MXFP8 without the scale output, and FP4 measured slower at some sizes in both kinds of call,
so those have no large range.  Every program gives the same bytes at every size.  Neither the last dimension, nor the
rank, nor the row count enters the kernel, so a new shape never rebuilds a library.  A call builds or loads only the
library of its own program, so build every library of the architecture before a CUDA graph capture: call
`build_and_load()` on each spec of `cake_jit.specs_for_arch(arch)`, or make one eager call for each format, scale output
and program (size range and kind of call) that the capture uses.

## Locality domains (`sm_107a`)

An R200 has two dies, each with its own SMs and its own memory (a locality domain).  Default memory alternates between
the dies every 4 KB, so about half of the bytes each SM reads come from the other die.
`cake_act_qdq_det_train_domain_empty(shape, dtype=torch.bfloat16, device=None)` maps one range of 2 MB chunks
(`cuMemCreate` in a locality domain), chunk `k` in die `k & 1`, and returns a torch tensor over it without a copy.  It
registers the range, so a call finds a tensor's chunk offset and die from its address alone, with no driver call.

The interleaved-domain program of a route (`cake_jit.DOMAIN_ROUTES`) has its own tiles of 128 threads, 1,024 vectors (8
each).  Its MXFP4 and MXFP8 programs keep each vector in packed BF16 pairs: the group maximum of the absolute values in
BF16, one BF16 multiply by the inverse power-of-two scale, and BF16-pair quantize and UE8M0-scaled dequantize
conversions, each giving its FP32 step's result ("Numerics" below), so the bytes are those of the FP32 steps.  A chunk
holds 128 tiles, and an item is one tile.  Each CTA reads its die at each claim, takes its next item from its own die's
counter, and takes the other die's items once its own are taken.  The items a CTA runs depend only on its claims, so any
grid gives the same bytes; the host launches one CTA per item, at most 7 (FP4) or 8 (MXFP4, MXFP8) per SM.  A call with
`locality="domain"` runs it when all of these hold (`cake_jit.domain_route`): the device is `sm_107a`; the call is out
of place with an `out` buffer; `x` and `out` lie in memory from the helper at the same offset in their chunks and in
chunks of the same die; and `numel / 8` is in the record's `vectors` range (8,388,608 to 2,147,482,624 vectors).  Every
other call runs the `"off"` program, with the same bytes, and never raises for its placement.
`cake_act_qdq_det_train_domain_plan` returns the route, the reason (`"placed"`, or the first of `"empty"`, `"arch"`,
`"in_place"`, `"no_out"`, `"size"`, `"x_memory"`, `"out_memory"`, `"chunk_offset"` and `"die_parity"` that holds), the
program, `first_tile`, `first_die` and the grid, without a launch.  An in-place call runs the `"off"` program: the
interleaved-domain records define no in-place range (`vectors[arch]["in_place"]` is `None`).  On VR200, the domain
program ran 1.11-1.30x as fast as the `"off"` program on the package benchmark's out-of-place MXFP4 and MXFP8 rows from
8,388,608 vectors, and 1.10-1.24x on FP4 `[131072, 512]` and `[1048576, 512]` (GPU time, median of 15 runs on 3 GPUs,
each run at least 1.09x; MXFP8 `[16384, 8, 4096]`: 162.5-162.6 µs off, 124.5-124.8 µs domain), and 1.09-1.31x in
back-to-back calls, eager or in a CUDA graph.  A call that falls back runs the `"off"` program on the helper's memory:
on the benchmark's fallback rows (out of place, 131,072 to 4,194,304 vectors) it ran 0.95-1.03x as fast as on default
memory (GPU time), the slowest for buffers of one chunk, which lie wholly in die 0; in-place calls and larger ones on
the helper's memory were not measured.

The claim counters are one unsigned 32-bit word per die (`cake_domain`; the record's `counters` field), each in its own
die's memory.  The helper's first call on a device maps one 2 MB chunk in each die for them (4 MB, kept for the
process, which `torch.cuda.memory_*` does not count), zeroes them and synchronizes once.  A workspace is 128 bytes in
each die's chunk and is handed out once; every launch leaves its counters zero, so a call on such a workspace launches
no fill.  Eager calls use one workspace per (device, stream), and per (device, thread) on the per-thread default stream
(`cudaStreamPerThread`, one handle that names each thread's own stream): 15,360 per device.  The calls of one CUDA
graph capture on one capture stream share one of 1,024 others.  Once the 15,360 are taken, a call on a new stream makes
a workspace in default memory and zeroes its 256 bytes; once the 1,024 are taken, each captured call takes one from the
graph's pool, which the graph zeroes at every replay (die memory cannot be mapped during a capture).  Either keeps both
counters in default memory, with the same bytes.  An eager launch that raises drops its workspace for good.  The
replays of one instantiated graph never overlap, but a capture instantiated more than once, or added to other graphs as
a child graph, shares its workspace between those copies: run them one at a time.  Allocate the buffers before a
capture (the helper refuses during a CUDA graph capture on any device in this thread), and make one eager call per
program the capture uses.  After the first call on a stream, a call does not synchronize the host.

Freeing the last view of a buffer synchronizes the device, then unmaps its chunks; a buffer freed during a CUDA graph
capture on any device in this thread is unmapped by the next helper call.  The helper reads torch's current stream of
each device: a capture begun with `cudaStreamBeginCapture` on a stream that is not torch's current stream is not
detected.  Do not allocate these buffers, or free the last view of one, in one thread while another thread captures a
CUDA graph: the driver calls would end that capture.  The driver maps this memory outside torch's caching allocator, so
the `torch.cuda.memory_*` counters do not count it.  When the device has no room, the helper returns torch's cached
free memory to the driver once and then raises `torch.cuda.OutOfMemoryError`.  The helper needs cuda-python with the
locality-domain names (`cuda-bindings` 13.5 or later: the package's `locality` extra), and the driver must report two
locality domains.  The `"off"` route never imports it.

## Numerics

For each group of `G` elements, with `amax` the largest magnitude in the group:

| format | element | `G` | scale `s` | `amax` floor | `Q` |
|---|---|---|---|---|---|
| `fp4_e4m3_g16` | E2M1 | 16 | `e4m3_rne_satfinite(max(amax, 6 * 2^-9) / 6)` | `6 * 2^-9` | 6 |
| `mxfp4_e8m0_g32` | E2M1 | 32 | `2^ceil(log2(max(amax, 6 * 2^-126) / 6))` | `6 * 2^-126` | 6 |
| `mxfp8_e8m0_g32` | E4M3 | 32 | `2^ceil(log2(max(amax, fp32(1e-4)) / 448))` | `fp32(1e-4)` | 448 |

Then `u = clamp(x / s, -Q, Q)` with the exact quotient, `q` is the element format's round-to-nearest-even value of
`u` with the sign of zero kept, and `y = bf16_rne(q * s)`.  The E4M3 scale saturates at 448.  The outputs equal this
rule evaluated in FP64, bit for bit, including the sign of zero.  The kernels use no FP64, and the FP32 and BF16 steps
below give the rule's outputs:

- **E8M0 scale.**  Write `amax = m * 2^e` with `m` in [1, 2), and `Q = mq * 2^c` (`6 = 1.5 * 2^2`,
  `448 = 1.75 * 2^8`).  Then `ceil(log2(amax / Q)) = e - c + [m > mq]`, read from the IEEE fields of `amax`.
  `ceil(log2(.))` is monotone, so the floor becomes a lower bound on the scale exponent: `2^-126` for MXFP4 and
  `2^-22` for MXFP8.
- **E8M0 quotient.**  `s` is a power of two, so `x * 2^-k` and the IEEE quotient `x / s` are the same correctly
  rounded number.  The kernels keep subnormals (no flush to zero), so the product equals the quotient bit for bit.
  A pair of elements takes this multiply, and the E4M3-scaled steps below, in packed two-lane FP32 arithmetic; each
  lane rounds as it would alone.  The `sm_107a` interleaved-domain programs of MXFP4 and MXFP8 instead take one BF16x2
  multiply of each BF16 pair by `2^-k`: BF16 has FP32's exponent range, so that product is the quotient from `2^-126`
  up, and a smaller quotient lies below the smallest E2M1 and E4M3 midpoints (`0.25` and `2^-10`) and converts to a
  signed zero either way.  Their BF16-pair conversions to E2M1 and E4M3 round each half as the FP32 conversion rounds
  its widened value.
- **E4M3 scale.**  The `sm_107a` programs multiply `max(amax, 6 * 2^-9)` by `RN(1/6)`, the FP32 value nearest
  `1/6`, which is `1/6` times `1 + 2^-25`; the `sm_100a` and `sm_103a` programs divide it by 6 (an IEEE
  round-to-nearest division).  Either FP32 result is within `2^-23` (relative) of the exact quotient
  `max(amax, 6 * 2^-9) / 6`, and equals it whenever that quotient is an FP32 value, as every E4M3 value and rounding
  midpoint is.  `amax` has 8 significant bits, so `amax / 3` is either exact or at least `2^-8` (relative) away from
  every 5-bit value that it does not equal.  So the conversion to E4M3, which rounds to nearest even and saturates
  at 448, gives the E4M3 value of the exact quotient, ties included.
- **E4M3-scaled quotient.**  `u` is never the plain product `x * RN(1 / s)` rounded to nearest.  Every program forms
  `inv = 1 / s` once per group (an IEEE division).  The `sm_107a` programs take `u = x * inv` rounded toward zero: `s`
  is a power of two times one of the eight E4M3 significands `m`, and the FP32 value nearest `1/m` is `1/m` or lies
  above it by at most `7 * 2^-27` (relative), so the exact product equals `x / s` or lies just beyond it (away from
  zero), within `2^-24` (relative).  The `sm_100a` and `sm_103a` programs take `u0 = x * inv` and correct it with one
  FMA residual step: `r = fma(u0, s, -x)` is exact, and the exact value `u0 - r * inv`, which `u = fma(-r, inv, u0)`
  rounds, is within a small fraction of an FP32 ulp of `x / s`.  In both forms `u` is `x / s` whenever that quotient
  is an FP32 value, as every E2M1 midpoint times `s` is, and within one FP32 ulp of it otherwise; in the `sm_100a` and
  `sm_103a` programs this holds for `|x|` of at least `2^-100`, and below that the residual may round, but such a
  quotient is below `2^-91` and becomes a signed zero either way.  The exact quotient of an 8-bit `x` by a 4-bit `s`
  is at least `2^-9` (relative) away from every E2M1 midpoint that it does not equal, so `u` rounds to the E2M1 value
  of the exact quotient.  A zero `x` keeps its sign.
- **Clamp.**  The saturating element conversion is the rule's clamp: a `|u|` above `Q` converts to `Q`.
- **Dequantize.**  Each pair of codes converts exactly to two BF16 values, and one BF16x2 multiply scales both by `s`.
  `s` is a BF16 value: it has at most 4 significant bits.  `q` has at most 2 (E2M1) or 4 (E4M3) significant bits, so the
  BF16 product `q * s` is exact, down to the MXFP4 subnormal `2^-127`, which the multiply keeps.  The `sm_107a`
  interleaved-domain programs of MXFP4 and MXFP8 convert each pair of codes and scale both by `s` in one UE8M0-scaled
  conversion, rounded once to BF16: the same exact product.  The `sm_107a` main programs of `mxfp8_e8m0_g32`, with and
  without the scale output, decode each pair to two FP16 values instead and multiply each by `s` in FP32; that product
  has at most 8 significant bits, so it and its BF16 rounding are exact too.  Above the BF16 range the product is
  `±inf`, as the rule's own BF16 rounding gives: an element with `|x| >= 248 * 2^120` (MXFP8) or `|x| >= 224 * 2^120`
  (MXFP4) becomes `±inf`, although `x` is finite.

The programs are compiled with IEEE division and square root and without flush to zero (`--prec-div=true`,
`--prec-sqrt=true`, `--ftz=false`, no `--use_fast_math`), as `build.json` records.

## Determinism

This is the `det` package of the op, and the family has no other package.  For identical inputs every output
repeats bit for bit, on every supported architecture, for any grid and in place or not:

| output | identical inputs |
|---|---|
| `y` (BF16 values) | repeats bit for bit |
| `scales` (FP32 scales) | repeats bit for bit |

Each group is quantized by its own `G / 8` threads, and the group maximum is exact in any order.  The kernels sum
nothing across threads, CTAs or calls.  Only the interleaved-domain programs use atomics: 32-bit integer claims that
choose which CTA computes a tile, never what it computes, so their outputs are the `"off"` programs' bytes.  No entry
takes a `deterministic` keyword, and no call alerts under `torch.use_deterministic_algorithms(True)`.  A
non-deterministic package is not planned: no step of this op has an order to relax.  There is no KCP
(context-parallel) package either: the op works on each token alone.

`tests/experimental/test_cake_act_qdq_det_train.py` compares every output bit for bit (signed zeros included) with
the FP64 rule on edge rows (format levels and midpoints, ties, signed zeros, floors, saturation, the infinities), on
every finite BF16 value under every scale each format reaches, and on random inputs of the model shapes, in place and
not.  It also checks five repeated calls, CUDA graph replay, a
checkpointed recompute, steady-state calls without host synchronization, and no rebuild on a new shape.
`tests/experimental/test_cake_act_qdq_det_train_locality.py` checks the `locality="domain"` route.  On an R200 it
checks every interleaved-domain program against the FP64 rule on any grid, the route's bytes against the `"off"` route
above its first size, each reason of the plan, the claim counters and the die of their memory, two streams, CUDA graph
replay and steady-state calls without host synchronization; with two R200s, a free and an allocation during a capture
on the other device.  On every architecture it checks that a call outside the helper's memory runs the `"off"` program,
and without a GPU the helper's unwind of a failed mapping and its error without cuda-python.  On an R200 these tests
need the `locality` extra.

## Limitations

- BF16 input and output only, with FP32 scales.  The quantized codes are not returned.
- Contiguous input only; the groups run along the last dimension, and `D` must be a multiple of `G`.
- Non-finite inputs (NaN, ±Inf) are outside the rule; their outputs are unspecified.
- Forward only: no backward and no autograd entry.  A straight-through estimator is the caller's choice.
- One call takes at most `8 * (2^31 - 256)` elements.
- The interleaved-domain route serves out-of-place calls on `sm_107a` over memory from
  `cake_act_qdq_det_train_domain_empty` only.  The copies of one capture that uses it (instantiated more than once, or
  added to other graphs as a child graph) must not run concurrently.
