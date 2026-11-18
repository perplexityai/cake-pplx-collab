# Cake Engram row lookup (FP8 E4M3 + E8M0 -> BF16, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_engram_lookup_det_train`: the Engram row lookup.
It gathers rows of a quantized table by row id and writes them as BF16:
`weight` is `float8_e4m3fn [R, 256]`, `scale` is `float8_e8m0fnu [R, 8]`
(one scale per 32 columns), `row_ids` is int64 of any shape, and `out` is
BF16 `[*row_ids.shape, 256]`.  The table is two contiguous tensors or two
views into one packed `[R, 264]` byte buffer (each row: its 256 value bytes,
then its 8 scale bytes).  `R` and the `row_ids` shape are run-time values:
no shape list, no recompilation, no device-to-host read, no synchronization.

The facade `cake_pplx.cake_engram_lookup_det_train` is a stable API; this
backend package is experimental (no compatibility guarantee for its internal
names).  SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the
supported architectures; another device raises `RuntimeError` for a non-empty
lookup.  Its library is loaded prebuilt from `aot/` when a builder compiled it
and JIT-built from the sources and flags `build.json` records otherwise.  Only
the JIT shim (`_cake_pplx_jit.py`) reads environment variables, and only to
build or find a library: `CUDA_HOME` / `CUDA_PATH` (else `nvcc` on `PATH`),
`CAKE_PPLX_NVCC`, `CXX`, `CAKE_PPLX_CACHE_DIR`, `CAKE_PPLX_DISABLE_JIT` and
`CAKE_PPLX_JIT_WORKERS`.  The rest of the package reads none.

The family has a forward step only: no backward kernel and no autograd entry.
The table's gradient does not depend on its storage format; the caller sums it.

## Opt-in

```python
from cake_pplx.cake_engram_lookup_det_train import (
    cake_engram_lookup_det_train_forward,
    cake_engram_lookup_det_train_split_packed,
)

# a contiguous shard: weight float8_e4m3fn [R, 256], scale float8_e8m0fnu [R, 8]
out = cake_engram_lookup_det_train_forward(weight, scale, row_ids)  # int64 [T, 24] -> BF16 [T, 24, 256]

# a packed uint8 [M, 264] buffer of rows fetched from other ranks
w, s = cake_engram_lookup_det_train_split_packed(fetched)
out_m = cake_engram_lookup_det_train_forward(w, s, ids)             # int64 [N] -> BF16 [N, 256]
```

## Contract

| Item | Value |
|---|---|
| Computes | `out[p, 32*g + j] = bf16(float(weight[r, 32*g + j]) * float(scale[r, g]))`, `r = row_ids[p]` |
| `weight`, `scale` | Unit column stride, row stride a multiple of 8 elements, start on an 8-byte boundary; `scale` has `weight`'s `R` |
| `row_ids` | Contiguous int64, any shape, fewer than 2^31 - 8 elements, values in `[0, R)`, repeats allowed |
| Output | New contiguous BF16 `[*row_ids.shape, 256]` |
| Workspace | 0 bytes: the call allocates only `out` (`row_ids.numel() * 512` bytes) |
| Retained state | None |
| Empty `row_ids` | Empty `out`, no launch, before the device's architecture is looked up |
| Programs | One per architecture (record `lookup`), for both layouts and every `row_ids` shape |
| Minimum CUDA | 12.8 for `sm_100a`, 12.9 for `sm_103a`, 13.5 for `sm_107a` (`build.json`); an older local toolkit raises `RuntimeError` when the library is built |
| Row ids | In `[0, R)`. An id outside (a negative id included) stops the launch with a device-side trap before its row is read: the call or the next synchronizing CUDA call raises, and the CUDA context is unusable afterwards. The host checks no id (that would synchronize). |

A violation of the operand rules is a `ValueError` before any launch.

## Kernel

One call launches one kernel on the current torch stream of the operands'
device, a plain stream-ordered launch: nothing cooperative, no PDL, no shared
memory.  A CTA has 8 warps; a warp handles 4 consecutive looked-up rows per
step and walks the rows grid-stride, so any grid covers every row.  For each
row a lane reads 8 value bytes and the scale byte of its 32-column group,
decodes them to FP32, multiplies in FP32 and stores 16 bytes of BF16.  The
row strides of `weight` and `scale` come from the tensors (264 for the packed
views; a one-row table goes in with 256 and 8, which change no address since
its only valid id is 0), and byte offsets are 64-bit (tables past 2^31 bytes and outputs past
2^31 elements are served).  The grid is one CTA per 32 looked-up rows, capped
at 8 CTAs per SM.

## Numerics

Each element is the exact product `weight * scale` rounded once to BF16 (round
to nearest even), the value of the FP64 product rounded to BF16:

- A finite E4M3 value has at most 4 significant bits and is a multiple of
  2^-9; a finite E8M0 scale is a power of two (2^-127 for the byte 0, an FP32
  subnormal).  So every product is exact in FP32 unless its magnitude reaches
  2^128, and one rounding to BF16 is the only rounding.
- FP32-subnormal products are kept (the multiply does not flush, and the
  library is built without fast math, `--ftz=false`).
- Products past the FP32 range are `±inf`, as in BF16.
- The E8M0 byte `0xFF` is NaN and makes its 32 columns NaN; the E4M3 codes
  `0x7F` and `0xFF` are NaN.  `-0 * scale` is `-0`.

The tests check every one of the 65,536 (E4M3 code, E8M0 byte) pairs bit for
bit against an FP64 reference, in both layouts.  That check is stricter than
a comparison with an error budget: the error is 0 for every element.

## Determinism

Every output element is one product rounded once, and nothing is summed
across threads, CTAs or calls, so the output repeats bit for bit for the same
table bytes and row ids, in either layout.  This is the family's only
package; the module name carries `det`, and no entry takes a `deterministic`
keyword.  The call never reads torch's deterministic-algorithms setting and
never alerts.  There is no `nondet` package because a gather has no sums, and
no KCP form because the lookup is a single-rank local lookup.

## Limitations

- Forward only; no backward and no autograd entry.
- Row width 256 and 8 scales per row only.
- `row_ids` must be contiguous int64 and hold fewer than 2^31 - 8 elements.
- `weight` and `scale` need unit column strides, row strides that are
  multiples of 8 elements and 8-byte aligned starts.
- No build for any architecture other than SM100, SM103 and SM107.
