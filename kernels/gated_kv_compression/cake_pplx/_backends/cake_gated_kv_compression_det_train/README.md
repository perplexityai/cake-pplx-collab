# Cake deterministic ratio-2 gated KV compression (width 512, SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_gated_kv_compression_det_train`: document-bounded pooling of
token pairs for training, forward and backward.  Token pairs `(2k, 2k + 1)` of each packed
document become one row, mixed per channel by a softmax over the two gate logits:

```text
p0, p1    = softmax(Z[t0, d], Z[t1, d])                          # FP32
out[r, d] = BF16(p0 * C[t0, d] + p1 * C[t1, d])                  # one rounding

g        = float32(dout[r, d])
dC[t0, d] = p0 * g                dC[t1, d] = p1 * g
dZ[t0, d] = p0 * p1 * (C[t0, d] - C[t1, d]) * g                  dZ[t1, d] = -dZ[t0, d]
```

For document `s` with bounds `cu_seqlens[s] <= cu_seqlens[s + 1]` and `g_s` pairs before
it, pair `k < floor(L_s / 2)` reads `t0 = cu_seqlens[s] + 2k`, `t1 = t0 + 1` and writes row
`r = g_s + k`.  Pairs never cross a document boundary.  The last token of an odd-length
document belongs to no pair: it changes no output, and its `dC` and `dZ` rows are exactly
zero (+0.0).  The backward treats the BF16 cast as the identity and uses the pairwise form of
`dZ`, which stays accurate when the two values of a pair are close.

The facade `cake_pplx.cake_gated_kv_compression_det_train` is a stable API; this backend
package is experimental (no compatibility guarantee for its internal names).  SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the supported architectures;
another device raises `RuntimeError`.  Its libraries are loaded prebuilt from `aot/` when a
builder compiled them and JIT-built from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Opt-in

```python
from cake_pplx.cake_gated_kv_compression_det_train import (
    cake_gated_kv_compression_det,
    cake_gated_kv_compression_det_cz,
    cake_gated_kv_compression_det_train_backward,
    cake_gated_kv_compression_det_train_forward,
)

# cz: FP32 [T, 1024] (C | Z) from one projection GEMM; cu_seqlens: int32 [S + 1] on the device;
# G: the pair count, sum over documents of floor(length / 2), known to the caller.
out = cake_gated_kv_compression_det_cz(cz, cu_seqlens, G)              # autograd; grad of cz in one [T, 1024] buffer
out = cake_gated_kv_compression_det(c, z, cu_seqlens, G)               # autograd; C, Z as two [T, 512] tensors

out = cake_gated_kv_compression_det_train_forward(c, z, cu_seqlens, G)  # or out=<BF16 [G, 512]>
dc, dz = cake_gated_kv_compression_det_train_backward(dout, c, z, cu_seqlens)
grad = torch.empty(T, 1024, device="cuda")                              # dC | dZ halves of one buffer
cake_gated_kv_compression_det_train_backward(dout, c, z, cu_seqlens, dc=grad[:, :512], dz=grad[:, 512:])
```

## Operands

| Operand | Shape | Dtype | Layout |
|---|---|---|---|
| `c`, `z` | `[T, 512]` | FP32 | `stride(1) == 1`, row stride a multiple of 8 elements and at least 512, base 32-byte aligned (the column halves of one `[T, 1024]` tensor qualify) |
| `cz` | `[T, 1024]` | FP32 | `stride(1) == 1`, row stride a multiple of 8 elements and at least 1024, base 32-byte aligned |
| `cu_seqlens` | `[S + 1]` | int32 | contiguous, on the operands' device, `0 = b_0 <= ... <= b_S = T` |
| `out`, `dout` | `[G, 512]` | BF16 | contiguous, base 16-byte aligned |
| `dc`, `dz` | `[T, 512]` | FP32 | as `c` / `z` (two halves of one `[T, 1024]` buffer qualify) |
| `pair_offsets` | `[S + 1]` | int32 | contiguous: the exclusive prefix of the pair counts (`cu_pairs` below) |

`G` is the caller's: `num_pairs`, or the rows of a caller `out` / `dout`.  The host never
reads `cu_seqlens` (no device-to-host copy, no synchronization): `T`, `S` and `G` are
run-time values, no shape list, no recompilation.  The caller keeps the `cu_seqlens`
contract (non-decreasing, `cu_seqlens[S] == T`); the host cannot check it.  A forward with
`G` above the pair count stores zero rows past it; a `G` below it returns the first `G`
rows.  The backward needs `G` (the rows of `dout`) at least the pair count: rows of `dout`
past it are not read, and the gradient rows of a pair at or past `G` are not written.  The
autograd entries need `num_pairs` at least the pair count too: their gradient buffers are not
zero-filled, so with a smaller `num_pairs` the gradient rows of the tokens of the pairs at or
past it are uninitialized.

A layout the kernels cannot read or write in place (a feature stride other than 1, a row
stride off a multiple of 8, a base off its alignment), a wrong dtype, shape or device, and a
caller output that overlaps an input or another output raise `ValueError` naming the
argument before any launch.  No call copies an operand silently; the autograd backward
reads a contiguous, aligned copy of an incoming gradient of another layout.

## Kernels

| Role | Kernel | Grid |
|---|---|---|
| `pair_table` | `cu_pairs[0] = 0`, `cu_pairs[s + 1] = cu_pairs[s] + (cu_seqlens[s + 1] - cu_seqlens[s]) // 2`, int32 `[S + 1]`, by one CTA of 1024 threads with a carried running total | `(1, 1, 1)` |
| `fwd` | per output row `r < G`: the document by a binary search of `cu_pairs`, the two token rows, the FP32 math above, one BF16 rounding; rows `r >= cu_pairs[S]` stored as zeros.  With `S <= fuse_docs` (256) the launch builds the pair table itself and does not search: row `r`'s first token is `cu_seqlens[0] + 2 r` plus the number of odd-length documents before its document, counted from `cu_seqlens`, and CTA 0 writes `cu_pairs` | rows `G` |
| `bwd` | virtual rows `v < G + S`: pair `v` writes the `dC` / `dZ` rows of its two tokens, its document by a binary search of `cu_pairs`.  With `S <= 16384` the search has two levels: each CTA with pair rows copies every 16th entry of `cu_pairs` to shared memory, and a row picks its window of 16 entries there and searches only that window; row `G + s` stores +0.0 into the tail row of document `s` when its length is odd | rows `G + S` |

Each role has one registry record per architecture.  The forward has two, one kernel in two
schedules, each the faster on its architectures: record `fwd` on SM100 / SM103 reads `C` and
`Z` in 128-bit lanes (4 FP32 channels, 128 lanes a row, 2 rows per 256-thread CTA); record
`fwd_sm_107a` on SM107 reads them in 256-bit lanes (8 channels, 64 lanes a row, 4 rows per
CTA).  `pair_table` and `bwd` (256-bit lanes) are one record each on all three.  A launch
picks the record of the operands' device.

A forward call launches `fwd`, after `pair_table` only when `S` is above the forward
record's `fuse_docs` (256; `cake_jit.fuse_docs`): a forward of at most 256 documents builds
the table itself and writes it to `cu_pairs`, the same bytes.  A backward call launches
`pair_table` (not when `pair_offsets` is given, as the autograd backward does) then `bwd`.  Every launch is a
plain stream-ordered launch on the current torch stream of the operands' device; nothing
synchronizes.  Numerics: FP32 throughout, IEEE round to nearest, no flush to zero; the
logit gap is `u = z1 - z0`, `e = exp(-|u|)` (precise `expf`), the larger-logit token's
weight `1 / (1 + e)` and the other's `e / (1 + e)`, both correctly rounded divisions; the
forward value is `fma(e, c_small, c_big) / (1 + e)`.  Gate gaps of about 87.3 and more give a
subnormal or zero `e` (zero from about 104), and exact ties give `e = 1`: every output stays
finite.  No FP64 arithmetic.

## Memory

| Item | Bytes | Held |
|---|---|---|
| pair table (`cu_pairs`) | `4 (S + 1)` | workspace of an explicit forward, and of an explicit backward without `pair_offsets`; freed on return |
| autograd node | `4 (S + 1)` | its forward's pair table, retained until its backward |
| inputs the autograd node references | `4096 T + 4 (S + 1)` | `C` and `Z` (or `cz`) and `cu_seqlens`, the caller's tensors: no copy |
| incoming-gradient copy | `1024 G` | workspace of the autograd backward only when its incoming gradient is not contiguous and 16-byte aligned (for example the expanded gradient of `out.sum().backward()`); freed on return |

`cake_gated_kv_compression_det_train_memory_bytes(num_tokens, num_docs)` returns the figures
of the first three rows; it takes no `G` and leaves out the incoming-gradient copy, the only
workspace of the autograd backward.  The autograd node saves its inputs and the pair table
only (no activation, no probability); under `torch.utils.checkpoint(..., use_reentrant=False)`
the recompute rebuilds the same bits.  The outputs (`out`, `dC`, `dZ`) are allocated from
PyTorch's caching allocator unless the caller passes them.

## Determinism

This is the `det` package of the family, and its only package.  Every output element is
computed from its own pair (or token) by one fixed chain of FP32 operations: nothing is
summed across threads, CTAs or calls and no atomic is used, so `out`, `dC` and `dZ` repeat
bit for bit for identical inputs, also in the recompute of an activation checkpoint and
whatever `T`, `S`, `G`, the grid or the SM count.  No non-deterministic step exists that a
`nondet` package could trade for speed, so the family has no `nondet` module.  No entry takes
a `deterministic` keyword and nothing alerts under `torch.use_deterministic_algorithms(True)`.

The family has no KCP (context-parallel) module: the kernel is single-rank, and under
context parallelism the caller moves a boundary token so that every rank holds whole pairs.

## Limitations

- Ratio 2 and width 512 only.
- `G <= 2^26` (67108864), `S <= 2^24` (16777216), `T` within int32.
- No build for any architecture other than SM100, SM103 and SM107.
