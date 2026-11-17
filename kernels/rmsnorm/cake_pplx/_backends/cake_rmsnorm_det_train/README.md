# cake_rmsnorm_det_train

Deterministic BF16 RMSNorm training kernels for SM100 (`sm_100a`), SM103 (`sm_103a`) and
SM107 (`sm_107a`). The forward writes `y`, `rstd` and, with a residual, `h_new`; the
backward writes `dx` and `dw`. Each route (architecture, hidden size, direction) selects
one program per call from the token count `T` (`cake_windows.PROGRAM_WINDOWS`).

## What repeats bit for bit

Two calls with identical inputs and the same hidden size, token count `T`, weight dtype
and residual setting, on devices of one architecture with the same SM count, return
identical bits in every output. No output depends on the order in which CTAs run: the
backward CTAs claim chunks of rows through counters, but each chunk writes its partial
`dw` row to its own slot, and the reduce sums the slots in a fixed order. Recomputing the
forward (activation checkpointing) gives the same `y`, `rstd` and `h_new` bits, so the
backward after a recomputation matches the backward after the saved forward.

| Output | Repeats bit for bit | What can change its bits between two calls |
| --- | --- | --- |
| `y` | yes | `T` across a forward window edge where the per-row thread group changes |
| `rstd` | yes | `T` across such a forward window edge |
| `h_new` | yes | nothing: `x + residual` rounded once to BF16 |
| `dx` | yes | `T` across a window edge whose class in `cake_windows.EDGES` is not `same` |
| `dw` | yes | `T` (it sums every row); the SM count, outside the column-task windows |

`cake_windows.EDGES` gives the measured class of the outputs at every window edge and
`cake_windows.NOTES` the differences measured there.

## Accuracy

Statistics are computed in FP32. Every sum is one FP32 word, except the chunk sums of
`dw` in the windows the tables below mark `compensated`, which add each block of rows
into an FP32 pair (a sum and its compensation word). `dx` is one BF16 rounding of FP32
math. With a residual, `h_new` is rounded to BF16 before the normalization reads it. The
package tests hold no tolerance. They judge every floating output (`y`, `rstd` and
`h_new` of the forward, `dx` and `dw` of the backward, with a BF16 and with an FP32
weight) by its error against an FP64 reference of the same math, beside the error of the
naive PyTorch implementation (the same ops in BF16 on the same BF16 inputs) against that
reference, over 32 independent random inputs of each case:

- in the L2 norm and in the max norm, the median and the largest error over the inputs
  are at most 1.005 times the naive implementation's;
- the error is 0 on every input where the naive implementation's error is 0.

## dw summation order per window

The backward splits the rows of a call over `ctas_per_sm` x SM count slots and cuts each
slot's share into chunks by the window's chunk table: one chunk per slot, or a halving
tail (`tail_div`, `max_halving`). Each chunk sums its rows in FP32 into one FP32 partial
row of `dw`, in row order or over blocks of rows; a `compensated` window adds each
block's sums into an FP32 pair and stores the sum less its compensation word as the
partial row. The reduce program then adds the partial rows of each column slab in FP32
in a fixed order (`chain` or `tree`), one FP32 word per sum. A row-strided window gives
slot `s` the rows `s`, `s + n`, `s + 2n`, ... with `n` = min(`T`, slots), and each slot
sums its rows plainly in row order into its partial row; where its kernel runs beside a
side stream, the window leaves eight SMs out of the slot count (`ctas_per_sm` x (SM
count - 8)). A column-task window sums each column over every row in one CTA in a fixed
order (FP32 sums per row lane, the lanes' sums added in FP32), so there `dw` does not
depend on the SM count. The FP32-weight routes use the same windows and orders, except
the `_w32` routes the tables list: they keep the programs of windows where the BF16
route runs a program without an FP32-weight build, or run another program's FP32-weight
build where that one is faster with an FP32 weight. Program and reduce names are the
first eight hex digits of the program name.

### sm_100a

| Route | Rows `T` | Program | Partial sums | Chunk table | Reduce |
| --- | --- | --- | --- | --- | --- |
| `bwd_h1536` | (0, 1] | f6233ff8 | column task | - | - |
| `bwd_h1536` | (1, 768] | 4176a24f | column task | - | - |
| `bwd_h1536` | (768, 1024] | 7dc29629 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree) |
| `bwd_h1536` | (1024, 4096] | c1a9efdf | plain | tail_div 6, max_halving 5 | 1863320f (tree) |
| `bwd_h1536` | (4096, 32768] | 21c8bb55 | plain | tail_div 12, max_halving 2 | 1863320f (tree) |
| `bwd_h1536` | (32768, 36352] | b457c4d1 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree) |
| `bwd_h1536` | (36352, inf) | b457c4d1 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree) |
| `bwd_h2048` | (0, 1] | b13b0c73 | column task | - | - |
| `bwd_h2048` | (1, 768] | 923e668c | column task | - | - |
| `bwd_h2048` | (768, 1024] | cc26e316 | plain | row-strided slots | 880768a0 (tree) |
| `bwd_h2048` | (1024, 2047] | c73074cf | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_h2048` | (2047, 22528] | cc26e316 | plain | row-strided slots | 880768a0 (tree) |
| `bwd_h2048` | (22528, 58880] | ceeef043 | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_h2048` | (58880, inf) | ceeef043 | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_w32_h2048` | (0, 1] | d22b3e8c | column task | - | - |
| `bwd_w32_h2048` | (1, 768] | 9030dbeb | column task | - | - |
| `bwd_w32_h2048` | (768, 1024] | 07f9f2fb | plain | tail_div 12, max_halving 5 | 880768a0 (tree) |
| `bwd_w32_h2048` | (1024, 16384] | b187e598 | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_w32_h2048` | (16384, 58880] | 7e6a0b63 | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_w32_h2048` | (58880, inf) | 7e6a0b63 | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_h256` | (0, 87] | 6959ea32 | column task | - | - |
| `bwd_h256` | (87, 2048] | a385ae16 | plain | tail_div 4, max_halving 5 | 72138dce (chain) |
| `bwd_h256` | (2048, 5120] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (5120, 12288] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (12288, 40960] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (40960, 80896] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (80896, 1572864] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (1572864, inf) | a385ae16 | plain | tail_div 4, max_halving 5 | 72138dce (chain) |
| `bwd_h4096` | (0, 203] | 4822793e | column task | - | - |
| `bwd_h4096` | (203, 256] | 3e644ea1 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree) |
| `bwd_h4096` | (256, 1024] | 59270fb2 | plain | tail_div 4, max_halving 5 | 5d75ea5f (tree) |
| `bwd_h4096` | (1024, 22528] | 5572b267 | plain | row-strided slots, 8 SMs left out | 5d75ea5f (tree) |
| `bwd_h4096` | (22528, 32768] | 5abcdc29 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree) |
| `bwd_h4096` | (32768, 44032] | 35a0b0ca | plain | tail_div 12, max_halving 5 | fcbfa14d (chain) |
| `bwd_h4096` | (44032, inf) | 35a0b0ca | plain | tail_div 12, max_halving 5 | fcbfa14d (chain) |
| `bwd_h512` | (0, 1] | f283bf47 | column task | - | - |
| `bwd_h512` | (1, 227] | 931757c8 | column task | - | - |
| `bwd_h512` | (227, 1024] | 09601d42 | plain | tail_div 12, max_halving 5 | 7097069b (tree) |
| `bwd_h512` | (1024, 4096] | 123db0b2 | plain | tail_div 12, max_halving 5 | fef03ce5 (tree) |
| `bwd_h512` | (4096, 32768] | ad1a5a3c | plain | tail_div 6, max_halving 5 | 7097069b (tree) |
| `bwd_h512` | (32768, inf) | 8877322a | plain | tail_div 12, max_halving 5 | 122721bd (tree) |
| `bwd_h5120` | (0, 1] | 080e0a6c | column task | - | - |
| `bwd_h5120` | (1, 256] | dd3a3573 | column task | - | - |
| `bwd_h5120` | (256, 6144] | b482c84d | plain | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_h5120` | (6144, 50688] | b482c84d | plain | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_h5120` | (50688, inf) | b482c84d | plain | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_w32_h5120` | (0, 1] | 747d86f3 | column task | - | - |
| `bwd_w32_h5120` | (1, 256] | 747d86f3 | column task | - | - |
| `bwd_w32_h5120` | (256, 6144] | 40227b2a | plain | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_w32_h5120` | (6144, 50688] | 9479b129 | compensated, 8-row blocks | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_w32_h5120` | (50688, inf) | 40227b2a | plain | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_h6144` | (0, 384] | f2271417 | column task | - | - |
| `bwd_h6144` | (384, 767] | 5c804aed | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_h6144` | (767, 4096] | 61421bc7 | plain | row-strided slots, 8 SMs left out | 5f5cdc51 (tree) |
| `bwd_h6144` | (4096, 42496] | b93bf060 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_h6144` | (42496, inf) | b93bf060 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_h6144` | (0, 384] | 8a280a7e | column task | - | - |
| `bwd_w32_h6144` | (384, 4096] | 41b80b6a | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_h6144` | (4096, 42496] | 62beb52f | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_h6144` | (42496, inf) | 62beb52f | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_residual_h6144` | (0, 1152] | 864e84f3 | column task | - | - |
| `bwd_residual_h6144` | (1152, 42496] | c1a3be94 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_residual_h6144` | (42496, inf) | 84385468 | compensated, 8-row blocks | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_residual_h6144` | (0, 1152] | 71d36617 | column task | - | - |
| `bwd_w32_residual_h6144` | (1152, 42496] | ab0e20fe | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_residual_h6144` | (42496, inf) | ab0e20fe | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |

### sm_103a

| Route | Rows `T` | Program | Partial sums | Chunk table | Reduce |
| --- | --- | --- | --- | --- | --- |
| `bwd_h1536` | (0, 1] | e254714a | column task | - | - |
| `bwd_h1536` | (1, 31] | 4aef8d1f | column task | - | - |
| `bwd_h1536` | (31, 214] | 4aef8d1f | column task | - | - |
| `bwd_h1536` | (214, 1024] | 7dc29629 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree) |
| `bwd_h1536` | (1024, 4096] | c1a9efdf | plain | tail_div 6, max_halving 5 | 1863320f (tree) |
| `bwd_h1536` | (4096, 14928] | 1df7db17 | plain | tail_div 12, max_halving 5 | 9b727c2c (tree) |
| `bwd_h1536` | (14928, 15566] | 4df08ce5 | plain | tail_div 12, max_halving 2 | 1863320f (tree) |
| `bwd_h1536` | (15566, 40448] | f522f5b8 | plain | tail_div 12, max_halving 5 | 1863320f (tree) |
| `bwd_h1536` | (40448, inf) | 9cd3c766 | 16-row blocks | tail_div 12, max_halving 5 | 1863320f (tree) |
| `bwd_h2048` | (0, 1] | a1c71cc3 | column task | - | - |
| `bwd_h2048` | (1, 2] | ed36a8d9 | column task | - | - |
| `bwd_h2048` | (2, 171] | ed36a8d9 | column task | - | - |
| `bwd_h2048` | (171, 1024] | caf6978e | plain | tail_div 12, max_halving 5 | 880768a0 (tree) |
| `bwd_h2048` | (1024, 16384] | cc26e316 | plain | row-strided slots, 8 SMs left out | 880768a0 (tree) |
| `bwd_h2048` | (16384, 32768] | caf6978e | plain | tail_div 12, max_halving 5 | 596b457a (tree) |
| `bwd_h2048` | (32768, 55296] | 302f64db | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_h2048` | (55296, inf) | 302f64db | plain | tail_div 12, max_halving 5 | d0960be9 (tree) |
| `bwd_h256` | (0, 1680] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (1680, 2048] | a385ae16 | plain | tail_div 4, max_halving 5 | 72138dce (chain) |
| `bwd_h256` | (2048, 5120] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (5120, 12288] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (12288, 40960] | af005462 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (40960, 99840] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (99840, 720896] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (720896, 1572864] | a385ae16 | plain | tail_div 4, max_halving 5 | affa1b95 (chain, column-split) |
| `bwd_h256` | (1572864, inf) | a385ae16 | plain | tail_div 4, max_halving 5 | 72138dce (chain) |
| `bwd_h4096` | (0, 512] | 5efbf78a | column task | - | - |
| `bwd_h4096` | (512, 1024] | ad2bb26e | plain | tail_div 6, max_halving 5 | 5d75ea5f (tree) |
| `bwd_h4096` | (1024, 24576] | 5572b267 | plain | row-strided slots, 8 SMs left out | 5d75ea5f (tree) |
| `bwd_h4096` | (24576, 32768] | 5abcdc29 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree) |
| `bwd_h4096` | (32768, 48640] | 35a0b0ca | plain | tail_div 12, max_halving 5 | fcbfa14d (chain) |
| `bwd_h4096` | (48640, 51200] | 24643edc | plain | tail_div 4, max_halving 2 | 5d75ea5f (tree) |
| `bwd_h4096` | (51200, inf) | 24643edc | plain | tail_div 4, max_halving 2 | 5d75ea5f (tree) |
| `bwd_w32_h4096` | (0, 1] | 27615d0e | column task | - | - |
| `bwd_w32_h4096` | (1, 512] | ff7de8d3 | column task | - | - |
| `bwd_w32_h4096` | (512, 1024] | 16545056 | plain | tail_div 6, max_halving 5 | 5d75ea5f (tree) |
| `bwd_w32_h4096` | (1024, 24576] | 47a01388 | plain | row-strided slots, 8 SMs left out | 5d75ea5f (tree) |
| `bwd_w32_h4096` | (24576, 32768] | a8eb7547 | plain | tail_div 12, max_halving 5 | 5d75ea5f (tree) |
| `bwd_w32_h4096` | (32768, 48640] | 626aed74 | plain | tail_div 12, max_halving 5 | fcbfa14d (chain) |
| `bwd_w32_h4096` | (48640, 51200] | cb164a27 | compensated, 8-row blocks | tail_div 4, max_halving 2 | 5d75ea5f (tree) |
| `bwd_w32_h4096` | (51200, inf) | 78c26a53 | plain | tail_div 4, max_halving 2 | 5d75ea5f (tree) |
| `bwd_h512` | (0, 1] | f283bf47 | column task | - | - |
| `bwd_h512` | (1, 4] | bdfd7789 | column task | - | - |
| `bwd_h512` | (4, 227] | bdfd7789 | column task | - | - |
| `bwd_h512` | (227, 1024] | 09601d42 | plain | tail_div 12, max_halving 5 | 7097069b (tree) |
| `bwd_h512` | (1024, 4096] | 8e7f4d81 | plain | tail_div 12, max_halving 5 | 7097069b (tree) |
| `bwd_h512` | (4096, 32768] | 274b0055 | plain | tail_div 6, max_halving 5 | 122721bd (tree) |
| `bwd_h512` | (32768, inf) | cfaae3ce | plain | tail_div 12, max_halving 5 | 122721bd (tree) |
| `bwd_h5120` | (0, 64] | dd3a3573 | column task | - | - |
| `bwd_h5120` | (64, 5120] | 2fc4932f | plain | tail_div 4, max_halving 2 | c5831b29 (tree) |
| `bwd_h5120` | (5120, inf) | 2fc4932f | plain | tail_div 4, max_halving 2 | c5831b29 (tree) |
| `bwd_h6144` | (0, 256] | f2271417 | column task | - | - |
| `bwd_h6144` | (256, 767] | 5c804aed | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_h6144` | (767, 4096] | 61421bc7 | plain | row-strided slots, 8 SMs left out | 5f5cdc51 (tree) |
| `bwd_h6144` | (4096, 48640] | 322574af | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_h6144` | (48640, inf) | 322574af | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_h6144` | (0, 1] | b3e6531e | column task | - | - |
| `bwd_w32_h6144` | (1, 256] | 8a280a7e | column task | - | - |
| `bwd_w32_h6144` | (256, 4096] | 41b80b6a | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_h6144` | (4096, 48640] | 9a8c0396 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_w32_h6144` | (48640, inf) | 9a8c0396 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_residual_h6144` | (0, 1216] | 864e84f3 | column task | - | - |
| `bwd_residual_h6144` | (1216, 48640] | 04ebec87 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |
| `bwd_residual_h6144` | (48640, inf) | 04ebec87 | plain | tail_div 12, max_halving 5 | 5f5cdc51 (tree) |

### sm_107a

| Route | Rows `T` | Program | Partial sums | Chunk table | Reduce |
| --- | --- | --- | --- | --- | --- |
| `bwd_h1536` | (0, 1] | c3b1b46a | column task | - | - |
| `bwd_h1536` | (1, 576] | 1207b056 | column task | - | - |
| `bwd_h1536` | (576, 1024] | 7dc29629 | plain | tail_div 12, max_halving 5 | 628248fe (tree) |
| `bwd_h1536` | (1024, 16384] | 6a428da2 | plain | tail_div 12, max_halving 2 | db5c0b67 (tree) |
| `bwd_h1536` | (16384, 45056] | 56d26996 | plain | tail_div 12, max_halving 5 | db5c0b67 (tree) |
| `bwd_h1536` | (45056, inf) | 411e270c | 4-row blocks | tail_div 12, max_halving 5 | db5c0b67 (tree) |
| `bwd_h2048` | (0, 1] | 92def3f9 | column task | - | - |
| `bwd_h2048` | (1, 256] | 6720b4a1 | column task | - | - |
| `bwd_h2048` | (256, 1024] | 845cdcd8 | plain | tail_div 12, max_halving 5 | 271082c5 (tree) |
| `bwd_h2048` | (1024, 16384] | 81c3d8ab | plain | tail_div 12, max_halving 2 | c4d8be67 (tree) |
| `bwd_h2048` | (16384, 50688] | 845cdcd8 | plain | tail_div 12, max_halving 5 | c4d8be67 (tree) |
| `bwd_h2048` | (50688, inf) | 21ee6fc9 | 8-row blocks | tail_div 12, max_halving 5 | c4d8be67 (tree) |
| `bwd_h256` | (0, 1572864] | a385ae16 | plain | tail_div 4, max_halving 5 | 312bbe08 (tree) |
| `bwd_h256` | (1572864, inf) | a385ae16 | plain | tail_div 4, max_halving 5 | 312bbe08 (tree) |
| `bwd_h4096` | (0, 256] | c1de7e41 | column task | - | - |
| `bwd_h4096` | (256, 1024] | a647fccc | plain | tail_div 12, max_halving 5 | 69bf6e10 (tree) |
| `bwd_h4096` | (1024, 4096] | d41a514f | plain | tail_div 4, max_halving 5 | 2030d808 (tree) |
| `bwd_h4096` | (4096, 24064] | ee4a9053 | plain | tail_div 12, max_halving 5 | 10e24516 (tree) |
| `bwd_h4096` | (24064, 50176] | e9e21bbe | 8-row blocks | tail_div 12, max_halving 5 | 10e24516 (tree) |
| `bwd_h4096` | (50176, inf) | e9e21bbe | 8-row blocks | tail_div 12, max_halving 5 | 10e24516 (tree) |
| `bwd_h512` | (0, 1] | f283bf47 | column task | - | - |
| `bwd_h512` | (1, 768] | b143f616 | column task | - | - |
| `bwd_h512` | (768, 1024] | 3feed502 | plain | tail_div 12, max_halving 5 | f94861c3 (chain) |
| `bwd_h512` | (1024, 4096] | 3feed502 | plain | tail_div 12, max_halving 5 | d75459d9 (tree) |
| `bwd_h512` | (4096, 16384] | dfaf378a | plain | tail_div 12, max_halving 5 | 8f032213 (tree) |
| `bwd_h512` | (16384, inf) | eb141cc9 | plain | tail_div 4, max_halving 5 | 8f032213 (tree) |
| `bwd_h5120` | (0, 1] | 080e0a6c | column task | - | - |
| `bwd_h5120` | (1, 256] | dd3a3573 | column task | - | - |
| `bwd_h5120` | (256, 1280] | 080e0a6c | column task | - | - |
| `bwd_h5120` | (1280, 4096] | 838c077d | plain | tail_div 12, max_halving 5 | 2af4cf2f (tree) |
| `bwd_h5120` | (4096, 19968] | 6816fb80 | plain | tail_div 4, max_halving 5 | 2af4cf2f (tree) |
| `bwd_h5120` | (19968, inf) | 47da668e | 8-row blocks | tail_div 4, max_halving 5 | c5831b29 (tree) |
| `bwd_h6144` | (0, 1] | 0dafdf09 | column task | - | - |
| `bwd_h6144` | (1, 256] | dffe53d5 | column task | - | - |
| `bwd_h6144` | (256, 1664] | 0dafdf09 | column task | - | - |
| `bwd_h6144` | (1664, 19456] | 5d292172 | plain | tail_div 12, max_halving 5 | fb3f2a96 (tree) |
| `bwd_h6144` | (19456, inf) | 9cc51adf | 8-row blocks | tail_div 12, max_halving 5 | fb3f2a96 (tree) |
| `bwd_residual_h6144` | (0, 368] | ee1be07a | column task | - | - |
| `bwd_residual_h6144` | (368, 19456] | 36cad444 | plain | tail_div 12, max_halving 5 | fb3f2a96 (tree) |
| `bwd_residual_h6144` | (19456, inf) | 2f149999 | 8-row blocks | tail_div 12, max_halving 5 | fb3f2a96 (tree) |
