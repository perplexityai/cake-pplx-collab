# cake-pplx-attn-res

Cake `attn_res` training kernels: the attention residual (AttnRes) of Kimi-K3, a softmax-weighted sum of 1 to 9 BF16
values of width 7168 (forward, backward and an autograd entry) on `sm_100a`, `sm_103a` and `sm_107a`. Public modules:

- `cake_pplx.cake_attn_res_det_train`
- `cake_pplx.cake_attn_res_nondet_train`

The values are separate BF16 `[T, 7168]` tensors (nothing is stacked) and the weight `w` is FP32 `[7168]`. `eps` is a
run-time Python float or int (1e-5 by default), finite and at least 2**-84 (another type is a `TypeError`), so an
all-zero value row gives a finite result. The forward keeps `p`, `r` and `d` (FP32 `[T, n]` each) for the backward. Caller-owned `out` and `dv` tensors
are contiguous and 16-byte aligned, and must not overlap the operands or each other. `T = 0` launches no Cake kernel.

The det backward sums the weight gradient in one fixed order, in a workspace of one FP32 `[7168]` row per cluster of
its grid (the size depends on `n`). In the nondet backward each CTA adds its tokens' weight-gradient sums into `dw`
once, with FP32 atomic reductions, so `dw` makes no bit promise. Under `torch.use_deterministic_algorithms(True)` a
nondet backward of `T > 0` tokens alerts; the forward and the det module never do. Details are in
[`docs/API.md`](../../docs/API.md#attn_res).

The GPU suites ran on B200 and GB200 (`sm_100a`), GB300 (`sm_103a`), VR200 and R200 (`sm_107a`), the benchmark on
B200, GB300, VR200 and R200. On VR200 and R200 the `sm_107a` kernels also pass compute-sanitizer (memcheck, racecheck,
synccheck and initcheck).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-attn-res @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/attn_res"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
