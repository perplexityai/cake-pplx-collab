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

The GPU suites and the benchmark ran on B200 and GB200 (`sm_100a`), GB300 (`sm_103a`), VR200 and R200 (`sm_107a`).
At 0.1.2, compute-sanitizer passes on GB300 for every program (memcheck, racecheck, synccheck and initcheck) and on
VR200 for the new forwards of 5 to 9 values (the same four tools). The other programs passed on VR200 and R200 at
0.1.0; 0.1.1 then added the `dw` memset to the nondet backward bindings. 0.1.2 replaces the forwards of 5 to 9 values:
one 448-thread CTA serves each token, all 7168 columns, with no cluster, one CTA per SM; the other programs are
unchanged. At 5 to 9 values the outputs and the gradients differ slightly from 0.1.1, because the new forwards sum in
another order; within 0.1.2 they repeat bit for bit as before (the nondet `dw` makes no bit promise).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-attn-res @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/attn_res"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
