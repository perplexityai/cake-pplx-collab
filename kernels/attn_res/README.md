# cake-pplx-attn-res

Cake `attn_res` training kernels: the attention residual (AttnRes) of Kimi-K3, a softmax-weighted sum of 1 to 9 BF16
values of width 7168 (forward, backward and an autograd entry) on `sm_100a`, `sm_103a` and `sm_107a`. Public modules:

- `cake_pplx.cake_attn_res_det_train`
- `cake_pplx.cake_attn_res_nondet_train`

The values are separate BF16 `[T, 7168]` tensors (nothing is stacked), the weight `w` is FP32 `[7168]` and `eps` is a
run-time float. The forward keeps `p`, `r` and `d` (FP32 `[T, n]` each) for the backward. The nondet backward adds the
weight gradient with FP32 atomics; the det backward sums it in one fixed order. Details are in
[`docs/API.md`](../../docs/API.md#attn_res).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-attn-res @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/attn_res"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
