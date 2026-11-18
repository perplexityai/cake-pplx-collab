# cake-pplx-clamped-swiglu

Cake `clamped_swiglu` training kernels: the clamped SwiGLU of a routed MoE expert, forward and backward, with the FP32
route weight applied before the one BF16 rounding (intermediate width 2304), on `sm_100a`, `sm_103a` and `sm_107a`.
Public module:

- `cake_pplx.cake_clamped_swiglu_det_train`

The family has a deterministic module only (no `nondet` module): every output is elementwise or a sum within one row.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-clamped-swiglu @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/clamped_swiglu"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
