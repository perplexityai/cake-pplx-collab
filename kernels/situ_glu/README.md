# cake-pplx-situ-glu

Cake `situ_glu` training kernels: the SiTU-GLU activation of an MLP, forward and backward, from BF16 gate and up
inputs with FP32 math and one BF16 rounding of each output, at any width that is a multiple of 256 (3072, 6144 and
33,792 among them), on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_situ_glu_det_train`

The family has a deterministic module only (no `nondet` module): every output is elementwise.

The kernels compile on first use: a JIT build needs a CUDA toolkit of at least 12.8 for `sm_100a`, 12.9 for `sm_103a`
and 13.5 for `sm_107a`.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-situ-glu @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/situ_glu"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
