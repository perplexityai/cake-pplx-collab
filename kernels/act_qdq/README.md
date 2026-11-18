# cake-pplx-act-qdq

Cake `act_qdq` training kernels: block-scaled quantize-dequantize of BF16 activations back to BF16, in three formats
(`fp4_e4m3_g16`, `mxfp4_e8m0_g32`, `mxfp8_e8m0_g32`), on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_act_qdq_det_train`

The family has a deterministic module only (no `nondet` module, no KCP module) and a forward step only (no backward,
no autograd entry).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-act-qdq @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/act_qdq"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
