# cake-pplx-act-qdq

Cake `act_qdq` training kernels: block-scaled quantize-dequantize of BF16 activations back to BF16, in three formats
(`fp4_e4m3_g16`, `mxfp4_e8m0_g32`, `mxfp8_e8m0_g32`), on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_act_qdq_det_train`

The family has a deterministic module only (no `nondet` module, no KCP module) and a forward step only (no backward,
no autograd entry).

On an R200 (`sm_107a`), `locality="domain"` runs a program whose CTAs work first on their own die's memory, for an
out-of-place call of 8,388,608 to 2,147,482,624 vectors for FP4 (`numel / 8`), to 2,147,483,136 for MXFP4 and MXFP8,
whose input and output lie in memory from `cake_act_qdq_det_train_domain_empty`, at the same offset in their 2 MB chunks
and in chunks of the same die; every other call runs the default program, and the bytes are the same.  This path needs
cuda-python: install the `locality` extra (`cake-pplx-act-qdq[locality]`).  See
[docs/API.md](../../docs/API.md#act_qdq).  On an R200 the locality tests need the extra too: install
`"./kernels/act_qdq[test,locality]"`; without cuda-python the SM107 tests of
`tests/experimental/test_cake_act_qdq_det_train_locality.py` fail.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-act-qdq @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/act_qdq"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
