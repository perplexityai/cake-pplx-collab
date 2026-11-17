# cake-pplx-fmha

Cake `fmha` training kernels: the backward of dense multi-head attention, head dimension 64 or 128, non-causal or
causal, on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_fmha_nondet_train`

The family has a non-deterministic module only (no `det` module) and a backward step only (no forward, no autograd
entry).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-fmha @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/fmha"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
