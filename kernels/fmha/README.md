# cake-pplx-fmha

Cake `fmha` training kernels: the backward of dense multi-head attention, head dimension 64 or 128, non-causal or
causal, on `sm_100a`, `sm_103a` and `sm_107a`. Public module:

- `cake_pplx.cake_fmha_nondet_train`

The family has a non-deterministic module only (no `det` module) and a backward step only (no forward, no autograd
entry). A head-dimension 128 call whose key length is a multiple of 256 runs a main kernel on clusters of two CTAs (the
pair route) for the long sequences where it is faster on the architecture (without the mask on `sm_100a`, `sm_103a` and
`sm_107a`, with the causal mask on `sm_103a` and `sm_107a`), and a small call without the mask runs one kernel for the
whole backward (the one-launch route) for the shapes where it is faster on the architecture (head dimension 64 and 128
on `sm_100a`, `sm_103a` and `sm_107a`); `dq` may differ in its last bits run to run on the three-kernel routes, and the
module stays non-deterministic on every route.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-fmha @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/fmha"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
