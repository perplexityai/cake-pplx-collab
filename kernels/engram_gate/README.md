# cake-pplx-engram-gate

Cake `engram_gate` training kernels: the four-stream Engram context gate of width 5120 (forward, backward and an
autograd entry) on `sm_100a`, `sm_103a` and `sm_107a`. Public modules:

- `cake_pplx.cake_engram_gate_det_train`
- `cake_pplx.cake_engram_gate_nondet_train`

Every entry takes `dot_sum`, the sum of the gate's dot product. The default, `"compensated"`, is a compensated FP32
dot product and meets every accuracy bound of the tests. `"fp32"` (opt-in) sums in one FP32 word; where the products
nearly cancel, its `dx`, `dkey` and weight gradients are about 1e-2 off in relative L2 error, more than the bounds
allow. The measured numbers are in [`docs/API.md`](../../docs/API.md#engram_gate).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-engram-gate @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/engram_gate"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
