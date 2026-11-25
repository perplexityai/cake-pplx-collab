# cake-pplx-engram-gate

Cake `engram_gate` training kernels: the four-stream Engram context gate of width 5120 (forward, backward and an
autograd entry) on `sm_100a`, `sm_103a` and `sm_107a`. Public modules:

- `cake_pplx.cake_engram_gate_det_train`
- `cake_pplx.cake_engram_gate_nondet_train`

The autograd entries and the explicit forward take `dot_sum`, the forward's sum of the gate's dot product. The
default, `"fp32"`, sums in one FP32 word. `"compensated"` (opt-in) is a compensated FP32 dot product: a slower forward
that rounds fewer output elements away from the correctly rounded result where the products nearly cancel. The
backward is the same for both. Both choices meet every accuracy bound of the tests, as measured on B200, GB200, GB300
and VR200; the GB300 numbers are in [`docs/API.md`](../../docs/API.md#engram_gate).

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-engram-gate @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/engram_gate"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
