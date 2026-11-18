# cake-pplx-engram-gate

Cake `engram_gate` training kernels: the four-stream Engram context gate of width 5120 (forward, backward and an
autograd entry) on `sm_100a`, `sm_103a` and `sm_107a`. Public modules:

- `cake_pplx.cake_engram_gate_det_train`
- `cake_pplx.cake_engram_gate_nondet_train`

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-engram-gate @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/engram_gate"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
