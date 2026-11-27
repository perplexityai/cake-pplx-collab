# cake-pplx-dsa-sparse-attention

Cake `dsa_sparse_attention` training kernels. Public modules:

- `cake_pplx.cake_dsa_sparse_attention_det_train`
- `cake_pplx.cake_dsa_sparse_attention_nondet_train`

The rope forms of both modules take 64, 96 or 128 query heads on `sm_100a`, `sm_103a` and `sm_107a`; the NoPE form
takes 64. In the det module, 96 and 128 heads run the single-pass forward and the det
head-window backward builds (both 64-head windows of every token in one launch), and every output repeats bit for bit as
at 64 heads. In the nondet module, at 96 and 128 heads the call runs, where the package ships them for the device
(the forward builds on `sm_100a` and `sm_103a`, the backward builds on all three architectures), the head-window
builds: one launch per stage over both 64-head windows, `[0, 64)` and `[64, 128)`, of every token (the heads past `H`
are padding), writing the `[T, H]` outputs directly; where the package ships it (`sm_100a`, `sm_103a`, `sm_107a`), the
forward runs the single-pass forward instead: one 2-CTA cluster per token over all its heads. Where the package ships
neither for a phase, that phase runs the 64-head kernels on two head windows, `[H-64, H)` and then `[0, 64)`; this
costs more than one native launch: at 96
heads the forward does the work of 128 heads and copies the windows' `lse` and `o_lo` into the outputs, and the
backward gathers the keys and drains the FP32 dK/dV accumulator once per window, and the workspace grows by the window
scratch and statistics. Both plans add, for a `dout` that is not dense over its rows (the head-major view a trainer
hands back), a dense `dout` copy of `H` KiB per token; see the [API reference](../../docs/API.md#dsa_sparse_attention)
for the tables.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-dsa-sparse-attention @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/dsa_sparse_attention"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
