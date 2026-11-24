# cake-pplx-dsa-sparse-attention

Cake `dsa_sparse_attention` training kernels. Public modules:

- `cake_pplx.cake_dsa_sparse_attention_det_train`
- `cake_pplx.cake_dsa_sparse_attention_nondet_train`

The nondet rope form takes 64, 96 or 128 query heads (96 and 128 on `sm_100a` and `sm_103a`); the det module and
the NoPE form take 64. At 96 and 128 heads the call runs the 64-head kernels on two head windows, `[H-64, H)` and then
`[0, 64)`. This costs more than one native launch: at 96 heads the forward does the work of 128 heads and copies the
windows' `lse` and `o_lo` into the outputs, and the backward gathers the keys and drains the FP32 dK/dV accumulator
once per window. The workspace grows by the window scratch and statistics; see the
[API reference](../../docs/API.md#dsa_sparse_attention) for the tables.

Install this family alone, pinned to a commit:

```bash
pip install "cake-pplx-dsa-sparse-attention @ git+https://github.com/perplexityai/cake-pplx-collab@<sha>#subdirectory=kernels/dsa_sparse_attention"
```

The repository rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
