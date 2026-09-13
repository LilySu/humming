# Dense GEMM isn't the target — fused MoE is

Source: `dev_probe/humming-ldmatrix-s4-moe-plan.md`, intro + §6/§8.

The committed `ptx94-ldmatrix-s8-s4` branch (`9c9f063`) implements
`ldmatrix.s8.s4` for dense GEMM only. The plan is explicit that this is an
*intermediate diagnostic*, not the acceptance target — the real target is
demonstrating correctness and useful performance through vLLM's existing
fused Humming MoE backend (§8's gate: "an actual vLLM MoE run demonstrably
uses the new path for the eligible sublayers and meets the
correctness/accuracy criteria").

Humming-side MoE correctness work (§6) hasn't started yet. Relevant if
you're deciding when a vLLM-side change here is "done" — dense-GEMM support
existing on the Humming side isn't evidence the MoE path works.
