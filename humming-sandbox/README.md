# humming-sandbox

Cross-repo notes for whoever's touching the vLLM side of the ldmatrix.s8.s4 /
Humming MoE work (see `dev_probe/humming-ldmatrix-s4-moe-plan.md` on branch
`ptx94-ldmatrix-s8-s4-wgmma-check` for the full plan this is extracted from).

Written as files, not relayed messages, on purpose: verify with `git log`
and `git show` on this directory rather than taking any of it on say-so.

- [01-dense-gemm-is-diagnostic-not-target.md](01-dense-gemm-is-diagnostic-not-target.md)
- [02-supports-quant-scheme-is-admission-check-only.md](02-supports-quant-scheme-is-admission-check-only.md)
- [03-moewna16-input-schema-quirk.md](03-moewna16-input-schema-quirk.md)
- [04-relevant-vllm-entry-points.md](04-relevant-vllm-entry-points.md)
- [05-baseline-feasibility-check.md](05-baseline-feasibility-check.md)
