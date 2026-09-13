# _supports_quant_scheme is an admission check, not a capability proof

Source: `dev_probe/humming-ldmatrix-s4-moe-plan.md`, §2 and §8.

Don't add a `SUPPORTED_W_A` entry just to "activate the instruction." In the
current WNA16 oracle route, `_supports_quant_scheme` is called with
`activation_key=None` — Humming resolves its actual activation dtype later,
internally, not at that check. Adding
`(kInt4Static, kInt8DynamicTokenSym)` to the table only changes whether vLLM
*considers* routing to Humming; it's not proof Humming's internal dtype
resolution produces a correct, working kernel for that pairing.

Plan's stated trigger for making this change (§8): "only change the vLLM
support table when the actual intended caller supplies a rejected but
correctly supported key pair" — confirmed by tracing the real call, not
inferred from the table alone.

Treating admission-check success as capability proof (or vice versa) is
called out by name in this repo's `CLAUDE.md` as a known false-signal
failure mode.
