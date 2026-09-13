# MoeWNA16Config doesn't take the usual input-schema path

Source: `dev_probe/humming-ldmatrix-s4-moe-plan.md`, §2.

`MoeWNA16Config`'s branch supplies a default `HummingInputSchema` explicitly.
It does *not* go through the same environment-derived input-schema branch
that the ordinary GPTQ preparation path uses.

Don't assume all Humming entry routes behave alike — select and verify the
intended checkpoint adapter for whichever route is actually being exercised,
rather than generalizing from the GPTQ path's behavior.
