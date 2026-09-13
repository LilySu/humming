# Baseline feasibility check — vLLM-side, no Humming kernel work needed

Source: `dev_probe/humming-ldmatrix-s4-moe-plan.md`, §2 ("Baseline feasibility
check").

Independent of any Humming kernel work, and can run in parallel with it
(does not gate or depend on the Phase 0 GPU work):

1. Load the exact target checkpoint revision through the existing,
   *unmodified* Humming backend — no new code.
2. Confirm the resolved activation dtype, zero-point representation, and
   scale grouping for both w13 and w2.
3. Record the actual expert execution variant selected (indexed /
   grouped-contiguous / grouped-masked) and the w13/w2 shapes.
4. Confirm the intended GPU setup can actually run it (driver, toolkit,
   memory).

This only needs to land before the plan's §7 kernel benchmarks (so they're
representative — separating pre-existing integration issues in the current
route from regressions a new loader would introduce) and before §8's real
vLLM validation. The full optimized-model run stays deferred to §8.
