"""Design proposal, not yet wired into the real package.

Addresses two of Codex's review points together, since they're the same
underlying problem:

  1. "Treat the layout as an explicit type/variant, not a loose boolean.
     The repacker, JIT specialization, kernel dispatch, and cached artifact
     key must all use the same layout identity."
  2. "Centralize the availability predicate: CUDA/PTX version, compile
     target, architecture-family target, runtime device, and fallback
     selection should not be independently reconstructed in several files."

Earlier drafts on this branch proposed a plain `use_ldmatrix_s4: bool` on
RepackWeightKernel, matching the existing `use_packed_k_layout`/
`use_native_dequant` boolean-flag style in humming/kernel/repack_weight.py.
That's consistent with the codebase's existing convention, but Codex's point
is sharper than a style preference: a bare bool can be independently
recomputed (correctly or not) at each of the four places that need to agree
on it — repack, JIT kernel-expr specialization, loader dispatch, and (if
repacked-weight caching across processes is ever introduced — the same
dormant risk flagged for both vLLM's PR and this one) a cache key. If any
one of those four recomputes the check slightly differently, you get
exactly the "wrong-layout buffer reaches the wrong kernel" failure mode
this same review calls out as needing a deliberate test.

So: one enum, one predicate function, both defined here once, imported
everywhere else needs them, instead of four independent boolean
computations that are supposed to stay in sync by convention.
"""

from __future__ import annotations

import enum


class WeightSmemLayout(enum.Enum):
    """Identity of the packed-weight layout a given tensor was repacked
    into. This is the thing that must travel identically through repack,
    kernel specialization, and dispatch — never re-derived independently at
    each site from (weight_bits, activation_bits, use_packed_k_layout, ...)
    tuples that could drift out of sync with each other.
    """

    LEGACY_DEQUANT_INTERLEAVED = "legacy_dequant_interleaved"
    WGMMA_A_LDMATRIX_S4 = "wgmma_a_ldmatrix_s4"


# Supported (SM family) targets for the ldmatrix.s8.s4 expanding load,
# mirroring vLLM PR #50096's CMake gate (cuda_archs_sm90plus) — kept as
# a plain tuple here, not re-derived per call site.
_LDMATRIX_S4_SM_FAMILIES = (90, 100, 103, 107, 110, 120, 121)
_LDMATRIX_S4_MIN_CUDA = (13, 4)


def ldmatrix_s4_available(sm_version: int, nvcc_version: tuple[int, int]) -> bool:
    """Single source of truth for whether the CUDA 13.4 ldmatrix.s8.s4
    expanding-load path can be used on this (toolkit, GPU) combination.

    Every call site that needs this answer — RepackWeightKernel deciding
    whether to instantiate the WGMMA_A_LDMATRIX_S4 template branch, the
    loader deciding which S2RMemoryLoaderB method to call, any future
    cache-key computation, and this probe's own gating — must call this
    function rather than reconstructing the (nvcc >= 13.4) and
    (sm_version in {...}) checks locally. That's the concrete mechanism for
    Codex's "should not be independently reconstructed in several files."
    """
    return nvcc_version >= _LDMATRIX_S4_MIN_CUDA and sm_version in _LDMATRIX_S4_SM_FAMILIES


def resolve_weight_smem_layout(
    *,
    weight_bits: int,
    activation_bits: int,
    use_packed_k_layout: bool,
    has_zero_point: bool,
    sm_version: int,
    nvcc_version: tuple[int, int],
) -> WeightSmemLayout:
    """The one place that turns (config, hardware) into a layout identity.
    Repack, kernel specialization, and dispatch should all call this and
    carry its *result* forward, not each re-derive their own boolean from
    the same inputs.
    """
    eligible = (
        use_packed_k_layout
        and weight_bits == 4
        and activation_bits == 8
        and not has_zero_point
        and ldmatrix_s4_available(sm_version, nvcc_version)
    )
    return (
        WeightSmemLayout.WGMMA_A_LDMATRIX_S4
        if eligible
        else WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED
    )


if __name__ == "__main__":
    # Quick self-check of the boundary cases — run directly, no test
    # framework needed, no GPU needed (this is pure arithmetic on inputs).
    cases = [
        # (weight_bits, activation_bits, packed_k, has_zp, sm, nvcc, expected)
        (4, 8, True, False, 90, (13, 4), WeightSmemLayout.WGMMA_A_LDMATRIX_S4),
        (4, 8, True, False, 90, (13, 3), WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED),  # toolkit too old
        (4, 8, True, False, 89, (13, 4), WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED),  # Ada, not SM90+
        (4, 8, True, True, 90, (13, 4), WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED),   # has zero-point
        (8, 8, True, False, 90, (13, 4), WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED),  # not 4-bit weight
        (4, 8, False, False, 90, (13, 4), WeightSmemLayout.LEGACY_DEQUANT_INTERLEAVED), # packed_k off
    ]
    for weight_bits, activation_bits, packed_k, has_zp, sm, nvcc, expected in cases:
        actual = resolve_weight_smem_layout(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            use_packed_k_layout=packed_k,
            has_zero_point=has_zp,
            sm_version=sm,
            nvcc_version=nvcc,
        )
        status = "ok" if actual == expected else "FAIL"
        print(f"[{status}] weight_bits={weight_bits} act_bits={activation_bits} "
              f"packed_k={packed_k} has_zp={has_zp} sm={sm} nvcc={nvcc} -> {actual.value}")
