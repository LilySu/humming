"""WGMMA m64n64k32.s32.s8.s8 fragment-layout oracle for Phase 0 Part 2.

This module answers, precisely and for every (warp, lane, register, byte),
two questions that were previously only checked in aggregate:

1. Given candidate 1b's addressing (m_subgroup = matrix_idx % 2,
   k_half = matrix_idx // 2, m_base = warp_id * 16), which real (m, k)
   position does `ldmatrix.s8.s4` physically deliver into each lane's
   a_regs[reg] byte?
2. Which (m, k) position does WGMMA's hardware A-operand register layout
   *require* to be in that same (warp, lane, reg, byte) slot?

If those two answers agree everywhere, candidate 1b's fragment mapping is
exactly correct at the per-element level -- not just "the aggregate sum
happens to match", which is all the GPU run so far has established. This
module makes that comparison directly, with no GPU involved: the ldmatrix
side is transcribed verbatim from Part 1's own hardware-validated formula
(ldmatrix_s4_layout.cu's `validate_layout` checking logic, SASS-confirmed
on real H100 hardware), not re-derived from memory; the WGMMA side is
derived from NVIDIA's tensor core register layout, cross-checked below
against NVIDIA's own documented example before being trusted.

Result of running this file directly: 0 mismatches across all 2048
(warp, lane, register, byte) combinations. See
dev_probe/WGMMA_D_FRAGMENT_LAYOUT.md for the full writeup and citations.
"""

from __future__ import annotations

import dataclasses
import sys


# ---------------------------------------------------------------------------
# WGMMA m64n64k32.s32.s8.s8 accumulator (C/D) fragment layout.
#
# Source: cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:433-435
#   template<int N>
#   using CLayout_64xN = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,Int<N/8>>>,
#                               Stride<Stride<_128,_1,_16>,Stride<_64,_8,   _512>>>;
# specialized here for N=64 (CLayout_64x64), which is CUTLASS's own tested,
# shipped production code for exactly this MMA shape family -- not a
# hand-derived guess. Decoded (see WGMMA_D_FRAGMENT_LAYOUT.md for the
# arithmetic) into:
#
#   m = 16*warp_id + (lane//4) + 8*v1
#   n = 2*(lane%4)  + v0       + 8*v2
#   register_index (0..31) = v0 + 2*v1 + 4*v2   (v0 fastest, CuTe compact order)
#
# Cross-checked against NVIDIA's own documented example (quoted via the
# PTX ISA docs / Colfax's WGMMA tutorial): "thread 0 holds the values at
# (0,0), (0,1), (8,0), (8,1) and repeated every 8 columns to the right" --
# this formula reproduces that exactly for warp_id=0, lane=0, R=0..4.
# ---------------------------------------------------------------------------


def d_fragment_mn(warp_id: int, lane: int, reg: int) -> tuple[int, int]:
    """Which (m, n) does accumulator register `reg` (0..31) of this thread hold."""
    v0 = reg % 2
    v1 = (reg // 2) % 2
    v2 = reg // 4
    m = 16 * warp_id + (lane // 4) + 8 * v1
    n = 2 * (lane % 4) + v0 + 8 * v2
    return m, n


# ---------------------------------------------------------------------------
# WGMMA m64n?k32 register-sourced A-operand layout (what a raw-PTX-a,
# register-sourced s8 operand's 4 registers x 4 bytes represent).
#
# Source: cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:454-456
#   // Register source layout for 8-bit (sparse 16-bit) value types
#   using ALayout_64x32 = Layout<Shape <Shape <  _4,_8, _4>,Shape < _4,_2,   _2>>,
#                                Stride<Stride<_256,_1,_16>,Stride<_64,_8,_1024>>>;
#
# Decoded the same way as CLayout above:
#   m = 16*warp_id + (lane//4) + 8*v1
#   k = 4*(lane%4)  + v0       + 16*v2
#   register_index (0..3) = v1 + 2*v2, byte_within_register (0..3) = v0
# ---------------------------------------------------------------------------


def wgmma_a_operand_mk(warp_id: int, lane: int, reg: int, byte: int) -> tuple[int, int]:
    """Which (m, k) does WGMMA's hardware require in (warp, lane, reg, byte)."""
    v1 = reg % 2
    v2 = reg // 2
    v0 = byte
    m = 16 * warp_id + (lane // 4) + 8 * v1
    k = 4 * (lane % 4) + v0 + 16 * v2
    return m, k


# ---------------------------------------------------------------------------
# ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4's ACTUAL physical
# distribution, transcribed (not re-derived) from Part 1's own validated
# checking logic:
#
#   dev_probe/ldmatrix_s4_layout.cu, validate_layout<4>():
#     lane = row * 4 + column / 4
#     byte = column % 4
#     actual = int8(output[lane*4 + matrix] >> (8*byte))
#     expected = nibble(matrix, row, column) - 8
#
# where `matrix` (0..3) and `row` (0..7) identify which of the 4 (8-row x
# 16-col) source tiles the data came from, and `column` (0..15) is the
# nibble's position within that tile's row. This is SASS-confirmed ground
# truth (Part 1 passed against an independently-published opcode/modifier
# encoding), not a guess -- do not re-derive it differently.
#
# wgmma_pairing_check.cu's specific way of calling this instruction (one
# ldmatrix.x4 call per warp, candidate 1b's fragment assignment) repurposes
# "matrix" and "row" to mean:
#   matrix = matrix_idx (0..3)   -- which of this warp's 4 sub-tiles
#   m_subgroup = matrix_idx % 2  -- candidate 1b
#   k_half     = matrix_idx // 2 -- candidate 1b
#   row (0..7) = row_in_matrix, directly a sub-row within m_subgroup's 8 rows
#   column (0..15) = nibble position within k_half's 16-wide K slice
# ---------------------------------------------------------------------------


def ldmatrix_actual_mk(
    out_lane: int, reg: int, byte: int, m_base: int, candidate: str = "1b"
) -> tuple[int, int]:
    """Which real (m, k) does ldmatrix.s8.s4 physically deliver here.

    `reg` here is Part 1's "matrix" (0..3): which of the 4 tiles this warp's
    ldmatrix.x4 call loaded -- i.e. this lane's own a_regs[reg].
    `out_lane` is the lane whose a_regs[] we're reading.

    `candidate` selects which matrix_idx -> (m_subgroup, k_half) assignment
    to simulate: "1b" is the one that passed on real hardware
    (m_subgroup=idx%2, k_half=idx//2); "1a" is the one that failed
    (m_subgroup=idx//2, k_half=idx%2) -- kept here as the negative control
    for item 5 (dev_probe/NEXT_STEPS.md, humming-ldmatrix-s4-moe-plan.md
    §3.5): a correctness check that can't also flag a known-wrong ordering
    as wrong isn't actually checking anything.
    """
    matrix = reg
    row = out_lane // 4
    column = 4 * (out_lane % 4) + byte
    if candidate == "1b":
        m_subgroup = matrix % 2
        k_half = matrix // 2
    elif candidate == "1a":
        m_subgroup = matrix // 2
        k_half = matrix % 2
    else:
        raise ValueError(candidate)
    m = m_base + m_subgroup * 8 + row
    k = k_half * 16 + column
    return m, k


@dataclasses.dataclass
class CrossCheckResult:
    total: int
    mismatches: list[tuple[int, int, int, int, tuple[int, int], tuple[int, int]]]

    @property
    def ok(self) -> bool:
        return not self.mismatches


def cross_check(candidate: str, num_warps: int = 4) -> CrossCheckResult:
    """Compare ldmatrix's actual delivery against WGMMA's hardware requirement.

    For every (warp, lane, reg, byte), checks that the (m,k) ldmatrix.s8.s4
    physically loads under `candidate`'s addressing equals the (m,k) WGMMA's
    A-operand register layout expects to find there. Any disagreement here
    is a real, exact bug -- not something an aggregate sum could hide.
    """
    mismatches = []
    total = 0
    for warp_id in range(num_warps):
        m_base = warp_id * 16  # candidate1_m_offset
        for lane in range(32):
            for reg in range(4):
                for byte in range(4):
                    total += 1
                    expected = wgmma_a_operand_mk(warp_id, lane, reg, byte)
                    actual = ldmatrix_actual_mk(lane, reg, byte, m_base, candidate=candidate)
                    if expected != actual:
                        mismatches.append((warp_id, lane, reg, byte, expected, actual))
    return CrossCheckResult(total=total, mismatches=mismatches)


def cross_check_candidate_1b(num_warps: int = 4) -> CrossCheckResult:
    return cross_check("1b", num_warps=num_warps)


def _verify_clayout_bijection_and_example() -> None:
    """Sanity checks run at import/main time, not just asserted in prose."""
    cells: dict[tuple[int, int], tuple[int, int, int]] = {}
    for warp_id in range(4):
        for lane in range(32):
            for reg in range(32):
                mn = d_fragment_mn(warp_id, lane, reg)
                assert mn not in cells, f"CLayout collision at {mn}: {cells[mn]} vs {(warp_id, lane, reg)}"
                cells[mn] = (warp_id, lane, reg)
    assert len(cells) == 64 * 64, f"CLayout gap: only {len(cells)}/{64*64} cells covered"

    # NVIDIA's documented example: "thread 0 holds the values at (0,0),
    # (0,1), (8,0), (8,1) and repeated every 8 columns to the right."
    expected_example = {0: (0, 0), 1: (0, 1), 2: (8, 0), 3: (8, 1), 4: (0, 8)}
    for reg, mn in expected_example.items():
        got = d_fragment_mn(warp_id=0, lane=0, reg=reg)
        assert got == mn, f"CLayout documented-example check failed at reg={reg}: got {got}, want {mn}"

    alayout_cells: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for warp_id in range(4):
        for lane in range(32):
            for reg in range(4):
                for byte in range(4):
                    mk = wgmma_a_operand_mk(warp_id, lane, reg, byte)
                    assert mk not in alayout_cells, f"ALayout collision at {mk}"
                    alayout_cells[mk] = (warp_id, lane, reg, byte)
    assert len(alayout_cells) == 64 * 32, f"ALayout gap: only {len(alayout_cells)}/{64*32} cells covered"


if __name__ == "__main__":
    _verify_clayout_bijection_and_example()
    print("CLayout_64x64 / ALayout_64x32: bijection + documented-example checks passed")
    print()

    for candidate in ("1b", "1a"):
        result = cross_check(candidate)
        print(f"candidate {candidate} cross-check: {result.total} (warp,lane,reg,byte) "
              f"combinations checked, {len(result.mismatches)} mismatches")
        if result.mismatches:
            for warp_id, lane, reg, byte, expected, actual in result.mismatches[:5]:
                print(f"  warp={warp_id} lane={lane} reg={reg} byte={byte}: "
                      f"wgmma expects {expected}, ldmatrix delivers {actual}")
        print()

    result_1b = cross_check("1b")
    result_1a = cross_check("1a")
    if result_1b.ok and not result_1a.ok:
        print("PASSED: candidate 1b matches exactly (0/2048 mismatches) and candidate 1a "
              f"does not ({len(result_1a.mismatches)}/2048 mismatches) -- the cross-check "
              "genuinely discriminates a known-correct ordering from a known-wrong one, "
              "not just reporting PASS regardless of input (item 5's negative control, "
              "at the derivation level -- the GPU-side compile-time negative control in "
              "wgmma_pairing_check.cu's WRONG_FRAGMENT_ORDERING build is the empirical "
              "counterpart of this same check).")
    else:
        print("UNEXPECTED: either candidate 1b failed its own check, or candidate 1a "
              "passed when it shouldn't have -- the discriminating power of this check "
              "itself is in question, investigate before trusting either result.")
        # BUG (found by external review): this branch used to only print and
        # fall through to a normal (zero) exit code, so a CI/driver script
        # checking this process's exit status would see PASS regardless of
        # outcome -- exactly the class of silent-false-pass bug this whole
        # file exists to rule out for the ldmatrix/WGMMA layout question
        # itself. Fixed by actually failing the process.
        sys.exit(1)
