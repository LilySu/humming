# WGMMA m64n64k32.s32.s8.s8 fragment layout — derivation and verification

Phase 0 follow-up item 2/3 (`dev_probe/NEXT_STEPS.md`, `humming-ldmatrix-s4-moe-plan.md` §3): replace the aggregate-sum check with an exact, per-element one. That requires knowing precisely which (m, n) each accumulator register holds, and which (m, k) each A-operand register holds — not just trusting that a sum came out right.

This document is the citation trail for `dev_probe/wgmma_fragment_layout.py`. Every formula in that module is either transcribed verbatim from already-validated code, or derived from a primary source and cross-checked computationally — none of it is asserted from memory alone.

## Why not just read NVIDIA's prose docs directly

Tried first. `WebFetch` against the PTX ISA page (`docs.nvidia.com/cuda/parallel-thread-execution`, sections 9.7.16.5.1.1.1–.3, "Matrix Fragments for `wgmma.mma_async.m64nNk32`") repeatedly failed to return the actual table — the page is large enough that the fetch's summarization step truncates before reaching that section. A web search surfaced a paraphrase (via Colfax Research's WGMMA tutorial) of one documented example: *"thread 0 holds the values at (0,0), (0,1), (8,0), (8,1) and repeated every 8 columns to the right."* That's real and useful (used as a cross-check below), but it's an example, not a formula — not enough to build a checker on by itself.

## Primary source used instead: CUTLASS's own production code

`cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp` is NVIDIA's own open-source, shipped, tested implementation of exactly this instruction family. Two layouts from it:

**Accumulator (C/D) layout, lines 433-435:**
```cpp
template<int N>
using CLayout_64xN = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,Int<N/8>>>,
                            Stride<Stride<_128,_1,_16>,Stride<_64,_8,   _512>>>;
```

**Register-sourced A-operand layout (8-bit values), lines 454-456:**
```cpp
// Register source layout for 8-bit (sparse 16-bit) value types
using ALayout_64x32 = Layout<Shape <Shape <  _4,_8, _4>,Shape < _4,_2,   _2>>,
                             Stride<Stride<_256,_1,_16>,Stride<_64,_8,_1024>>>;
```

Both use CuTe's standard `Layout<Shape, Stride>` convention: a coordinate `(t0,t1,t2,v0,v1,v2)` (nested exactly like the Shape) maps to a flat index via `index = Σ coord[i] * stride[i]`. The first nested shape (`Shape<_4,_8,_4>`, 4·8·4 = 128) is the *thread* mode — matching a warpgroup's 128 threads exactly. The second is the *value* mode — how many registers/bytes each thread holds.

## Decoding CLayout_64x64 (N=64)

Substituting the strides and separating which terms land in the low 6 bits (mod 64, → M) versus the high bits (→ N, since the tile is 64×64):

- `16*t2`, `t1`, `8*v1` are all `< 64` on their own and sum to at most `48+7+8=63` → these are the **M** contribution.
- `128*t0`, `64*v0`, `512*v2` are all multiples of 64 → these divide out to the **N** contribution: `2*t0 + v0 + 8*v2`.

With the warpgroup thread decomposition `tid = warp_id*32 + lane`, and CuTe's compact (first-listed-fastest) coordinate convention: `t0 = lane%4`, `t1 = lane//4`, `t2 = warp_id`. So:

```
m = 16*warp_id + (lane//4) + 8*v1
n = 2*(lane%4)  + v0       + 8*v2
register_index (0..31) = v0 + 2*v1 + 4*v2
```

**Verified two ways**, both in `wgmma_fragment_layout.py`'s `__main__`:
1. **Bijection check** — enumerating all `(warp_id, lane, reg)` combinations produces exactly 4096 distinct `(m,n)` pairs, no collisions, no gaps (a real accumulator layout must be a bijection over the tile; this confirms the formula isn't just "an" answer but a *complete, non-overlapping* one).
2. **Documented-example check** — `d_fragment_mn(warp_id=0, lane=0, reg=0..4)` reproduces `(0,0), (0,1), (8,0), (8,1), (0,8)` exactly, matching NVIDIA's own quoted example.

## Decoding ALayout_64x32 the same way

Same method: `t1`, `16*t2`, `8*v1` are the **M** terms (max `7+48+8=63`); `256*t0`, `64*v0`, `1024*v2` are multiples of 64, dividing out to **K**: `4*t0 + v0 + 16*v2`.

```
m = 16*warp_id + (lane//4) + 8*v1
k = 4*(lane%4)  + v0        + 16*v2
register_index (a_regs[0..3]) = v1 + 2*v2
byte_within_register (0..3)   = v0
```

Note the `m` formula is *identical* to CLayout's — expected, since the A operand and the accumulator must agree on which physical rows a given thread owns. Verified the same way: bijection over all 2048 `(m,k)` cells, zero collisions/gaps.

## The actual cross-check: does candidate 1b deliver what WGMMA expects

Part 1 (`ldmatrix_s4_layout.cu`) already validated, at the SASS level against an independently-published opcode/modifier encoding, exactly what `ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4` physically does. Its own checking code (`validate_layout<4>`) states this precisely:

```cpp
int lane = row * 4 + column / 4;
int byte = column % 4;
// output[lane*4 + matrix] holds nibble(matrix, row, column) - 8, at byte `byte`
```

`wgmma_pairing_check.cu`'s candidate 1b calls this instruction with lane-dependent addresses that repurpose "matrix"/"row" to mean `m_subgroup = matrix_idx % 2`, `k_half = matrix_idx // 2`, `row = row_in_matrix`. Composing that addressing scheme with Part 1's transcribed (not re-derived) output-distribution formula gives, for any `(warp_id, out_lane, reg, byte)`, the *actual* `(m, k)` that ends up there — this is `ldmatrix_actual_mk()` in the module.

`cross_check_candidate_1b()` then directly compares, for all `4 warps × 32 lanes × 4 registers × 4 bytes = 2048` combinations, `ldmatrix_actual_mk(...)` against `wgmma_a_operand_mk(...)` (what CUTLASS's `ALayout_64x32` says WGMMA's hardware expects there).

**Result: 0 mismatches out of 2048.** Every single element candidate 1b delivers is exactly where WGMMA's hardware A-operand layout expects it to be — not just "the aggregate total happens to match," which is all the earlier GPU run established.

## What this does and doesn't prove

**Proves:** candidate 1b's fragment mapping is exactly correct at the per-element level, *given* that (a) Part 1's transcribed ldmatrix formula is right (SASS-verified on real hardware) and (b) CUTLASS's `CLayout_64x64`/`ALayout_64x32` correctly reflect the hardware-mandated register layout for this instruction family (verified via bijection + the documented-example cross-check, and corroborated by CUTLASS being NVIDIA's own shipped, tested implementation of this exact instruction).

**Doesn't prove:** that this is empirically confirmed by actually running `wgmma` on silicon and reading the registers back — that's still the GPU-bound half of item 2/3, and the derivation here is exactly what that GPU run's "expected" table should be built from once hardware access is available again. It also doesn't independently confirm that Humming's own generated PTX (`WgmmaOpClassImpl.generate_ptx()`) lists its `{%0,...,%31}`/`{%32,...,%35}` operands in the same order this derivation assumes — that operand-list ordering is dictated by the instruction's own ISA semantics (not a library-specific choice), so it should match, but this hasn't been independently re-derived from Humming's own generator output the way the fma() signature itself was in earlier work.
