# Phase 0 follow-up: no-GPU work completed this session

Status against `NEXT_STEPS.md`'s "Phase 0 follow-up: exact pairing correctness" list (same six items as `humming-ldmatrix-s4-moe-plan.md` §3). This is everything that could be done without the rented H100 — the remaining GPU-bound confirmation runs are listed at the bottom.

## 1. Preserve immutable source copies and hashes — done
`SHA256SUMS.txt` in this directory, covering the current `wgmma_pairing_check.cu`, `wgmma_pairing_check_isolate.cu`, `ldmatrix_s4_layout.cu`, `wgmma_fragment_layout.py`, `run_probe.py`.

## 2. Reconstruct every output element, diff against an independent CPU GEMM — done, code written
- `wgmma_fragment_layout.py` derives the exact WGMMA accumulator (C/D) and register-sourced A-operand layouts from CUTLASS's own tested production code (`cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp`), verifies each is a real bijection (no gaps/collisions) over the full 4096-cell / 2048-cell tile, and cross-checks the C-layout formula against NVIDIA's own documented example. See `WGMMA_D_FRAGMENT_LAYOUT.md` for the full derivation and citation trail.
- `wgmma_pairing_check.cu`'s `validate_wgmma()` now decodes all 4096 raw outputs via `d_fragment_mn()` (transcribed from the above), diffs every element against an independently-computed reference GEMM, and additionally checks the decode itself is a bijection (every (m,n) cell written exactly once) before trusting the value comparison. The aggregate sum is now printed as a diagnostic only, per the plan's explicit note that it isn't the pass criterion.
- **Not yet done: running it.** This needs the H100.

## 3. Cross-check expected input registers by lane/register/byte — done, at the derivation level
`wgmma_fragment_layout.py`'s `cross_check()` compares, for every one of 2048 `(warp, lane, register, byte)` combinations, what `ldmatrix.s8.s4` actually delivers there (Part 1's own hardware-validated formula, transcribed from `ldmatrix_s4_layout.cu`) against what WGMMA's hardware A-operand layout requires there (CUTLASS's `ALayout_64x32`). **Result: 0 mismatches for candidate 1b.** This is real evidence at the per-element level, derived and verified without a GPU — not a substitute for the GPU run in item 2, but strong grounds to expect it will pass.

## 4. Coverage: nibble values, both positions, K-halves, rows/cols, single-element, random — mostly already satisfied, verified
Checked computationally (not just asserted): the existing `a_value` formula already produces all 16 signed nibble values across the tile, and there is not a single `(m, k)` pair anywhere where a packed byte's low and high nibble come out equal (0 out of 1024 packed bytes, including at the K-half boundary specifically) — so a low/high nibble extraction bug would already be visible in the existing dense data, without needing a dedicated degenerate-value test.

**Not done:** a dedicated single-element localization kernel (set exactly one source value nonzero, confirm only the expected output responds), mirroring Part 1's `validate_every_source_element()`. This needs a kernel-level refactor (parameterizing which values get packed) that carries more risk to write correctly untested than the value it adds given the check above already covers the specific degeneracy concerns item 4 was guarding against. Left as a scoped follow-up, not a blocker.

## 5. Deliberately-wrong ordering that must be rejected — done, both levels
- **Derivation level (no GPU):** `wgmma_fragment_layout.py --candidate 1a` runs the identical cross-check against the already-known-wrong candidate 1a ordering: **1024/2048 mismatches**, cleanly distinguishing it from candidate 1b's 0/2048. The check has real, demonstrated discriminating power — it doesn't just report PASS regardless of input.
- **GPU level (code written, not yet run):** `wgmma_pairing_check.cu` now has a `WRONG_FRAGMENT_ORDERING` compile-time switch that swaps in candidate 1a's addressing. Building with `-DWRONG_FRAGMENT_ORDERING` and confirming the elementwise check reports failure is the empirical counterpart of the derivation-level result above — needs the H100.

## 6. Memcheck + real exit-code discipline — done
`run_probe.py` now runs the built executable under `compute-sanitizer --tool memcheck` in addition to the plain run, and requires *its* exit code (not a text-search of its output) for overall success. If `compute-sanitizer` isn't on PATH, that's now reported as a failure with a clear message, not silently skipped. Both runs' full output are written to `plain_run.txt` / `compute_sanitizer.txt` in the artifacts directory.

(The CRLF-tolerant remote-completion-check issue from earlier in this project was in ad hoc, uncommitted SSH polling scripts written during the interactive session, not in any file in this repo -- there was nothing to fix here beyond making sure this driver's own exit-code handling, which the remote scripts wrapped, is itself correct.)

## What's left, and needs the H100

1. Rebuild `wgmma_pairing_check.cu` (now with the exact elementwise check) and run it — confirm `validate_wgmma()` reports PASSED with 0 mismatches, not just a matching aggregate sum.
2. Rebuild with `-DWRONG_FRAGMENT_ORDERING` and confirm it reports FAILED with mismatches, empirically confirming the negative control.
3. Run both under `compute-sanitizer` via the updated `run_probe.py` and confirm 0 sanitizer errors on the correct build.
4. Re-archive the resulting artifacts into `vllm-wip/humming-phase0-ldmatrix-s4/` alongside the existing evidence, updating that bundle's README to reflect exact (not aggregate) closure.
