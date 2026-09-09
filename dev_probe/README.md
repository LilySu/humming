# ptx94-ldmatrix-s8-s4-wgmma-check — dev probe (not a standing test)

This directory is **not** meant to be committed to `tests/`. It mirrors the
precedent from vLLM PR #50096: WorldExplored's own Phase 0 layout probe
(`marlin_ldmatrix_s4_layout.cu` / `test_marlin_ldmatrix_s4_layout.py`, commit
`3406be9f0d7`) was written as a standalone, throwaway verification kernel,
reviewed once, and explicitly deleted (`removed tests`, commit `2bea40c6be9`)
once its one job — confirming the instruction behaves as documented on real
hardware — was done. Harry-Chen's own words on it: verifying an ISA-level fact
"is not likely to change once the ISA is published," so it isn't worth
keeping as a permanent CI test.

Same idea here, adapted for Humming's WGMMA path instead of vLLM's Marlin
(`mma.sync`) path — see the whole `ptx94-ldmatrix-s8-s4-wgmma-check` branch
history for why the two aren't the same question.

## Files

- `ldmatrix_s4_layout.cu` — **Part 1, complete and ready to run.** Validates
  `ldmatrix.sync.aligned.m8n16.x{1,2,4}.shared::cta.s8.s4` itself: does the
  instruction sign-extend and place bytes exactly where PTX says it will, on
  this specific toolkit and this specific GPU. Independent of any MMA
  pairing — same instruction, same guarantee, regardless of whether it
  eventually feeds `mma.sync` or `wgmma`. Adapted from WorldExplored's
  probe, plus two categories of coverage his didn't need to worry about
  (Humming's WGMMA mainloop iterates a K-loop and reuses smem buffers
  across iterations, which vLLM's `mma.sync` probe never exercised):
  - `validate_alignment_boundaries()` — the tile sitting behind unrelated
    padding (0/16/32 bytes), not just at a freshly-allocated buffer's
    natural offset-0 start.
  - `validate_sequential_row_transitions()` — two independent tiles loaded
    back-to-back by the same warp in one launch, matching a real K-loop's
    repeated-call shape rather than a single isolated load.

- `wgmma_pairing_check.cu` — **Part 2, scaffolded, two pieces intentionally
  left open — see the TODO blocks in the file itself.** Builds the actual
  `wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8` call using the exact
  signature extracted from Humming's own `WgmmaOpClassImpl.generate_ptx()`
  (`humming/config/mma.py`), feeds it `ldmatrix.s8.s4`-loaded data as the
  register-`a` operand (confirmed via that generator's own output that
  Humming's weight registers land in raw-PTX `a`, not `b` — WGMMA's `b` is
  always a descriptor, never registers, for any dtype), and is meant to
  check the result against an independent reference GEMM. **Do not treat a
  clean run of this file as Phase 0 closing** — per Codex's review: "keep
  Phase 0 closed only if it proves semantic layout, not merely that
  literal PTX was emitted... 'compiler accepted the instruction' and
  'every lane receives the consumer's expected bytes' are separate
  milestones." This file currently only gets you to the first milestone.
  The two remaining gaps:
  1. The activation-side (`b`, smem-descriptor-sourced) tile needs
     Humming's real, already-tested swizzle write path, not a plain
     row-major placeholder — see the file's TODO for why this isn't
     hand-derived.
  2. The accumulator (`d`) readback needs Humming's actual D-fragment
     thread-to-element mapping (from its epilogue code) before the kernel's
     output can be diffed against the reference GEMM at all — right now it
     only prints a sample value, it doesn't assert pass/fail.

- `run_probe.py` — compiles and runs whichever `.cu` file you point it at.
  Beyond WorldExplored's original build-and-check-stdout structure, this
  also captures, per Codex's "separate PTX acceptance from hardware proof":
  the generated PTX, the cubin, its SASS disassembly, and `ptxas -v`'s
  register-count/spill report — written to `artifacts/<probe-name>/` along
  with a `metadata.txt` recording the exact nvcc version, target arch, GPU,
  and every command run. "It compiled and ran" and "here's the SASS proving
  what actually got emitted" are different claims; this keeps both on hand.

- `layout_identity.py` — **pure Python, no GPU needed, already verified
  running locally.** Codex: "treat the layout as an explicit type/variant,
  not a loose boolean" + "centralize the availability predicate... should
  not be independently reconstructed in several files." Defines
  `WeightSmemLayout` (an enum, not a bool) and `resolve_weight_smem_layout()`
  as the one function every one of repack / JIT specialization / dispatch
  should call and carry the *result* of forward, rather than each
  re-deriving their own (nvcc-version, sm-family, dtype) check independently.

- `test_wrong_layout_pairing.py` — **pure Python, no GPU needed, already
  verified running locally.** Codex: "test 'wrong-layout pairing'
  deliberately... should be structurally impossible or caught immediately."
  Exercises the concrete failure mode this matters for: a weight repacked
  elsewhere (SM90 + CUDA 13.4) later loaded on a box that can't produce
  that layout itself — e.g. once repacked-weight caching across processes
  exists, which is a dormant risk flagged for both this branch and vLLM's
  PR. Confirms the mismatch raises loudly rather than silently
  misinterpreting bytes.

- `SHARED_MEMORY_CONTRACT.md` — Codex: "make the no-.trans consequence
  explicit in the shared-memory contract: logical dimensions, physical
  strides, swizzle, required alignment, and bank mapping." Written out as
  its own document rather than left implicit in code comments, since this
  is exactly the kind of fact a future contributor could otherwise
  silently violate by "simplifying" the layout.

## What's still open, honestly

- The two TODOs inside `wgmma_pairing_check.cu` (swizzle write, D-fragment
  readback) — needed before Phase 0 can actually close, not optional
  polish.
- Codex's remaining points that aren't addressed by anything in this
  directory yet, because they need either real hardware runs or belong to
  later phases, not this one:
  - **Negative compile tests** (unsupported target/toolkit still builds
    the fallback kernel cleanly) — needs a *second* build environment with
    an older CUDA toolkit or non-SM90 target to actually exercise, which
    this branch's current pod access doesn't cover by itself.
  - **Benchmark beyond latency** (register pressure, occupancy, spills,
    bank conflicts, break-even by shape) — explicitly a Phase 1.5 concern,
    matching vLLM's own issue's phase split; premature before Phase 0 and
    Phase 1 land. `ptxas -v`'s output (captured by `run_probe.py` above) is
    a start on register pressure specifically, not the full picture.

## Running this

Needs an SM90 GPU + CUDA 13.4 toolkit. This branch has that covered —
`~/wsl_git/vllm-wip/blog_draft_cuda134_marlin.md` documents the exact
RunPod setup already used earlier in this work: an H100, the
`nvcr.io/nvidia/cuda-dl-base:26.08-cuda13.4-devel-ubuntu24.04` container
image, and the SSH/verification steps to confirm the environment before
spending time on anything. Same pattern applies here — confirm the
environment first, then run:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # confirm compute capability 9.0
nvcc --version                                          # confirm release 13.4+

python3 dev_probe/layout_identity.py                    # pure Python, run first, no GPU needed
python3 dev_probe/test_wrong_layout_pairing.py          # pure Python, run first, no GPU needed

python3 dev_probe/run_probe.py dev_probe/ldmatrix_s4_layout.cu
# once wgmma_pairing_check.cu's two TODOs are resolved:
python3 dev_probe/run_probe.py dev_probe/wgmma_pairing_check.cu

ls dev_probe/artifacts/       # PTX, cubin, SASS, register/spill reports, metadata.txt per probe
```

Once Part 1 and Part 2 both genuinely pass (not just run without crashing —
see "what's still open" above), delete this directory before opening the
real PR — same as WorldExplored did. This is investigation, not the
feature.
