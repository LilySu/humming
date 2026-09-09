# Next steps after the Humming Phase 0 experiment

Prepared on 2026-09-07 against Humming HEAD 0e85810a235bc71023b5cb5efa6b4941f1d8e617. This is a proposed plan, not a record of production implementation or new GPU testing.

## Starting point

The evidence bundle is /home/lily/wsl_git/vllm-wip/humming-phase0-ldmatrix-s4/. It records an H100 80GB HBM3, CUDA 13.4.59, and an explicit compute_90a / sm_90a target.

- The isolated ldmatrix.s8.s4 tests passed their pattern, element sensitivity, alignment-offset, and sequential-load checks.
- The WGMMA pairing probe stopped crashing after its activation shared-memory allocation, addressing, and visibility contract were corrected. Increasing alignment alone had failed.
- Removing an inappropriate sign-bit XOR from already-signed test data made candidate 1b produce the expected total, -4096, with zero reported memcheck errors.
- That result checks an aggregate sum. It does not establish that every output element is correct, that the production mainloop works, or that the change is faster.

The checkout has an untracked dev_probe/ directory. Its README describes an earlier scaffold state; use the archived source and recorded outputs to identify the passing baseline. Preserve that archive and distinguish new raw logs from older annotated or abbreviated records.

## Recommended initial scope

Start with one H100 WGMMA configuration using int8 activations and symmetric four-bit weights, without explicit zero points, through dense GEMM. Choose one compatible existing block geometry. This keeps the first implementation close to the tested arithmetic.

Determine the production four-bit encoding first: unsigned quantization codes and already-signed nibbles require different handling. Removing XOR from the signed probe is not a rule to remove it from every production packer. Preserve the existing scale semantics for the selected configuration.

Initially require explicit selection of the experimental variant, with the existing path available for comparison and fallback. Expand FP8, zero points, MoE, additional geometries, and other GPU families separately.

## Phase 0 follow-up: exact pairing correctness

1. Preserve immutable source copies and hashes for the passing v5 baseline before changing the diagnostic.
2. Reconstruct all 64 by 64 output elements and compare against an independent CPU int32 GEMM. Derive the raw WGMMA accumulator mapping from NVIDIA's documentation and cross-check Humming's epilogue convention. Document the translation between project operand names and raw PTX names.
3. Test signed extrema, both nibble positions, distinct K halves, distinct rows and columns, targeted single-element inputs, and deterministic random inputs. Keep the total sum only as a diagnostic.
4. Include a deliberately wrong register ordering that the targeted test must reject. If exact comparison fails, dump input register fragments to separate loading errors from output reconstruction errors.
5. Run normally and under memcheck. Capture actual process statuses and use a nonzero sanitizer error exit code. Use bounded, job-specific completion checks that tolerate CRLF and record failure as well as success.

Completion gate: every output element matches exactly across the selected cases; the deliberate wrong ordering fails; the correct run reports no memcheck errors. Archive exact source hashes, commands, statuses, and raw output.

This can remain a small standalone diagnostic; it does not require porting the entire production epilogue. References: [NVIDIA WGMMA fragments](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n32), [Compute Sanitizer memcheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool).

## Phase 1: one integrated configuration

1. Define a distinct weight-layout variant covering logical indices, nibble encoding, physical shared-memory offsets, alignment, and WGMMA register order. The existing packed-K variant is not automatically interchangeable with it.
2. Resolve availability once from compiler/target, dtype, geometry, and supported quantization options. Carry that resolved choice through preparation, specialization, and launch. Include it in generated kernel identity so JIT caching cannot reuse a different weight interpretation.
3. Make incompatible prepared-weight and consumer combinations impossible or reject them before launch. Select fallback before repacking. Switching representations later requires compatible prepared weights or explicit repacking from a canonical representation.
4. Change producer and consumer together: weight repacking, global-to-shared placement, expanding shared-to-register load, and WGMMA fragment use.
5. Bypass duplicate unpacking for values already expanded to signed int8, while preserving required scaling and arithmetic.
6. Respect production stage sizes and synchronization. Instruction K and staged block K are different. The standalone activation allocation bug does not establish that production activation storage needs the same change.

Likely implementation locations:

| Responsibility | Current code |
| --- | --- |
| Selection and generated context | humming/config/config.py; humming/kernel/humming.py |
| Weight preparation and packing | humming/transform.py; humming/kernel/repack_weight.py; humming/include/humming/kernel/process.cuh |
| Weight staging | humming/include/humming/memory/g2s_loader/loader_b.cuh |
| Expanding load | humming/include/humming/memory/s2r_loader/loader_b.cuh |
| Conversion and scale handling | humming/include/humming/mma/wgmma.cuh |
| Target and specialization verification | humming/jit/compiler.py |

The existing use_native_dequant property selects other MMA/UMMA conversion paths. Do not silently repurpose it for this WGMMA variant. Reuse suitable infrastructure without redesigning unrelated paths or the cache.

Completion gate: the real weight transformation and GEMM entrypoint produce correct results with the new variant explicitly selected. The existing path also works with separately prepared compatible weights.

## Phase 1 validation: production correctness and fallback

- Add focused regressions through the real preparation and execution APIs under tests/kernels/humming/. Compare both variants with an independent reference using the same logical inputs. Use exact comparison for exposed int32 results and established tolerances for scaled outputs.
- Cover positive and negative extrema, zeros, nibble boundaries, multiple output tiles, supported padding/tails, and K lengths that exercise multiple pipeline stages and buffer reuse.
- Test incompatible layout rejection and fallback selection before preparation. Exercise prepared-weight reuse wherever the current API permits it.
- Verify fallback builds for an unsupported target and an older toolkit without requiring the new instruction. Record the actual build matrix and distinguish compile checks from GPU execution. Force the existing path on H100 as an additional runtime check.
- Run representative integrated cases under memcheck. Inspect PTX/SASS for the intended instruction and record register count, shared-memory use, and spills.

Completion gate: the supported scope passes correctness and fallback checks, including repeated-stage execution. Unsupported combinations fail early or use their existing valid path. List untested environments explicitly.

Keep standalone ISA probes as archived evidence. Permanent tests should protect production behavior.

## Phase 1.5: performance and enablement

1. Compare both variants on the same commit, GPU, toolchain, logical inputs, and scales, preparing each representation separately. Warm up and use synchronized device timing with repeated samples.
2. First match kernel geometry to isolate the load change; then compare the best valid configurations for each path. Include small-token and larger-batch M values such as 1, 8, 32, 64, and 128, using representative supported N/K dimensions.
3. Measure steady-state GEMM separately from preparation and cold JIT cost. Report variability and the reuse count needed to recover any additional preparation cost.
4. Examine registers, spills, shared-memory use, and occupancy. Investigate bank conflicts or scheduling when needed to explain timing.
5. Enable automatic selection only where improvement is reproducible beyond measurement noise and regression behavior is acceptable. If there is no useful gain, keep the variant experimental or stop promotion.

Completion gate: a reproducible shape-by-shape comparison supports a specific enablement region. The standalone probe's zero-spill result is not a production performance result.

## Review and later expansion

Prepare a narrow review package with exact commit/base, diff scope, layout contract, capability/fallback rules, correctness results, benchmarks, and unrun checks. Preserve matching source hashes, environment, commands, raw logs, PTX/cubin/SASS, and compiler resource reports.

Keep integration separate from earlier Marlin and AWQ/zero-point work. Audit proposed changes before seeking approval for commits, pushes, or PR creation. Archive diagnostic work and omit it from the production PR; this planning task does not delete files or mutate Git history.

After the first path earns enablement, evaluate FP8 conversion, explicit zero points, MoE, other block geometries, and other architectures individually, with correctness and performance evidence for each.

Immediate next task: extend the pairing probe to exact elementwise comparison, add the deliberate wrong-order check, and prepare a reproducible script locally. Then execute one bounded H100 validation batch. Production integration follows that result.
