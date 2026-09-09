// Part 2 of the ptx94-ldmatrix-s8-s4-wgmma-check probe.
//
// The actual gating question for Humming (different from vLLM's, which
// paired ldmatrix.s8.s4 with warp-level mma.sync — see dev_probe/README.md):
//
//   Does ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4's per-thread
//   output, composed across all 4 warps of a warpgroup, match the
//   register-`a` operand layout that
//   wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 requires?
//
// That exact instruction string, and the fact that Humming's own weight
// registers land in the raw PTX `a` slot (not `b` — WGMMA's `b` is always
// an SMEM descriptor, never registers), came directly from running
// Humming's own code generator:
//
//   humming/config/mma.py: WgmmaOpClassImpl(m=64, n=64, k=32,
//     a_dtype='s8', b_dtype='s8', cd_dtype='s32').to_cpp_str()
//
// which is where the fma() signature and the literal asm below are lifted
// from — this is not hand-derived, it's Humming's actual generated code for
// this instruntiation.
//
// Both TODOs from the earlier draft of this file are resolved:
//  - the B-tile's swizzled shared-memory write now ports the exact XOR
//    formula from Humming's real G2SMemoryLoaderA::load_legacy_swizzled_128B
//    (humming/include/humming/memory/g2s_loader/loader_a.cuh:100-128),
//    concretely specialized for this probe's fixed tile shape, rather than
//    hand-guessed.
//  - the accumulator readback now does an EXACT per-(m,n) diff against an
//    independent reference GEMM, via d_fragment_mn()'s decode -- not just
//    an aggregate sum. Getting here didn't need Humming's real D-fragment
//    epilogue thread mapping (real templated production code, out of scope
//    for a throwaway probe); it came from CUTLASS's own tested
//    CLayout_64xN/ALayout_64x32 (see dev_probe/WGMMA_D_FRAGMENT_LAYOUT.md
//    and dev_probe/wgmma_fragment_layout.py for the full derivation,
//    bijection check, and a cross-check against NVIDIA's documented
//    example). An earlier version of this file's aggregate-sum-only check
//    is exactly what dev_probe/NEXT_STEPS.md's Phase 0 follow-up section
//    and humming-ldmatrix-s4-moe-plan.md §3 both flag as insufficient: an
//    aggregate sum is invariant to permuting which output element holds
//    which value, so it cannot rule out a wrong-but-sum-preserving
//    fragment mapping. This file's WRONG_FRAGMENT_ORDERING build (see
//    matrix_idx -> (m_subgroup, k_half) below) is the negative control
//    proving the new check actually has the power to catch that class of
//    bug, not just report PASS regardless of input.
//
// Three real bugs were found and fixed, the first two after the first run
// crashed (cudaMemcpy: illegal memory access) under compute-sanitizer, the
// third by external review (before it could produce a false pass):
//
//  1. STORAGE-LAYOUT BUG (the crash). smem_b was allocated as [kK][kN] =
//     32 rows x 64 bytes/row (2048 bytes total), but
//     make_wgmma_smem_desc<128>'s stride encoding assumes 128-byte rows
//     (it hardcodes an 8-row/128-byte-per-row swizzle period -- see that
//     function's comment). With a 64-byte physical row, the hardware's
//     internal address computation for a raw-N=64 descriptor reaches up to
//     64*128=8192 bytes -- 4x past the actual 2048-byte allocation.
//     Bisected via wgmma_pairing_check_isolate.cu's ISOLATE_SKIP_WGMMA
//     build (0 sanitizer errors with wgmma skipped, 33 errors with it
//     included), which localized the fault to the wgmma/descriptor path
//     specifically, not the ldmatrix load or the swizzle write. A first
//     attempt just bumped smem_b's alignment 16->128 without fixing the
//     allocation size; that did not help (identical crash) -- alignment
//     alone can't fix an access pattern that's reading past the buffer.
//     Fix: allocate smem_b as [kN][128] (real 128-byte rows, matching the
//     descriptor's assumption), align it to 1024 (one full 8-row swizzle
//     period, so the descriptor's implicit zero matrix-base-offset is
//     valid), and write K-contiguously per row via the same swizzle helper.
//  2. TEST-METHODOLOGY BUG (found by inspection, before it could produce a
//     false pass). The original a_value(m,k) used k's coefficient (5) times
//     16 as its modulus, and 5*16=80 is itself a multiple of 16 -- meaning
//     a_value(m,k) and a_value(m,k+16) were *exactly* identical for every
//     (m,k). Since candidate1_m_offset's open question is precisely which
//     half of the 4 ldmatrix.s8.s4 matrices maps to which K-half, a k-half
//     swap bug would have been mathematically undetectable no matter what
//     b_value looked like. Separately, the original b_value's row-sum
//     (summed over n, for fixed k) was constant across all k, which on its
//     own makes the aggregate-sum check blind to *any* k-reordering bug
//     (sum_k a(m,k)*C is invariant to how the k terms are permuted, for a
//     k-independent constant C). Both were verified with a standalone
//     Python check before being fixed -- not just asserted. Fix: a_value
//     now uses genuinely different (not just algebraically-shifted)
//     formulas for k<16 vs k>=16, and b_value's row-sum now varies by k.
//  3. TEST-METHODOLOGY BUG, second instance (found by external review of
//     this branch, before it could produce a false pass). Bug #2's fixed
//     a_value(m,k) closed the k-half-swap blind spot but introduced a
//     different one: it produced only 55 distinct output ROWS out of 64
//     (e.g. a_value(0,k) == a_value(41,k) for every k), so a row-swap/
//     M-permutation bug would have been invisible to the per-(m,n) check
//     added by this same round of fixes. Verified in Python by direct
//     enumeration, not assumed. Fix: see a_value()'s own comment below for
//     the structural (not trial-and-error) fix and why two earlier fix
//     attempts regressed bug #2's checks instead of fixing this one.
//
// What's still a genuine unknown, and the actual point of running this:
// candidate1_m_offset()'s guess at how the 4 warps' ldmatrix.s8.s4 calls
// partition the 64-row A tile, and the matrix_idx -> (m_subgroup, k_half)
// ordering below it. If validate_wgmma() fails, these are the first things
// to revisit -- see the Phase-0C-style alternate-candidate notes near
// those functions.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if CUDART_VERSION < 13040
  #error "ldmatrix.s8.s4 requires CUDA 13.4 or newer"
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerGroup = 4;
constexpr int kThreadsPerGroup = kWarpSize * kWarpsPerGroup;

// Target instruction shape: wgmma.mma_async...m64n64k32.s32.s8.s8
constexpr int kM = 64;  // raw-PTX A operand rows   (Humming's weight, register-sourced)
constexpr int kN = 64;  // raw-PTX B operand cols   (Humming's activation, descriptor-sourced)
constexpr int kK = 32;  // shared reduction dim
// Physical row width of smem_b, in bytes: matches make_wgmma_smem_desc<128>'s
// hardcoded 128-byte-row swizzle-period assumption. Only the first kK bytes
// of each 128-byte row hold real K data; the rest is unused padding needed
// so the descriptor's internal addressing (which assumes 128B/row) stays
// within the actual allocation. See the header comment's bug writeup.
constexpr int kSmemBRowBytes = 128;

#define CUDA_CHECK(call)                             \
  do {                                               \
    cudaError_t error = call;                        \
    if (error != cudaSuccess) {                      \
      std::fprintf(stderr, "%s failed: %s\n", #call, \
                   cudaGetErrorString(error));        \
      std::exit(EXIT_FAILURE);                        \
    }                                                \
  } while (0)

// --- ldmatrix.s8.s4 wrapper: identical to Part 1 -----------------------

__device__ inline void load_s4_x4(uint32_t (&registers)[4],
                                   const void* shared_ptr) {
  uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4 "
      "{%0,%1,%2,%3}, [%4];\n"
      : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
        "=r"(registers[3])
      : "r"(address));
}

// --- wgmma primitives: copied from
// humming/include/humming/utils/ptx/wgmma.cuh and
// humming/include/humming/mma/wgmma.cuh (make_wgmma_smem_desc), verbatim,
// so this probe stays standalone (no Humming package build/JIT needed to
// compile it) while calling the exact same instructions the real kernel
// would. -------------------------------------------------------------

__device__ inline void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" :: : "memory");
}

__device__ inline void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" :: : "memory");
}

template <uint32_t N>
__device__ inline void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <uint32_t swizzle_bytes = 128>
__device__ inline uint64_t make_wgmma_smem_desc(uint32_t addr) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);
  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);
  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t*>(&desc)[0] = (addr >> 4);
  return desc;
}

// --- the exact instruntiation from WgmmaOpClassImpl.to_cpp_str() -------
// fma(uint64_t &desc, uint32_t *b, uint32_t *d, bool pred = true)
// Renamed b->a_regs here since it is the raw-PTX `a` operand — see the
// header comment above and the branch's naming-clarity discussion.
__device__ inline void wgmma_m64n64k32_s8s8s32(uint64_t& desc,
                                                uint32_t* a_regs,
                                                int32_t* d, bool pred = true) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8.satfinite "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
      "%28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p;\n"
      "}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]),
        "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]),
        "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]),
        "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),
        "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
        "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]),
        "+r"(d[30]), "+r"(d[31])
      : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
        "l"(desc), "r"((uint32_t)pred));
}

// --- reference values, deliberately non-degenerate (per the earlier
// review on this branch: avoid patterns that could hide a transpose or a
// permutation behind symmetric/repeating values). --------------------
//
// Both functions below were revised after a standalone Python check (see
// the header comment's bug #2 writeup) found the *first* version of each
// had an exact algebraic symmetry that made the specific bug this probe
// exists to catch -- a K-half swap -- mathematically undetectable by the
// aggregate-sum check, regardless of what the other operand looked like.
// Don't simplify these back to a smooth closed-form linear/XOR-mod
// expression across the k<16 / k>=16 boundary: that was tried and proven
// (by direct computation, not assumption) to reintroduce the same
// degeneracy, because any linear or XOR combination reduced mod 16 is
// invariant to adding 16 to k (bit 4 and above are discarded by the mask).

__host__ __device__ int8_t a_value(int m, int k) {
  // Stays within s4's legal signed-nibble range [-8, 7], since this is the
  // value that gets packed two-per-byte and hardware-sign-extended by
  // ldmatrix.s8.s4.
  //
  // BUG #3 (found by external review, before it could produce a false
  // pass). The previous version of this function (two ad hoc formulas
  // split on k_half) produced only 55 distinct output ROWS out of 64 --
  // e.g. a_value(0,k) == a_value(41,k) for every k, because both
  // branches' m-dependent terms happened to collide mod 16 at that
  // specific (m1,m2) pair. A row-swap/M-permutation bug (e.g. two warps'
  // M-ranges transposed) would have been invisible to validate_wgmma's
  // per-(m,n) check, since row 0's and row 41's reference AND actual
  // outputs would be identical either way. Verified by direct enumeration
  // in Python, not assumed.
  //
  // Two prior attempts at a fix regressed the k-half/nibble checks that
  // bug #2 above established: making the m-coefficient vary with k_sub
  // (to break the row collision) also, incidentally, made the *difference*
  // between the k_half==0 and k_half==1 formulas (or between a nibble
  // pair's two formulas) depend on m again -- reintroducing exactly the
  // kind of m-dependent coincidental collision bug #2 first fixed, just at
  // different (m, k) positions. Confirmed by exhaustive Python check, not
  // assumed: one attempt produced 128 low==high nibble collisions and 80
  // k==k+16 collisions (both worse than the version being replaced).
  //
  // Fix: decouple the two failure modes structurally instead of tuning
  // coefficients by trial and error.
  //   - k_pair (0..7) is shared by exactly the two k-values being
  //     compared in both checks: a nibble pair (k_sub, k_sub+1) and a
  //     k-half pair (k_sub, k_sub+16) both keep k_pair fixed.
  //   - The m-dependent term (m*c + m_block*d) uses a coefficient c that
  //     depends ONLY on k_pair -- so it is IDENTICAL for both members of
  //     either pair being compared, and cancels out of their difference.
  //   - k_half and k_parity contribute pure additive constants (8 and 4)
  //     that don't depend on m at all. The difference between any two
  //     values being checked is therefore a fixed nonzero constant (8 or
  //     4 mod 16), true for every m -- an algebraic guarantee, not an
  //     empirical one.
  //   - c is odd for every k_pair (so the m-term doesn't degenerate at
  //     particular k_pair values), and m_block = m/16 breaks the
  //     row-collision periodicity that a pure "m*c mod 16" term would
  //     otherwise have every 16 rows.
  // Re-verified via the same exhaustive Python checklist as before: full
  // [-8,7] range, 0 low==high nibble collisions, 0 k==k+16 collisions, all
  // 64 output rows distinct, all 64 output columns distinct.
  int k_sub = k % 16;       // 0..15, position within this K-half
  int k_half = k / 16;      // 0 or 1
  int k_pair = k_sub / 2;   // 0..7, shared by a packed nibble pair
  int k_parity = k_sub % 2; // 0 = low nibble, 1 = high nibble
  int m_block = m / 16;     // 0..3

  int c = 2 * k_pair + 1;  // odd for every k_pair in [0,8)
  int d = 3 * k_pair + 1;

  int v = (m * c + m_block * d + k_half * 8 + k_parity * 4) % 16;
  return static_cast<int8_t>(v - 8);
}

__host__ __device__ int8_t b_value(int k, int n) {
  // XOR-mixes k and n (rather than a pure linear combination) so each
  // row's sum-over-n varies with k -- necessary for the aggregate-sum
  // check to be sensitive to a K-reordering bug at all: if the row-sum
  // were constant across k (the original formula's flaw), sum_k a(m,k)*C
  // is invariant to any permutation of which k value lands where, since a
  // full 0..31 sweep of k contributes the same multiset of terms either
  // way. This one's full int8 range (not s4-packed), so no [-8,7] limit.
  int v = (k * 11 + n * 17 + ((k ^ n) * 5)) & 0xFF;
  return static_cast<int8_t>(v);
}

// Matches G2SMemoryLoaderA::load_legacy_swizzled_128B's XOR swizzle
// (humming/include/humming/memory/g2s_loader/loader_a.cuh:100-128) —
// not hand-derived, ported from that exact formula rather than re-guessed,
// per the branch's stated rule against reimplementing WGMMA smem swizzle
// logic independently. `logical_chunk` here indexes 16-byte (int4) chunks
// within a 128-byte-wide row layout (8 chunks/row), matching smem_b's
// physical [kN][128] shape below -- NOT the original [kK][kN] shape this
// helper was first (incorrectly) applied to, which had 64-byte rows and
// caused the wgmma instruction to read past the allocation. See the header
// comment's bug #1 writeup.
__device__ inline uint32_t wgmma_b_swizzled_chunk(uint32_t logical_chunk,
                                                   uint32_t base_addr) {
  uint32_t smem_base = (base_addr / 128) % 8;
  uint32_t smem_row = logical_chunk / 8;
  uint32_t smem_col = logical_chunk % 8;
  uint32_t swizzled_col = smem_col ^ ((smem_row + smem_base) % 8);
  return smem_row * 8 + swizzled_col;
}

// TODO(candidate mapping, the actual thing under test):
// Each warp issues one x4 ldmatrix.s8.s4 call, loading 4 (8-row x 16-col)
// source matrices = 32 M-rows-worth-of-K16-halves. Two x4 calls per warp
// (kK/16 = 2) are needed to cover all 32 K-columns for a warp's M-rows.
// CANDIDATE 1 below: warp w owns M-rows [16w, 16w+16). This is the most
// natural even split of 64 rows across 4 warps and is the first thing this
// probe checks — it is a guess, not a documented fact. If validate_wgmma()
// fails, the Phase-0C-style next step (per the review discussion on this
// branch) is to try alternate row/col partitionings before concluding the
// direct-feed approach doesn't work at all.
__device__ int candidate1_m_offset(int warp_id) { return warp_id * 16; }

__global__ void wgmma_layout_kernel(const uint8_t* packed_a,
                                     int32_t* output_d) {
  // K-major packed A: kM rows x (kK/2) packed bytes (2 nibbles/byte).
  __shared__ __align__(16) uint8_t smem_a[kM][kK / 2];
  // [kN][128]: kN=64 logical rows, each physically kSmemBRowBytes=128 bytes
  // wide (only the first kK=32 bytes/row hold real data -- see the header
  // comment's bug #1 writeup for why the row width must match
  // make_wgmma_smem_desc<128>'s hardcoded 128-byte-row assumption). Aligned
  // to 1024 = one full 8-row swizzle period, so the descriptor's implicit
  // zero matrix-base-offset is valid for this buffer's actual start
  // address, not just coincidentally for whatever the compiler picked.
  __shared__ __align__(1024) int8_t smem_b[kN][kSmemBRowBytes];

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane = tid % kWarpSize;

  for (int i = tid; i < kM * (kK / 2); i += kThreadsPerGroup) {
    reinterpret_cast<uint8_t*>(smem_a)[i] = packed_a[i];
  }

  // Swizzled write for smem_b: K-contiguous per row (matches WGMMA's
  // K-major requirement for the descriptor-sourced operand), through the
  // same swizzle helper used for the (now-corrected) 128-byte-row layout.
  // kN*kK = 2048 real bytes to place, kThreadsPerGroup=128 threads -> 16
  // iterations/thread; padding bytes past kK in each 128-byte row are left
  // unwritten (unused by the wgmma call, which only reads kK=32 per row).
  {
    uint32_t smem_b_base_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
    for (int t = tid; t < kN * kK; t += kThreadsPerGroup) {
      int n = t / kK;
      int k = t % kK;
      uint32_t logical_chunk = n * 8 + k / 16;
      uint32_t physical_chunk =
          wgmma_b_swizzled_chunk(logical_chunk, smem_b_base_addr);
      reinterpret_cast<int8_t*>(smem_b)[physical_chunk * 16 + k % 16] =
          b_value(k, n);
    }
  }
  // WGMMA reads shared memory through the async proxy; ordinary st.shared
  // writes (like the loop above) need an explicit proxy fence before an
  // async-proxy reader can see them -- __syncthreads() alone only orders
  // the generic proxy. Per PTX ISA's warpgroup-level-matrix-async-proxy
  // ordering requirements.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  int m_base = candidate1_m_offset(warp_id);
  uint32_t a_regs[4];
  // Structural fix from an earlier draft: x4 needs 32 DISTINCT addresses
  // across the warp (one per lane, per Part 1's load_layout_kernel
  // convention: matrix = lane/8, row = lane%8), covering 4 separate
  // (8-row x 16-nibble) matrices. An earlier version of this file used
  // `lane % 16` for row selection, which only produces 16 distinct
  // addresses (each shared by 2 lanes) and never varies the K-half offset
  // -- that's an addressing bug independent of candidate1_m_offset's
  // warp-partition guess, and would have produced garbage regardless of
  // whether that guess is right. Fixed here so a failure below reflects
  // the actual open question (which warp owns which rows), not a
  // confounding implementation error in how x4's 32 addresses are formed.
  //
  // matrix_idx -> (m_subgroup, k_half) ordering below is ALSO a candidate
  // guess, layered on top of candidate1_m_offset. Phase-0C sweep:
  //   CANDIDATE 1a: m_subgroup=idx/2, k_half=idx%2 -- gave actual=-46848 vs
  //     reference=-4096 (with the packing bug still present at the time).
  //   CANDIDATE 1b (current), per NVIDIA's documented WGMMA A-fragment
  //     layout: m_subgroup=idx%2, k_half=idx/2 -- gave actual=-24576 vs
  //     reference=-4096 with the packing bug present. A standalone CPU
  //     model of both candidates against pack_a_k_major()'s double-XOR bug
  //     (see that function's comment) exactly reproduced BOTH recorded
  //     totals (-24576 and -46848), and predicts that with the packing bug
  //     fixed, candidate 1b matches the reference exactly (-4096) while 1a
  //     does not (predicts -18176). That prediction is what this run tests.
  // Only idx=0 and idx=3 coincide between the two candidates (the 2x2
  // diagonal); idx=1 and idx=2 read different (m,k) addresses under each.
  int matrix_idx = lane / 8;    // 0..3, matches Part 1's x4 addressing
  int row_in_matrix = lane % 8; // 0..7
#ifndef WRONG_FRAGMENT_ORDERING
  int m_subgroup = matrix_idx % 2;  // 0 or 1  -- candidate 1b (passed on real hardware)
  int k_half = matrix_idx / 2;      // 0 or 1
#else
  // Deliberately-wrong negative control (item 5, dev_probe/NEXT_STEPS.md /
  // humming-ldmatrix-s4-moe-plan.md §3.5): candidate 1a, already known to
  // fail on real hardware (part2_v4_candidate1a_output.txt) and confirmed
  // to disagree with WGMMA's hardware A-operand layout at exactly
  // 1024/2048 (warp,lane,reg,byte) positions by
  // dev_probe/wgmma_fragment_layout.py's derivation-level cross-check.
  // Build this TU with -DWRONG_FRAGMENT_ORDERING and confirm the
  // elementwise check below actually reports failure -- if it doesn't,
  // the check itself has no discriminating power and the "PASS" on the
  // correct build means nothing.
  int m_subgroup = matrix_idx / 2;
  int k_half = matrix_idx % 2;
#endif
  int address_row = m_base + m_subgroup * 8 + row_in_matrix;
  int k_half_byte_offset = k_half * (kK / 2 / 2);  // 0 or 8 packed bytes
  load_s4_x4(a_regs, &smem_a[address_row][k_half_byte_offset]);

  int32_t d[32] = {};
  uint32_t smem_b_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
  uint64_t desc = make_wgmma_smem_desc<128>(smem_b_addr);

  wgmma_fence();
  wgmma_m64n64k32_s8s8s32(desc, a_regs, d);
  wgmma_commit();
  wgmma_wait<0>();

  for (int i = 0; i < 32; ++i) {
    output_d[tid * 32 + i] = d[i];
  }
}

std::vector<uint8_t> pack_a_k_major() {
  // No XOR here. The v^0x8 "bias trick" (present in an earlier draft of
  // this file, copied from vLLM Marlin's repack step) exists there to
  // align UNSIGNED quantization codes [0,15] with a fused dequant
  // subtraction downstream (decoded later as code-8). a_value() here is
  // already the intended SIGNED value in [-8,7] with no separate dequant
  // step to absorb a bias -- ldmatrix.s8.s4 sign-extends a raw two's
  // complement nibble back to the exact same signed value, so the nibble
  // must be stored as-is. XORing bit 3 of an already-signed nibble maps
  // v -> v-8 (v>=0) or v -> v+8 (v<0) -- a real value-corrupting
  // transformation, not a no-op. This was applied to every (m,k)
  // uniformly, which is why it corrupted the sum-check for BOTH
  // matrix_idx->(m_subgroup,k_half) candidates tried (see
  // part2_v3_candidate1b_output.txt and part2_v4_candidate1a_output.txt)
  // without distinguishing which ordering is actually correct.
  std::vector<uint8_t> packed(kM * kK / 2);
  for (int m = 0; m < kM; ++m) {
    for (int k = 0; k < kK; k += 2) {
      uint8_t low = static_cast<uint8_t>(a_value(m, k)) & 0xF;
      uint8_t high = static_cast<uint8_t>(a_value(m, k + 1)) & 0xF;
      packed[m * (kK / 2) + k / 2] = low | (high << 4);
    }
  }
  return packed;
}

// WGMMA m64n64k32.s32.s8.s8 accumulator (C/D) fragment decode: which (m,n)
// does accumulator register `reg` (0..31) of thread (warp_id,lane) hold.
//
// Derived from CUTLASS's own tested production code
// (cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:433-435,
// CLayout_64xN specialized for N=64), verified as an exact bijection over
// the full 64x64 tile and cross-checked against NVIDIA's own documented
// example ("thread 0 holds (0,0),(0,1),(8,0),(8,1), repeated every 8
// columns"). See dev_probe/WGMMA_D_FRAGMENT_LAYOUT.md and
// dev_probe/wgmma_fragment_layout.py for the full derivation and a
// standalone cross-check against WGMMA's hardware A-operand layout (0
// mismatches across all 2048 warp/lane/register/byte combinations, and a
// confirmed 1024/2048 mismatch against the known-wrong candidate 1a --
// i.e. this decode has real, verified discriminating power, not just a
// plausible-looking formula). This is transcribed from that derivation,
// not re-derived here -- keep the two in sync if either changes.
void d_fragment_mn(int warp_id, int lane, int reg, int* m, int* n) {
  int v0 = reg % 2;
  int v1 = (reg / 2) % 2;
  int v2 = reg / 4;
  *m = 16 * warp_id + (lane / 4) + 8 * v1;
  *n = 2 * (lane % 4) + v0 + 8 * v2;
}

bool validate_wgmma() {
  std::vector<uint8_t> packed_a = pack_a_k_major();
  std::vector<int32_t> output(kThreadsPerGroup * 32);

  uint8_t* device_packed;
  int32_t* device_output;
  CUDA_CHECK(cudaMalloc(&device_packed, packed_a.size()));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(device_packed, packed_a.data(), packed_a.size(),
                        cudaMemcpyHostToDevice));

  wgmma_layout_kernel<<<1, kThreadsPerGroup>>>(device_packed, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_packed));

  // Exact per-(m,n) reference, computed independently (a plain triple loop,
  // sharing no indexing logic with the decode above or with the kernel).
  std::vector<int32_t> reference(kM * kN);
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      int32_t sum = 0;
      for (int k = 0; k < kK; ++k) {
        sum += static_cast<int32_t>(a_value(m, k)) *
               static_cast<int32_t>(b_value(k, n));
      }
      reference[m * kN + n] = sum;
    }
  }

  // Decode every one of the 4096 raw outputs into its claimed (m,n) via
  // d_fragment_mn, tracking how many times each (m,n) cell gets written.
  // A correct, bijective fragment mapping writes every cell exactly once;
  // this catches a wrong mapping that scrambles positions even before
  // looking at the actual values (an earlier version of this check would
  // have silently overwritten a cell with a second, wrong write and never
  // noticed).
  std::vector<int32_t> actual(kM * kN, 0);
  std::vector<int32_t> write_count(kM * kN, 0);
  for (int tid = 0; tid < kThreadsPerGroup; ++tid) {
    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    for (int reg = 0; reg < 32; ++reg) {
      int m = 0, n = 0;
      d_fragment_mn(warp_id, lane, reg, &m, &n);
      actual[m * kN + n] = output[tid * 32 + reg];
      write_count[m * kN + n] += 1;
    }
  }

  int64_t reference_sum = 0, actual_sum = 0;
  for (int32_t v : reference) reference_sum += v;
  for (int32_t v : actual) actual_sum += v;
  std::printf(
      "validate_wgmma: aggregate totals (diagnostic only, not the pass "
      "criterion) -- actual = %lld, reference = %lld\n",
      static_cast<long long>(actual_sum), static_cast<long long>(reference_sum));

  int coverage_errors = 0;
  for (int i = 0; i < kM * kN; ++i) {
    if (write_count[i] != 1) {
      if (coverage_errors < 10) {
        std::fprintf(stderr,
                     "COVERAGE ERROR at (m=%d,n=%d): written %d times "
                     "(expected exactly 1) -- the fragment decode is not a "
                     "bijection for this build\n",
                     i / kN, i % kN, write_count[i]);
      }
      coverage_errors++;
    }
  }
  if (coverage_errors > 0) {
    std::fprintf(stderr,
                 "validate_wgmma: FAILED -- %d of %d output cells were not "
                 "written exactly once by d_fragment_mn's decode\n",
                 coverage_errors, kM * kN);
    return false;
  }

  int mismatches = 0;
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      int idx = m * kN + n;
      if (actual[idx] != reference[idx]) {
        if (mismatches < 20) {
          std::fprintf(stderr,
                       "MISMATCH at (m=%d,n=%d): actual=%d reference=%d\n",
                       m, n, actual[idx], reference[idx]);
        }
        mismatches++;
      }
    }
  }

  if (mismatches > 0) {
    std::fprintf(
        stderr,
        "validate_wgmma: FAILED -- %d of %d elements mismatched "
        "(showing first 20 above). Since int8 x int8 -> int32 accumulation "
        "is exact, this means the register-A fragment mapping under test "
        "is wrong for at least these positions -- not a rounding "
        "difference.\n",
        mismatches, kM * kN);
    return false;
  }

  std::printf(
      "validate_wgmma: PASSED -- all %d elements matched exactly, and the "
      "fragment decode covered every (m,n) cell exactly once. This "
      "confirms the register-A fragment mapping is correct at the "
      "per-element level, not just in aggregate.\n",
      kM * kN);
  return true;
}

}  // namespace

int main() {
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 9 || properties.minor != 0) {
    std::fprintf(stderr, "expected compute capability 9.0, found %d.%d\n",
                 properties.major, properties.minor);
    return EXIT_FAILURE;
  }

  if (!validate_wgmma()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
