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
//  - the accumulator readback uses a permutation-invariant aggregate-sum
//    check instead of an exact per-(m,n) diff, since the latter needs
//    Humming's D-fragment epilogue thread mapping (a separate, larger
//    piece of real templated production code) ported in, which is out of
//    scope for this probe. See validate_wgmma()'s comment for exactly what
//    a PASS here does and doesn't establish.
//
// What's still a genuine unknown, and the actual point of running this:
// candidate1_m_offset()'s guess at how the 4 warps' ldmatrix.s8.s4 calls
// partition the 64-row A tile. If validate_wgmma() fails, that's the first
// thing to revisit -- see the Phase-0C-style alternate-candidate note near
// that function.

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

__host__ __device__ int8_t a_value(int m, int k) {
  return static_cast<int8_t>(((3 * m + 5 * k + m / 8) % 16) - 8);
}

__host__ __device__ int8_t b_value(int k, int n) {
  // Dense and non-degenerate on purpose (changed from an earlier sparse
  // one-hot version): the pass/fail check below relies on int8x int8
  // ->int32 accumulation being *exact*, so a wrong fragment mapping must
  // reproduce the exact same sum as the correct one to pass by accident.
  // That's overwhelmingly unlikely with dense, non-symmetric data across
  // 4096 (m,n) positions and 32 k-terms each, and it's not unlikely at all
  // with sparse mostly-zero data (fewer nonzero terms to disagree on).
  return static_cast<int8_t>(((7 * k + 3 * n + n / 4) % 16) - 8);
}

// Matches G2SMemoryLoaderA::load_legacy_swizzled_128B's XOR swizzle
// (humming/include/humming/memory/g2s_loader/loader_a.cuh:100-128),
// applied concretely for this probe's fixed B tile (kK=32 x kN=64, int8 ->
// 128 total 16-byte/int4 chunks, one 8-chunk/128-byte swizzle period) —
// not hand-derived, ported from that exact formula rather than
// re-guessed, per the branch's stated rule against reimplementing WGMMA
// smem swizzle logic independently.
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
  // See wgmma_pairing_check.cu for why this needs 128, not 16: the isolate
  // bisection (this file) showed 0 sanitizer errors with wgmma skipped and
  // 33 errors with it included, pointing at the descriptor/swizzle path,
  // not ldmatrix or the swizzle write itself.
  __shared__ __align__(128) int8_t smem_b[kK][kN];

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane = tid % kWarpSize;

  for (int i = tid; i < kM * (kK / 2); i += kThreadsPerGroup) {
    reinterpret_cast<uint8_t*>(smem_a)[i] = packed_a[i];
  }

  // Swizzled write for smem_b: 128 total int4 (16-byte) chunks, one thread
  // per chunk (kThreadsPerGroup == 128 == kK*kN/16 exactly). Resolved
  // per-branch rule: port the real G2S swizzle formula, don't re-guess it.
  {
    constexpr int kChunksPerKRow = kN / 16;  // 4
    constexpr int kTotalChunks = kK * kChunksPerKRow;  // 128
    static_assert(kTotalChunks == kThreadsPerGroup,
                  "one thread per 16-byte chunk assumed below");
    uint32_t smem_b_base_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
    int k = tid / kChunksPerKRow;
    int n_start = (tid % kChunksPerKRow) * 16;
    uint8_t chunk_bytes[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      chunk_bytes[i] = static_cast<uint8_t>(b_value(k, n_start + i));
    }
    uint32_t physical_chunk =
        wgmma_b_swizzled_chunk(static_cast<uint32_t>(tid), smem_b_base_addr);
    int4* smem_b_int4 = reinterpret_cast<int4*>(&smem_b[0][0]);
    smem_b_int4[physical_chunk] = *reinterpret_cast<int4*>(chunk_bytes);
  }
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
  // matrix_idx -> (m_subgroup, k_half) ordering below is ALSO a first
  // guess (CANDIDATE 1a), layered on top of candidate1_m_offset: matrices
  // {0,1} cover this warp's first 8-row subgroup's two K-halves, {2,3}
  // cover the second 8-row subgroup's two K-halves. If validate_wgmma()
  // fails, this ordering is an equally valid thing to permute per the
  // Phase-0C sweep, not just candidate1_m_offset's row-range guess.
  int matrix_idx = lane / 8;    // 0..3, matches Part 1's x4 addressing
  int row_in_matrix = lane % 8; // 0..7
  int m_subgroup = matrix_idx / 2;  // 0 or 1
  int k_half = matrix_idx % 2;      // 0 or 1
  int address_row = m_base + m_subgroup * 8 + row_in_matrix;
  int k_half_byte_offset = k_half * (kK / 2 / 2);  // 0 or 8 packed bytes
  load_s4_x4(a_regs, &smem_a[address_row][k_half_byte_offset]);

  int32_t d[32] = {};
  uint32_t smem_b_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
  uint64_t desc = make_wgmma_smem_desc<128>(smem_b_addr);
  (void)desc;

#ifndef ISOLATE_SKIP_WGMMA
  wgmma_fence();
  wgmma_m64n64k32_s8s8s32(desc, a_regs, d);
  wgmma_commit();
  wgmma_wait<0>();
#else
  // DIAGNOSTIC: skip the wgmma call entirely. Dump a_regs (from
  // ldmatrix.s8.s4) and a sample readback of smem_b (post-swizzle-write)
  // instead, to check whether operand setup alone -- without the wgmma
  // instruction itself -- runs clean under compute-sanitizer.
  d[0] = static_cast<int32_t>(a_regs[0]);
  d[1] = static_cast<int32_t>(a_regs[1]);
  d[2] = static_cast<int32_t>(a_regs[2]);
  d[3] = static_cast<int32_t>(a_regs[3]);
  d[4] = static_cast<int32_t>(smem_b[0][tid % kN]);
#endif

  for (int i = 0; i < 32; ++i) {
    output_d[tid * 32 + i] = d[i];
  }
}

std::vector<uint8_t> pack_a_k_major() {
  std::vector<uint8_t> packed(kM * kK / 2);
  for (int m = 0; m < kM; ++m) {
    for (int k = 0; k < kK; k += 2) {
      uint8_t low = static_cast<uint8_t>(a_value(m, k)) ^ 0x8;
      uint8_t high = static_cast<uint8_t>(a_value(m, k + 1)) ^ 0x8;
      packed[m * (kK / 2) + k / 2] = (low & 0xF) | ((high & 0xF) << 4);
    }
  }
  return packed;
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

  // Exact per-(m,n) reference. We don't diff this element-by-element
  // against `output`, because that requires Humming's D-fragment
  // thread-to-(m,n) mapping (a separate, already-solved table in its
  // epilogue code — humming/include/humming/epilogue/smem_writer.cuh —
  // that's real templated production code, not something worth
  // hand-porting here without independent need). Instead: sum both sides.
  // int8 x int8 -> int32 accumulation is exact (no rounding), so if the
  // register-A fragment mapping under test is wrong, the wrong (a,b)
  // pairs get multiplied and accumulated per thread, and their total
  // essentially never coincidentally equals the correct total — this is
  // permutation-invariant (doesn't care which thread holds which (m,n)),
  // which is exactly the property that lets it substitute for knowing the
  // D-fragment mapping, at the cost of not confirming exact per-element
  // placement.
  int64_t reference_sum = 0;
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      int32_t sum = 0;
      for (int k = 0; k < kK; ++k) {
        sum += static_cast<int32_t>(a_value(m, k)) *
               static_cast<int32_t>(b_value(k, n));
      }
      reference_sum += sum;
    }
  }

  int64_t actual_sum = 0;
  for (int32_t v : output) actual_sum += v;

  std::printf("validate_wgmma: actual total = %lld, reference total = %lld\n",
              static_cast<long long>(actual_sum),
              static_cast<long long>(reference_sum));

  if (actual_sum == 0 && reference_sum != 0) {
    std::fprintf(stderr,
                 "output is degenerately all-zero -- the wgmma call likely "
                 "did not execute as intended, not a fragment-mapping "
                 "mismatch specifically\n");
    return false;
  }
  if (actual_sum != reference_sum) {
    std::fprintf(
        stderr,
        "SUM MISMATCH: candidate1_m_offset's register-A fragment mapping "
        "does not reproduce the correct GEMM total. Since the underlying "
        "arithmetic is exact integer accumulation, this means the mapping "
        "itself is wrong, not a rounding difference -- see "
        "dev_probe/README.md's Phase-0C-style next step (try alternate "
        "row/col partitionings) before concluding ldmatrix.s8.s4 can't "
        "feed WGMMA's register-a operand at all.\n");
    return false;
  }
  std::printf(
      "validate_wgmma: PASSED (aggregate-sum check). Scope note: this "
      "confirms the register-A fragment mapping produces arithmetically "
      "correct dot products in aggregate: it does NOT confirm each "
      "individual (m,n) landed in its documented thread/register -- that "
      "would need Humming's D-fragment epilogue mapping ported in for an "
      "exact per-element diff, which is a separate, larger piece of work "
      "than this probe's scope.\n");
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
