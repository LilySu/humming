#pragma once

#include <humming/utils/all.cuh>
#include <humming/utils/ptx/ldmatrix_s4.cuh>


template <class Ctx>
class S2RMemoryLoaderB {
private:
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;

  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;
  static constexpr uint32_t K_WARPS = Ctx::K_WARPS;

  static constexpr bool kIsWarpHalfGroup = WarpShape::N == ElementA::kBits * 2;
  static constexpr bool kLoadHalfGroup = ElementB::kBits % 2 == 0 && kIsWarpHalfGroup && !Ctx::kUsePackedKLayout;
  static constexpr uint32_t TRUE_N_WARPS = kIsWarpHalfGroup ? N_WARPS / 2 : N_WARPS;
  static constexpr uint32_t kSmemStride = BlockShape::N * Ctx::kPartMmaShapeK * ElementB::kBits / 32 / 4 * (Ctx::kUsePackedKLayout ? 2 : 1);
  static constexpr uint32_t kNumIntsPerThread = ElementB::kBits / (kLoadHalfGroup ? 2 : 1);
  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;
  static constexpr uint32_t kLoadIters = kNumIntsPerThread / (sizeof(LoadType) / 4);

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderB(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load_packed_k(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    static_assert(Ctx::kPartMmaShapeK == 32);
    static_assert(ElementA::kBits == 8);
    static_assert(Ctx::kUseWgmma);
    static_assert(Ctx::kWarpIters == WarpShape::N / 16);
    static_assert(WarpShape::K == 128);

    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t lane_id = ctx.lane_id();

    constexpr uint32_t kKChunks = WarpShape::K / 64;
    constexpr uint32_t kPlane = N_WARPS * (WarpShape::N / 16) * 32;

    uint32_t idx = n_warp_id * (WarpShape::N / 16) + iter_id;
    idx = idx * 32 + lane_id;

    if constexpr (K_WARPS > 1) {
      idx += kPlane * kKChunks * ctx.k_warp_id();
    }

    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t kc = 0; kc < kKChunks; kc++) {
      uint32_t smem_start_idx = (idx + kc * kPlane) * kLoadIters;
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        reg_ptr_load[kc * kLoadIters + j] = smem_ptr_load[smem_start_idx + j];
      }
    }
  }

  CUDA_INLINE
  void load_ldmatrix_s4(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    static_assert(Ctx::kPartMmaShapeK == 32);
    static_assert(ElementA::kBits == 8);
    static_assert(Ctx::kUseWgmma);
    static_assert(WarpShape::K % 64 == 0, "load_ldmatrix_s4 requires WarpShape::K a multiple of 64");
    static_assert(BlockShape::N % 16 == 0,
      "signed-S4 slot swizzle assumes 16-row-aligned N tiles (tile row == global row mod 16)");
    static_assert(!Ctx::kUseSignedS4KMajorLayout || kLdmatrixS4Ptx94Available,
                  "signed-S4 K-major layout requires CUDA 13.4+ (PTX ISA 9.4)");

    constexpr uint32_t kNumKChunks = WarpShape::K / 64;
    constexpr uint32_t kPlaneBytes = kSmemStride * sizeof(int4);   // one 64-K plane
    static_assert(kPlaneBytes * (BlockShape::K / 64) == sizeof(Ctx::SharedStorage::StageStorage::b),
                  "S4 planes must exactly fill the B stage buffer");

    uint32_t lane_id = ctx.lane_id();
    uint32_t matrix_idx = lane_id / 8;
    uint32_t row_in_matrix = lane_id % 8;
    uint32_t m_subgroup = matrix_idx % 2;
    uint32_t k_half = matrix_idx / 2;
    uint32_t m_base = ctx.n_warp_offset() + iter_id * 16;
    uint32_t address_row = m_base + m_subgroup * 8 + row_in_matrix;

    const uint8_t *smem_bytes = reinterpret_cast<const uint8_t *>(smem_ptr);
    PRAGMA_UNROLL
    for (uint32_t kc = 0; kc < kNumKChunks; kc++) {
      uint32_t plane_id = ctx.k_warp_id() * kNumKChunks + kc;
      const uint8_t *plane = smem_bytes + plane_id * kPlaneBytes;
      PRAGMA_UNROLL
      for (uint32_t slab = 0; slab < 2; slab++) {
        // Logical slot = slab*2 + k_half; physical slot after the per-row XOR
        // (must match weight_repack_nk's signed-S4 branch).
        uint32_t slot = signed_s4_slot(address_row, slab * 2 + k_half);
        ld_shared_s4x4(&plane[address_row * 32 + slot * 8], &regs_ptr[kc * 8 + slab * 4]);
      }
    }
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    if constexpr (Ctx::kUseSignedS4KMajorLayout) return load_ldmatrix_s4(smem_ptr, regs_ptr, iter_id);
    if constexpr (Ctx::kUsePackedKLayout) return load_packed_k(smem_ptr, regs_ptr, iter_id);
    uint32_t warp_id = ctx.warp_id();
    uint32_t n_warp_id = ctx.n_warp_id();
    if (kIsWarpHalfGroup) n_warp_id = n_warp_id / 2;
    uint32_t lane_id = ctx.lane_id();
    constexpr uint32_t warp_weight_blocks = MAX(WarpShape::N / (ElementA::kBits * 4), 1);
    uint32_t idx = warp_weight_blocks * 32 * n_warp_id + lane_id;

    if constexpr (K_WARPS > 1) {
      idx = TRUE_N_WARPS * 32 * warp_weight_blocks * Ctx::kWarpIters * ctx.k_warp_id() + idx;
    }

    uint32_t smem_start_idx = idx * kLoadIters;
    smem_ptr = smem_ptr + kSmemStride * iter_id;
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < warp_weight_blocks; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        if constexpr (kLoadHalfGroup) {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[(smem_start_idx + 32 * kLoadIters * i) * 2 + warp_id % 2 * kLoadIters + j];
        } else {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[smem_start_idx + 32 * kLoadIters * i + j];
        }
      }
    }
  };
};
