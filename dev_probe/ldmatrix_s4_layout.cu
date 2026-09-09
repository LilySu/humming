// Part 1 of the ptx94-ldmatrix-s8-s4-wgmma-check probe.
//
// Validates ldmatrix.sync.aligned.m8n16.x{1,2,4}.shared::cta.s8.s4 in
// isolation: does the instruction sign-extend and place bytes exactly where
// PTX 9.4 documents it will, on this specific toolkit + GPU. Independent of
// any MMA pairing (mma.sync or wgmma) — same instruction, same guarantee
// either way.
//
// Adapted from WorldExplored's Phase 0 probe for vLLM PR #50096
// (tests/kernels/quantization/marlin_ldmatrix_s4_layout.cu, commit
// 3406be9f0d7), with the mma.sync-specific second half removed — that half
// answered a different question (see wgmma_pairing_check.cu for the one
// that actually matters for Humming's WGMMA path).

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
constexpr int kMatrixRows = 8;
constexpr int kMatrixColumns = 16;
constexpr int kPackedRowBytes = kMatrixColumns / 2;
constexpr int kMaxMatrices = 4;

#define CUDA_CHECK(call)                             \
  do {                                               \
    cudaError_t error = call;                        \
    if (error != cudaSuccess) {                      \
      std::fprintf(stderr, "%s failed: %s\n", #call, \
                   cudaGetErrorString(error));        \
      std::exit(EXIT_FAILURE);                        \
    }                                                \
  } while (0)

template <int NumMatrices>
__device__ inline void load_s4(uint32_t (&registers)[NumMatrices],
                                const void* shared_ptr) {
  uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
  if constexpr (NumMatrices == 1) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4 {%0}, [%1];\n"
        : "=r"(registers[0])
        : "r"(address));
  } else if constexpr (NumMatrices == 2) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared::cta.s8.s4 {%0,%1}, [%2];\n"
        : "=r"(registers[0]), "=r"(registers[1])
        : "r"(address));
  } else {
    static_assert(NumMatrices == 4);
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
          "=r"(registers[3])
        : "r"(address));
  }
}

template <int NumMatrices>
__global__ void load_layout_kernel(const uint8_t* packed, uint32_t* output) {
  __shared__ __align__(16)
      uint8_t shared[NumMatrices][kMatrixRows][kPackedRowBytes];

  int lane = threadIdx.x;
  for (int index = lane; index < NumMatrices * kMatrixRows * kPackedRowBytes;
       index += kWarpSize) {
    reinterpret_cast<uint8_t*>(shared)[index] = packed[index];
  }
  __syncthreads();

  int address_lane = lane < NumMatrices * kMatrixRows ? lane : 0;
  int matrix = address_lane / kMatrixRows;
  int row = address_lane % kMatrixRows;

  uint32_t registers[NumMatrices];
  load_s4(registers, &shared[matrix][row][0]);

#pragma unroll
  for (int i = 0; i < NumMatrices; ++i) {
    output[lane * kMaxMatrices + i] = registers[i];
  }
}

// Same load, but the tile sits `padding_bytes` into a larger shared buffer
// instead of at offset 0 — catches address arithmetic that quietly assumed
// "tile starts at 0".
__global__ void load_layout_kernel_padded(const uint8_t* packed,
                                           uint32_t* output,
                                           int padding_bytes) {
  extern __shared__ uint8_t raw[];
  uint8_t* tile = raw + padding_bytes;

  int lane = threadIdx.x;
  constexpr int tile_bytes = kMaxMatrices * kMatrixRows * kPackedRowBytes;
  for (int index = lane; index < tile_bytes; index += kWarpSize) {
    tile[index] = packed[index];
  }
  __syncthreads();

  int address_lane = lane < kMaxMatrices * kMatrixRows ? lane : 0;
  int matrix = address_lane / kMatrixRows;
  int row = address_lane % kMatrixRows;

  uint32_t registers[kMaxMatrices];
  load_s4(registers, tile + (matrix * kMatrixRows + row) * kPackedRowBytes);

#pragma unroll
  for (int i = 0; i < kMaxMatrices; ++i) {
    output[lane * kMaxMatrices + i] = registers[i];
  }
}

// Two independent tiles, loaded back-to-back by the same warp in one
// kernel launch — the actual shape of a K-loop mainloop's repeated calls,
// not a single isolated one.
__global__ void sequential_load_kernel(const uint8_t* packed_a,
                                        const uint8_t* packed_b,
                                        uint32_t* output_a,
                                        uint32_t* output_b) {
  __shared__ __align__(16)
      uint8_t shared_a[kMaxMatrices][kMatrixRows][kPackedRowBytes];
  __shared__ __align__(16)
      uint8_t shared_b[kMaxMatrices][kMatrixRows][kPackedRowBytes];

  int lane = threadIdx.x;
  constexpr int tile_bytes = kMaxMatrices * kMatrixRows * kPackedRowBytes;
  for (int index = lane; index < tile_bytes; index += kWarpSize) {
    reinterpret_cast<uint8_t*>(shared_a)[index] = packed_a[index];
    reinterpret_cast<uint8_t*>(shared_b)[index] = packed_b[index];
  }
  __syncthreads();

  int address_lane = lane < kMaxMatrices * kMatrixRows ? lane : 0;
  int matrix = address_lane / kMatrixRows;
  int row = address_lane % kMatrixRows;

  uint32_t regs_a[kMaxMatrices];
  load_s4(regs_a, &shared_a[matrix][row][0]);
  uint32_t regs_b[kMaxMatrices];
  load_s4(regs_b, &shared_b[matrix][row][0]);

#pragma unroll
  for (int i = 0; i < kMaxMatrices; ++i) {
    output_a[lane * kMaxMatrices + i] = regs_a[i];
    output_b[lane * kMaxMatrices + i] = regs_b[i];
  }
}

uint8_t get_nibble(const std::vector<uint8_t>& values, int matrix, int row,
                    int column) {
  int index = (matrix * kMatrixRows + row) * kMatrixColumns + column;
  return values[index];
}

std::vector<uint8_t> pack_values(const std::vector<uint8_t>& values,
                                  int num_matrices) {
  std::vector<uint8_t> packed(num_matrices * kMatrixRows * kPackedRowBytes);
  for (int matrix = 0; matrix < num_matrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; column += 2) {
        // The XOR-0x8 bias is the same repack trick the real Humming
        // integration would use: storing v^8 lets s8.s4's hardware sign
        // extension produce v-8 directly, matching a symmetric-int4
        // decode with no separate bias/mask/xor sequence at load time.
        uint8_t low = get_nibble(values, matrix, row, column) ^ 0x8;
        uint8_t high = get_nibble(values, matrix, row, column + 1) ^ 0x8;
        int index =
            (matrix * kMatrixRows + row) * kPackedRowBytes + column / 2;
        packed[index] = low | (high << 4);
      }
    }
  }
  return packed;
}

template <int NumMatrices>
std::vector<uint32_t> run_layout(const std::vector<uint8_t>& values) {
  std::vector<uint8_t> packed = pack_values(values, NumMatrices);
  std::vector<uint32_t> output(kWarpSize * kMaxMatrices);

  uint8_t* device_packed;
  uint32_t* device_output;
  CUDA_CHECK(cudaMalloc(&device_packed, packed.size()));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_packed, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice));

  load_layout_kernel<NumMatrices>
      <<<1, kWarpSize>>>(device_packed, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_packed));
  return output;
}

template <int NumMatrices>
bool validate_layout(const std::vector<uint8_t>& values) {
  std::vector<uint32_t> output = run_layout<NumMatrices>(values);
  for (int matrix = 0; matrix < NumMatrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; ++column) {
        int lane = row * 4 + column / 4;
        int byte = column % 4;
        int8_t actual = static_cast<int8_t>(
            output[lane * kMaxMatrices + matrix] >> (8 * byte));
        int8_t expected =
            static_cast<int8_t>(get_nibble(values, matrix, row, column) - 8);
        if (actual != expected) {
          std::fprintf(
              stderr,
              "x%d mismatch: matrix=%d row=%d column=%d lane=%d byte=%d "
              "expected=%d actual=%d\n",
              NumMatrices, matrix, row, column, lane, byte, expected,
              actual);
          return false;
        }
      }
    }
  }
  return true;
}

template <int NumMatrices>
bool validate_pattern() {
  std::vector<uint8_t> values(NumMatrices * kMatrixRows * kMatrixColumns);
  for (int matrix = 0; matrix < NumMatrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; ++column) {
        int index = (matrix * kMatrixRows + row) * kMatrixColumns + column;
        values[index] =
            static_cast<uint8_t>((column + row * 3 + matrix * 5) & 0xF);
      }
    }
  }
  return validate_layout<NumMatrices>(values);
}

// Exhaustive single-element sensitivity sweep: every source position gets
// set to the one value (15) that's distinguishable from the all-8 baseline
// (8^8=0 after the bias XOR, then 0-8=-8; 15^8=7, then 7-8=-1) one at a
// time. Catches byte/lane swaps that a repeating or symmetric test pattern
// could hide.
bool validate_every_source_element() {
  constexpr int num_values = kMaxMatrices * kMatrixRows * kMatrixColumns;
  std::vector<uint8_t> values(num_values, 8);
  for (int index = 0; index < num_values; ++index) {
    values[index] = 15;
    if (!validate_layout<kMaxMatrices>(values)) {
      std::fprintf(stderr, "source element %d failed\n", index);
      return false;
    }
    values[index] = 8;
  }
  return true;
}

// Row-transition / alignment-boundary coverage (per the review on this
// branch): the tests above always place the tile at the natural start of a
// freshly-sized shared buffer. That can hide an address-computation bug
// that only shows up once the tile sits behind unrelated data, or once a
// warp issues a second ldmatrix call against a different address in the
// same kernel invocation (the real mainloop's actual usage pattern, since
// it iterates over K in a loop, not once).
template <int NumMatrices>
bool validate_layout_with_padding(const std::vector<uint8_t>& values,
                                   int padding_bytes) {
  // Same kernel/addressing as load_layout_kernel, but the tile sits
  // `padding_bytes` into a larger shared buffer instead of at offset 0.
  static_assert(NumMatrices == 4, "padding variant only wired for x4 here");
  std::vector<uint8_t> packed = pack_values(values, NumMatrices);
  std::vector<uint32_t> output(kWarpSize * kMaxMatrices);

  uint8_t* device_packed;
  uint32_t* device_output;
  CUDA_CHECK(cudaMalloc(&device_packed, packed.size()));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_packed, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice));

  constexpr int tile_bytes = kMaxMatrices * kMatrixRows * kPackedRowBytes;
  int shared_bytes = padding_bytes + tile_bytes;
  load_layout_kernel_padded<<<1, kWarpSize, shared_bytes>>>(
      device_packed, device_output, padding_bytes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_packed));

  for (int matrix = 0; matrix < NumMatrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; ++column) {
        int lane = row * 4 + column / 4;
        int byte = column % 4;
        int8_t actual = static_cast<int8_t>(
            output[lane * kMaxMatrices + matrix] >> (8 * byte));
        int8_t expected =
            static_cast<int8_t>(get_nibble(values, matrix, row, column) - 8);
        if (actual != expected) {
          std::fprintf(stderr,
                       "padded(%d) mismatch: matrix=%d row=%d column=%d "
                       "expected=%d actual=%d\n",
                       padding_bytes, matrix, row, column, expected, actual);
          return false;
        }
      }
    }
  }
  return true;
}

bool validate_alignment_boundaries() {
  std::vector<uint8_t> values(kMaxMatrices * kMatrixRows * kMatrixColumns);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint8_t>((i * 5 + i / 7) & 0xF);
  }
  // 16 and 32-byte offsets: the tile no longer starts at a trivially
  // convenient address; if per-thread address arithmetic quietly assumed
  // "tile starts at 0", this is what would catch it.
  for (int padding : {0, 16, 32}) {
    if (!validate_layout_with_padding<4>(values, padding)) return false;
  }
  return true;
}

bool validate_sequential_row_transitions() {
  // Two independent tiles, loaded back-to-back by the same warp in one
  // kernel launch, each with a distinct recognizable pattern — the actual
  // usage shape of a K-loop mainloop, not a single isolated call.
  std::vector<uint8_t> values_a(kMaxMatrices * kMatrixRows * kMatrixColumns);
  std::vector<uint8_t> values_b(kMaxMatrices * kMatrixRows * kMatrixColumns);
  for (size_t i = 0; i < values_a.size(); ++i) {
    values_a[i] = static_cast<uint8_t>((i * 3) & 0xF);
    values_b[i] = static_cast<uint8_t>((i * 11 + 4) & 0xF);
  }
  std::vector<uint8_t> packed_a = pack_values(values_a, 4);
  std::vector<uint8_t> packed_b = pack_values(values_b, 4);
  std::vector<uint32_t> output_a(kWarpSize * kMaxMatrices);
  std::vector<uint32_t> output_b(kWarpSize * kMaxMatrices);

  uint8_t *device_a, *device_b;
  uint32_t *device_out_a, *device_out_b;
  CUDA_CHECK(cudaMalloc(&device_a, packed_a.size()));
  CUDA_CHECK(cudaMalloc(&device_b, packed_b.size()));
  CUDA_CHECK(cudaMalloc(&device_out_a, output_a.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&device_out_b, output_b.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_a, packed_a.data(), packed_a.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, packed_b.data(), packed_b.size(),
                        cudaMemcpyHostToDevice));

  sequential_load_kernel<<<1, kWarpSize>>>(device_a, device_b, device_out_a,
                                           device_out_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output_a.data(), device_out_a,
                        output_a.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output_b.data(), device_out_b,
                        output_b.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_out_a));
  CUDA_CHECK(cudaFree(device_out_b));

  auto check = [](const std::vector<uint8_t>& values,
                  const std::vector<uint32_t>& output, const char* label) {
    for (int matrix = 0; matrix < 4; ++matrix) {
      for (int row = 0; row < kMatrixRows; ++row) {
        for (int column = 0; column < kMatrixColumns; ++column) {
          int lane = row * 4 + column / 4;
          int byte = column % 4;
          int8_t actual = static_cast<int8_t>(
              output[lane * kMaxMatrices + matrix] >> (8 * byte));
          int8_t expected = static_cast<int8_t>(
              get_nibble(values, matrix, row, column) - 8);
          if (actual != expected) {
            std::fprintf(stderr,
                         "sequential[%s] mismatch: matrix=%d row=%d "
                         "column=%d expected=%d actual=%d\n",
                         label, matrix, row, column, expected, actual);
            return false;
          }
        }
      }
    }
    return true;
  };
  return check(values_a, output_a, "first") && check(values_b, output_b, "second");
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

  if (!validate_pattern<1>() || !validate_pattern<2>() ||
      !validate_pattern<4>() || !validate_every_source_element() ||
      !validate_alignment_boundaries() ||
      !validate_sequential_row_transitions()) {
    return EXIT_FAILURE;
  }

  std::printf("ldmatrix.s8.s4 layout matches documented per-thread mapping\n");
  return EXIT_SUCCESS;
}
