/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gated_kv_compression_det_train_68cbef2c67f1d183366f(int* cu_seqlens, int* cu_pairs, int num_docs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* warp_tot = reinterpret_cast<int*>(smem_raw + 0);
    const int warp_tot_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int carry = 0;
    #pragma unroll 1
    for (int base = 0; base < num_docs; base += 256) {
        int doc = base + tid;
        int half = 0;
        if (doc < num_docs) {
            half = cu_seqlens[doc + 1] - cu_seqlens[doc] >> 1;
        }
        // CtaScanSum: threads=256, warps=8
        int prefix;
        int total;
        {
            const uint32_t prefix_own = (uint32_t)(half);
            uint32_t prefix_ws = prefix_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, prefix_ws, 1, 32); if (lane >= 1) { prefix_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, prefix_ws, 2, 32); if (lane >= 2) { prefix_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, prefix_ws, 4, 32); if (lane >= 4) { prefix_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, prefix_ws, 8, 32); if (lane >= 8) { prefix_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, prefix_ws, 16, 32); if (lane >= 16) { prefix_ws += peer; } }
            if (lane == 31) { warp_tot[warp] = (int)prefix_ws; }
            __syncthreads();
            uint32_t prefix_before = 0;
            uint32_t total_sum = 0;
            { const uint32_t t = (uint32_t)warp_tot[0]; total_sum += t;
              if (warp > 0) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[1]; total_sum += t;
              if (warp > 1) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[2]; total_sum += t;
              if (warp > 2) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[3]; total_sum += t;
              if (warp > 3) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[4]; total_sum += t;
              if (warp > 4) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[5]; total_sum += t;
              if (warp > 5) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[6]; total_sum += t;
              if (warp > 6) { prefix_before += t; } }
            { const uint32_t t = (uint32_t)warp_tot[7]; total_sum += t;
              if (warp > 7) { prefix_before += t; } }
            total = (int)total_sum;
            prefix = (int)(prefix_ws - prefix_own + prefix_before);
        }
        if (doc < num_docs) {
            cu_pairs[doc] = carry + prefix;
        }
        carry = carry + total;
        __syncthreads();
    }
    if (tid == 0) {
        cu_pairs[num_docs] = carry;
    }
}

} // extern "C"
