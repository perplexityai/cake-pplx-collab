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
#if defined(__CUDACC_RTC__)
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

__global__ __launch_bounds__(1024) void
kernel_cake_gated_kv_compression_det_train_226d77a045d657b1ed90(int* cu_seqlens, int* workspace, int num_docs)
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
    int chunk = bid * 1024;
    int first = chunk + warp * 32;
    int lower[2];
    #pragma unroll
    for (int j = 0; j < 1; j++) {
        int _min_0 = ((first + 32 * j + lane) < (num_docs) ? (first + 32 * j + lane) : (num_docs));
        lower[j] = cu_seqlens[_min_0];
    }
    int _min_1 = ((first + 32) < (num_docs) ? (first + 32) : (num_docs));
    lower[1] = cu_seqlens[_min_1];
    int own = 0;
    #pragma unroll
    for (int j_1 = 0; j_1 < 1; j_1++) {
        int handed = ((lane == 0) ? lower[j_1 + 1] : lower[j_1]);
        int upper = __shfl_sync(0xFFFFFFFF, handed, lane + 1 & 31);
        own = own + (upper - lower[j_1] >> 1);
    }
    // CtaScanSum: threads=1024, warps=32
    int own_prefix;
    int chunk_total;
    {
        const uint32_t own_prefix_own = (uint32_t)(own);
        uint32_t own_prefix_ws = own_prefix_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_ws, 1, 32); if (lane >= 1) { own_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_ws, 2, 32); if (lane >= 2) { own_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_ws, 4, 32); if (lane >= 4) { own_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_ws, 8, 32); if (lane >= 8) { own_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_ws, 16, 32); if (lane >= 16) { own_prefix_ws += peer; } }
        if (lane == 31) { warp_tot[warp] = (int)own_prefix_ws; }
        __syncthreads();
        const uint32_t own_prefix_wt = (uint32_t)warp_tot[lane];
        uint32_t own_prefix_wsum = own_prefix_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_wsum, 1, 32); if (lane >= 1) { own_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_wsum, 2, 32); if (lane >= 2) { own_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_wsum, 4, 32); if (lane >= 4) { own_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_wsum, 8, 32); if (lane >= 8) { own_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, own_prefix_wsum, 16, 32); if (lane >= 16) { own_prefix_wsum += peer; } }
        chunk_total = (int)__shfl_sync(0xFFFFFFFF, own_prefix_wsum, 31);
        const uint32_t own_prefix_before = __shfl_sync(0xFFFFFFFF, own_prefix_wsum - own_prefix_wt, warp);
        own_prefix = (int)(own_prefix_ws - own_prefix_own + own_prefix_before);
    }
    if (tid == 0) {
        workspace[bid] = chunk_total;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
