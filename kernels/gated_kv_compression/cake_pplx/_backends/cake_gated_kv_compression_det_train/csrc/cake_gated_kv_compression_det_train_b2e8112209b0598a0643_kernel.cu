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

__global__ __launch_bounds__(1024) void
kernel_cake_gated_kv_compression_det_train_b2e8112209b0598a0643(int* cu_seqlens, int* cu_pairs, int* workspace, int num_docs)
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
    int* red_tot = reinterpret_cast<int*>(smem_raw + 128);
    const int red_tot_addr = smem + 128;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int chunk = bid * 8192;
    int first = chunk + warp * 256;
    int lower[9];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int _min_0 = ((first + 32 * j + lane) < (num_docs) ? (first + 32 * j + lane) : (num_docs));
        lower[j] = cu_seqlens[_min_0];
    }
    int _min_1 = ((first + 256) < (num_docs) ? (first + 256) : (num_docs));
    lower[8] = cu_seqlens[_min_1];
    int before = 0;
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        int earlier_chunk = tid + 1024 * k;
        if (earlier_chunk < bid) {
            before = before + workspace[earlier_chunk];
        }
    }
    // CtaScanSum: threads=1024, warps=32
    int before_prefix;
    int carry;
    {
        const uint32_t before_prefix_own = (uint32_t)(before);
        uint32_t before_prefix_ws = before_prefix_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_ws, 1, 32); if (lane >= 1) { before_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_ws, 2, 32); if (lane >= 2) { before_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_ws, 4, 32); if (lane >= 4) { before_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_ws, 8, 32); if (lane >= 8) { before_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_ws, 16, 32); if (lane >= 16) { before_prefix_ws += peer; } }
        if (lane == 31) { red_tot[warp] = (int)before_prefix_ws; }
        __syncthreads();
        const uint32_t before_prefix_wt = (uint32_t)red_tot[lane];
        uint32_t before_prefix_wsum = before_prefix_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_wsum, 1, 32); if (lane >= 1) { before_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_wsum, 2, 32); if (lane >= 2) { before_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_wsum, 4, 32); if (lane >= 4) { before_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_wsum, 8, 32); if (lane >= 8) { before_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, before_prefix_wsum, 16, 32); if (lane >= 16) { before_prefix_wsum += peer; } }
        carry = (int)__shfl_sync(0xFFFFFFFF, before_prefix_wsum, 31);
        const uint32_t before_prefix_before = __shfl_sync(0xFFFFFFFF, before_prefix_wsum - before_prefix_wt, warp);
        before_prefix = (int)(before_prefix_ws - before_prefix_own + before_prefix_before);
    }
    int entries[8];
    int warp_carry = 0;
    #pragma unroll
    for (int j_1 = 0; j_1 < 8; j_1++) {
        int handed = ((lane == 0) ? lower[j_1 + 1] : lower[j_1]);
        int upper = __shfl_sync(0xFFFFFFFF, handed, lane + 1 & 31);
        int row_half = upper - lower[j_1] >> 1;
        int row_prefix = row_half;
        {
            uint32_t row_prefix_scan = (uint32_t)row_prefix;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, row_prefix_scan, 1, 32); if (lane >= 1) { row_prefix_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, row_prefix_scan, 2, 32); if (lane >= 2) { row_prefix_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, row_prefix_scan, 4, 32); if (lane >= 4) { row_prefix_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, row_prefix_scan, 8, 32); if (lane >= 8) { row_prefix_scan += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, row_prefix_scan, 16, 32); if (lane >= 16) { row_prefix_scan += peer; } }
            row_prefix = (int)row_prefix_scan;
        }
        entries[j_1] = warp_carry + row_prefix - row_half;
        int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, row_prefix, 31);
        warp_carry = warp_carry + _shfl_sync_2;
    }
    int warp_total = ((lane == 31) ? warp_carry : 0);
    // CtaScanSum: threads=1024, warps=32
    int offset;
    int chunk_total;
    {
        const uint32_t offset_own = (uint32_t)(warp_total);
        uint32_t offset_ws = offset_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_ws, 1, 32); if (lane >= 1) { offset_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_ws, 2, 32); if (lane >= 2) { offset_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_ws, 4, 32); if (lane >= 4) { offset_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_ws, 8, 32); if (lane >= 8) { offset_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_ws, 16, 32); if (lane >= 16) { offset_ws += peer; } }
        if (lane == 31) { warp_tot[warp] = (int)offset_ws; }
        __syncthreads();
        const uint32_t offset_wt = (uint32_t)warp_tot[lane];
        uint32_t offset_wsum = offset_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_wsum, 1, 32); if (lane >= 1) { offset_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_wsum, 2, 32); if (lane >= 2) { offset_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_wsum, 4, 32); if (lane >= 4) { offset_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_wsum, 8, 32); if (lane >= 8) { offset_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, offset_wsum, 16, 32); if (lane >= 16) { offset_wsum += peer; } }
        chunk_total = (int)__shfl_sync(0xFFFFFFFF, offset_wsum, 31);
        const uint32_t offset_before = __shfl_sync(0xFFFFFFFF, offset_wsum - offset_wt, warp);
        offset = (int)(offset_ws - offset_own + offset_before);
    }
    if (first < num_docs) {
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            int row_doc = first + 32 * j_2 + lane;
            if (row_doc < num_docs) {
                cu_pairs[row_doc] = carry + offset + entries[j_2];
            }
        }
    }
    if (tid == 0) {
        if (chunk + 8192 >= num_docs) {
            cu_pairs[num_docs] = carry + chunk_total;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
