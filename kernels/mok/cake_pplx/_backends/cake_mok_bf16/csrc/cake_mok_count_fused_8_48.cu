/*
 * Copyright 2026 Cursor Research
 * Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from the Apache-2.0 Mixture of Kittens BF16 training kernels.
 * Modified: generated CUDA implementation and standalone TVM-FFI bindings.
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
kernel_cake_mok_count_fused_8_48(int* __restrict__ topk_all, int* __restrict__ counts, int* __restrict__ tokens_per_expert, int* __restrict__ num_tokens, int* __restrict__ schedule_peer_rank, int* __restrict__ layout, int* __restrict__ partials, unsigned int* __restrict__ done, int local_tokens, int top_k, int capacity, int rank)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partials));
    __builtin_assume(__isGlobal(partials));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
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
    int* hist = reinterpret_cast<int*>(smem_raw + 0);
    const int hist_addr = smem + 0;
    int* sums = reinterpret_cast<int*>(smem_raw + 224);
    const int sums_addr = smem + 224;
    int* padded = reinterpret_cast<int*>(smem_raw + 1792);
    const int padded_addr = smem + 1792;
    int* real = reinterpret_cast<int*>(smem_raw + 1984);
    const int real_addr = smem + 1984;
    int* offsets = reinterpret_cast<int*>(smem_raw + 2176);
    const int offsets_addr = smem + 2176;
    int* flag = reinterpret_cast<int*>(smem_raw + 2368);
    const int flag_addr = smem + 2368;

    // === Task calls (declaration order) ===
    int rank_stride = local_tokens * top_k;
    int first_expert = rank * 48;
    int last_expert = first_expert + 48;
    for (int i = bid * 1024 + tid; i < capacity; i += num_bids * 1024) {
        schedule_peer_rank[i] = -1;
    }
    int column = tid % 32;
    for (int chunk = bid; chunk < 256; chunk += num_bids) {
        int peer = chunk / 32;
        for (int i_1 = tid; i_1 < 56; i_1 += 1024) {
            hist[i_1] = 0;
        }
        __syncthreads();
        for (int slot = tid / 32 * 1024 + chunk % 32 * 32 + column; slot < rank_stride; slot += 32 * 1024) {
            int expert = topk_all[peer * rank_stride + slot];
            if (expert >= first_expert && expert < last_expert) {
                atomicAdd(&hist[expert - first_expert], 1);
            }
            if (expert >= 0 && expert < 384) {
                atomicAdd(&hist[48 + expert / 48], 1);
            }
        }
        __syncthreads();
        for (int i_2 = tid; i_2 < 56; i_2 += 1024) {
            partials[chunk * 56 + i_2] = hist[i_2];
        }
        __syncthreads();
    }
    for (int i_3 = tid; i_3 < 392; i_3 += 1024) {
        sums[i_3] = 0;
    }
    if (tid == 0) {
        int is_last = 0;
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            is_last = 1;
        }
        flag[0] = is_last;
    }
    __syncthreads();
    int last = flag[0];
    if (last != 0) {
        for (int i_4 = tid; i_4 < 14336; i_4 += 1024) {
            int value = partials[i_4];
            if (value != 0) {
                int word = i_4 % 56;
                if (word < 48) {
                    atomicAdd(&sums[word * 8 + i_4 / 1792], value);
                } else {
                    atomicAdd(&sums[384 + (word - 48)], value);
                }
            }
        }
    }
    __syncthreads();
    if (last != 0) {
        for (int i_5 = tid; i_5 < 392; i_5 += 1024) {
            counts[i_5] = sums[i_5];
        }
    }
    if (last != 0) {
        #pragma unroll 1
        for (int e = tid; e < 48; e += 1024) {
            int total = 0;
            #pragma unroll 1
            for (int q = 0; q < 8; q++) {
                total = total + sums[e * 8 + q];
            }
            int rounded = (total + 255) / 256 * 256;
            real[e] = total;
            padded[e] = rounded;
            tokens_per_expert[e] = rounded;
            layout[e] = rounded;
            layout[96 + e] = total;
        }
    }
    __syncthreads();
    if (last != 0) {
        if (tid / 32 == 0) {
            int lane_1 = tid % 32;
            int carry = 0;
            #pragma unroll 1
            for (int chunk_1 = 0; chunk_1 < 48; chunk_1 += 64) {
                int lo_e = chunk_1 + lane_1;
                int hi_e = chunk_1 + 32 + lane_1;
                int lo_c = 0;
                int hi_c = 0;
                if (lo_e < 48) {
                    lo_c = padded[lo_e];
                }
                if (hi_e < 48) {
                    hi_c = padded[hi_e];
                }
                uint32_t _warp_scan_sum_0 = (unsigned int)lo_c;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 1, 32); if (lane >= 1) { _warp_scan_sum_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 2, 32); if (lane >= 2) { _warp_scan_sum_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 4, 32); if (lane >= 4) { _warp_scan_sum_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 8, 32); if (lane >= 8) { _warp_scan_sum_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 16, 32); if (lane >= 16) { _warp_scan_sum_0 += peer; } }
                int lo_s = (int)_warp_scan_sum_0;
                uint32_t _warp_scan_sum_1 = (unsigned int)hi_c;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 1, 32); if (lane >= 1) { _warp_scan_sum_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 2, 32); if (lane >= 2) { _warp_scan_sum_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 4, 32); if (lane >= 4) { _warp_scan_sum_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 8, 32); if (lane >= 8) { _warp_scan_sum_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 16, 32); if (lane >= 16) { _warp_scan_sum_1 += peer; } }
                int hi_s = (int)_warp_scan_sum_1;
                int lo_t = __shfl_sync(0xFFFFFFFF, lo_s, 31);
                int hi_t = __shfl_sync(0xFFFFFFFF, hi_s, 31);
                if (lo_e < 48) {
                    offsets[lo_e] = carry + lo_s - lo_c;
                    layout[48 + lo_e] = carry + lo_s - lo_c;
                }
                if (hi_e < 48) {
                    offsets[hi_e] = carry + lo_t + hi_s - hi_c;
                    layout[48 + hi_e] = carry + lo_t + hi_s - hi_c;
                }
                carry = carry + lo_t + hi_t;
            }
            if (lane_1 == 0) {
                num_tokens[0] = carry;
            }
        }
    }
    __syncthreads();
    if (last != 0) {
        #pragma unroll 1
        for (int e_1 = tid; e_1 < 48; e_1 += 1024) {
            int start = offsets[e_1];
            int end = ((start + padded[e_1]) < (capacity) ? (start + padded[e_1]) : (capacity));
            #pragma unroll 1
            for (int block = start / 256; block < end / 256; block++) {
                layout[144 + block] = e_1;
            }
        }
    }
}

} // extern "C"
