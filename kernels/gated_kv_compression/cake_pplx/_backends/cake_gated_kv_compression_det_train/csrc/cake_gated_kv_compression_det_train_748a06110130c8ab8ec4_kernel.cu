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

#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_kv_compression_det_train_748a06110130c8ab8ec4(float* c, float* z, int* cu_seqlens, int* cu_pairs, __nv_bfloat16* out, int num_docs, int G, long long c_ld, long long z_ld)
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
    int* sseq = reinterpret_cast<int*>(smem_raw + 32);
    const int sseq_addr = smem + 32;
    int* spairs = reinterpret_cast<int*>(smem_raw + 1056);
    const int spairs_addr = smem + 1056;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int slot = tid >> 7;
    long long col = (long long)((tid & 127) * 4);
    int pairs = 0;
    int _count_leading_zeros_0 = __clz(num_docs - 1);
    int steps = 32 - _count_leading_zeros_0;
    if (num_docs <= 256) {
        int half = 0;
        if (tid < num_docs) {
            int begin = cu_seqlens[tid];
            half = cu_seqlens[tid + 1] - begin >> 1;
            sseq[tid] = begin;
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
        if (tid < num_docs) {
            spairs[tid] = prefix;
        }
        if (bid == 0) {
            if (tid < num_docs) {
                cu_pairs[tid] = prefix;
            }
            if (tid == 0) {
                cu_pairs[num_docs] = total;
            }
        }
        __syncthreads();
        pairs = total;
    } else {
        pairs = cu_pairs[num_docs];
    }
    int pooled_rows = ((G) < (pairs) ? (G) : (pairs));
    int nblocks = (G + 1) / 2;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 2 + slot;
        float vals_0[16];
        if (row_0 < pooled_rows) {
            int start_0 = 0;
            if (num_docs <= 256) {
                int doc = 0;
                int last = num_docs - 1;
                #pragma unroll 1
                for (int _trip = 0; _trip < steps; _trip++) {
                    int mid = doc + last + 1 >> 1;
                    if (row_0 >= spairs[mid]) {
                        doc = mid;
                    } else {
                        last = mid - 1;
                    }
                }
                int start = sseq[doc] + 2 * (row_0 - spairs[doc]);
                start_0 = start;
            } else {
                int doc_1 = 0;
                int last_1 = num_docs - 1;
                #pragma unroll 1
                for (int _trip_1 = 0; _trip_1 < steps; _trip_1++) {
                    int mid_1 = doc_1 + last_1 + 1 >> 1;
                    if (row_0 >= cu_pairs[mid_1]) {
                        doc_1 = mid_1;
                    } else {
                        last_1 = mid_1 - 1;
                    }
                }
                int start_1 = cu_seqlens[doc_1] + 2 * (row_0 - cu_pairs[doc_1]);
                start_0 = start_1;
            }
            long long at_c_0 = (long long)start_0 * c_ld + col;
            long long at_z_0 = (long long)start_0 * z_ld + col;
            {
                unsigned _v4_1_0;
                unsigned _v4_1_1;
                unsigned _v4_1_2;
                unsigned _v4_1_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
                vals_0[0 + 0] = __uint_as_float(_v4_1_0);
                vals_0[0 + 1] = __uint_as_float(_v4_1_1);
                vals_0[0 + 2] = __uint_as_float(_v4_1_2);
                vals_0[0 + 3] = __uint_as_float(_v4_1_3);
            }
            {
                unsigned _v4_2_0;
                unsigned _v4_2_1;
                unsigned _v4_2_2;
                unsigned _v4_2_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
                vals_0[4 + 0] = __uint_as_float(_v4_2_0);
                vals_0[4 + 1] = __uint_as_float(_v4_2_1);
                vals_0[4 + 2] = __uint_as_float(_v4_2_2);
                vals_0[4 + 3] = __uint_as_float(_v4_2_3);
            }
            {
                unsigned _v4_3_0;
                unsigned _v4_3_1;
                unsigned _v4_3_2;
                unsigned _v4_3_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
                vals_0[8 + 0] = __uint_as_float(_v4_3_0);
                vals_0[8 + 1] = __uint_as_float(_v4_3_1);
                vals_0[8 + 2] = __uint_as_float(_v4_3_2);
                vals_0[8 + 3] = __uint_as_float(_v4_3_3);
            }
            {
                unsigned _v4_4_0;
                unsigned _v4_4_1;
                unsigned _v4_4_2;
                unsigned _v4_4_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
                vals_0[12 + 0] = __uint_as_float(_v4_4_0);
                vals_0[12 + 1] = __uint_as_float(_v4_4_1);
                vals_0[12 + 2] = __uint_as_float(_v4_4_2);
                vals_0[12 + 3] = __uint_as_float(_v4_4_3);
            }
        }
        unsigned int words[2] = {0};
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[12]), "f"(vals_0[8]));
            float _abs_6;
            asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(gap));
            float _neg_5;
            asm("neg.f32 %0, %1;" : "=f"(_neg_5) : "f"(_abs_6));
            float e = __nv_expf(_neg_5);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float big = ((first == 1) ? vals_0[0] : vals_0[4]);
            float small = ((first == 1) ? vals_0[4] : vals_0[0]);
            float _fma_8;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(e), "f"(small), "f"(big));
            float _fdiv_7 = __fdiv_rn(_fma_8, den);
            vals_0[0] = _fdiv_7;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[13]), "f"(vals_0[9]));
            float _abs_10;
            asm("abs.f32 %0, %1;" : "=f"(_abs_10) : "f"(gap_0));
            float _neg_9;
            asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(_abs_10));
            float e_0 = __nv_expf(_neg_9);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float big_0 = ((first_0 == 1) ? vals_0[1] : vals_0[5]);
            float small_0 = ((first_0 == 1) ? vals_0[5] : vals_0[1]);
            float _fma_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(e_0), "f"(small_0), "f"(big_0));
            float _fdiv_11 = __fdiv_rn(_fma_12, den_0);
            vals_0[1] = _fdiv_11;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[14]), "f"(vals_0[10]));
            float _abs_14;
            asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(gap_1));
            float _neg_13;
            asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(_abs_14));
            float e_1 = __nv_expf(_neg_13);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float big_1 = ((first_1 == 1) ? vals_0[2] : vals_0[6]);
            float small_1 = ((first_1 == 1) ? vals_0[6] : vals_0[2]);
            float _fma_16;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(e_1), "f"(small_1), "f"(big_1));
            float _fdiv_15 = __fdiv_rn(_fma_16, den_1);
            vals_0[2] = _fdiv_15;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[15]), "f"(vals_0[11]));
            float _abs_18;
            asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(gap_2));
            float _neg_17;
            asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(_abs_18));
            float e_2 = __nv_expf(_neg_17);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float big_2 = ((first_2 == 1) ? vals_0[3] : vals_0[7]);
            float small_2 = ((first_2 == 1) ? vals_0[7] : vals_0[3]);
            float _fma_20;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(e_2), "f"(small_2), "f"(big_2));
            float _fdiv_19 = __fdiv_rn(_fma_20, den_2);
            vals_0[3] = _fdiv_19;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                words[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 2], vals_0[_lp*2+1 + 2]));
                words[_lp + 1] = *(uint32_t*)&_bf2;
            }
        }
        if (row_0 < G) {
            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(out + ((long long)row_0 * 512ll + col)))[0] = reinterpret_cast<int2*>(words)[0];
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
