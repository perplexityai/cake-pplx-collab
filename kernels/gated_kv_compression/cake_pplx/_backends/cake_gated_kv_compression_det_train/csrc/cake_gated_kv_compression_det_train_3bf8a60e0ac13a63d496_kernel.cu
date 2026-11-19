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

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gated_kv_compression_det_train_3bf8a60e0ac13a63d496(float* c, float* z, int* cu_seqlens, int* cu_pairs, __nv_bfloat16* out, int num_docs, int G, long long c_ld, long long z_ld)
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
    int slot = tid >> 6;
    long long col = (long long)((tid & 63) * 8);
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
    int nblocks = (G + 3) / 4;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 4 + slot;
        float vals_0[32];
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
                unsigned _ldv8_1_0;
                unsigned _ldv8_1_1;
                unsigned _ldv8_1_2;
                unsigned _ldv8_1_3;
                unsigned _ldv8_1_4;
                unsigned _ldv8_1_5;
                unsigned _ldv8_1_6;
                unsigned _ldv8_1_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
                vals_0[0 + 0] = __uint_as_float(_ldv8_1_0);
                vals_0[0 + 1] = __uint_as_float(_ldv8_1_1);
                vals_0[0 + 2] = __uint_as_float(_ldv8_1_2);
                vals_0[0 + 3] = __uint_as_float(_ldv8_1_3);
                vals_0[0 + 4] = __uint_as_float(_ldv8_1_4);
                vals_0[0 + 5] = __uint_as_float(_ldv8_1_5);
                vals_0[0 + 6] = __uint_as_float(_ldv8_1_6);
                vals_0[0 + 7] = __uint_as_float(_ldv8_1_7);
            }
            {
                unsigned _ldv8_2_0;
                unsigned _ldv8_2_1;
                unsigned _ldv8_2_2;
                unsigned _ldv8_2_3;
                unsigned _ldv8_2_4;
                unsigned _ldv8_2_5;
                unsigned _ldv8_2_6;
                unsigned _ldv8_2_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
                vals_0[8 + 0] = __uint_as_float(_ldv8_2_0);
                vals_0[8 + 1] = __uint_as_float(_ldv8_2_1);
                vals_0[8 + 2] = __uint_as_float(_ldv8_2_2);
                vals_0[8 + 3] = __uint_as_float(_ldv8_2_3);
                vals_0[8 + 4] = __uint_as_float(_ldv8_2_4);
                vals_0[8 + 5] = __uint_as_float(_ldv8_2_5);
                vals_0[8 + 6] = __uint_as_float(_ldv8_2_6);
                vals_0[8 + 7] = __uint_as_float(_ldv8_2_7);
            }
            {
                unsigned _ldv8_3_0;
                unsigned _ldv8_3_1;
                unsigned _ldv8_3_2;
                unsigned _ldv8_3_3;
                unsigned _ldv8_3_4;
                unsigned _ldv8_3_5;
                unsigned _ldv8_3_6;
                unsigned _ldv8_3_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
                vals_0[16 + 0] = __uint_as_float(_ldv8_3_0);
                vals_0[16 + 1] = __uint_as_float(_ldv8_3_1);
                vals_0[16 + 2] = __uint_as_float(_ldv8_3_2);
                vals_0[16 + 3] = __uint_as_float(_ldv8_3_3);
                vals_0[16 + 4] = __uint_as_float(_ldv8_3_4);
                vals_0[16 + 5] = __uint_as_float(_ldv8_3_5);
                vals_0[16 + 6] = __uint_as_float(_ldv8_3_6);
                vals_0[16 + 7] = __uint_as_float(_ldv8_3_7);
            }
            {
                unsigned _ldv8_4_0;
                unsigned _ldv8_4_1;
                unsigned _ldv8_4_2;
                unsigned _ldv8_4_3;
                unsigned _ldv8_4_4;
                unsigned _ldv8_4_5;
                unsigned _ldv8_4_6;
                unsigned _ldv8_4_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
                vals_0[24 + 0] = __uint_as_float(_ldv8_4_0);
                vals_0[24 + 1] = __uint_as_float(_ldv8_4_1);
                vals_0[24 + 2] = __uint_as_float(_ldv8_4_2);
                vals_0[24 + 3] = __uint_as_float(_ldv8_4_3);
                vals_0[24 + 4] = __uint_as_float(_ldv8_4_4);
                vals_0[24 + 5] = __uint_as_float(_ldv8_4_5);
                vals_0[24 + 6] = __uint_as_float(_ldv8_4_6);
                vals_0[24 + 7] = __uint_as_float(_ldv8_4_7);
            }
        }
        unsigned int words[4] = {0};
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[24]), "f"(vals_0[16]));
            float _abs_6;
            asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(gap));
            float _neg_5;
            asm("neg.f32 %0, %1;" : "=f"(_neg_5) : "f"(_abs_6));
            float e = __nv_expf(_neg_5);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float big = ((first == 1) ? vals_0[0] : vals_0[8]);
            float small = ((first == 1) ? vals_0[8] : vals_0[0]);
            float _fma_8;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(e), "f"(small), "f"(big));
            float _fdiv_7 = __fdiv_rn(_fma_8, den);
            vals_0[0] = _fdiv_7;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[25]), "f"(vals_0[17]));
            float _abs_10;
            asm("abs.f32 %0, %1;" : "=f"(_abs_10) : "f"(gap_0));
            float _neg_9;
            asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(_abs_10));
            float e_0 = __nv_expf(_neg_9);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float big_0 = ((first_0 == 1) ? vals_0[1] : vals_0[9]);
            float small_0 = ((first_0 == 1) ? vals_0[9] : vals_0[1]);
            float _fma_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(e_0), "f"(small_0), "f"(big_0));
            float _fdiv_11 = __fdiv_rn(_fma_12, den_0);
            vals_0[1] = _fdiv_11;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[26]), "f"(vals_0[18]));
            float _abs_14;
            asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(gap_1));
            float _neg_13;
            asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(_abs_14));
            float e_1 = __nv_expf(_neg_13);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float big_1 = ((first_1 == 1) ? vals_0[2] : vals_0[10]);
            float small_1 = ((first_1 == 1) ? vals_0[10] : vals_0[2]);
            float _fma_16;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(e_1), "f"(small_1), "f"(big_1));
            float _fdiv_15 = __fdiv_rn(_fma_16, den_1);
            vals_0[2] = _fdiv_15;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[27]), "f"(vals_0[19]));
            float _abs_18;
            asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(gap_2));
            float _neg_17;
            asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(_abs_18));
            float e_2 = __nv_expf(_neg_17);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float big_2 = ((first_2 == 1) ? vals_0[3] : vals_0[11]);
            float small_2 = ((first_2 == 1) ? vals_0[11] : vals_0[3]);
            float _fma_20;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(e_2), "f"(small_2), "f"(big_2));
            float _fdiv_19 = __fdiv_rn(_fma_20, den_2);
            vals_0[3] = _fdiv_19;
            float gap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_3) : "f"(vals_0[28]), "f"(vals_0[20]));
            float _abs_22;
            asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(gap_3));
            float _neg_21;
            asm("neg.f32 %0, %1;" : "=f"(_neg_21) : "f"(_abs_22));
            float e_3 = __nv_expf(_neg_21);
            float den_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_3) : "f"(e_3), "f"(1.0f));
            int first_3 = ((gap_3 <= 0.0f) ? 1 : 0);
            float big_3 = ((first_3 == 1) ? vals_0[4] : vals_0[12]);
            float small_3 = ((first_3 == 1) ? vals_0[12] : vals_0[4]);
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(e_3), "f"(small_3), "f"(big_3));
            float _fdiv_23 = __fdiv_rn(_fma_24, den_3);
            vals_0[4] = _fdiv_23;
            float gap_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_4) : "f"(vals_0[29]), "f"(vals_0[21]));
            float _abs_26;
            asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(gap_4));
            float _neg_25;
            asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(_abs_26));
            float e_4 = __nv_expf(_neg_25);
            float den_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_4) : "f"(e_4), "f"(1.0f));
            int first_4 = ((gap_4 <= 0.0f) ? 1 : 0);
            float big_4 = ((first_4 == 1) ? vals_0[5] : vals_0[13]);
            float small_4 = ((first_4 == 1) ? vals_0[13] : vals_0[5]);
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(e_4), "f"(small_4), "f"(big_4));
            float _fdiv_27 = __fdiv_rn(_fma_28, den_4);
            vals_0[5] = _fdiv_27;
            float gap_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_5) : "f"(vals_0[30]), "f"(vals_0[22]));
            float _abs_30;
            asm("abs.f32 %0, %1;" : "=f"(_abs_30) : "f"(gap_5));
            float _neg_29;
            asm("neg.f32 %0, %1;" : "=f"(_neg_29) : "f"(_abs_30));
            float e_5 = __nv_expf(_neg_29);
            float den_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_5) : "f"(e_5), "f"(1.0f));
            int first_5 = ((gap_5 <= 0.0f) ? 1 : 0);
            float big_5 = ((first_5 == 1) ? vals_0[6] : vals_0[14]);
            float small_5 = ((first_5 == 1) ? vals_0[14] : vals_0[6]);
            float _fma_32;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(e_5), "f"(small_5), "f"(big_5));
            float _fdiv_31 = __fdiv_rn(_fma_32, den_5);
            vals_0[6] = _fdiv_31;
            float gap_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_6) : "f"(vals_0[31]), "f"(vals_0[23]));
            float _abs_34;
            asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(gap_6));
            float _neg_33;
            asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(_abs_34));
            float e_6 = __nv_expf(_neg_33);
            float den_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_6) : "f"(e_6), "f"(1.0f));
            int first_6 = ((gap_6 <= 0.0f) ? 1 : 0);
            float big_6 = ((first_6 == 1) ? vals_0[7] : vals_0[15]);
            float small_6 = ((first_6 == 1) ? vals_0[15] : vals_0[7]);
            float _fma_36;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(e_6), "f"(small_6), "f"(big_6));
            float _fdiv_35 = __fdiv_rn(_fma_36, den_6);
            vals_0[7] = _fdiv_35;
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
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 4], vals_0[_lp*2+1 + 4]));
                words[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 6], vals_0[_lp*2+1 + 6]));
                words[_lp + 3] = *(uint32_t*)&_bf2;
            }
        }
        if (row_0 < G) {
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + ((long long)row_0 * 512ll + col)))[0] = reinterpret_cast<int4*>(words)[0];
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
