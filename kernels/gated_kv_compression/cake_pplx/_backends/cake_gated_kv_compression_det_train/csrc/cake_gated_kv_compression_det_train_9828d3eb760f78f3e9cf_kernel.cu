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
kernel_cake_gated_kv_compression_det_train_9828d3eb760f78f3e9cf(float* c, float* z, int* cu_seqlens, int* cu_pairs, __nv_bfloat16* out, int num_docs, int G, long long c_ld, long long z_ld)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int slot = tid >> 6;
    long long col = (long long)((tid & 63) * 8);
    int pooled_rows = ((G) < (cu_pairs[num_docs]) ? (G) : (cu_pairs[num_docs]));
    int _count_leading_zeros_0 = __clz(num_docs - 1 | 1);
    int rounds = (36 - _count_leading_zeros_0) / 5;
    int nblocks = (G + 3) / 4;
    float held_vals_0[32];
    float held_vals_1[32];
    int row_0 = bid * 4 + slot;
    if (row_0 < pooled_rows) {
        int lo = 0;
        int shift = 5 * (rounds - 1);
        int hits = 1;
        int entry = 0;
        int origin = 0;
        #pragma unroll 1
        for (int _trip = 0; _trip < rounds; _trip++) {
            int probe = lo + (lane << shift);
            int hit = 0;
            if (probe < num_docs) {
                entry = cu_pairs[probe];
                if (shift == 0) {
                    origin = cu_seqlens[probe];
                }
                if (entry <= row_0) {
                    hit = 1;
                }
            }
            int _warp_redux_1;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_1) : "r"(hit));
            hits = (int)_warp_redux_1;
            lo = lo + (hits - 1 << shift);
            shift = shift - 5;
        }
        int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, origin, hits - 1);
        int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, entry, hits - 1);
        int start = _shfl_sync_2 + 2 * (row_0 - _shfl_sync_3);
        int start_0 = start;
        long long at_c_0 = (long long)start_0 * c_ld + col;
        long long at_z_0 = (long long)start_0 * z_ld + col;
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
                : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
            held_vals_0[0 + 0] = __uint_as_float(_ldv8_4_0);
            held_vals_0[0 + 1] = __uint_as_float(_ldv8_4_1);
            held_vals_0[0 + 2] = __uint_as_float(_ldv8_4_2);
            held_vals_0[0 + 3] = __uint_as_float(_ldv8_4_3);
            held_vals_0[0 + 4] = __uint_as_float(_ldv8_4_4);
            held_vals_0[0 + 5] = __uint_as_float(_ldv8_4_5);
            held_vals_0[0 + 6] = __uint_as_float(_ldv8_4_6);
            held_vals_0[0 + 7] = __uint_as_float(_ldv8_4_7);
        }
        {
            unsigned _ldv8_5_0;
            unsigned _ldv8_5_1;
            unsigned _ldv8_5_2;
            unsigned _ldv8_5_3;
            unsigned _ldv8_5_4;
            unsigned _ldv8_5_5;
            unsigned _ldv8_5_6;
            unsigned _ldv8_5_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
            held_vals_0[8 + 0] = __uint_as_float(_ldv8_5_0);
            held_vals_0[8 + 1] = __uint_as_float(_ldv8_5_1);
            held_vals_0[8 + 2] = __uint_as_float(_ldv8_5_2);
            held_vals_0[8 + 3] = __uint_as_float(_ldv8_5_3);
            held_vals_0[8 + 4] = __uint_as_float(_ldv8_5_4);
            held_vals_0[8 + 5] = __uint_as_float(_ldv8_5_5);
            held_vals_0[8 + 6] = __uint_as_float(_ldv8_5_6);
            held_vals_0[8 + 7] = __uint_as_float(_ldv8_5_7);
        }
        {
            unsigned _ldv8_6_0;
            unsigned _ldv8_6_1;
            unsigned _ldv8_6_2;
            unsigned _ldv8_6_3;
            unsigned _ldv8_6_4;
            unsigned _ldv8_6_5;
            unsigned _ldv8_6_6;
            unsigned _ldv8_6_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_6_0), "=r"(_ldv8_6_1), "=r"(_ldv8_6_2), "=r"(_ldv8_6_3), "=r"(_ldv8_6_4), "=r"(_ldv8_6_5), "=r"(_ldv8_6_6), "=r"(_ldv8_6_7) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
            held_vals_0[16 + 0] = __uint_as_float(_ldv8_6_0);
            held_vals_0[16 + 1] = __uint_as_float(_ldv8_6_1);
            held_vals_0[16 + 2] = __uint_as_float(_ldv8_6_2);
            held_vals_0[16 + 3] = __uint_as_float(_ldv8_6_3);
            held_vals_0[16 + 4] = __uint_as_float(_ldv8_6_4);
            held_vals_0[16 + 5] = __uint_as_float(_ldv8_6_5);
            held_vals_0[16 + 6] = __uint_as_float(_ldv8_6_6);
            held_vals_0[16 + 7] = __uint_as_float(_ldv8_6_7);
        }
        {
            unsigned _ldv8_7_0;
            unsigned _ldv8_7_1;
            unsigned _ldv8_7_2;
            unsigned _ldv8_7_3;
            unsigned _ldv8_7_4;
            unsigned _ldv8_7_5;
            unsigned _ldv8_7_6;
            unsigned _ldv8_7_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_7_0), "=r"(_ldv8_7_1), "=r"(_ldv8_7_2), "=r"(_ldv8_7_3), "=r"(_ldv8_7_4), "=r"(_ldv8_7_5), "=r"(_ldv8_7_6), "=r"(_ldv8_7_7) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
            held_vals_0[24 + 0] = __uint_as_float(_ldv8_7_0);
            held_vals_0[24 + 1] = __uint_as_float(_ldv8_7_1);
            held_vals_0[24 + 2] = __uint_as_float(_ldv8_7_2);
            held_vals_0[24 + 3] = __uint_as_float(_ldv8_7_3);
            held_vals_0[24 + 4] = __uint_as_float(_ldv8_7_4);
            held_vals_0[24 + 5] = __uint_as_float(_ldv8_7_5);
            held_vals_0[24 + 6] = __uint_as_float(_ldv8_7_6);
            held_vals_0[24 + 7] = __uint_as_float(_ldv8_7_7);
        }
    }
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += 2 * num_bids) {
        int row_1 = (blk + num_bids) * 4 + slot;
        if (row_1 < pooled_rows) {
            int lo_1 = 0;
            int shift_1 = 5 * (rounds - 1);
            int hits_1 = 1;
            int entry_1 = 0;
            int origin_1 = 0;
            #pragma unroll 1
            for (int _trip_1 = 0; _trip_1 < rounds; _trip_1++) {
                int probe_1 = lo_1 + (lane << shift_1);
                int hit_1 = 0;
                if (probe_1 < num_docs) {
                    entry_1 = cu_pairs[probe_1];
                    if (shift_1 == 0) {
                        origin_1 = cu_seqlens[probe_1];
                    }
                    if (entry_1 <= row_1) {
                        hit_1 = 1;
                    }
                }
                int _warp_redux_8;
                asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_8) : "r"(hit_1));
                hits_1 = (int)_warp_redux_8;
                lo_1 = lo_1 + (hits_1 - 1 << shift_1);
                shift_1 = shift_1 - 5;
            }
            int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, origin_1, hits_1 - 1);
            int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, entry_1, hits_1 - 1);
            int start_1 = _shfl_sync_9 + 2 * (row_1 - _shfl_sync_10);
            int start_1_1 = start_1;
            long long at_c_1 = (long long)start_1_1 * c_ld + col;
            long long at_z_1 = (long long)start_1_1 * z_ld + col;
            {
                unsigned _ldv8_11_0;
                unsigned _ldv8_11_1;
                unsigned _ldv8_11_2;
                unsigned _ldv8_11_3;
                unsigned _ldv8_11_4;
                unsigned _ldv8_11_5;
                unsigned _ldv8_11_6;
                unsigned _ldv8_11_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_11_0), "=r"(_ldv8_11_1), "=r"(_ldv8_11_2), "=r"(_ldv8_11_3), "=r"(_ldv8_11_4), "=r"(_ldv8_11_5), "=r"(_ldv8_11_6), "=r"(_ldv8_11_7) : "l"((const void*)(c + at_c_1 + (0))) : "memory");
                held_vals_1[0 + 0] = __uint_as_float(_ldv8_11_0);
                held_vals_1[0 + 1] = __uint_as_float(_ldv8_11_1);
                held_vals_1[0 + 2] = __uint_as_float(_ldv8_11_2);
                held_vals_1[0 + 3] = __uint_as_float(_ldv8_11_3);
                held_vals_1[0 + 4] = __uint_as_float(_ldv8_11_4);
                held_vals_1[0 + 5] = __uint_as_float(_ldv8_11_5);
                held_vals_1[0 + 6] = __uint_as_float(_ldv8_11_6);
                held_vals_1[0 + 7] = __uint_as_float(_ldv8_11_7);
            }
            {
                unsigned _ldv8_12_0;
                unsigned _ldv8_12_1;
                unsigned _ldv8_12_2;
                unsigned _ldv8_12_3;
                unsigned _ldv8_12_4;
                unsigned _ldv8_12_5;
                unsigned _ldv8_12_6;
                unsigned _ldv8_12_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_12_0), "=r"(_ldv8_12_1), "=r"(_ldv8_12_2), "=r"(_ldv8_12_3), "=r"(_ldv8_12_4), "=r"(_ldv8_12_5), "=r"(_ldv8_12_6), "=r"(_ldv8_12_7) : "l"((const void*)(c + (at_c_1 + c_ld) + (0))) : "memory");
                held_vals_1[8 + 0] = __uint_as_float(_ldv8_12_0);
                held_vals_1[8 + 1] = __uint_as_float(_ldv8_12_1);
                held_vals_1[8 + 2] = __uint_as_float(_ldv8_12_2);
                held_vals_1[8 + 3] = __uint_as_float(_ldv8_12_3);
                held_vals_1[8 + 4] = __uint_as_float(_ldv8_12_4);
                held_vals_1[8 + 5] = __uint_as_float(_ldv8_12_5);
                held_vals_1[8 + 6] = __uint_as_float(_ldv8_12_6);
                held_vals_1[8 + 7] = __uint_as_float(_ldv8_12_7);
            }
            {
                unsigned _ldv8_13_0;
                unsigned _ldv8_13_1;
                unsigned _ldv8_13_2;
                unsigned _ldv8_13_3;
                unsigned _ldv8_13_4;
                unsigned _ldv8_13_5;
                unsigned _ldv8_13_6;
                unsigned _ldv8_13_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_13_0), "=r"(_ldv8_13_1), "=r"(_ldv8_13_2), "=r"(_ldv8_13_3), "=r"(_ldv8_13_4), "=r"(_ldv8_13_5), "=r"(_ldv8_13_6), "=r"(_ldv8_13_7) : "l"((const void*)(z + at_z_1 + (0))) : "memory");
                held_vals_1[16 + 0] = __uint_as_float(_ldv8_13_0);
                held_vals_1[16 + 1] = __uint_as_float(_ldv8_13_1);
                held_vals_1[16 + 2] = __uint_as_float(_ldv8_13_2);
                held_vals_1[16 + 3] = __uint_as_float(_ldv8_13_3);
                held_vals_1[16 + 4] = __uint_as_float(_ldv8_13_4);
                held_vals_1[16 + 5] = __uint_as_float(_ldv8_13_5);
                held_vals_1[16 + 6] = __uint_as_float(_ldv8_13_6);
                held_vals_1[16 + 7] = __uint_as_float(_ldv8_13_7);
            }
            {
                unsigned _ldv8_14_0;
                unsigned _ldv8_14_1;
                unsigned _ldv8_14_2;
                unsigned _ldv8_14_3;
                unsigned _ldv8_14_4;
                unsigned _ldv8_14_5;
                unsigned _ldv8_14_6;
                unsigned _ldv8_14_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_14_0), "=r"(_ldv8_14_1), "=r"(_ldv8_14_2), "=r"(_ldv8_14_3), "=r"(_ldv8_14_4), "=r"(_ldv8_14_5), "=r"(_ldv8_14_6), "=r"(_ldv8_14_7) : "l"((const void*)(z + (at_z_1 + z_ld) + (0))) : "memory");
                held_vals_1[24 + 0] = __uint_as_float(_ldv8_14_0);
                held_vals_1[24 + 1] = __uint_as_float(_ldv8_14_1);
                held_vals_1[24 + 2] = __uint_as_float(_ldv8_14_2);
                held_vals_1[24 + 3] = __uint_as_float(_ldv8_14_3);
                held_vals_1[24 + 4] = __uint_as_float(_ldv8_14_4);
                held_vals_1[24 + 5] = __uint_as_float(_ldv8_14_5);
                held_vals_1[24 + 6] = __uint_as_float(_ldv8_14_6);
                held_vals_1[24 + 7] = __uint_as_float(_ldv8_14_7);
            }
        }
        unsigned int words[4] = {0};
        int row_2 = blk * 4 + slot;
        if (row_2 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(held_vals_0[24]), "f"(held_vals_0[16]));
            float _abs_16;
            asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(gap));
            float _neg_15;
            asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(_abs_16));
            float e = __nv_expf(_neg_15);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float big = ((first == 1) ? held_vals_0[0] : held_vals_0[8]);
            float small = ((first == 1) ? held_vals_0[8] : held_vals_0[0]);
            float _fma_18;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(e), "f"(small), "f"(big));
            float _fdiv_17 = __fdiv_rn(_fma_18, den);
            held_vals_0[0] = _fdiv_17;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(held_vals_0[25]), "f"(held_vals_0[17]));
            float _abs_20;
            asm("abs.f32 %0, %1;" : "=f"(_abs_20) : "f"(gap_0));
            float _neg_19;
            asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(_abs_20));
            float e_0 = __nv_expf(_neg_19);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float big_0 = ((first_0 == 1) ? held_vals_0[1] : held_vals_0[9]);
            float small_0 = ((first_0 == 1) ? held_vals_0[9] : held_vals_0[1]);
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(e_0), "f"(small_0), "f"(big_0));
            float _fdiv_21 = __fdiv_rn(_fma_22, den_0);
            held_vals_0[1] = _fdiv_21;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(held_vals_0[26]), "f"(held_vals_0[18]));
            float _abs_24;
            asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(gap_1));
            float _neg_23;
            asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(_abs_24));
            float e_1 = __nv_expf(_neg_23);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float big_1 = ((first_1 == 1) ? held_vals_0[2] : held_vals_0[10]);
            float small_1 = ((first_1 == 1) ? held_vals_0[10] : held_vals_0[2]);
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(e_1), "f"(small_1), "f"(big_1));
            float _fdiv_25 = __fdiv_rn(_fma_26, den_1);
            held_vals_0[2] = _fdiv_25;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(held_vals_0[27]), "f"(held_vals_0[19]));
            float _abs_28;
            asm("abs.f32 %0, %1;" : "=f"(_abs_28) : "f"(gap_2));
            float _neg_27;
            asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(_abs_28));
            float e_2 = __nv_expf(_neg_27);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float big_2 = ((first_2 == 1) ? held_vals_0[3] : held_vals_0[11]);
            float small_2 = ((first_2 == 1) ? held_vals_0[11] : held_vals_0[3]);
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(e_2), "f"(small_2), "f"(big_2));
            float _fdiv_29 = __fdiv_rn(_fma_30, den_2);
            held_vals_0[3] = _fdiv_29;
            float gap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_3) : "f"(held_vals_0[28]), "f"(held_vals_0[20]));
            float _abs_32;
            asm("abs.f32 %0, %1;" : "=f"(_abs_32) : "f"(gap_3));
            float _neg_31;
            asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(_abs_32));
            float e_3 = __nv_expf(_neg_31);
            float den_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_3) : "f"(e_3), "f"(1.0f));
            int first_3 = ((gap_3 <= 0.0f) ? 1 : 0);
            float big_3 = ((first_3 == 1) ? held_vals_0[4] : held_vals_0[12]);
            float small_3 = ((first_3 == 1) ? held_vals_0[12] : held_vals_0[4]);
            float _fma_34;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(e_3), "f"(small_3), "f"(big_3));
            float _fdiv_33 = __fdiv_rn(_fma_34, den_3);
            held_vals_0[4] = _fdiv_33;
            float gap_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_4) : "f"(held_vals_0[29]), "f"(held_vals_0[21]));
            float _abs_36;
            asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(gap_4));
            float _neg_35;
            asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(_abs_36));
            float e_4 = __nv_expf(_neg_35);
            float den_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_4) : "f"(e_4), "f"(1.0f));
            int first_4 = ((gap_4 <= 0.0f) ? 1 : 0);
            float big_4 = ((first_4 == 1) ? held_vals_0[5] : held_vals_0[13]);
            float small_4 = ((first_4 == 1) ? held_vals_0[13] : held_vals_0[5]);
            float _fma_38;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(e_4), "f"(small_4), "f"(big_4));
            float _fdiv_37 = __fdiv_rn(_fma_38, den_4);
            held_vals_0[5] = _fdiv_37;
            float gap_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_5) : "f"(held_vals_0[30]), "f"(held_vals_0[22]));
            float _abs_40;
            asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(gap_5));
            float _neg_39;
            asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(_abs_40));
            float e_5 = __nv_expf(_neg_39);
            float den_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_5) : "f"(e_5), "f"(1.0f));
            int first_5 = ((gap_5 <= 0.0f) ? 1 : 0);
            float big_5 = ((first_5 == 1) ? held_vals_0[6] : held_vals_0[14]);
            float small_5 = ((first_5 == 1) ? held_vals_0[14] : held_vals_0[6]);
            float _fma_42;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(e_5), "f"(small_5), "f"(big_5));
            float _fdiv_41 = __fdiv_rn(_fma_42, den_5);
            held_vals_0[6] = _fdiv_41;
            float gap_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_6) : "f"(held_vals_0[31]), "f"(held_vals_0[23]));
            float _abs_44;
            asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(gap_6));
            float _neg_43;
            asm("neg.f32 %0, %1;" : "=f"(_neg_43) : "f"(_abs_44));
            float e_6 = __nv_expf(_neg_43);
            float den_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_6) : "f"(e_6), "f"(1.0f));
            int first_6 = ((gap_6 <= 0.0f) ? 1 : 0);
            float big_6 = ((first_6 == 1) ? held_vals_0[7] : held_vals_0[15]);
            float small_6 = ((first_6 == 1) ? held_vals_0[15] : held_vals_0[7]);
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(e_6), "f"(small_6), "f"(big_6));
            float _fdiv_45 = __fdiv_rn(_fma_46, den_6);
            held_vals_0[7] = _fdiv_45;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_0[_lp*2 + 0], held_vals_0[_lp*2+1 + 0]));
                words[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_0[_lp*2 + 2], held_vals_0[_lp*2+1 + 2]));
                words[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_0[_lp*2 + 4], held_vals_0[_lp*2+1 + 4]));
                words[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_0[_lp*2 + 6], held_vals_0[_lp*2+1 + 6]));
                words[_lp + 3] = *(uint32_t*)&_bf2;
            }
        }
        if (row_2 < G) {
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + ((long long)row_2 * 512ll + col)))[0] = reinterpret_cast<int4*>(words)[0];
        }
        int row_3 = (blk + 2 * num_bids) * 4 + slot;
        if (row_3 < pooled_rows) {
            int lo_2 = 0;
            int shift_2 = 5 * (rounds - 1);
            int hits_2 = 1;
            int entry_2 = 0;
            int origin_2 = 0;
            #pragma unroll 1
            for (int _trip_2 = 0; _trip_2 < rounds; _trip_2++) {
                int probe_2 = lo_2 + (lane << shift_2);
                int hit_2 = 0;
                if (probe_2 < num_docs) {
                    entry_2 = cu_pairs[probe_2];
                    if (shift_2 == 0) {
                        origin_2 = cu_seqlens[probe_2];
                    }
                    if (entry_2 <= row_3) {
                        hit_2 = 1;
                    }
                }
                int _warp_redux_47;
                asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_47) : "r"(hit_2));
                hits_2 = (int)_warp_redux_47;
                lo_2 = lo_2 + (hits_2 - 1 << shift_2);
                shift_2 = shift_2 - 5;
            }
            int _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, origin_2, hits_2 - 1);
            int _shfl_sync_49 = __shfl_sync(0xFFFFFFFF, entry_2, hits_2 - 1);
            int start_2 = _shfl_sync_48 + 2 * (row_3 - _shfl_sync_49);
            int start_2_1 = start_2;
            long long at_c_2 = (long long)start_2_1 * c_ld + col;
            long long at_z_2 = (long long)start_2_1 * z_ld + col;
            {
                unsigned _ldv8_50_0;
                unsigned _ldv8_50_1;
                unsigned _ldv8_50_2;
                unsigned _ldv8_50_3;
                unsigned _ldv8_50_4;
                unsigned _ldv8_50_5;
                unsigned _ldv8_50_6;
                unsigned _ldv8_50_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_50_0), "=r"(_ldv8_50_1), "=r"(_ldv8_50_2), "=r"(_ldv8_50_3), "=r"(_ldv8_50_4), "=r"(_ldv8_50_5), "=r"(_ldv8_50_6), "=r"(_ldv8_50_7) : "l"((const void*)(c + at_c_2 + (0))) : "memory");
                held_vals_0[0 + 0] = __uint_as_float(_ldv8_50_0);
                held_vals_0[0 + 1] = __uint_as_float(_ldv8_50_1);
                held_vals_0[0 + 2] = __uint_as_float(_ldv8_50_2);
                held_vals_0[0 + 3] = __uint_as_float(_ldv8_50_3);
                held_vals_0[0 + 4] = __uint_as_float(_ldv8_50_4);
                held_vals_0[0 + 5] = __uint_as_float(_ldv8_50_5);
                held_vals_0[0 + 6] = __uint_as_float(_ldv8_50_6);
                held_vals_0[0 + 7] = __uint_as_float(_ldv8_50_7);
            }
            {
                unsigned _ldv8_51_0;
                unsigned _ldv8_51_1;
                unsigned _ldv8_51_2;
                unsigned _ldv8_51_3;
                unsigned _ldv8_51_4;
                unsigned _ldv8_51_5;
                unsigned _ldv8_51_6;
                unsigned _ldv8_51_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_51_0), "=r"(_ldv8_51_1), "=r"(_ldv8_51_2), "=r"(_ldv8_51_3), "=r"(_ldv8_51_4), "=r"(_ldv8_51_5), "=r"(_ldv8_51_6), "=r"(_ldv8_51_7) : "l"((const void*)(c + (at_c_2 + c_ld) + (0))) : "memory");
                held_vals_0[8 + 0] = __uint_as_float(_ldv8_51_0);
                held_vals_0[8 + 1] = __uint_as_float(_ldv8_51_1);
                held_vals_0[8 + 2] = __uint_as_float(_ldv8_51_2);
                held_vals_0[8 + 3] = __uint_as_float(_ldv8_51_3);
                held_vals_0[8 + 4] = __uint_as_float(_ldv8_51_4);
                held_vals_0[8 + 5] = __uint_as_float(_ldv8_51_5);
                held_vals_0[8 + 6] = __uint_as_float(_ldv8_51_6);
                held_vals_0[8 + 7] = __uint_as_float(_ldv8_51_7);
            }
            {
                unsigned _ldv8_52_0;
                unsigned _ldv8_52_1;
                unsigned _ldv8_52_2;
                unsigned _ldv8_52_3;
                unsigned _ldv8_52_4;
                unsigned _ldv8_52_5;
                unsigned _ldv8_52_6;
                unsigned _ldv8_52_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_52_0), "=r"(_ldv8_52_1), "=r"(_ldv8_52_2), "=r"(_ldv8_52_3), "=r"(_ldv8_52_4), "=r"(_ldv8_52_5), "=r"(_ldv8_52_6), "=r"(_ldv8_52_7) : "l"((const void*)(z + at_z_2 + (0))) : "memory");
                held_vals_0[16 + 0] = __uint_as_float(_ldv8_52_0);
                held_vals_0[16 + 1] = __uint_as_float(_ldv8_52_1);
                held_vals_0[16 + 2] = __uint_as_float(_ldv8_52_2);
                held_vals_0[16 + 3] = __uint_as_float(_ldv8_52_3);
                held_vals_0[16 + 4] = __uint_as_float(_ldv8_52_4);
                held_vals_0[16 + 5] = __uint_as_float(_ldv8_52_5);
                held_vals_0[16 + 6] = __uint_as_float(_ldv8_52_6);
                held_vals_0[16 + 7] = __uint_as_float(_ldv8_52_7);
            }
            {
                unsigned _ldv8_53_0;
                unsigned _ldv8_53_1;
                unsigned _ldv8_53_2;
                unsigned _ldv8_53_3;
                unsigned _ldv8_53_4;
                unsigned _ldv8_53_5;
                unsigned _ldv8_53_6;
                unsigned _ldv8_53_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_53_0), "=r"(_ldv8_53_1), "=r"(_ldv8_53_2), "=r"(_ldv8_53_3), "=r"(_ldv8_53_4), "=r"(_ldv8_53_5), "=r"(_ldv8_53_6), "=r"(_ldv8_53_7) : "l"((const void*)(z + (at_z_2 + z_ld) + (0))) : "memory");
                held_vals_0[24 + 0] = __uint_as_float(_ldv8_53_0);
                held_vals_0[24 + 1] = __uint_as_float(_ldv8_53_1);
                held_vals_0[24 + 2] = __uint_as_float(_ldv8_53_2);
                held_vals_0[24 + 3] = __uint_as_float(_ldv8_53_3);
                held_vals_0[24 + 4] = __uint_as_float(_ldv8_53_4);
                held_vals_0[24 + 5] = __uint_as_float(_ldv8_53_5);
                held_vals_0[24 + 6] = __uint_as_float(_ldv8_53_6);
                held_vals_0[24 + 7] = __uint_as_float(_ldv8_53_7);
            }
        }
        unsigned int words_0[4] = {0};
        int row_4 = (blk + num_bids) * 4 + slot;
        if (row_4 < pooled_rows) {
            float gap_7;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_7) : "f"(held_vals_1[24]), "f"(held_vals_1[16]));
            float _abs_55;
            asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(gap_7));
            float _neg_54;
            asm("neg.f32 %0, %1;" : "=f"(_neg_54) : "f"(_abs_55));
            float e_7 = __nv_expf(_neg_54);
            float den_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_7) : "f"(e_7), "f"(1.0f));
            int first_7 = ((gap_7 <= 0.0f) ? 1 : 0);
            float big_7 = ((first_7 == 1) ? held_vals_1[0] : held_vals_1[8]);
            float small_7 = ((first_7 == 1) ? held_vals_1[8] : held_vals_1[0]);
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(e_7), "f"(small_7), "f"(big_7));
            float _fdiv_56 = __fdiv_rn(_fma_57, den_7);
            held_vals_1[0] = _fdiv_56;
            float gap_7_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_7_1) : "f"(held_vals_1[25]), "f"(held_vals_1[17]));
            float _abs_59;
            asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(gap_7_1));
            float _neg_58;
            asm("neg.f32 %0, %1;" : "=f"(_neg_58) : "f"(_abs_59));
            float e_7_1 = __nv_expf(_neg_58);
            float den_7_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_7_1) : "f"(e_7_1), "f"(1.0f));
            int first_7_1 = ((gap_7_1 <= 0.0f) ? 1 : 0);
            float big_7_1 = ((first_7_1 == 1) ? held_vals_1[1] : held_vals_1[9]);
            float small_7_1 = ((first_7_1 == 1) ? held_vals_1[9] : held_vals_1[1]);
            float _fma_61;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(e_7_1), "f"(small_7_1), "f"(big_7_1));
            float _fdiv_60 = __fdiv_rn(_fma_61, den_7_1);
            held_vals_1[1] = _fdiv_60;
            float gap_8;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_8) : "f"(held_vals_1[26]), "f"(held_vals_1[18]));
            float _abs_63;
            asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(gap_8));
            float _neg_62;
            asm("neg.f32 %0, %1;" : "=f"(_neg_62) : "f"(_abs_63));
            float e_8 = __nv_expf(_neg_62);
            float den_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_8) : "f"(e_8), "f"(1.0f));
            int first_8 = ((gap_8 <= 0.0f) ? 1 : 0);
            float big_8 = ((first_8 == 1) ? held_vals_1[2] : held_vals_1[10]);
            float small_8 = ((first_8 == 1) ? held_vals_1[10] : held_vals_1[2]);
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(e_8), "f"(small_8), "f"(big_8));
            float _fdiv_64 = __fdiv_rn(_fma_65, den_8);
            held_vals_1[2] = _fdiv_64;
            float gap_9;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_9) : "f"(held_vals_1[27]), "f"(held_vals_1[19]));
            float _abs_67;
            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(gap_9));
            float _neg_66;
            asm("neg.f32 %0, %1;" : "=f"(_neg_66) : "f"(_abs_67));
            float e_9 = __nv_expf(_neg_66);
            float den_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_9) : "f"(e_9), "f"(1.0f));
            int first_9 = ((gap_9 <= 0.0f) ? 1 : 0);
            float big_9 = ((first_9 == 1) ? held_vals_1[3] : held_vals_1[11]);
            float small_9 = ((first_9 == 1) ? held_vals_1[11] : held_vals_1[3]);
            float _fma_69;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(e_9), "f"(small_9), "f"(big_9));
            float _fdiv_68 = __fdiv_rn(_fma_69, den_9);
            held_vals_1[3] = _fdiv_68;
            float gap_10;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_10) : "f"(held_vals_1[28]), "f"(held_vals_1[20]));
            float _abs_71;
            asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(gap_10));
            float _neg_70;
            asm("neg.f32 %0, %1;" : "=f"(_neg_70) : "f"(_abs_71));
            float e_10 = __nv_expf(_neg_70);
            float den_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_10) : "f"(e_10), "f"(1.0f));
            int first_10 = ((gap_10 <= 0.0f) ? 1 : 0);
            float big_10 = ((first_10 == 1) ? held_vals_1[4] : held_vals_1[12]);
            float small_10 = ((first_10 == 1) ? held_vals_1[12] : held_vals_1[4]);
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(e_10), "f"(small_10), "f"(big_10));
            float _fdiv_72 = __fdiv_rn(_fma_73, den_10);
            held_vals_1[4] = _fdiv_72;
            float gap_11;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_11) : "f"(held_vals_1[29]), "f"(held_vals_1[21]));
            float _abs_75;
            asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(gap_11));
            float _neg_74;
            asm("neg.f32 %0, %1;" : "=f"(_neg_74) : "f"(_abs_75));
            float e_11 = __nv_expf(_neg_74);
            float den_11;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_11) : "f"(e_11), "f"(1.0f));
            int first_11 = ((gap_11 <= 0.0f) ? 1 : 0);
            float big_11 = ((first_11 == 1) ? held_vals_1[5] : held_vals_1[13]);
            float small_11 = ((first_11 == 1) ? held_vals_1[13] : held_vals_1[5]);
            float _fma_77;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(e_11), "f"(small_11), "f"(big_11));
            float _fdiv_76 = __fdiv_rn(_fma_77, den_11);
            held_vals_1[5] = _fdiv_76;
            float gap_12;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_12) : "f"(held_vals_1[30]), "f"(held_vals_1[22]));
            float _abs_79;
            asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(gap_12));
            float _neg_78;
            asm("neg.f32 %0, %1;" : "=f"(_neg_78) : "f"(_abs_79));
            float e_12 = __nv_expf(_neg_78);
            float den_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_12) : "f"(e_12), "f"(1.0f));
            int first_12 = ((gap_12 <= 0.0f) ? 1 : 0);
            float big_12 = ((first_12 == 1) ? held_vals_1[6] : held_vals_1[14]);
            float small_12 = ((first_12 == 1) ? held_vals_1[14] : held_vals_1[6]);
            float _fma_81;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(e_12), "f"(small_12), "f"(big_12));
            float _fdiv_80 = __fdiv_rn(_fma_81, den_12);
            held_vals_1[6] = _fdiv_80;
            float gap_13;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_13) : "f"(held_vals_1[31]), "f"(held_vals_1[23]));
            float _abs_83;
            asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(gap_13));
            float _neg_82;
            asm("neg.f32 %0, %1;" : "=f"(_neg_82) : "f"(_abs_83));
            float e_13 = __nv_expf(_neg_82);
            float den_13;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_13) : "f"(e_13), "f"(1.0f));
            int first_13 = ((gap_13 <= 0.0f) ? 1 : 0);
            float big_13 = ((first_13 == 1) ? held_vals_1[7] : held_vals_1[15]);
            float small_13 = ((first_13 == 1) ? held_vals_1[15] : held_vals_1[7]);
            float _fma_85;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(e_13), "f"(small_13), "f"(big_13));
            float _fdiv_84 = __fdiv_rn(_fma_85, den_13);
            held_vals_1[7] = _fdiv_84;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_1[_lp*2 + 0], held_vals_1[_lp*2+1 + 0]));
                words_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_1[_lp*2 + 2], held_vals_1[_lp*2+1 + 2]));
                words_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_1[_lp*2 + 4], held_vals_1[_lp*2+1 + 4]));
                words_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(held_vals_1[_lp*2 + 6], held_vals_1[_lp*2+1 + 6]));
                words_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
        }
        if (row_4 < G) {
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + ((long long)row_4 * 512ll + col)))[0] = reinterpret_cast<int4*>(words_0)[0];
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
