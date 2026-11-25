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

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gated_kv_compression_det_train_64aae35cfb0e308d103a(float* c, float* z, int* cu_seqlens, int* cu_pairs, __nv_bfloat16* out, int num_docs, int G, long long c_ld, long long z_ld)
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
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 4 + slot;
        float vals_0[32];
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
                vals_0[0 + 0] = __uint_as_float(_ldv8_4_0);
                vals_0[0 + 1] = __uint_as_float(_ldv8_4_1);
                vals_0[0 + 2] = __uint_as_float(_ldv8_4_2);
                vals_0[0 + 3] = __uint_as_float(_ldv8_4_3);
                vals_0[0 + 4] = __uint_as_float(_ldv8_4_4);
                vals_0[0 + 5] = __uint_as_float(_ldv8_4_5);
                vals_0[0 + 6] = __uint_as_float(_ldv8_4_6);
                vals_0[0 + 7] = __uint_as_float(_ldv8_4_7);
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
                vals_0[8 + 0] = __uint_as_float(_ldv8_5_0);
                vals_0[8 + 1] = __uint_as_float(_ldv8_5_1);
                vals_0[8 + 2] = __uint_as_float(_ldv8_5_2);
                vals_0[8 + 3] = __uint_as_float(_ldv8_5_3);
                vals_0[8 + 4] = __uint_as_float(_ldv8_5_4);
                vals_0[8 + 5] = __uint_as_float(_ldv8_5_5);
                vals_0[8 + 6] = __uint_as_float(_ldv8_5_6);
                vals_0[8 + 7] = __uint_as_float(_ldv8_5_7);
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
                vals_0[16 + 0] = __uint_as_float(_ldv8_6_0);
                vals_0[16 + 1] = __uint_as_float(_ldv8_6_1);
                vals_0[16 + 2] = __uint_as_float(_ldv8_6_2);
                vals_0[16 + 3] = __uint_as_float(_ldv8_6_3);
                vals_0[16 + 4] = __uint_as_float(_ldv8_6_4);
                vals_0[16 + 5] = __uint_as_float(_ldv8_6_5);
                vals_0[16 + 6] = __uint_as_float(_ldv8_6_6);
                vals_0[16 + 7] = __uint_as_float(_ldv8_6_7);
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
                vals_0[24 + 0] = __uint_as_float(_ldv8_7_0);
                vals_0[24 + 1] = __uint_as_float(_ldv8_7_1);
                vals_0[24 + 2] = __uint_as_float(_ldv8_7_2);
                vals_0[24 + 3] = __uint_as_float(_ldv8_7_3);
                vals_0[24 + 4] = __uint_as_float(_ldv8_7_4);
                vals_0[24 + 5] = __uint_as_float(_ldv8_7_5);
                vals_0[24 + 6] = __uint_as_float(_ldv8_7_6);
                vals_0[24 + 7] = __uint_as_float(_ldv8_7_7);
            }
        }
        unsigned int words[4] = {0};
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[24]), "f"(vals_0[16]));
            float _abs_9;
            asm("abs.f32 %0, %1;" : "=f"(_abs_9) : "f"(gap));
            float _neg_8;
            asm("neg.f32 %0, %1;" : "=f"(_neg_8) : "f"(_abs_9));
            float e = __nv_expf(_neg_8);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float big = ((first == 1) ? vals_0[0] : vals_0[8]);
            float small = ((first == 1) ? vals_0[8] : vals_0[0]);
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(e), "f"(small), "f"(big));
            float _fdiv_10 = __fdiv_rn(_fma_11, den);
            vals_0[0] = _fdiv_10;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[25]), "f"(vals_0[17]));
            float _abs_13;
            asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(gap_0));
            float _neg_12;
            asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(_abs_13));
            float e_0 = __nv_expf(_neg_12);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float big_0 = ((first_0 == 1) ? vals_0[1] : vals_0[9]);
            float small_0 = ((first_0 == 1) ? vals_0[9] : vals_0[1]);
            float _fma_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(e_0), "f"(small_0), "f"(big_0));
            float _fdiv_14 = __fdiv_rn(_fma_15, den_0);
            vals_0[1] = _fdiv_14;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[26]), "f"(vals_0[18]));
            float _abs_17;
            asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(gap_1));
            float _neg_16;
            asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
            float e_1 = __nv_expf(_neg_16);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float big_1 = ((first_1 == 1) ? vals_0[2] : vals_0[10]);
            float small_1 = ((first_1 == 1) ? vals_0[10] : vals_0[2]);
            float _fma_19;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(e_1), "f"(small_1), "f"(big_1));
            float _fdiv_18 = __fdiv_rn(_fma_19, den_1);
            vals_0[2] = _fdiv_18;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[27]), "f"(vals_0[19]));
            float _abs_21;
            asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(gap_2));
            float _neg_20;
            asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
            float e_2 = __nv_expf(_neg_20);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float big_2 = ((first_2 == 1) ? vals_0[3] : vals_0[11]);
            float small_2 = ((first_2 == 1) ? vals_0[11] : vals_0[3]);
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(e_2), "f"(small_2), "f"(big_2));
            float _fdiv_22 = __fdiv_rn(_fma_23, den_2);
            vals_0[3] = _fdiv_22;
            float gap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_3) : "f"(vals_0[28]), "f"(vals_0[20]));
            float _abs_25;
            asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(gap_3));
            float _neg_24;
            asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(_abs_25));
            float e_3 = __nv_expf(_neg_24);
            float den_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_3) : "f"(e_3), "f"(1.0f));
            int first_3 = ((gap_3 <= 0.0f) ? 1 : 0);
            float big_3 = ((first_3 == 1) ? vals_0[4] : vals_0[12]);
            float small_3 = ((first_3 == 1) ? vals_0[12] : vals_0[4]);
            float _fma_27;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(e_3), "f"(small_3), "f"(big_3));
            float _fdiv_26 = __fdiv_rn(_fma_27, den_3);
            vals_0[4] = _fdiv_26;
            float gap_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_4) : "f"(vals_0[29]), "f"(vals_0[21]));
            float _abs_29;
            asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(gap_4));
            float _neg_28;
            asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
            float e_4 = __nv_expf(_neg_28);
            float den_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_4) : "f"(e_4), "f"(1.0f));
            int first_4 = ((gap_4 <= 0.0f) ? 1 : 0);
            float big_4 = ((first_4 == 1) ? vals_0[5] : vals_0[13]);
            float small_4 = ((first_4 == 1) ? vals_0[13] : vals_0[5]);
            float _fma_31;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(e_4), "f"(small_4), "f"(big_4));
            float _fdiv_30 = __fdiv_rn(_fma_31, den_4);
            vals_0[5] = _fdiv_30;
            float gap_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_5) : "f"(vals_0[30]), "f"(vals_0[22]));
            float _abs_33;
            asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(gap_5));
            float _neg_32;
            asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(_abs_33));
            float e_5 = __nv_expf(_neg_32);
            float den_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_5) : "f"(e_5), "f"(1.0f));
            int first_5 = ((gap_5 <= 0.0f) ? 1 : 0);
            float big_5 = ((first_5 == 1) ? vals_0[6] : vals_0[14]);
            float small_5 = ((first_5 == 1) ? vals_0[14] : vals_0[6]);
            float _fma_35;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(e_5), "f"(small_5), "f"(big_5));
            float _fdiv_34 = __fdiv_rn(_fma_35, den_5);
            vals_0[6] = _fdiv_34;
            float gap_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_6) : "f"(vals_0[31]), "f"(vals_0[23]));
            float _abs_37;
            asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(gap_6));
            float _neg_36;
            asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(_abs_37));
            float e_6 = __nv_expf(_neg_36);
            float den_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_6) : "f"(e_6), "f"(1.0f));
            int first_6 = ((gap_6 <= 0.0f) ? 1 : 0);
            float big_6 = ((first_6 == 1) ? vals_0[7] : vals_0[15]);
            float small_6 = ((first_6 == 1) ? vals_0[15] : vals_0[7]);
            float _fma_39;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(e_6), "f"(small_6), "f"(big_6));
            float _fdiv_38 = __fdiv_rn(_fma_39, den_6);
            vals_0[7] = _fdiv_38;
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
