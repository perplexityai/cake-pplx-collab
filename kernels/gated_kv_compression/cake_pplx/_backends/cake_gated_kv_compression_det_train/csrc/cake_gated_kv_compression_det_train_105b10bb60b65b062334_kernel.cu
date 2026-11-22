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
kernel_cake_gated_kv_compression_det_train_105b10bb60b65b062334(float* c, float* z, int* cu_seqlens, int* cu_pairs, __nv_bfloat16* out, int num_docs, int G, long long c_ld, long long z_ld)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int slot = tid >> 7;
    long long col = (long long)((tid & 127) * 4);
    int pooled_rows = ((G) < (cu_pairs[num_docs]) ? (G) : (cu_pairs[num_docs]));
    int _count_leading_zeros_0 = __clz(num_docs - 1 | 1);
    int rounds = (36 - _count_leading_zeros_0) / 5;
    int nblocks = (G + 3) / 4;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 4 + slot;
        float vals_0[16];
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
                unsigned _v4_4_0;
                unsigned _v4_4_1;
                unsigned _v4_4_2;
                unsigned _v4_4_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
                vals_0[0 + 0] = __uint_as_float(_v4_4_0);
                vals_0[0 + 1] = __uint_as_float(_v4_4_1);
                vals_0[0 + 2] = __uint_as_float(_v4_4_2);
                vals_0[0 + 3] = __uint_as_float(_v4_4_3);
            }
            {
                unsigned _v4_5_0;
                unsigned _v4_5_1;
                unsigned _v4_5_2;
                unsigned _v4_5_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
                vals_0[4 + 0] = __uint_as_float(_v4_5_0);
                vals_0[4 + 1] = __uint_as_float(_v4_5_1);
                vals_0[4 + 2] = __uint_as_float(_v4_5_2);
                vals_0[4 + 3] = __uint_as_float(_v4_5_3);
            }
            {
                unsigned _v4_6_0;
                unsigned _v4_6_1;
                unsigned _v4_6_2;
                unsigned _v4_6_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
                vals_0[8 + 0] = __uint_as_float(_v4_6_0);
                vals_0[8 + 1] = __uint_as_float(_v4_6_1);
                vals_0[8 + 2] = __uint_as_float(_v4_6_2);
                vals_0[8 + 3] = __uint_as_float(_v4_6_3);
            }
            {
                unsigned _v4_7_0;
                unsigned _v4_7_1;
                unsigned _v4_7_2;
                unsigned _v4_7_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
                vals_0[12 + 0] = __uint_as_float(_v4_7_0);
                vals_0[12 + 1] = __uint_as_float(_v4_7_1);
                vals_0[12 + 2] = __uint_as_float(_v4_7_2);
                vals_0[12 + 3] = __uint_as_float(_v4_7_3);
            }
        }
        int row_1 = blk * 4 + (2 + slot);
        float vals_1[16];
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
                unsigned _v4_11_0;
                unsigned _v4_11_1;
                unsigned _v4_11_2;
                unsigned _v4_11_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(c + at_c_1 + (0))) : "memory");
                vals_1[0 + 0] = __uint_as_float(_v4_11_0);
                vals_1[0 + 1] = __uint_as_float(_v4_11_1);
                vals_1[0 + 2] = __uint_as_float(_v4_11_2);
                vals_1[0 + 3] = __uint_as_float(_v4_11_3);
            }
            {
                unsigned _v4_12_0;
                unsigned _v4_12_1;
                unsigned _v4_12_2;
                unsigned _v4_12_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(c + (at_c_1 + c_ld) + (0))) : "memory");
                vals_1[4 + 0] = __uint_as_float(_v4_12_0);
                vals_1[4 + 1] = __uint_as_float(_v4_12_1);
                vals_1[4 + 2] = __uint_as_float(_v4_12_2);
                vals_1[4 + 3] = __uint_as_float(_v4_12_3);
            }
            {
                unsigned _v4_13_0;
                unsigned _v4_13_1;
                unsigned _v4_13_2;
                unsigned _v4_13_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3) : "l"((const void*)(z + at_z_1 + (0))) : "memory");
                vals_1[8 + 0] = __uint_as_float(_v4_13_0);
                vals_1[8 + 1] = __uint_as_float(_v4_13_1);
                vals_1[8 + 2] = __uint_as_float(_v4_13_2);
                vals_1[8 + 3] = __uint_as_float(_v4_13_3);
            }
            {
                unsigned _v4_14_0;
                unsigned _v4_14_1;
                unsigned _v4_14_2;
                unsigned _v4_14_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_14_0), "=r"(_v4_14_1), "=r"(_v4_14_2), "=r"(_v4_14_3) : "l"((const void*)(z + (at_z_1 + z_ld) + (0))) : "memory");
                vals_1[12 + 0] = __uint_as_float(_v4_14_0);
                vals_1[12 + 1] = __uint_as_float(_v4_14_1);
                vals_1[12 + 2] = __uint_as_float(_v4_14_2);
                vals_1[12 + 3] = __uint_as_float(_v4_14_3);
            }
        }
        unsigned int words[4] = {0};
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[12]), "f"(vals_0[8]));
            float _abs_16;
            asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(gap));
            float _neg_15;
            asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(_abs_16));
            float e = __nv_expf(_neg_15);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float big = ((first == 1) ? vals_0[0] : vals_0[4]);
            float small = ((first == 1) ? vals_0[4] : vals_0[0]);
            float _fma_18;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(e), "f"(small), "f"(big));
            float _fdiv_17 = __fdiv_rn(_fma_18, den);
            vals_0[0] = _fdiv_17;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[13]), "f"(vals_0[9]));
            float _abs_20;
            asm("abs.f32 %0, %1;" : "=f"(_abs_20) : "f"(gap_0));
            float _neg_19;
            asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(_abs_20));
            float e_0 = __nv_expf(_neg_19);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float big_0 = ((first_0 == 1) ? vals_0[1] : vals_0[5]);
            float small_0 = ((first_0 == 1) ? vals_0[5] : vals_0[1]);
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(e_0), "f"(small_0), "f"(big_0));
            float _fdiv_21 = __fdiv_rn(_fma_22, den_0);
            vals_0[1] = _fdiv_21;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[14]), "f"(vals_0[10]));
            float _abs_24;
            asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(gap_1));
            float _neg_23;
            asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(_abs_24));
            float e_1 = __nv_expf(_neg_23);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float big_1 = ((first_1 == 1) ? vals_0[2] : vals_0[6]);
            float small_1 = ((first_1 == 1) ? vals_0[6] : vals_0[2]);
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(e_1), "f"(small_1), "f"(big_1));
            float _fdiv_25 = __fdiv_rn(_fma_26, den_1);
            vals_0[2] = _fdiv_25;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[15]), "f"(vals_0[11]));
            float _abs_28;
            asm("abs.f32 %0, %1;" : "=f"(_abs_28) : "f"(gap_2));
            float _neg_27;
            asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(_abs_28));
            float e_2 = __nv_expf(_neg_27);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float big_2 = ((first_2 == 1) ? vals_0[3] : vals_0[7]);
            float small_2 = ((first_2 == 1) ? vals_0[7] : vals_0[3]);
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(e_2), "f"(small_2), "f"(big_2));
            float _fdiv_29 = __fdiv_rn(_fma_30, den_2);
            vals_0[3] = _fdiv_29;
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
        if (row_1 < pooled_rows) {
            float gap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_3) : "f"(vals_1[12]), "f"(vals_1[8]));
            float _abs_32;
            asm("abs.f32 %0, %1;" : "=f"(_abs_32) : "f"(gap_3));
            float _neg_31;
            asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(_abs_32));
            float e_3 = __nv_expf(_neg_31);
            float den_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_3) : "f"(e_3), "f"(1.0f));
            int first_3 = ((gap_3 <= 0.0f) ? 1 : 0);
            float big_3 = ((first_3 == 1) ? vals_1[0] : vals_1[4]);
            float small_3 = ((first_3 == 1) ? vals_1[4] : vals_1[0]);
            float _fma_34;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(e_3), "f"(small_3), "f"(big_3));
            float _fdiv_33 = __fdiv_rn(_fma_34, den_3);
            vals_1[0] = _fdiv_33;
            float gap_3_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_3_1) : "f"(vals_1[13]), "f"(vals_1[9]));
            float _abs_36;
            asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(gap_3_1));
            float _neg_35;
            asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(_abs_36));
            float e_3_1 = __nv_expf(_neg_35);
            float den_3_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_3_1) : "f"(e_3_1), "f"(1.0f));
            int first_3_1 = ((gap_3_1 <= 0.0f) ? 1 : 0);
            float big_3_1 = ((first_3_1 == 1) ? vals_1[1] : vals_1[5]);
            float small_3_1 = ((first_3_1 == 1) ? vals_1[5] : vals_1[1]);
            float _fma_38;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(e_3_1), "f"(small_3_1), "f"(big_3_1));
            float _fdiv_37 = __fdiv_rn(_fma_38, den_3_1);
            vals_1[1] = _fdiv_37;
            float gap_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_4) : "f"(vals_1[14]), "f"(vals_1[10]));
            float _abs_40;
            asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(gap_4));
            float _neg_39;
            asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(_abs_40));
            float e_4 = __nv_expf(_neg_39);
            float den_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_4) : "f"(e_4), "f"(1.0f));
            int first_4 = ((gap_4 <= 0.0f) ? 1 : 0);
            float big_4 = ((first_4 == 1) ? vals_1[2] : vals_1[6]);
            float small_4 = ((first_4 == 1) ? vals_1[6] : vals_1[2]);
            float _fma_42;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(e_4), "f"(small_4), "f"(big_4));
            float _fdiv_41 = __fdiv_rn(_fma_42, den_4);
            vals_1[2] = _fdiv_41;
            float gap_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_5) : "f"(vals_1[15]), "f"(vals_1[11]));
            float _abs_44;
            asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(gap_5));
            float _neg_43;
            asm("neg.f32 %0, %1;" : "=f"(_neg_43) : "f"(_abs_44));
            float e_5 = __nv_expf(_neg_43);
            float den_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_5) : "f"(e_5), "f"(1.0f));
            int first_5 = ((gap_5 <= 0.0f) ? 1 : 0);
            float big_5 = ((first_5 == 1) ? vals_1[3] : vals_1[7]);
            float small_5 = ((first_5 == 1) ? vals_1[7] : vals_1[3]);
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(e_5), "f"(small_5), "f"(big_5));
            float _fdiv_45 = __fdiv_rn(_fma_46, den_5);
            vals_1[3] = _fdiv_45;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                words[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 2], vals_1[_lp*2+1 + 2]));
                words[_lp + 3] = *(uint32_t*)&_bf2;
            }
        }
        if (row_1 < G) {
            reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(out + ((long long)row_1 * 512ll + col)))[0] = reinterpret_cast<int2*>(words + (2))[0];
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
