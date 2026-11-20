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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gated_kv_compression_det_train_aa24263114554decff8e(__nv_bfloat16* dout, float* c, float* z, int* cu_seqlens, int* cu_pairs, float* dc, float* dz, int num_docs, int G, long long c_ld, long long z_ld, long long dc_ld, long long dz_ld)
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
    int rows_end = G + num_docs;
    float zeros[8] = {0};
    int _count_leading_zeros_0 = __clz(num_docs - 1 | 1);
    int rounds = (36 - _count_leading_zeros_0) / 5;
    int nblocks = (rows_end + 3) / 4;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 4 + slot;
        float vals_0[32];
        float grads_0[8];
        int start_0 = 0;
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
            start_0 = start;
            long long at_c_0 = (long long)start_0 * c_ld + col;
            long long at_z_0 = (long long)start_0 * z_ld + col;
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(dout + ((long long)row_0 * 512ll + col) + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vld_4[_blk].x), "=r"(_vld_4[_blk].y), "=r"(_vld_4[_blk].z), "=r"(_vld_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)) : "memory");
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&grads_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&grads_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
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
                    : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
                vals_0[0 + 0] = __uint_as_float(_ldv8_5_0);
                vals_0[0 + 1] = __uint_as_float(_ldv8_5_1);
                vals_0[0 + 2] = __uint_as_float(_ldv8_5_2);
                vals_0[0 + 3] = __uint_as_float(_ldv8_5_3);
                vals_0[0 + 4] = __uint_as_float(_ldv8_5_4);
                vals_0[0 + 5] = __uint_as_float(_ldv8_5_5);
                vals_0[0 + 6] = __uint_as_float(_ldv8_5_6);
                vals_0[0 + 7] = __uint_as_float(_ldv8_5_7);
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
                    : "=r"(_ldv8_6_0), "=r"(_ldv8_6_1), "=r"(_ldv8_6_2), "=r"(_ldv8_6_3), "=r"(_ldv8_6_4), "=r"(_ldv8_6_5), "=r"(_ldv8_6_6), "=r"(_ldv8_6_7) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
                vals_0[8 + 0] = __uint_as_float(_ldv8_6_0);
                vals_0[8 + 1] = __uint_as_float(_ldv8_6_1);
                vals_0[8 + 2] = __uint_as_float(_ldv8_6_2);
                vals_0[8 + 3] = __uint_as_float(_ldv8_6_3);
                vals_0[8 + 4] = __uint_as_float(_ldv8_6_4);
                vals_0[8 + 5] = __uint_as_float(_ldv8_6_5);
                vals_0[8 + 6] = __uint_as_float(_ldv8_6_6);
                vals_0[8 + 7] = __uint_as_float(_ldv8_6_7);
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
                    : "=r"(_ldv8_7_0), "=r"(_ldv8_7_1), "=r"(_ldv8_7_2), "=r"(_ldv8_7_3), "=r"(_ldv8_7_4), "=r"(_ldv8_7_5), "=r"(_ldv8_7_6), "=r"(_ldv8_7_7) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
                vals_0[16 + 0] = __uint_as_float(_ldv8_7_0);
                vals_0[16 + 1] = __uint_as_float(_ldv8_7_1);
                vals_0[16 + 2] = __uint_as_float(_ldv8_7_2);
                vals_0[16 + 3] = __uint_as_float(_ldv8_7_3);
                vals_0[16 + 4] = __uint_as_float(_ldv8_7_4);
                vals_0[16 + 5] = __uint_as_float(_ldv8_7_5);
                vals_0[16 + 6] = __uint_as_float(_ldv8_7_6);
                vals_0[16 + 7] = __uint_as_float(_ldv8_7_7);
            }
            {
                unsigned _ldv8_8_0;
                unsigned _ldv8_8_1;
                unsigned _ldv8_8_2;
                unsigned _ldv8_8_3;
                unsigned _ldv8_8_4;
                unsigned _ldv8_8_5;
                unsigned _ldv8_8_6;
                unsigned _ldv8_8_7;
                asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_8_0), "=r"(_ldv8_8_1), "=r"(_ldv8_8_2), "=r"(_ldv8_8_3), "=r"(_ldv8_8_4), "=r"(_ldv8_8_5), "=r"(_ldv8_8_6), "=r"(_ldv8_8_7) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
                vals_0[24 + 0] = __uint_as_float(_ldv8_8_0);
                vals_0[24 + 1] = __uint_as_float(_ldv8_8_1);
                vals_0[24 + 2] = __uint_as_float(_ldv8_8_2);
                vals_0[24 + 3] = __uint_as_float(_ldv8_8_3);
                vals_0[24 + 4] = __uint_as_float(_ldv8_8_4);
                vals_0[24 + 5] = __uint_as_float(_ldv8_8_5);
                vals_0[24 + 6] = __uint_as_float(_ldv8_8_6);
                vals_0[24 + 7] = __uint_as_float(_ldv8_8_7);
            }
        }
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[24]), "f"(vals_0[16]));
            float _abs_10;
            asm("abs.f32 %0, %1;" : "=f"(_abs_10) : "f"(gap));
            float _neg_9;
            asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(_abs_10));
            float e = __nv_expf(_neg_9);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float p_big = __frcp_rn(den);
            float p_small = __fdiv_rn(e, den);
            float _sub_11;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(vals_0[0]), "f"(vals_0[8]));
            float dz_first;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big, p_small), _sub_11)), "f"(grads_0[0]));
            float dc_first;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first) : "f"(((first == 1) ? p_big : p_small)), "f"(grads_0[0]));
            float dc_second;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second) : "f"(((first == 1) ? p_small : p_big)), "f"(grads_0[0]));
            vals_0[0] = dc_first;
            vals_0[8] = dc_second;
            vals_0[16] = dz_first;
            float _neg_12;
            asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(dz_first));
            vals_0[24] = _neg_12;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[25]), "f"(vals_0[17]));
            float _abs_14;
            asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(gap_0));
            float _neg_13;
            asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(_abs_14));
            float e_0 = __nv_expf(_neg_13);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float p_big_0 = __frcp_rn(den_0);
            float p_small_0 = __fdiv_rn(e_0, den_0);
            float _sub_15;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(vals_0[1]), "f"(vals_0[9]));
            float dz_first_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_0) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_0, p_small_0), _sub_15)), "f"(grads_0[1]));
            float dc_first_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_0) : "f"(((first_0 == 1) ? p_big_0 : p_small_0)), "f"(grads_0[1]));
            float dc_second_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_0) : "f"(((first_0 == 1) ? p_small_0 : p_big_0)), "f"(grads_0[1]));
            vals_0[1] = dc_first_0;
            vals_0[9] = dc_second_0;
            vals_0[17] = dz_first_0;
            float _neg_16;
            asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(dz_first_0));
            vals_0[25] = _neg_16;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[26]), "f"(vals_0[18]));
            float _abs_18;
            asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(gap_1));
            float _neg_17;
            asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(_abs_18));
            float e_1 = __nv_expf(_neg_17);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float p_big_1 = __frcp_rn(den_1);
            float p_small_1 = __fdiv_rn(e_1, den_1);
            float _sub_19;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(vals_0[2]), "f"(vals_0[10]));
            float dz_first_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_1) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_1, p_small_1), _sub_19)), "f"(grads_0[2]));
            float dc_first_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_1) : "f"(((first_1 == 1) ? p_big_1 : p_small_1)), "f"(grads_0[2]));
            float dc_second_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_1) : "f"(((first_1 == 1) ? p_small_1 : p_big_1)), "f"(grads_0[2]));
            vals_0[2] = dc_first_1;
            vals_0[10] = dc_second_1;
            vals_0[18] = dz_first_1;
            float _neg_20;
            asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(dz_first_1));
            vals_0[26] = _neg_20;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[27]), "f"(vals_0[19]));
            float _abs_22;
            asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(gap_2));
            float _neg_21;
            asm("neg.f32 %0, %1;" : "=f"(_neg_21) : "f"(_abs_22));
            float e_2 = __nv_expf(_neg_21);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float p_big_2 = __frcp_rn(den_2);
            float p_small_2 = __fdiv_rn(e_2, den_2);
            float _sub_23;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(vals_0[3]), "f"(vals_0[11]));
            float dz_first_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_2) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_2, p_small_2), _sub_23)), "f"(grads_0[3]));
            float dc_first_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_2) : "f"(((first_2 == 1) ? p_big_2 : p_small_2)), "f"(grads_0[3]));
            float dc_second_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_2) : "f"(((first_2 == 1) ? p_small_2 : p_big_2)), "f"(grads_0[3]));
            vals_0[3] = dc_first_2;
            vals_0[11] = dc_second_2;
            vals_0[19] = dz_first_2;
            float _neg_24;
            asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(dz_first_2));
            vals_0[27] = _neg_24;
            float gap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_3) : "f"(vals_0[28]), "f"(vals_0[20]));
            float _abs_26;
            asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(gap_3));
            float _neg_25;
            asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(_abs_26));
            float e_3 = __nv_expf(_neg_25);
            float den_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_3) : "f"(e_3), "f"(1.0f));
            int first_3 = ((gap_3 <= 0.0f) ? 1 : 0);
            float p_big_3 = __frcp_rn(den_3);
            float p_small_3 = __fdiv_rn(e_3, den_3);
            float _sub_27;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(vals_0[4]), "f"(vals_0[12]));
            float dz_first_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_3) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_3, p_small_3), _sub_27)), "f"(grads_0[4]));
            float dc_first_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_3) : "f"(((first_3 == 1) ? p_big_3 : p_small_3)), "f"(grads_0[4]));
            float dc_second_3;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_3) : "f"(((first_3 == 1) ? p_small_3 : p_big_3)), "f"(grads_0[4]));
            vals_0[4] = dc_first_3;
            vals_0[12] = dc_second_3;
            vals_0[20] = dz_first_3;
            float _neg_28;
            asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(dz_first_3));
            vals_0[28] = _neg_28;
            float gap_4;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_4) : "f"(vals_0[29]), "f"(vals_0[21]));
            float _abs_30;
            asm("abs.f32 %0, %1;" : "=f"(_abs_30) : "f"(gap_4));
            float _neg_29;
            asm("neg.f32 %0, %1;" : "=f"(_neg_29) : "f"(_abs_30));
            float e_4 = __nv_expf(_neg_29);
            float den_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_4) : "f"(e_4), "f"(1.0f));
            int first_4 = ((gap_4 <= 0.0f) ? 1 : 0);
            float p_big_4 = __frcp_rn(den_4);
            float p_small_4 = __fdiv_rn(e_4, den_4);
            float _sub_31;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(vals_0[5]), "f"(vals_0[13]));
            float dz_first_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_4) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_4, p_small_4), _sub_31)), "f"(grads_0[5]));
            float dc_first_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_4) : "f"(((first_4 == 1) ? p_big_4 : p_small_4)), "f"(grads_0[5]));
            float dc_second_4;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_4) : "f"(((first_4 == 1) ? p_small_4 : p_big_4)), "f"(grads_0[5]));
            vals_0[5] = dc_first_4;
            vals_0[13] = dc_second_4;
            vals_0[21] = dz_first_4;
            float _neg_32;
            asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(dz_first_4));
            vals_0[29] = _neg_32;
            float gap_5;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_5) : "f"(vals_0[30]), "f"(vals_0[22]));
            float _abs_34;
            asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(gap_5));
            float _neg_33;
            asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(_abs_34));
            float e_5 = __nv_expf(_neg_33);
            float den_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_5) : "f"(e_5), "f"(1.0f));
            int first_5 = ((gap_5 <= 0.0f) ? 1 : 0);
            float p_big_5 = __frcp_rn(den_5);
            float p_small_5 = __fdiv_rn(e_5, den_5);
            float _sub_35;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(vals_0[6]), "f"(vals_0[14]));
            float dz_first_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_5) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_5, p_small_5), _sub_35)), "f"(grads_0[6]));
            float dc_first_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_5) : "f"(((first_5 == 1) ? p_big_5 : p_small_5)), "f"(grads_0[6]));
            float dc_second_5;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_5) : "f"(((first_5 == 1) ? p_small_5 : p_big_5)), "f"(grads_0[6]));
            vals_0[6] = dc_first_5;
            vals_0[14] = dc_second_5;
            vals_0[22] = dz_first_5;
            float _neg_36;
            asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(dz_first_5));
            vals_0[30] = _neg_36;
            float gap_6;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_6) : "f"(vals_0[31]), "f"(vals_0[23]));
            float _abs_38;
            asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(gap_6));
            float _neg_37;
            asm("neg.f32 %0, %1;" : "=f"(_neg_37) : "f"(_abs_38));
            float e_6 = __nv_expf(_neg_37);
            float den_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_6) : "f"(e_6), "f"(1.0f));
            int first_6 = ((gap_6 <= 0.0f) ? 1 : 0);
            float p_big_6 = __frcp_rn(den_6);
            float p_small_6 = __fdiv_rn(e_6, den_6);
            float _sub_39;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(vals_0[7]), "f"(vals_0[15]));
            float dz_first_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_6) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_6, p_small_6), _sub_39)), "f"(grads_0[7]));
            float dc_first_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_6) : "f"(((first_6 == 1) ? p_big_6 : p_small_6)), "f"(grads_0[7]));
            float dc_second_6;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_6) : "f"(((first_6 == 1) ? p_small_6 : p_big_6)), "f"(grads_0[7]));
            vals_0[7] = dc_first_6;
            vals_0[15] = dc_second_6;
            vals_0[23] = dz_first_6;
            float _neg_40;
            asm("neg.f32 %0, %1;" : "=f"(_neg_40) : "f"(dz_first_6));
            vals_0[31] = _neg_40;
            long long at_dc_0 = (long long)start_0 * dc_ld + col;
            long long at_dz_0 = (long long)start_0 * dz_ld + col;
            {
                unsigned _stv8_41_0 = __float_as_uint(vals_0[0 + 0]);
                unsigned _stv8_41_1 = __float_as_uint(vals_0[0 + 1]);
                unsigned _stv8_41_2 = __float_as_uint(vals_0[0 + 2]);
                unsigned _stv8_41_3 = __float_as_uint(vals_0[0 + 3]);
                unsigned _stv8_41_4 = __float_as_uint(vals_0[0 + 4]);
                unsigned _stv8_41_5 = __float_as_uint(vals_0[0 + 5]);
                unsigned _stv8_41_6 = __float_as_uint(vals_0[0 + 6]);
                unsigned _stv8_41_7 = __float_as_uint(vals_0[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dc + at_dc_0 + (0))), "r"(_stv8_41_0), "r"(_stv8_41_1), "r"(_stv8_41_2), "r"(_stv8_41_3), "r"(_stv8_41_4), "r"(_stv8_41_5), "r"(_stv8_41_6), "r"(_stv8_41_7) : "memory");
            }
            {
                unsigned _stv8_42_0 = __float_as_uint(vals_0[8 + 0]);
                unsigned _stv8_42_1 = __float_as_uint(vals_0[8 + 1]);
                unsigned _stv8_42_2 = __float_as_uint(vals_0[8 + 2]);
                unsigned _stv8_42_3 = __float_as_uint(vals_0[8 + 3]);
                unsigned _stv8_42_4 = __float_as_uint(vals_0[8 + 4]);
                unsigned _stv8_42_5 = __float_as_uint(vals_0[8 + 5]);
                unsigned _stv8_42_6 = __float_as_uint(vals_0[8 + 6]);
                unsigned _stv8_42_7 = __float_as_uint(vals_0[8 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dc + (at_dc_0 + dc_ld) + (0))), "r"(_stv8_42_0), "r"(_stv8_42_1), "r"(_stv8_42_2), "r"(_stv8_42_3), "r"(_stv8_42_4), "r"(_stv8_42_5), "r"(_stv8_42_6), "r"(_stv8_42_7) : "memory");
            }
            {
                unsigned _stv8_43_0 = __float_as_uint(vals_0[16 + 0]);
                unsigned _stv8_43_1 = __float_as_uint(vals_0[16 + 1]);
                unsigned _stv8_43_2 = __float_as_uint(vals_0[16 + 2]);
                unsigned _stv8_43_3 = __float_as_uint(vals_0[16 + 3]);
                unsigned _stv8_43_4 = __float_as_uint(vals_0[16 + 4]);
                unsigned _stv8_43_5 = __float_as_uint(vals_0[16 + 5]);
                unsigned _stv8_43_6 = __float_as_uint(vals_0[16 + 6]);
                unsigned _stv8_43_7 = __float_as_uint(vals_0[16 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dz + at_dz_0 + (0))), "r"(_stv8_43_0), "r"(_stv8_43_1), "r"(_stv8_43_2), "r"(_stv8_43_3), "r"(_stv8_43_4), "r"(_stv8_43_5), "r"(_stv8_43_6), "r"(_stv8_43_7) : "memory");
            }
            {
                unsigned _stv8_44_0 = __float_as_uint(vals_0[24 + 0]);
                unsigned _stv8_44_1 = __float_as_uint(vals_0[24 + 1]);
                unsigned _stv8_44_2 = __float_as_uint(vals_0[24 + 2]);
                unsigned _stv8_44_3 = __float_as_uint(vals_0[24 + 3]);
                unsigned _stv8_44_4 = __float_as_uint(vals_0[24 + 4]);
                unsigned _stv8_44_5 = __float_as_uint(vals_0[24 + 5]);
                unsigned _stv8_44_6 = __float_as_uint(vals_0[24 + 6]);
                unsigned _stv8_44_7 = __float_as_uint(vals_0[24 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dz + (at_dz_0 + dz_ld) + (0))), "r"(_stv8_44_0), "r"(_stv8_44_1), "r"(_stv8_44_2), "r"(_stv8_44_3), "r"(_stv8_44_4), "r"(_stv8_44_5), "r"(_stv8_44_6), "r"(_stv8_44_7) : "memory");
            }
        } else if (row_0 >= G) {
            if (row_0 < rows_end) {
                int doc_0 = row_0 - G;
                int end_0 = cu_seqlens[doc_0 + 1];
                if ((end_0 - cu_seqlens[doc_0] & 1) == 1) {
                    long long last_0 = (long long)(end_0 - 1);
                    {
                        unsigned _stv8_45_0 = __float_as_uint(zeros[0 + 0]);
                        unsigned _stv8_45_1 = __float_as_uint(zeros[0 + 1]);
                        unsigned _stv8_45_2 = __float_as_uint(zeros[0 + 2]);
                        unsigned _stv8_45_3 = __float_as_uint(zeros[0 + 3]);
                        unsigned _stv8_45_4 = __float_as_uint(zeros[0 + 4]);
                        unsigned _stv8_45_5 = __float_as_uint(zeros[0 + 5]);
                        unsigned _stv8_45_6 = __float_as_uint(zeros[0 + 6]);
                        unsigned _stv8_45_7 = __float_as_uint(zeros[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dc + (last_0 * dc_ld + col) + (0))), "r"(_stv8_45_0), "r"(_stv8_45_1), "r"(_stv8_45_2), "r"(_stv8_45_3), "r"(_stv8_45_4), "r"(_stv8_45_5), "r"(_stv8_45_6), "r"(_stv8_45_7) : "memory");
                    }
                    {
                        unsigned _stv8_46_0 = __float_as_uint(zeros[0 + 0]);
                        unsigned _stv8_46_1 = __float_as_uint(zeros[0 + 1]);
                        unsigned _stv8_46_2 = __float_as_uint(zeros[0 + 2]);
                        unsigned _stv8_46_3 = __float_as_uint(zeros[0 + 3]);
                        unsigned _stv8_46_4 = __float_as_uint(zeros[0 + 4]);
                        unsigned _stv8_46_5 = __float_as_uint(zeros[0 + 5]);
                        unsigned _stv8_46_6 = __float_as_uint(zeros[0 + 6]);
                        unsigned _stv8_46_7 = __float_as_uint(zeros[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dz + (last_0 * dz_ld + col) + (0))), "r"(_stv8_46_0), "r"(_stv8_46_1), "r"(_stv8_46_2), "r"(_stv8_46_3), "r"(_stv8_46_4), "r"(_stv8_46_5), "r"(_stv8_46_6), "r"(_stv8_46_7) : "memory");
                    }
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
