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
kernel_cake_gated_kv_compression_det_train_16dc016d084d18b8be56(__nv_bfloat16* dout, float* c, float* z, int* cu_seqlens, int* cu_pairs, float* dc, float* dz, int num_docs, int G, long long c_ld, long long z_ld, long long dc_ld, long long dz_ld)
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
    int rows_end = G + num_docs;
    float zeros[4] = {0};
    int _count_leading_zeros_0 = __clz(num_docs - 1 | 1);
    int rounds = (36 - _count_leading_zeros_0) / 5;
    int nblocks = (rows_end + 1) / 2;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 2 + slot;
        float vals_0[16];
        float grads_0[4];
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
                uint2 _vld_4;
                asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_vld_4.x), "=r"(_vld_4.y) : "l"((const void*)(dout + ((long long)row_0 * 512ll + col) + 0)) : "memory");
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&grads_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&grads_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
            {
                unsigned _v4_5_0;
                unsigned _v4_5_1;
                unsigned _v4_5_2;
                unsigned _v4_5_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
                vals_0[0 + 0] = __uint_as_float(_v4_5_0);
                vals_0[0 + 1] = __uint_as_float(_v4_5_1);
                vals_0[0 + 2] = __uint_as_float(_v4_5_2);
                vals_0[0 + 3] = __uint_as_float(_v4_5_3);
            }
            {
                unsigned _v4_6_0;
                unsigned _v4_6_1;
                unsigned _v4_6_2;
                unsigned _v4_6_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
                vals_0[4 + 0] = __uint_as_float(_v4_6_0);
                vals_0[4 + 1] = __uint_as_float(_v4_6_1);
                vals_0[4 + 2] = __uint_as_float(_v4_6_2);
                vals_0[4 + 3] = __uint_as_float(_v4_6_3);
            }
            {
                unsigned _v4_7_0;
                unsigned _v4_7_1;
                unsigned _v4_7_2;
                unsigned _v4_7_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
                vals_0[8 + 0] = __uint_as_float(_v4_7_0);
                vals_0[8 + 1] = __uint_as_float(_v4_7_1);
                vals_0[8 + 2] = __uint_as_float(_v4_7_2);
                vals_0[8 + 3] = __uint_as_float(_v4_7_3);
            }
            {
                unsigned _v4_8_0;
                unsigned _v4_8_1;
                unsigned _v4_8_2;
                unsigned _v4_8_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
                vals_0[12 + 0] = __uint_as_float(_v4_8_0);
                vals_0[12 + 1] = __uint_as_float(_v4_8_1);
                vals_0[12 + 2] = __uint_as_float(_v4_8_2);
                vals_0[12 + 3] = __uint_as_float(_v4_8_3);
            }
        }
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[12]), "f"(vals_0[8]));
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
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(vals_0[0]), "f"(vals_0[4]));
            float dz_first;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big, p_small), _sub_11)), "f"(grads_0[0]));
            float dc_first;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first) : "f"(((first == 1) ? p_big : p_small)), "f"(grads_0[0]));
            float dc_second;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second) : "f"(((first == 1) ? p_small : p_big)), "f"(grads_0[0]));
            vals_0[0] = dc_first;
            vals_0[4] = dc_second;
            vals_0[8] = dz_first;
            float _neg_12;
            asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(dz_first));
            vals_0[12] = _neg_12;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[13]), "f"(vals_0[9]));
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
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(vals_0[1]), "f"(vals_0[5]));
            float dz_first_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_0) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_0, p_small_0), _sub_15)), "f"(grads_0[1]));
            float dc_first_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_0) : "f"(((first_0 == 1) ? p_big_0 : p_small_0)), "f"(grads_0[1]));
            float dc_second_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_0) : "f"(((first_0 == 1) ? p_small_0 : p_big_0)), "f"(grads_0[1]));
            vals_0[1] = dc_first_0;
            vals_0[5] = dc_second_0;
            vals_0[9] = dz_first_0;
            float _neg_16;
            asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(dz_first_0));
            vals_0[13] = _neg_16;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[14]), "f"(vals_0[10]));
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
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(vals_0[2]), "f"(vals_0[6]));
            float dz_first_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_1) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_1, p_small_1), _sub_19)), "f"(grads_0[2]));
            float dc_first_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_1) : "f"(((first_1 == 1) ? p_big_1 : p_small_1)), "f"(grads_0[2]));
            float dc_second_1;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_1) : "f"(((first_1 == 1) ? p_small_1 : p_big_1)), "f"(grads_0[2]));
            vals_0[2] = dc_first_1;
            vals_0[6] = dc_second_1;
            vals_0[10] = dz_first_1;
            float _neg_20;
            asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(dz_first_1));
            vals_0[14] = _neg_20;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[15]), "f"(vals_0[11]));
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
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(vals_0[3]), "f"(vals_0[7]));
            float dz_first_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz_first_2) : "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(p_big_2, p_small_2), _sub_23)), "f"(grads_0[3]));
            float dc_first_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_first_2) : "f"(((first_2 == 1) ? p_big_2 : p_small_2)), "f"(grads_0[3]));
            float dc_second_2;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(dc_second_2) : "f"(((first_2 == 1) ? p_small_2 : p_big_2)), "f"(grads_0[3]));
            vals_0[3] = dc_first_2;
            vals_0[7] = dc_second_2;
            vals_0[11] = dz_first_2;
            float _neg_24;
            asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(dz_first_2));
            vals_0[15] = _neg_24;
            long long at_dc_0 = (long long)start_0 * dc_ld + col;
            long long at_dz_0 = (long long)start_0 * dz_ld + col;
            {
                float4 _v4 = make_float4(vals_0[0 + 0], vals_0[0 + 1], vals_0[0 + 2], vals_0[0 + 3]);
                *reinterpret_cast<float4*>((dc + at_dc_0) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(vals_0[4 + 0], vals_0[4 + 1], vals_0[4 + 2], vals_0[4 + 3]);
                *reinterpret_cast<float4*>((dc + (at_dc_0 + dc_ld)) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(vals_0[8 + 0], vals_0[8 + 1], vals_0[8 + 2], vals_0[8 + 3]);
                *reinterpret_cast<float4*>((dz + at_dz_0) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(vals_0[12 + 0], vals_0[12 + 1], vals_0[12 + 2], vals_0[12 + 3]);
                *reinterpret_cast<float4*>((dz + (at_dz_0 + dz_ld)) + 0) = _v4;
            }
        } else if (row_0 >= G) {
            if (row_0 < rows_end) {
                int doc_0 = row_0 - G;
                int end_0 = cu_seqlens[doc_0 + 1];
                if ((end_0 - cu_seqlens[doc_0] & 1) == 1) {
                    long long last_0 = (long long)(end_0 - 1);
                    {
                        float4 _v4 = make_float4(zeros[0 + 0], zeros[0 + 1], zeros[0 + 2], zeros[0 + 3]);
                        *reinterpret_cast<float4*>((dc + (last_0 * dc_ld + col)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(zeros[0 + 0], zeros[0 + 1], zeros[0 + 2], zeros[0 + 3]);
                        *reinterpret_cast<float4*>((dz + (last_0 * dz_ld + col)) + 0) = _v4;
                    }
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
