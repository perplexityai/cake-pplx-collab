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
kernel_cake_gated_kv_compression_det_train_0ef98d6170a08bd594f9(float* c, float* z, int* cu_seqlens, int* cu_pairs, __nv_bfloat16* out, int num_docs, int G, long long c_ld, long long z_ld)
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
    int nblocks = (G + 1) / 2;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 2 + slot;
        float vals_0[16];
        if (row_0 < pooled_rows) {
            int doc = 0;
            int doc_hi = num_docs - 1;
            while (doc < doc_hi) {
                int mid = doc + doc_hi + 1 >> 1;
                if (row_0 >= cu_pairs[mid]) {
                    doc = mid;
                } else {
                    doc_hi = mid - 1;
                }
            }
            int start = cu_seqlens[doc] + 2 * (row_0 - cu_pairs[doc]);
            int start_0 = start;
            long long at_c_0 = (long long)start_0 * c_ld + col;
            long long at_z_0 = (long long)start_0 * z_ld + col;
            {
                unsigned _v4_0_0;
                unsigned _v4_0_1;
                unsigned _v4_0_2;
                unsigned _v4_0_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(c + at_c_0 + (0))) : "memory");
                vals_0[0 + 0] = __uint_as_float(_v4_0_0);
                vals_0[0 + 1] = __uint_as_float(_v4_0_1);
                vals_0[0 + 2] = __uint_as_float(_v4_0_2);
                vals_0[0 + 3] = __uint_as_float(_v4_0_3);
            }
            {
                unsigned _v4_1_0;
                unsigned _v4_1_1;
                unsigned _v4_1_2;
                unsigned _v4_1_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(c + (at_c_0 + c_ld) + (0))) : "memory");
                vals_0[4 + 0] = __uint_as_float(_v4_1_0);
                vals_0[4 + 1] = __uint_as_float(_v4_1_1);
                vals_0[4 + 2] = __uint_as_float(_v4_1_2);
                vals_0[4 + 3] = __uint_as_float(_v4_1_3);
            }
            {
                unsigned _v4_2_0;
                unsigned _v4_2_1;
                unsigned _v4_2_2;
                unsigned _v4_2_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(z + at_z_0 + (0))) : "memory");
                vals_0[8 + 0] = __uint_as_float(_v4_2_0);
                vals_0[8 + 1] = __uint_as_float(_v4_2_1);
                vals_0[8 + 2] = __uint_as_float(_v4_2_2);
                vals_0[8 + 3] = __uint_as_float(_v4_2_3);
            }
            {
                unsigned _v4_3_0;
                unsigned _v4_3_1;
                unsigned _v4_3_2;
                unsigned _v4_3_3;
                asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(z + (at_z_0 + z_ld) + (0))) : "memory");
                vals_0[12 + 0] = __uint_as_float(_v4_3_0);
                vals_0[12 + 1] = __uint_as_float(_v4_3_1);
                vals_0[12 + 2] = __uint_as_float(_v4_3_2);
                vals_0[12 + 3] = __uint_as_float(_v4_3_3);
            }
        }
        unsigned int words[2] = {0};
        if (row_0 < pooled_rows) {
            float gap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap) : "f"(vals_0[12]), "f"(vals_0[8]));
            float _abs_5;
            asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(gap));
            float _neg_4;
            asm("neg.f32 %0, %1;" : "=f"(_neg_4) : "f"(_abs_5));
            float e = __nv_expf(_neg_4);
            float den;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den) : "f"(e), "f"(1.0f));
            int first = ((gap <= 0.0f) ? 1 : 0);
            float big = ((first == 1) ? vals_0[0] : vals_0[4]);
            float small = ((first == 1) ? vals_0[4] : vals_0[0]);
            float _fma_7;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(e), "f"(small), "f"(big));
            float _fdiv_6 = __fdiv_rn(_fma_7, den);
            vals_0[0] = _fdiv_6;
            float gap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_0) : "f"(vals_0[13]), "f"(vals_0[9]));
            float _abs_9;
            asm("abs.f32 %0, %1;" : "=f"(_abs_9) : "f"(gap_0));
            float _neg_8;
            asm("neg.f32 %0, %1;" : "=f"(_neg_8) : "f"(_abs_9));
            float e_0 = __nv_expf(_neg_8);
            float den_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_0) : "f"(e_0), "f"(1.0f));
            int first_0 = ((gap_0 <= 0.0f) ? 1 : 0);
            float big_0 = ((first_0 == 1) ? vals_0[1] : vals_0[5]);
            float small_0 = ((first_0 == 1) ? vals_0[5] : vals_0[1]);
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(e_0), "f"(small_0), "f"(big_0));
            float _fdiv_10 = __fdiv_rn(_fma_11, den_0);
            vals_0[1] = _fdiv_10;
            float gap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_1) : "f"(vals_0[14]), "f"(vals_0[10]));
            float _abs_13;
            asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(gap_1));
            float _neg_12;
            asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(_abs_13));
            float e_1 = __nv_expf(_neg_12);
            float den_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_1) : "f"(e_1), "f"(1.0f));
            int first_1 = ((gap_1 <= 0.0f) ? 1 : 0);
            float big_1 = ((first_1 == 1) ? vals_0[2] : vals_0[6]);
            float small_1 = ((first_1 == 1) ? vals_0[6] : vals_0[2]);
            float _fma_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(e_1), "f"(small_1), "f"(big_1));
            float _fdiv_14 = __fdiv_rn(_fma_15, den_1);
            vals_0[2] = _fdiv_14;
            float gap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(gap_2) : "f"(vals_0[15]), "f"(vals_0[11]));
            float _abs_17;
            asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(gap_2));
            float _neg_16;
            asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
            float e_2 = __nv_expf(_neg_16);
            float den_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(den_2) : "f"(e_2), "f"(1.0f));
            int first_2 = ((gap_2 <= 0.0f) ? 1 : 0);
            float big_2 = ((first_2 == 1) ? vals_0[3] : vals_0[7]);
            float small_2 = ((first_2 == 1) ? vals_0[7] : vals_0[3]);
            float _fma_19;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(e_2), "f"(small_2), "f"(big_2));
            float _fdiv_18 = __fdiv_rn(_fma_19, den_2);
            vals_0[3] = _fdiv_18;
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
