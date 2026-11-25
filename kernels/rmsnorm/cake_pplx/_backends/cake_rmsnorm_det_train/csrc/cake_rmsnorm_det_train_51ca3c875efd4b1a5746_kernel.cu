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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_rmsnorm_det_train_51ca3c875efd4b1a5746(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = lane * 8;
    float wreg[40];
    {
        unsigned _ldv8_0_0;
        unsigned _ldv8_0_1;
        unsigned _ldv8_0_2;
        unsigned _ldv8_0_3;
        unsigned _ldv8_0_4;
        unsigned _ldv8_0_5;
        unsigned _ldv8_0_6;
        unsigned _ldv8_0_7;
        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + col + (0))) : "memory");
        wreg[0 + 0] = __uint_as_float(_ldv8_0_0);
        wreg[0 + 1] = __uint_as_float(_ldv8_0_1);
        wreg[0 + 2] = __uint_as_float(_ldv8_0_2);
        wreg[0 + 3] = __uint_as_float(_ldv8_0_3);
        wreg[0 + 4] = __uint_as_float(_ldv8_0_4);
        wreg[0 + 5] = __uint_as_float(_ldv8_0_5);
        wreg[0 + 6] = __uint_as_float(_ldv8_0_6);
        wreg[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    {
        unsigned _ldv8_1_0;
        unsigned _ldv8_1_1;
        unsigned _ldv8_1_2;
        unsigned _ldv8_1_3;
        unsigned _ldv8_1_4;
        unsigned _ldv8_1_5;
        unsigned _ldv8_1_6;
        unsigned _ldv8_1_7;
        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + (col + 256) + (0))) : "memory");
        wreg[8 + 0] = __uint_as_float(_ldv8_1_0);
        wreg[8 + 1] = __uint_as_float(_ldv8_1_1);
        wreg[8 + 2] = __uint_as_float(_ldv8_1_2);
        wreg[8 + 3] = __uint_as_float(_ldv8_1_3);
        wreg[8 + 4] = __uint_as_float(_ldv8_1_4);
        wreg[8 + 5] = __uint_as_float(_ldv8_1_5);
        wreg[8 + 6] = __uint_as_float(_ldv8_1_6);
        wreg[8 + 7] = __uint_as_float(_ldv8_1_7);
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
        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + (col + 512) + (0))) : "memory");
        wreg[16 + 0] = __uint_as_float(_ldv8_2_0);
        wreg[16 + 1] = __uint_as_float(_ldv8_2_1);
        wreg[16 + 2] = __uint_as_float(_ldv8_2_2);
        wreg[16 + 3] = __uint_as_float(_ldv8_2_3);
        wreg[16 + 4] = __uint_as_float(_ldv8_2_4);
        wreg[16 + 5] = __uint_as_float(_ldv8_2_5);
        wreg[16 + 6] = __uint_as_float(_ldv8_2_6);
        wreg[16 + 7] = __uint_as_float(_ldv8_2_7);
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
        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(w + (col + 768) + (0))) : "memory");
        wreg[24 + 0] = __uint_as_float(_ldv8_3_0);
        wreg[24 + 1] = __uint_as_float(_ldv8_3_1);
        wreg[24 + 2] = __uint_as_float(_ldv8_3_2);
        wreg[24 + 3] = __uint_as_float(_ldv8_3_3);
        wreg[24 + 4] = __uint_as_float(_ldv8_3_4);
        wreg[24 + 5] = __uint_as_float(_ldv8_3_5);
        wreg[24 + 6] = __uint_as_float(_ldv8_3_6);
        wreg[24 + 7] = __uint_as_float(_ldv8_3_7);
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
        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(w + (col + 1024) + (0))) : "memory");
        wreg[32 + 0] = __uint_as_float(_ldv8_4_0);
        wreg[32 + 1] = __uint_as_float(_ldv8_4_1);
        wreg[32 + 2] = __uint_as_float(_ldv8_4_2);
        wreg[32 + 3] = __uint_as_float(_ldv8_4_3);
        wreg[32 + 4] = __uint_as_float(_ldv8_4_4);
        wreg[32 + 5] = __uint_as_float(_ldv8_4_5);
        wreg[32 + 6] = __uint_as_float(_ldv8_4_6);
        wreg[32 + 7] = __uint_as_float(_ldv8_4_7);
    }
    int nblocks = (T + 7) / 8;
    int lfirst = warp * 2 + 0;
    unsigned int xa[40];
    unsigned int xb[40];
    if (bid < nblocks) {
        int _min_5 = ((bid * 8 + lfirst) < (T - 1) ? (bid * 8 + lfirst) : (T - 1));
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + ((long long)_min_5 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&xa[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
        int _min_7 = ((bid * 8 + lfirst) < (T - 1) ? (bid * 8 + lfirst) : (T - 1));
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + ((long long)_min_7 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&xa[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
        int _min_9 = ((bid * 8 + lfirst) < (T - 1) ? (bid * 8 + lfirst) : (T - 1));
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + ((long long)_min_9 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&xa[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
        int _min_11 = ((bid * 8 + lfirst) < (T - 1) ? (bid * 8 + lfirst) : (T - 1));
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + ((long long)_min_11 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_12 = reinterpret_cast<uint4*>(&xa[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_12[_blk] = _vptr_12[_blk];
            }
        }
        int _min_13 = ((bid * 8 + lfirst) < (T - 1) ? (bid * 8 + lfirst) : (T - 1));
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + ((long long)_min_13 * x_stride_p0 + (long long)(col + 1024)) + 0);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&xa[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
        int _min_15 = ((bid * 8 + lfirst + 1) < (T - 1) ? (bid * 8 + lfirst + 1) : (T - 1));
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(x + ((long long)_min_15 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_16 = reinterpret_cast<uint4*>(&xa[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_16[_blk] = _vptr_16[_blk];
            }
        }
        int _min_17 = ((bid * 8 + lfirst + 1) < (T - 1) ? (bid * 8 + lfirst + 1) : (T - 1));
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(x + ((long long)_min_17 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_18 = reinterpret_cast<uint4*>(&xa[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_18[_blk] = _vptr_18[_blk];
            }
        }
        int _min_19 = ((bid * 8 + lfirst + 1) < (T - 1) ? (bid * 8 + lfirst + 1) : (T - 1));
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(x + ((long long)_min_19 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_20 = reinterpret_cast<uint4*>(&xa[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_20[_blk] = _vptr_20[_blk];
            }
        }
        int _min_21 = ((bid * 8 + lfirst + 1) < (T - 1) ? (bid * 8 + lfirst + 1) : (T - 1));
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(x + ((long long)_min_21 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_22 = reinterpret_cast<uint4*>(&xa[32]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_22[_blk] = _vptr_22[_blk];
            }
        }
        int _min_23 = ((bid * 8 + lfirst + 1) < (T - 1) ? (bid * 8 + lfirst + 1) : (T - 1));
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(x + ((long long)_min_23 * x_stride_p0 + (long long)(col + 1024)) + 0);
            uint4* _vdst_24 = reinterpret_cast<uint4*>(&xa[36]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_24[_blk] = _vptr_24[_blk];
            }
        }
    }
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids * 2) {
        int nb = blk + num_bids;
        if (nb < nblocks) {
            int _min_25 = ((nb * 8 + lfirst) < (T - 1) ? (nb * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_26 = reinterpret_cast<const uint4*>(x + ((long long)_min_25 * x_stride_p0 + (long long)col) + 0);
                uint4* _vdst_26 = reinterpret_cast<uint4*>(&xb[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_26[_blk] = _vptr_26[_blk];
                }
            }
            int _min_27 = ((nb * 8 + lfirst) < (T - 1) ? (nb * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_28 = reinterpret_cast<const uint4*>(x + ((long long)_min_27 * x_stride_p0 + (long long)(col + 256)) + 0);
                uint4* _vdst_28 = reinterpret_cast<uint4*>(&xb[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_28[_blk] = _vptr_28[_blk];
                }
            }
            int _min_29 = ((nb * 8 + lfirst) < (T - 1) ? (nb * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_30 = reinterpret_cast<const uint4*>(x + ((long long)_min_29 * x_stride_p0 + (long long)(col + 512)) + 0);
                uint4* _vdst_30 = reinterpret_cast<uint4*>(&xb[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_30[_blk] = _vptr_30[_blk];
                }
            }
            int _min_31 = ((nb * 8 + lfirst) < (T - 1) ? (nb * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_32 = reinterpret_cast<const uint4*>(x + ((long long)_min_31 * x_stride_p0 + (long long)(col + 768)) + 0);
                uint4* _vdst_32 = reinterpret_cast<uint4*>(&xb[12]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_32[_blk] = _vptr_32[_blk];
                }
            }
            int _min_33 = ((nb * 8 + lfirst) < (T - 1) ? (nb * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_34 = reinterpret_cast<const uint4*>(x + ((long long)_min_33 * x_stride_p0 + (long long)(col + 1024)) + 0);
                uint4* _vdst_34 = reinterpret_cast<uint4*>(&xb[16]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_34[_blk] = _vptr_34[_blk];
                }
            }
            int _min_35 = ((nb * 8 + lfirst + 1) < (T - 1) ? (nb * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_36 = reinterpret_cast<const uint4*>(x + ((long long)_min_35 * x_stride_p0 + (long long)col) + 0);
                uint4* _vdst_36 = reinterpret_cast<uint4*>(&xb[20]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_36[_blk] = _vptr_36[_blk];
                }
            }
            int _min_37 = ((nb * 8 + lfirst + 1) < (T - 1) ? (nb * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_38 = reinterpret_cast<const uint4*>(x + ((long long)_min_37 * x_stride_p0 + (long long)(col + 256)) + 0);
                uint4* _vdst_38 = reinterpret_cast<uint4*>(&xb[24]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_38[_blk] = _vptr_38[_blk];
                }
            }
            int _min_39 = ((nb * 8 + lfirst + 1) < (T - 1) ? (nb * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(x + ((long long)_min_39 * x_stride_p0 + (long long)(col + 512)) + 0);
                uint4* _vdst_40 = reinterpret_cast<uint4*>(&xb[28]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_40[_blk] = _vptr_40[_blk];
                }
            }
            int _min_41 = ((nb * 8 + lfirst + 1) < (T - 1) ? (nb * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(x + ((long long)_min_41 * x_stride_p0 + (long long)(col + 768)) + 0);
                uint4* _vdst_42 = reinterpret_cast<uint4*>(&xb[32]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_42[_blk] = _vptr_42[_blk];
                }
            }
            int _min_43 = ((nb * 8 + lfirst + 1) < (T - 1) ? (nb * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_44 = reinterpret_cast<const uint4*>(x + ((long long)_min_43 * x_stride_p0 + (long long)(col + 1024)) + 0);
                uint4* _vdst_44 = reinterpret_cast<uint4*>(&xb[36]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_44[_blk] = _vptr_44[_blk];
                }
            }
        }
        float tot[2];
        float sum_sq_0 = 0.0f;
        float _dot_45;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_45) : "r"(xa[0]), "r"(xa[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_45;
        float _dot_46;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_46) : "r"(xa[1]), "r"(xa[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_46;
        float _dot_47;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_47) : "r"(xa[2]), "r"(xa[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_47;
        float _dot_48;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_48) : "r"(xa[3]), "r"(xa[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_48;
        float _dot_49;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_49) : "r"(xa[4]), "r"(xa[4]), "f"(sum_sq_0));
        sum_sq_0 = _dot_49;
        float _dot_50;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_50) : "r"(xa[5]), "r"(xa[5]), "f"(sum_sq_0));
        sum_sq_0 = _dot_50;
        float _dot_51;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_51) : "r"(xa[6]), "r"(xa[6]), "f"(sum_sq_0));
        sum_sq_0 = _dot_51;
        float _dot_52;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_52) : "r"(xa[7]), "r"(xa[7]), "f"(sum_sq_0));
        sum_sq_0 = _dot_52;
        float _dot_53;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_53) : "r"(xa[8]), "r"(xa[8]), "f"(sum_sq_0));
        sum_sq_0 = _dot_53;
        float _dot_54;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_54) : "r"(xa[9]), "r"(xa[9]), "f"(sum_sq_0));
        sum_sq_0 = _dot_54;
        float _dot_55;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_55) : "r"(xa[10]), "r"(xa[10]), "f"(sum_sq_0));
        sum_sq_0 = _dot_55;
        float _dot_56;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_56) : "r"(xa[11]), "r"(xa[11]), "f"(sum_sq_0));
        sum_sq_0 = _dot_56;
        float _dot_57;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_57) : "r"(xa[12]), "r"(xa[12]), "f"(sum_sq_0));
        sum_sq_0 = _dot_57;
        float _dot_58;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_58) : "r"(xa[13]), "r"(xa[13]), "f"(sum_sq_0));
        sum_sq_0 = _dot_58;
        float _dot_59;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_59) : "r"(xa[14]), "r"(xa[14]), "f"(sum_sq_0));
        sum_sq_0 = _dot_59;
        float _dot_60;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_60) : "r"(xa[15]), "r"(xa[15]), "f"(sum_sq_0));
        sum_sq_0 = _dot_60;
        float _dot_61;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_61) : "r"(xa[16]), "r"(xa[16]), "f"(sum_sq_0));
        sum_sq_0 = _dot_61;
        float _dot_62;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_62) : "r"(xa[17]), "r"(xa[17]), "f"(sum_sq_0));
        sum_sq_0 = _dot_62;
        float _dot_63;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_63) : "r"(xa[18]), "r"(xa[18]), "f"(sum_sq_0));
        sum_sq_0 = _dot_63;
        float _dot_64;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_64) : "r"(xa[19]), "r"(xa[19]), "f"(sum_sq_0));
        sum_sq_0 = _dot_64;
        float _warp_reduce_65 = sum_sq_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_65 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_65, offset);
        tot[0] = _warp_reduce_65;
        float sum_sq_1 = 0.0f;
        float _dot_66;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_66) : "r"(xa[20]), "r"(xa[20]), "f"(sum_sq_1));
        sum_sq_1 = _dot_66;
        float _dot_67;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_67) : "r"(xa[21]), "r"(xa[21]), "f"(sum_sq_1));
        sum_sq_1 = _dot_67;
        float _dot_68;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_68) : "r"(xa[22]), "r"(xa[22]), "f"(sum_sq_1));
        sum_sq_1 = _dot_68;
        float _dot_69;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_69) : "r"(xa[23]), "r"(xa[23]), "f"(sum_sq_1));
        sum_sq_1 = _dot_69;
        float _dot_70;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_70) : "r"(xa[24]), "r"(xa[24]), "f"(sum_sq_1));
        sum_sq_1 = _dot_70;
        float _dot_71;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_71) : "r"(xa[25]), "r"(xa[25]), "f"(sum_sq_1));
        sum_sq_1 = _dot_71;
        float _dot_72;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_72) : "r"(xa[26]), "r"(xa[26]), "f"(sum_sq_1));
        sum_sq_1 = _dot_72;
        float _dot_73;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_73) : "r"(xa[27]), "r"(xa[27]), "f"(sum_sq_1));
        sum_sq_1 = _dot_73;
        float _dot_74;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_74) : "r"(xa[28]), "r"(xa[28]), "f"(sum_sq_1));
        sum_sq_1 = _dot_74;
        float _dot_75;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_75) : "r"(xa[29]), "r"(xa[29]), "f"(sum_sq_1));
        sum_sq_1 = _dot_75;
        float _dot_76;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_76) : "r"(xa[30]), "r"(xa[30]), "f"(sum_sq_1));
        sum_sq_1 = _dot_76;
        float _dot_77;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_77) : "r"(xa[31]), "r"(xa[31]), "f"(sum_sq_1));
        sum_sq_1 = _dot_77;
        float _dot_78;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_78) : "r"(xa[32]), "r"(xa[32]), "f"(sum_sq_1));
        sum_sq_1 = _dot_78;
        float _dot_79;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_79) : "r"(xa[33]), "r"(xa[33]), "f"(sum_sq_1));
        sum_sq_1 = _dot_79;
        float _dot_80;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_80) : "r"(xa[34]), "r"(xa[34]), "f"(sum_sq_1));
        sum_sq_1 = _dot_80;
        float _dot_81;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_81) : "r"(xa[35]), "r"(xa[35]), "f"(sum_sq_1));
        sum_sq_1 = _dot_81;
        float _dot_82;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_82) : "r"(xa[36]), "r"(xa[36]), "f"(sum_sq_1));
        sum_sq_1 = _dot_82;
        float _dot_83;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_83) : "r"(xa[37]), "r"(xa[37]), "f"(sum_sq_1));
        sum_sq_1 = _dot_83;
        float _dot_84;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_84) : "r"(xa[38]), "r"(xa[38]), "f"(sum_sq_1));
        sum_sq_1 = _dot_84;
        float _dot_85;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_85) : "r"(xa[39]), "r"(xa[39]), "f"(sum_sq_1));
        sum_sq_1 = _dot_85;
        float _warp_reduce_86 = sum_sq_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_86 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_86, offset);
        tot[1] = _warp_reduce_86;
        float own = tot[0];
        if (lane == 1) {
            own = tot[1];
        }
        float mean = __fdiv_rn(own, 1280.0f);
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_87;
        asm("neg.f32 %0, %1;" : "=f"(_neg_87) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_87));
        float _neg_88;
        asm("neg.f32 %0, %1;" : "=f"(_neg_88) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_88), "f"(rs_y), "f"(1.0f));
        float _neg_90;
        asm("neg.f32 %0, %1;" : "=f"(_neg_90) : "f"(rs_tl));
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(_neg_90), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_89;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_91;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_91), "f"(rs_h), "f"(rs_y));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_92), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_94;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_94) : "f"(rs_a));
            float _rcp_93 = approx_rcp(_sqrt_94);
            rs_y = _rcp_93;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_96;
            asm("neg.f32 %0, %1;" : "=f"(_neg_96) : "f"(rs_t));
            float _fma_95;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(rs_a), "f"(rs_y), "f"(_neg_96));
            rs_tl = _fma_95;
            float _neg_98;
            asm("neg.f32 %0, %1;" : "=f"(_neg_98) : "f"(rs_t));
            float _fma_97;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(_neg_98), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_97;
            float _neg_100;
            asm("neg.f32 %0, %1;" : "=f"(_neg_100) : "f"(rs_tl));
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(_neg_100), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_99;
            float _fma_101;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_101);
            unsigned long long rs_m = (unsigned long long)((rs_fb & 8388607) << 1 | 16777217);
            unsigned int rs_an = ((rs_ab < 8388608) ? __as_u32((float)rs_ab) : rs_ab);
            unsigned long long rs_p = rs_m * rs_m * (unsigned long long)(rs_an & 8388607 | 8388608);
            rs_r = ((rs_ab - 1 < 2139095039) ? __uint_as_float(rs_fb + (unsigned int)(rs_p >> 63ull)) : rs_y);
        }
        float rstd = rs_r;
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, 0);
        float out_0[8];
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[0]));
        out_0[0] = xv_0.x * rk_0 * wreg[0];
        out_0[1] = xv_0.y * rk_0 * wreg[1];
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[1]));
        out_0[2] = xv_1.x * rk_0 * wreg[2];
        out_0[3] = xv_1.y * rk_0 * wreg[3];
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[2]));
        out_0[4] = xv_2.x * rk_0 * wreg[4];
        out_0[5] = xv_2.y * rk_0 * wreg[5];
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[3]));
        out_0[6] = xv_3.x * rk_0 * wreg[6];
        out_0[7] = xv_3.y * rk_0 * wreg[7];
        unsigned int packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 2], out_0[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 4], out_0[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 6], out_0[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst) * 1280ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        float out_1[8];
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[4]));
        out_1[0] = xv_4.x * rk_0 * wreg[8];
        out_1[1] = xv_4.y * rk_0 * wreg[9];
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[5]));
        out_1[2] = xv_5.x * rk_0 * wreg[10];
        out_1[3] = xv_5.y * rk_0 * wreg[11];
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[6]));
        out_1[4] = xv_6.x * rk_0 * wreg[12];
        out_1[5] = xv_6.y * rk_0 * wreg[13];
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[7]));
        out_1[6] = xv_7.x * rk_0 * wreg[14];
        out_1[7] = xv_7.y * rk_0 * wreg[15];
        unsigned int packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
            packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 2], out_1[_lp*2+1 + 2]));
            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 4], out_1[_lp*2+1 + 4]));
            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 6], out_1[_lp*2+1 + 6]));
            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst) * 1280ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        float out_2[8];
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[8]));
        out_2[0] = xv_8.x * rk_0 * wreg[16];
        out_2[1] = xv_8.y * rk_0 * wreg[17];
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[9]));
        out_2[2] = xv_9.x * rk_0 * wreg[18];
        out_2[3] = xv_9.y * rk_0 * wreg[19];
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[10]));
        out_2[4] = xv_10.x * rk_0 * wreg[20];
        out_2[5] = xv_10.y * rk_0 * wreg[21];
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[11]));
        out_2[6] = xv_11.x * rk_0 * wreg[22];
        out_2[7] = xv_11.y * rk_0 * wreg[23];
        unsigned int packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
            packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 2], out_2[_lp*2+1 + 2]));
            packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 4], out_2[_lp*2+1 + 4]));
            packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 6], out_2[_lp*2+1 + 6]));
            packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst) * 1280ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_2)[0];
        }
        float out_3[8];
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[12]));
        out_3[0] = xv_12.x * rk_0 * wreg[24];
        out_3[1] = xv_12.y * rk_0 * wreg[25];
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[13]));
        out_3[2] = xv_13.x * rk_0 * wreg[26];
        out_3[3] = xv_13.y * rk_0 * wreg[27];
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[14]));
        out_3[4] = xv_14.x * rk_0 * wreg[28];
        out_3[5] = xv_14.y * rk_0 * wreg[29];
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[15]));
        out_3[6] = xv_15.x * rk_0 * wreg[30];
        out_3[7] = xv_15.y * rk_0 * wreg[31];
        unsigned int packed_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
            packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 2], out_3[_lp*2+1 + 2]));
            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 4], out_3[_lp*2+1 + 4]));
            packed_3[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 6], out_3[_lp*2+1 + 6]));
            packed_3[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst) * 1280ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        float out_4[8];
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[16]));
        out_4[0] = xv_16.x * rk_0 * wreg[32];
        out_4[1] = xv_16.y * rk_0 * wreg[33];
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[17]));
        out_4[2] = xv_17.x * rk_0 * wreg[34];
        out_4[3] = xv_17.y * rk_0 * wreg[35];
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[18]));
        out_4[4] = xv_18.x * rk_0 * wreg[36];
        out_4[5] = xv_18.y * rk_0 * wreg[37];
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[19]));
        out_4[6] = xv_19.x * rk_0 * wreg[38];
        out_4[7] = xv_19.y * rk_0 * wreg[39];
        unsigned int packed_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 0], out_4[_lp*2+1 + 0]));
            packed_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 2], out_4[_lp*2+1 + 2]));
            packed_4[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 4], out_4[_lp*2+1 + 4]));
            packed_4[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 6], out_4[_lp*2+1 + 6]));
            packed_4[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst) * 1280ll + (long long)(col + 1024)))[0] = reinterpret_cast<int4*>(packed_4)[0];
        }
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, 1);
        float out_5[8];
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[20]));
        out_5[0] = xv_20.x * rk_1 * wreg[0];
        out_5[1] = xv_20.y * rk_1 * wreg[1];
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[21]));
        out_5[2] = xv_21.x * rk_1 * wreg[2];
        out_5[3] = xv_21.y * rk_1 * wreg[3];
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[22]));
        out_5[4] = xv_22.x * rk_1 * wreg[4];
        out_5[5] = xv_22.y * rk_1 * wreg[5];
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[23]));
        out_5[6] = xv_23.x * rk_1 * wreg[6];
        out_5[7] = xv_23.y * rk_1 * wreg[7];
        unsigned int packed_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 0], out_5[_lp*2+1 + 0]));
            packed_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 2], out_5[_lp*2+1 + 2]));
            packed_5[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 4], out_5[_lp*2+1 + 4]));
            packed_5[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 6], out_5[_lp*2+1 + 6]));
            packed_5[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst + 1) * 1280ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_5)[0];
        }
        float out_6[8];
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[24]));
        out_6[0] = xv_24.x * rk_1 * wreg[8];
        out_6[1] = xv_24.y * rk_1 * wreg[9];
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[25]));
        out_6[2] = xv_25.x * rk_1 * wreg[10];
        out_6[3] = xv_25.y * rk_1 * wreg[11];
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[26]));
        out_6[4] = xv_26.x * rk_1 * wreg[12];
        out_6[5] = xv_26.y * rk_1 * wreg[13];
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[27]));
        out_6[6] = xv_27.x * rk_1 * wreg[14];
        out_6[7] = xv_27.y * rk_1 * wreg[15];
        unsigned int packed_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 0], out_6[_lp*2+1 + 0]));
            packed_6[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 2], out_6[_lp*2+1 + 2]));
            packed_6[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 4], out_6[_lp*2+1 + 4]));
            packed_6[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 6], out_6[_lp*2+1 + 6]));
            packed_6[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst + 1) * 1280ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_6)[0];
        }
        float out_7[8];
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[28]));
        out_7[0] = xv_28.x * rk_1 * wreg[16];
        out_7[1] = xv_28.y * rk_1 * wreg[17];
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[29]));
        out_7[2] = xv_29.x * rk_1 * wreg[18];
        out_7[3] = xv_29.y * rk_1 * wreg[19];
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[30]));
        out_7[4] = xv_30.x * rk_1 * wreg[20];
        out_7[5] = xv_30.y * rk_1 * wreg[21];
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[31]));
        out_7[6] = xv_31.x * rk_1 * wreg[22];
        out_7[7] = xv_31.y * rk_1 * wreg[23];
        unsigned int packed_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 0], out_7[_lp*2+1 + 0]));
            packed_7[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 2], out_7[_lp*2+1 + 2]));
            packed_7[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 4], out_7[_lp*2+1 + 4]));
            packed_7[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 6], out_7[_lp*2+1 + 6]));
            packed_7[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst + 1) * 1280ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_7)[0];
        }
        float out_8[8];
        float2 xv_32 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[32]));
        out_8[0] = xv_32.x * rk_1 * wreg[24];
        out_8[1] = xv_32.y * rk_1 * wreg[25];
        float2 xv_33 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[33]));
        out_8[2] = xv_33.x * rk_1 * wreg[26];
        out_8[3] = xv_33.y * rk_1 * wreg[27];
        float2 xv_34 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[34]));
        out_8[4] = xv_34.x * rk_1 * wreg[28];
        out_8[5] = xv_34.y * rk_1 * wreg[29];
        float2 xv_35 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[35]));
        out_8[6] = xv_35.x * rk_1 * wreg[30];
        out_8[7] = xv_35.y * rk_1 * wreg[31];
        unsigned int packed_8[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_8[_lp*2 + 0], out_8[_lp*2+1 + 0]));
            packed_8[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_8[_lp*2 + 2], out_8[_lp*2+1 + 2]));
            packed_8[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_8[_lp*2 + 4], out_8[_lp*2+1 + 4]));
            packed_8[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_8[_lp*2 + 6], out_8[_lp*2+1 + 6]));
            packed_8[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst + 1) * 1280ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_8)[0];
        }
        float out_9[8];
        float2 xv_36 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[36]));
        out_9[0] = xv_36.x * rk_1 * wreg[32];
        out_9[1] = xv_36.y * rk_1 * wreg[33];
        float2 xv_37 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[37]));
        out_9[2] = xv_37.x * rk_1 * wreg[34];
        out_9[3] = xv_37.y * rk_1 * wreg[35];
        float2 xv_38 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[38]));
        out_9[4] = xv_38.x * rk_1 * wreg[36];
        out_9[5] = xv_38.y * rk_1 * wreg[37];
        float2 xv_39 = cake_bf16x2_to_f32x2(__as_bf16x2(xa[39]));
        out_9[6] = xv_39.x * rk_1 * wreg[38];
        out_9[7] = xv_39.y * rk_1 * wreg[39];
        unsigned int packed_9[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_9[_lp*2 + 0], out_9[_lp*2+1 + 0]));
            packed_9[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_9[_lp*2 + 2], out_9[_lp*2+1 + 2]));
            packed_9[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_9[_lp*2 + 4], out_9[_lp*2+1 + 4]));
            packed_9[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_9[_lp*2 + 6], out_9[_lp*2+1 + 6]));
            packed_9[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (blk * 8 + lfirst + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 8 + lfirst + 1) * 1280ll + (long long)(col + 1024)))[0] = reinterpret_cast<int4*>(packed_9)[0];
        }
        if ((lane & 0) == 0) {
            if (lane < 2) {
                if (blk * 8 + lfirst + lane < T) {
                    r[blk * 8 + lfirst + lane] = rstd;
                }
            }
        }
        if (nb + num_bids < nblocks) {
            int _min_102 = (((nb + num_bids) * 8 + lfirst) < (T - 1) ? ((nb + num_bids) * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_103 = reinterpret_cast<const uint4*>(x + ((long long)_min_102 * x_stride_p0 + (long long)col) + 0);
                uint4* _vdst_103 = reinterpret_cast<uint4*>(&xa[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_103[_blk] = _vptr_103[_blk];
                }
            }
            int _min_104 = (((nb + num_bids) * 8 + lfirst) < (T - 1) ? ((nb + num_bids) * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_105 = reinterpret_cast<const uint4*>(x + ((long long)_min_104 * x_stride_p0 + (long long)(col + 256)) + 0);
                uint4* _vdst_105 = reinterpret_cast<uint4*>(&xa[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_105[_blk] = _vptr_105[_blk];
                }
            }
            int _min_106 = (((nb + num_bids) * 8 + lfirst) < (T - 1) ? ((nb + num_bids) * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_107 = reinterpret_cast<const uint4*>(x + ((long long)_min_106 * x_stride_p0 + (long long)(col + 512)) + 0);
                uint4* _vdst_107 = reinterpret_cast<uint4*>(&xa[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_107[_blk] = _vptr_107[_blk];
                }
            }
            int _min_108 = (((nb + num_bids) * 8 + lfirst) < (T - 1) ? ((nb + num_bids) * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_109 = reinterpret_cast<const uint4*>(x + ((long long)_min_108 * x_stride_p0 + (long long)(col + 768)) + 0);
                uint4* _vdst_109 = reinterpret_cast<uint4*>(&xa[12]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_109[_blk] = _vptr_109[_blk];
                }
            }
            int _min_110 = (((nb + num_bids) * 8 + lfirst) < (T - 1) ? ((nb + num_bids) * 8 + lfirst) : (T - 1));
            {
                const uint4* _vptr_111 = reinterpret_cast<const uint4*>(x + ((long long)_min_110 * x_stride_p0 + (long long)(col + 1024)) + 0);
                uint4* _vdst_111 = reinterpret_cast<uint4*>(&xa[16]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_111[_blk] = _vptr_111[_blk];
                }
            }
            int _min_112 = (((nb + num_bids) * 8 + lfirst + 1) < (T - 1) ? ((nb + num_bids) * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_113 = reinterpret_cast<const uint4*>(x + ((long long)_min_112 * x_stride_p0 + (long long)col) + 0);
                uint4* _vdst_113 = reinterpret_cast<uint4*>(&xa[20]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_113[_blk] = _vptr_113[_blk];
                }
            }
            int _min_114 = (((nb + num_bids) * 8 + lfirst + 1) < (T - 1) ? ((nb + num_bids) * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_115 = reinterpret_cast<const uint4*>(x + ((long long)_min_114 * x_stride_p0 + (long long)(col + 256)) + 0);
                uint4* _vdst_115 = reinterpret_cast<uint4*>(&xa[24]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_115[_blk] = _vptr_115[_blk];
                }
            }
            int _min_116 = (((nb + num_bids) * 8 + lfirst + 1) < (T - 1) ? ((nb + num_bids) * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_117 = reinterpret_cast<const uint4*>(x + ((long long)_min_116 * x_stride_p0 + (long long)(col + 512)) + 0);
                uint4* _vdst_117 = reinterpret_cast<uint4*>(&xa[28]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_117[_blk] = _vptr_117[_blk];
                }
            }
            int _min_118 = (((nb + num_bids) * 8 + lfirst + 1) < (T - 1) ? ((nb + num_bids) * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_119 = reinterpret_cast<const uint4*>(x + ((long long)_min_118 * x_stride_p0 + (long long)(col + 768)) + 0);
                uint4* _vdst_119 = reinterpret_cast<uint4*>(&xa[32]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_119[_blk] = _vptr_119[_blk];
                }
            }
            int _min_120 = (((nb + num_bids) * 8 + lfirst + 1) < (T - 1) ? ((nb + num_bids) * 8 + lfirst + 1) : (T - 1));
            {
                const uint4* _vptr_121 = reinterpret_cast<const uint4*>(x + ((long long)_min_120 * x_stride_p0 + (long long)(col + 1024)) + 0);
                uint4* _vdst_121 = reinterpret_cast<uint4*>(&xa[36]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_121[_blk] = _vptr_121[_blk];
                }
            }
        }
        if (nb < nblocks) {
            float tot_0[2];
            float sum_sq_2 = 0.0f;
            float _dot_122;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_122) : "r"(xb[0]), "r"(xb[0]), "f"(sum_sq_2));
            sum_sq_2 = _dot_122;
            float _dot_123;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_123) : "r"(xb[1]), "r"(xb[1]), "f"(sum_sq_2));
            sum_sq_2 = _dot_123;
            float _dot_124;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_124) : "r"(xb[2]), "r"(xb[2]), "f"(sum_sq_2));
            sum_sq_2 = _dot_124;
            float _dot_125;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_125) : "r"(xb[3]), "r"(xb[3]), "f"(sum_sq_2));
            sum_sq_2 = _dot_125;
            float _dot_126;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_126) : "r"(xb[4]), "r"(xb[4]), "f"(sum_sq_2));
            sum_sq_2 = _dot_126;
            float _dot_127;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_127) : "r"(xb[5]), "r"(xb[5]), "f"(sum_sq_2));
            sum_sq_2 = _dot_127;
            float _dot_128;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_128) : "r"(xb[6]), "r"(xb[6]), "f"(sum_sq_2));
            sum_sq_2 = _dot_128;
            float _dot_129;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_129) : "r"(xb[7]), "r"(xb[7]), "f"(sum_sq_2));
            sum_sq_2 = _dot_129;
            float _dot_130;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_130) : "r"(xb[8]), "r"(xb[8]), "f"(sum_sq_2));
            sum_sq_2 = _dot_130;
            float _dot_131;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_131) : "r"(xb[9]), "r"(xb[9]), "f"(sum_sq_2));
            sum_sq_2 = _dot_131;
            float _dot_132;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_132) : "r"(xb[10]), "r"(xb[10]), "f"(sum_sq_2));
            sum_sq_2 = _dot_132;
            float _dot_133;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_133) : "r"(xb[11]), "r"(xb[11]), "f"(sum_sq_2));
            sum_sq_2 = _dot_133;
            float _dot_134;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_134) : "r"(xb[12]), "r"(xb[12]), "f"(sum_sq_2));
            sum_sq_2 = _dot_134;
            float _dot_135;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_135) : "r"(xb[13]), "r"(xb[13]), "f"(sum_sq_2));
            sum_sq_2 = _dot_135;
            float _dot_136;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_136) : "r"(xb[14]), "r"(xb[14]), "f"(sum_sq_2));
            sum_sq_2 = _dot_136;
            float _dot_137;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_137) : "r"(xb[15]), "r"(xb[15]), "f"(sum_sq_2));
            sum_sq_2 = _dot_137;
            float _dot_138;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_138) : "r"(xb[16]), "r"(xb[16]), "f"(sum_sq_2));
            sum_sq_2 = _dot_138;
            float _dot_139;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_139) : "r"(xb[17]), "r"(xb[17]), "f"(sum_sq_2));
            sum_sq_2 = _dot_139;
            float _dot_140;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_140) : "r"(xb[18]), "r"(xb[18]), "f"(sum_sq_2));
            sum_sq_2 = _dot_140;
            float _dot_141;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_141) : "r"(xb[19]), "r"(xb[19]), "f"(sum_sq_2));
            sum_sq_2 = _dot_141;
            float _warp_reduce_142 = sum_sq_2;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_142 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_142, offset);
            tot_0[0] = _warp_reduce_142;
            float sum_sq_3 = 0.0f;
            float _dot_143;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_143) : "r"(xb[20]), "r"(xb[20]), "f"(sum_sq_3));
            sum_sq_3 = _dot_143;
            float _dot_144;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_144) : "r"(xb[21]), "r"(xb[21]), "f"(sum_sq_3));
            sum_sq_3 = _dot_144;
            float _dot_145;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_145) : "r"(xb[22]), "r"(xb[22]), "f"(sum_sq_3));
            sum_sq_3 = _dot_145;
            float _dot_146;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_146) : "r"(xb[23]), "r"(xb[23]), "f"(sum_sq_3));
            sum_sq_3 = _dot_146;
            float _dot_147;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_147) : "r"(xb[24]), "r"(xb[24]), "f"(sum_sq_3));
            sum_sq_3 = _dot_147;
            float _dot_148;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_148) : "r"(xb[25]), "r"(xb[25]), "f"(sum_sq_3));
            sum_sq_3 = _dot_148;
            float _dot_149;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_149) : "r"(xb[26]), "r"(xb[26]), "f"(sum_sq_3));
            sum_sq_3 = _dot_149;
            float _dot_150;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_150) : "r"(xb[27]), "r"(xb[27]), "f"(sum_sq_3));
            sum_sq_3 = _dot_150;
            float _dot_151;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_151) : "r"(xb[28]), "r"(xb[28]), "f"(sum_sq_3));
            sum_sq_3 = _dot_151;
            float _dot_152;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_152) : "r"(xb[29]), "r"(xb[29]), "f"(sum_sq_3));
            sum_sq_3 = _dot_152;
            float _dot_153;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_153) : "r"(xb[30]), "r"(xb[30]), "f"(sum_sq_3));
            sum_sq_3 = _dot_153;
            float _dot_154;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_154) : "r"(xb[31]), "r"(xb[31]), "f"(sum_sq_3));
            sum_sq_3 = _dot_154;
            float _dot_155;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_155) : "r"(xb[32]), "r"(xb[32]), "f"(sum_sq_3));
            sum_sq_3 = _dot_155;
            float _dot_156;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_156) : "r"(xb[33]), "r"(xb[33]), "f"(sum_sq_3));
            sum_sq_3 = _dot_156;
            float _dot_157;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_157) : "r"(xb[34]), "r"(xb[34]), "f"(sum_sq_3));
            sum_sq_3 = _dot_157;
            float _dot_158;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_158) : "r"(xb[35]), "r"(xb[35]), "f"(sum_sq_3));
            sum_sq_3 = _dot_158;
            float _dot_159;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_159) : "r"(xb[36]), "r"(xb[36]), "f"(sum_sq_3));
            sum_sq_3 = _dot_159;
            float _dot_160;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_160) : "r"(xb[37]), "r"(xb[37]), "f"(sum_sq_3));
            sum_sq_3 = _dot_160;
            float _dot_161;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_161) : "r"(xb[38]), "r"(xb[38]), "f"(sum_sq_3));
            sum_sq_3 = _dot_161;
            float _dot_162;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_162) : "r"(xb[39]), "r"(xb[39]), "f"(sum_sq_3));
            sum_sq_3 = _dot_162;
            float _warp_reduce_163 = sum_sq_3;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_163 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_163, offset);
            tot_0[1] = _warp_reduce_163;
            float own_0 = tot_0[0];
            if (lane == 1) {
                own_0 = tot_0[1];
            }
            float mean_0 = __fdiv_rn(own_0, 1280.0f);
            float rs_a_0 = mean_0 + eps;
            unsigned int rs_ab_0 = __as_u32(rs_a_0);
            float rs_y_0;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y_0) : "f"(rs_a_0));
            float rs_t_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t_0) : "f"(rs_a_0), "f"(rs_y_0));
            float _neg_164;
            asm("neg.f32 %0, %1;" : "=f"(_neg_164) : "f"(rs_t_0));
            float rs_tl_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl_0) : "f"(rs_a_0), "f"(rs_y_0), "f"(_neg_164));
            float _neg_165;
            asm("neg.f32 %0, %1;" : "=f"(_neg_165) : "f"(rs_t_0));
            float rs_e_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e_0) : "f"(_neg_165), "f"(rs_y_0), "f"(1.0f));
            float _neg_167;
            asm("neg.f32 %0, %1;" : "=f"(_neg_167) : "f"(rs_tl_0));
            float _fma_166;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(_neg_167), "f"(rs_y_0), "f"(rs_e_0));
            rs_e_0 = _fma_166;
            float rs_h_0;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h_0) : "f"(rs_y_0), "f"(0.5f));
            float _sub_168;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(rs_e_0), "f"(3.637979e-12f));
            float rs_lo_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo_0) : "f"(_sub_168), "f"(rs_h_0), "f"(rs_y_0));
            float _add_169;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(rs_e_0), "f"(3.637979e-12f));
            float rs_hi_0;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi_0) : "f"(_add_169), "f"(rs_h_0), "f"(rs_y_0));
            float rs_r_0 = rs_lo_0;
            unsigned int rs_slow_0 = ((rs_ab_0 - 8388608 >= 2130706432) ? 1 : 0);
            if (__as_u32(rs_lo_0) != __as_u32(rs_hi_0)) {
                rs_slow_0 = 1;
            }
            if (rs_slow_0 != 0) {
                float _sqrt_171;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_171) : "f"(rs_a_0));
                float _rcp_170 = approx_rcp(_sqrt_171);
                rs_y_0 = _rcp_170;
                rs_t_0 = mul_f32_rn_noftz(rs_a_0, rs_y_0);
                float _neg_173;
                asm("neg.f32 %0, %1;" : "=f"(_neg_173) : "f"(rs_t_0));
                float _fma_172;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(rs_a_0), "f"(rs_y_0), "f"(_neg_173));
                rs_tl_0 = _fma_172;
                float _neg_175;
                asm("neg.f32 %0, %1;" : "=f"(_neg_175) : "f"(rs_t_0));
                float _fma_174;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(_neg_175), "f"(rs_y_0), "f"(1.0f));
                rs_e_0 = _fma_174;
                float _neg_177;
                asm("neg.f32 %0, %1;" : "=f"(_neg_177) : "f"(rs_tl_0));
                float _fma_176;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(_neg_177), "f"(rs_y_0), "f"(rs_e_0));
                rs_e_0 = _fma_176;
                float _fma_178;
                asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(rs_e_0), "f"(mul_f32_rn_noftz(rs_y_0, 0.5f)), "f"(rs_y_0));
                unsigned int rs_fb_1 = __as_u32(_fma_178);
                unsigned long long rs_m_1 = (unsigned long long)((rs_fb_1 & 8388607) << 1 | 16777217);
                unsigned int rs_an_1 = ((rs_ab_0 < 8388608) ? __as_u32((float)rs_ab_0) : rs_ab_0);
                unsigned long long rs_p_1 = rs_m_1 * rs_m_1 * (unsigned long long)(rs_an_1 & 8388607 | 8388608);
                rs_r_0 = ((rs_ab_0 - 1 < 2139095039) ? __uint_as_float(rs_fb_1 + (unsigned int)(rs_p_1 >> 63ull)) : rs_y_0);
            }
            float rstd_0 = rs_r_0;
            float rk_2 = __shfl_sync(0xFFFFFFFF, rstd_0, 0);
            float out_10[8];
            float2 xv_40 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[0]));
            out_10[0] = xv_40.x * rk_2 * wreg[0];
            out_10[1] = xv_40.y * rk_2 * wreg[1];
            float2 xv_41 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[1]));
            out_10[2] = xv_41.x * rk_2 * wreg[2];
            out_10[3] = xv_41.y * rk_2 * wreg[3];
            float2 xv_42 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[2]));
            out_10[4] = xv_42.x * rk_2 * wreg[4];
            out_10[5] = xv_42.y * rk_2 * wreg[5];
            float2 xv_43 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[3]));
            out_10[6] = xv_43.x * rk_2 * wreg[6];
            out_10[7] = xv_43.y * rk_2 * wreg[7];
            unsigned int packed_10[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_10[_lp*2 + 0], out_10[_lp*2+1 + 0]));
                packed_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_10[_lp*2 + 2], out_10[_lp*2+1 + 2]));
                packed_10[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_10[_lp*2 + 4], out_10[_lp*2+1 + 4]));
                packed_10[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_10[_lp*2 + 6], out_10[_lp*2+1 + 6]));
                packed_10[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst) * 1280ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_10)[0];
            }
            float out_11[8];
            float2 xv_44 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[4]));
            out_11[0] = xv_44.x * rk_2 * wreg[8];
            out_11[1] = xv_44.y * rk_2 * wreg[9];
            float2 xv_45 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[5]));
            out_11[2] = xv_45.x * rk_2 * wreg[10];
            out_11[3] = xv_45.y * rk_2 * wreg[11];
            float2 xv_46 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[6]));
            out_11[4] = xv_46.x * rk_2 * wreg[12];
            out_11[5] = xv_46.y * rk_2 * wreg[13];
            float2 xv_47 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[7]));
            out_11[6] = xv_47.x * rk_2 * wreg[14];
            out_11[7] = xv_47.y * rk_2 * wreg[15];
            unsigned int packed_11[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_11[_lp*2 + 0], out_11[_lp*2+1 + 0]));
                packed_11[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_11[_lp*2 + 2], out_11[_lp*2+1 + 2]));
                packed_11[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_11[_lp*2 + 4], out_11[_lp*2+1 + 4]));
                packed_11[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_11[_lp*2 + 6], out_11[_lp*2+1 + 6]));
                packed_11[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst) * 1280ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_11)[0];
            }
            float out_12[8];
            float2 xv_48 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[8]));
            out_12[0] = xv_48.x * rk_2 * wreg[16];
            out_12[1] = xv_48.y * rk_2 * wreg[17];
            float2 xv_49 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[9]));
            out_12[2] = xv_49.x * rk_2 * wreg[18];
            out_12[3] = xv_49.y * rk_2 * wreg[19];
            float2 xv_50 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[10]));
            out_12[4] = xv_50.x * rk_2 * wreg[20];
            out_12[5] = xv_50.y * rk_2 * wreg[21];
            float2 xv_51 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[11]));
            out_12[6] = xv_51.x * rk_2 * wreg[22];
            out_12[7] = xv_51.y * rk_2 * wreg[23];
            unsigned int packed_12[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_12[_lp*2 + 0], out_12[_lp*2+1 + 0]));
                packed_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_12[_lp*2 + 2], out_12[_lp*2+1 + 2]));
                packed_12[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_12[_lp*2 + 4], out_12[_lp*2+1 + 4]));
                packed_12[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_12[_lp*2 + 6], out_12[_lp*2+1 + 6]));
                packed_12[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst) * 1280ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_12)[0];
            }
            float out_13[8];
            float2 xv_52 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[12]));
            out_13[0] = xv_52.x * rk_2 * wreg[24];
            out_13[1] = xv_52.y * rk_2 * wreg[25];
            float2 xv_53 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[13]));
            out_13[2] = xv_53.x * rk_2 * wreg[26];
            out_13[3] = xv_53.y * rk_2 * wreg[27];
            float2 xv_54 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[14]));
            out_13[4] = xv_54.x * rk_2 * wreg[28];
            out_13[5] = xv_54.y * rk_2 * wreg[29];
            float2 xv_55 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[15]));
            out_13[6] = xv_55.x * rk_2 * wreg[30];
            out_13[7] = xv_55.y * rk_2 * wreg[31];
            unsigned int packed_13[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_13[_lp*2 + 0], out_13[_lp*2+1 + 0]));
                packed_13[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_13[_lp*2 + 2], out_13[_lp*2+1 + 2]));
                packed_13[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_13[_lp*2 + 4], out_13[_lp*2+1 + 4]));
                packed_13[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_13[_lp*2 + 6], out_13[_lp*2+1 + 6]));
                packed_13[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst) * 1280ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_13)[0];
            }
            float out_14[8];
            float2 xv_56 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[16]));
            out_14[0] = xv_56.x * rk_2 * wreg[32];
            out_14[1] = xv_56.y * rk_2 * wreg[33];
            float2 xv_57 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[17]));
            out_14[2] = xv_57.x * rk_2 * wreg[34];
            out_14[3] = xv_57.y * rk_2 * wreg[35];
            float2 xv_58 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[18]));
            out_14[4] = xv_58.x * rk_2 * wreg[36];
            out_14[5] = xv_58.y * rk_2 * wreg[37];
            float2 xv_59 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[19]));
            out_14[6] = xv_59.x * rk_2 * wreg[38];
            out_14[7] = xv_59.y * rk_2 * wreg[39];
            unsigned int packed_14[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_14[_lp*2 + 0], out_14[_lp*2+1 + 0]));
                packed_14[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_14[_lp*2 + 2], out_14[_lp*2+1 + 2]));
                packed_14[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_14[_lp*2 + 4], out_14[_lp*2+1 + 4]));
                packed_14[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_14[_lp*2 + 6], out_14[_lp*2+1 + 6]));
                packed_14[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst) * 1280ll + (long long)(col + 1024)))[0] = reinterpret_cast<int4*>(packed_14)[0];
            }
            float rk_3 = __shfl_sync(0xFFFFFFFF, rstd_0, 1);
            float out_15[8];
            float2 xv_60 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[20]));
            out_15[0] = xv_60.x * rk_3 * wreg[0];
            out_15[1] = xv_60.y * rk_3 * wreg[1];
            float2 xv_61 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[21]));
            out_15[2] = xv_61.x * rk_3 * wreg[2];
            out_15[3] = xv_61.y * rk_3 * wreg[3];
            float2 xv_62 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[22]));
            out_15[4] = xv_62.x * rk_3 * wreg[4];
            out_15[5] = xv_62.y * rk_3 * wreg[5];
            float2 xv_63 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[23]));
            out_15[6] = xv_63.x * rk_3 * wreg[6];
            out_15[7] = xv_63.y * rk_3 * wreg[7];
            unsigned int packed_15[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_15[_lp*2 + 0], out_15[_lp*2+1 + 0]));
                packed_15[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_15[_lp*2 + 2], out_15[_lp*2+1 + 2]));
                packed_15[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_15[_lp*2 + 4], out_15[_lp*2+1 + 4]));
                packed_15[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_15[_lp*2 + 6], out_15[_lp*2+1 + 6]));
                packed_15[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst + 1 < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst + 1) * 1280ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_15)[0];
            }
            float out_16[8];
            float2 xv_64 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[24]));
            out_16[0] = xv_64.x * rk_3 * wreg[8];
            out_16[1] = xv_64.y * rk_3 * wreg[9];
            float2 xv_65 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[25]));
            out_16[2] = xv_65.x * rk_3 * wreg[10];
            out_16[3] = xv_65.y * rk_3 * wreg[11];
            float2 xv_66 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[26]));
            out_16[4] = xv_66.x * rk_3 * wreg[12];
            out_16[5] = xv_66.y * rk_3 * wreg[13];
            float2 xv_67 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[27]));
            out_16[6] = xv_67.x * rk_3 * wreg[14];
            out_16[7] = xv_67.y * rk_3 * wreg[15];
            unsigned int packed_16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_16[_lp*2 + 0], out_16[_lp*2+1 + 0]));
                packed_16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_16[_lp*2 + 2], out_16[_lp*2+1 + 2]));
                packed_16[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_16[_lp*2 + 4], out_16[_lp*2+1 + 4]));
                packed_16[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_16[_lp*2 + 6], out_16[_lp*2+1 + 6]));
                packed_16[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst + 1 < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst + 1) * 1280ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_16)[0];
            }
            float out_17[8];
            float2 xv_68 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[28]));
            out_17[0] = xv_68.x * rk_3 * wreg[16];
            out_17[1] = xv_68.y * rk_3 * wreg[17];
            float2 xv_69 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[29]));
            out_17[2] = xv_69.x * rk_3 * wreg[18];
            out_17[3] = xv_69.y * rk_3 * wreg[19];
            float2 xv_70 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[30]));
            out_17[4] = xv_70.x * rk_3 * wreg[20];
            out_17[5] = xv_70.y * rk_3 * wreg[21];
            float2 xv_71 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[31]));
            out_17[6] = xv_71.x * rk_3 * wreg[22];
            out_17[7] = xv_71.y * rk_3 * wreg[23];
            unsigned int packed_17[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_17[_lp*2 + 0], out_17[_lp*2+1 + 0]));
                packed_17[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_17[_lp*2 + 2], out_17[_lp*2+1 + 2]));
                packed_17[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_17[_lp*2 + 4], out_17[_lp*2+1 + 4]));
                packed_17[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_17[_lp*2 + 6], out_17[_lp*2+1 + 6]));
                packed_17[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst + 1 < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst + 1) * 1280ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_17)[0];
            }
            float out_18[8];
            float2 xv_72 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[32]));
            out_18[0] = xv_72.x * rk_3 * wreg[24];
            out_18[1] = xv_72.y * rk_3 * wreg[25];
            float2 xv_73 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[33]));
            out_18[2] = xv_73.x * rk_3 * wreg[26];
            out_18[3] = xv_73.y * rk_3 * wreg[27];
            float2 xv_74 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[34]));
            out_18[4] = xv_74.x * rk_3 * wreg[28];
            out_18[5] = xv_74.y * rk_3 * wreg[29];
            float2 xv_75 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[35]));
            out_18[6] = xv_75.x * rk_3 * wreg[30];
            out_18[7] = xv_75.y * rk_3 * wreg[31];
            unsigned int packed_18[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_18[_lp*2 + 0], out_18[_lp*2+1 + 0]));
                packed_18[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_18[_lp*2 + 2], out_18[_lp*2+1 + 2]));
                packed_18[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_18[_lp*2 + 4], out_18[_lp*2+1 + 4]));
                packed_18[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_18[_lp*2 + 6], out_18[_lp*2+1 + 6]));
                packed_18[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst + 1 < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst + 1) * 1280ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_18)[0];
            }
            float out_19[8];
            float2 xv_76 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[36]));
            out_19[0] = xv_76.x * rk_3 * wreg[32];
            out_19[1] = xv_76.y * rk_3 * wreg[33];
            float2 xv_77 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[37]));
            out_19[2] = xv_77.x * rk_3 * wreg[34];
            out_19[3] = xv_77.y * rk_3 * wreg[35];
            float2 xv_78 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[38]));
            out_19[4] = xv_78.x * rk_3 * wreg[36];
            out_19[5] = xv_78.y * rk_3 * wreg[37];
            float2 xv_79 = cake_bf16x2_to_f32x2(__as_bf16x2(xb[39]));
            out_19[6] = xv_79.x * rk_3 * wreg[38];
            out_19[7] = xv_79.y * rk_3 * wreg[39];
            unsigned int packed_19[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_19[_lp*2 + 0], out_19[_lp*2+1 + 0]));
                packed_19[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_19[_lp*2 + 2], out_19[_lp*2+1 + 2]));
                packed_19[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_19[_lp*2 + 4], out_19[_lp*2+1 + 4]));
                packed_19[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_19[_lp*2 + 6], out_19[_lp*2+1 + 6]));
                packed_19[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (nb * 8 + lfirst + 1 < T) {
                reinterpret_cast<int4*>(y + ((long long)(nb * 8 + lfirst + 1) * 1280ll + (long long)(col + 1024)))[0] = reinterpret_cast<int4*>(packed_19)[0];
            }
            if ((lane & 0) == 0) {
                if (lane < 2) {
                    if (nb * 8 + lfirst + lane < T) {
                        r[nb * 8 + lfirst + lane] = rstd_0;
                    }
                }
            }
        }
    }
}

} // extern "C"
