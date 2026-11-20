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
kernel_cake_rmsnorm_det_train_d99009bfce4043a5dffe(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = lane * 8;
    float wreg[32];
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
    int nblocks = (T + 7) / 8;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 8 + warp * 2;
        unsigned int xs[32];
        int _min_4 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + ((long long)_min_4 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_5 = reinterpret_cast<uint4*>(&xs[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_5[_blk] = _vptr_5[_blk];
            }
        }
        int _min_6 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + ((long long)_min_6 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_7 = reinterpret_cast<uint4*>(&xs[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_7[_blk] = _vptr_7[_blk];
            }
        }
        int _min_8 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + ((long long)_min_8 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_9 = reinterpret_cast<uint4*>(&xs[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_9[_blk] = _vptr_9[_blk];
            }
        }
        int _min_10 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(x + ((long long)_min_10 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_11 = reinterpret_cast<uint4*>(&xs[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_11[_blk] = _vptr_11[_blk];
            }
        }
        int _min_12 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + ((long long)_min_12 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_13 = reinterpret_cast<uint4*>(&xs[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_13[_blk] = _vptr_13[_blk];
            }
        }
        int _min_14 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + ((long long)_min_14 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_15 = reinterpret_cast<uint4*>(&xs[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_15[_blk] = _vptr_15[_blk];
            }
        }
        int _min_16 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + ((long long)_min_16 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_17 = reinterpret_cast<uint4*>(&xs[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_17[_blk] = _vptr_17[_blk];
            }
        }
        int _min_18 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + ((long long)_min_18 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_19 = reinterpret_cast<uint4*>(&xs[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_19[_blk] = _vptr_19[_blk];
            }
        }
        float tot[2];
        float sum_sq_0 = 0.0f;
        float _dot_20;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_20) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_20;
        float _dot_21;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_21) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_21;
        float _dot_22;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_22) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_22;
        float _dot_23;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_23) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_23;
        float _dot_24;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_24) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_0));
        sum_sq_0 = _dot_24;
        float _dot_25;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_25) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_0));
        sum_sq_0 = _dot_25;
        float _dot_26;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_26) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_0));
        sum_sq_0 = _dot_26;
        float _dot_27;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_27) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_0));
        sum_sq_0 = _dot_27;
        float _dot_28;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_28) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_0));
        sum_sq_0 = _dot_28;
        float _dot_29;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_29) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_0));
        sum_sq_0 = _dot_29;
        float _dot_30;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_30) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_0));
        sum_sq_0 = _dot_30;
        float _dot_31;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_31) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_0));
        sum_sq_0 = _dot_31;
        float _dot_32;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_32) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_0));
        sum_sq_0 = _dot_32;
        float _dot_33;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_33) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_0));
        sum_sq_0 = _dot_33;
        float _dot_34;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_34) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_0));
        sum_sq_0 = _dot_34;
        float _dot_35;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_35) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_0));
        sum_sq_0 = _dot_35;
        float _warp_reduce_36 = sum_sq_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_36 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_36, offset);
        tot[0] = _warp_reduce_36;
        float sum_sq_1 = 0.0f;
        float _dot_37;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_37) : "r"(xs[16]), "r"(xs[16]), "f"(sum_sq_1));
        sum_sq_1 = _dot_37;
        float _dot_38;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_38) : "r"(xs[17]), "r"(xs[17]), "f"(sum_sq_1));
        sum_sq_1 = _dot_38;
        float _dot_39;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_39) : "r"(xs[18]), "r"(xs[18]), "f"(sum_sq_1));
        sum_sq_1 = _dot_39;
        float _dot_40;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_40) : "r"(xs[19]), "r"(xs[19]), "f"(sum_sq_1));
        sum_sq_1 = _dot_40;
        float _dot_41;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_41) : "r"(xs[20]), "r"(xs[20]), "f"(sum_sq_1));
        sum_sq_1 = _dot_41;
        float _dot_42;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_42) : "r"(xs[21]), "r"(xs[21]), "f"(sum_sq_1));
        sum_sq_1 = _dot_42;
        float _dot_43;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_43) : "r"(xs[22]), "r"(xs[22]), "f"(sum_sq_1));
        sum_sq_1 = _dot_43;
        float _dot_44;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_44) : "r"(xs[23]), "r"(xs[23]), "f"(sum_sq_1));
        sum_sq_1 = _dot_44;
        float _dot_45;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_45) : "r"(xs[24]), "r"(xs[24]), "f"(sum_sq_1));
        sum_sq_1 = _dot_45;
        float _dot_46;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_46) : "r"(xs[25]), "r"(xs[25]), "f"(sum_sq_1));
        sum_sq_1 = _dot_46;
        float _dot_47;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_47) : "r"(xs[26]), "r"(xs[26]), "f"(sum_sq_1));
        sum_sq_1 = _dot_47;
        float _dot_48;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_48) : "r"(xs[27]), "r"(xs[27]), "f"(sum_sq_1));
        sum_sq_1 = _dot_48;
        float _dot_49;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_49) : "r"(xs[28]), "r"(xs[28]), "f"(sum_sq_1));
        sum_sq_1 = _dot_49;
        float _dot_50;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_50) : "r"(xs[29]), "r"(xs[29]), "f"(sum_sq_1));
        sum_sq_1 = _dot_50;
        float _dot_51;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_51) : "r"(xs[30]), "r"(xs[30]), "f"(sum_sq_1));
        sum_sq_1 = _dot_51;
        float _dot_52;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_52) : "r"(xs[31]), "r"(xs[31]), "f"(sum_sq_1));
        sum_sq_1 = _dot_52;
        float _warp_reduce_53 = sum_sq_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_53 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_53, offset);
        tot[1] = _warp_reduce_53;
        float own = tot[0];
        if (lane == 1) {
            own = tot[1];
        }
        float mean = own * 0.0009765625f;
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_54;
        asm("neg.f32 %0, %1;" : "=f"(_neg_54) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_54));
        float _neg_55;
        asm("neg.f32 %0, %1;" : "=f"(_neg_55) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_55), "f"(rs_y), "f"(1.0f));
        float _neg_57;
        asm("neg.f32 %0, %1;" : "=f"(_neg_57) : "f"(rs_tl));
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(_neg_57), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_56;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_58;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_58), "f"(rs_h), "f"(rs_y));
        float _add_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_59), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_61;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_61) : "f"(rs_a));
            float _rcp_60 = approx_rcp(_sqrt_61);
            rs_y = _rcp_60;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_63;
            asm("neg.f32 %0, %1;" : "=f"(_neg_63) : "f"(rs_t));
            float _fma_62;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(rs_a), "f"(rs_y), "f"(_neg_63));
            rs_tl = _fma_62;
            float _neg_65;
            asm("neg.f32 %0, %1;" : "=f"(_neg_65) : "f"(rs_t));
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(_neg_65), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_64;
            float _neg_67;
            asm("neg.f32 %0, %1;" : "=f"(_neg_67) : "f"(rs_tl));
            float _fma_66;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(_neg_67), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_66;
            float _fma_68;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_68);
            unsigned long long rs_m = (unsigned long long)((rs_fb & 8388607) << 1 | 16777217);
            unsigned int rs_an = ((rs_ab < 8388608) ? __as_u32((float)rs_ab) : rs_ab);
            unsigned long long rs_p = rs_m * rs_m * (unsigned long long)(rs_an & 8388607 | 8388608);
            rs_r = ((rs_ab - 1 < 2139095039) ? __uint_as_float(rs_fb + (unsigned int)(rs_p >> 63ull)) : rs_y);
        }
        float rstd = rs_r;
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, 0);
        float out_0[8];
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[0]));
        out_0[0] = xv_0.x * rk_0 * wreg[0];
        out_0[1] = xv_0.y * rk_0 * wreg[1];
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        out_0[2] = xv_1.x * rk_0 * wreg[2];
        out_0[3] = xv_1.y * rk_0 * wreg[3];
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        out_0[4] = xv_2.x * rk_0 * wreg[4];
        out_0[5] = xv_2.y * rk_0 * wreg[5];
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
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
        if (row0 < T) {
            reinterpret_cast<int4*>(y + ((long long)row0 * 1024ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        float out_1[8];
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        out_1[0] = xv_4.x * rk_0 * wreg[8];
        out_1[1] = xv_4.y * rk_0 * wreg[9];
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        out_1[2] = xv_5.x * rk_0 * wreg[10];
        out_1[3] = xv_5.y * rk_0 * wreg[11];
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        out_1[4] = xv_6.x * rk_0 * wreg[12];
        out_1[5] = xv_6.y * rk_0 * wreg[13];
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
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
        if (row0 < T) {
            reinterpret_cast<int4*>(y + ((long long)row0 * 1024ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        float out_2[8];
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        out_2[0] = xv_8.x * rk_0 * wreg[16];
        out_2[1] = xv_8.y * rk_0 * wreg[17];
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        out_2[2] = xv_9.x * rk_0 * wreg[18];
        out_2[3] = xv_9.y * rk_0 * wreg[19];
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        out_2[4] = xv_10.x * rk_0 * wreg[20];
        out_2[5] = xv_10.y * rk_0 * wreg[21];
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
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
        if (row0 < T) {
            reinterpret_cast<int4*>(y + ((long long)row0 * 1024ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_2)[0];
        }
        float out_3[8];
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        out_3[0] = xv_12.x * rk_0 * wreg[24];
        out_3[1] = xv_12.y * rk_0 * wreg[25];
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        out_3[2] = xv_13.x * rk_0 * wreg[26];
        out_3[3] = xv_13.y * rk_0 * wreg[27];
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        out_3[4] = xv_14.x * rk_0 * wreg[28];
        out_3[5] = xv_14.y * rk_0 * wreg[29];
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
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
        if (row0 < T) {
            reinterpret_cast<int4*>(y + ((long long)row0 * 1024ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, 1);
        float out_4[8];
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[16]));
        out_4[0] = xv_16.x * rk_1 * wreg[0];
        out_4[1] = xv_16.y * rk_1 * wreg[1];
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[17]));
        out_4[2] = xv_17.x * rk_1 * wreg[2];
        out_4[3] = xv_17.y * rk_1 * wreg[3];
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[18]));
        out_4[4] = xv_18.x * rk_1 * wreg[4];
        out_4[5] = xv_18.y * rk_1 * wreg[5];
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[19]));
        out_4[6] = xv_19.x * rk_1 * wreg[6];
        out_4[7] = xv_19.y * rk_1 * wreg[7];
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
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1024ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_4)[0];
        }
        float out_5[8];
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[20]));
        out_5[0] = xv_20.x * rk_1 * wreg[8];
        out_5[1] = xv_20.y * rk_1 * wreg[9];
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[21]));
        out_5[2] = xv_21.x * rk_1 * wreg[10];
        out_5[3] = xv_21.y * rk_1 * wreg[11];
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[22]));
        out_5[4] = xv_22.x * rk_1 * wreg[12];
        out_5[5] = xv_22.y * rk_1 * wreg[13];
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[23]));
        out_5[6] = xv_23.x * rk_1 * wreg[14];
        out_5[7] = xv_23.y * rk_1 * wreg[15];
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
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1024ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_5)[0];
        }
        float out_6[8];
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[24]));
        out_6[0] = xv_24.x * rk_1 * wreg[16];
        out_6[1] = xv_24.y * rk_1 * wreg[17];
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[25]));
        out_6[2] = xv_25.x * rk_1 * wreg[18];
        out_6[3] = xv_25.y * rk_1 * wreg[19];
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[26]));
        out_6[4] = xv_26.x * rk_1 * wreg[20];
        out_6[5] = xv_26.y * rk_1 * wreg[21];
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[27]));
        out_6[6] = xv_27.x * rk_1 * wreg[22];
        out_6[7] = xv_27.y * rk_1 * wreg[23];
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
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1024ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_6)[0];
        }
        float out_7[8];
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[28]));
        out_7[0] = xv_28.x * rk_1 * wreg[24];
        out_7[1] = xv_28.y * rk_1 * wreg[25];
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[29]));
        out_7[2] = xv_29.x * rk_1 * wreg[26];
        out_7[3] = xv_29.y * rk_1 * wreg[27];
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[30]));
        out_7[4] = xv_30.x * rk_1 * wreg[28];
        out_7[5] = xv_30.y * rk_1 * wreg[29];
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[31]));
        out_7[6] = xv_31.x * rk_1 * wreg[30];
        out_7[7] = xv_31.y * rk_1 * wreg[31];
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
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1024ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_7)[0];
        }
        if ((lane & 0) == 0) {
            if (lane < 2) {
                if (row0 + lane < T) {
                    r[row0 + lane] = rstd;
                }
            }
        }
    }
}

} // extern "C"
