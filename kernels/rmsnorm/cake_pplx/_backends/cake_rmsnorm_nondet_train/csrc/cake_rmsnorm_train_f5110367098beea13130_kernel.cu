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


__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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
kernel_cake_rmsnorm_train_f5110367098beea13130(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
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
    int nblocks = (T + 15) / 16;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 16 + warp * 4;
        unsigned int xs[64];
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
        int _min_20 = ((row0 + 2) < (T - 1) ? (row0 + 2) : (T - 1));
        {
            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(x + ((long long)_min_20 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_21 = reinterpret_cast<uint4*>(&xs[32]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_21[_blk] = _vptr_21[_blk];
            }
        }
        int _min_22 = ((row0 + 2) < (T - 1) ? (row0 + 2) : (T - 1));
        {
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(x + ((long long)_min_22 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_23 = reinterpret_cast<uint4*>(&xs[36]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_23[_blk] = _vptr_23[_blk];
            }
        }
        int _min_24 = ((row0 + 2) < (T - 1) ? (row0 + 2) : (T - 1));
        {
            const uint4* _vptr_25 = reinterpret_cast<const uint4*>(x + ((long long)_min_24 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_25 = reinterpret_cast<uint4*>(&xs[40]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_25[_blk] = _vptr_25[_blk];
            }
        }
        int _min_26 = ((row0 + 2) < (T - 1) ? (row0 + 2) : (T - 1));
        {
            const uint4* _vptr_27 = reinterpret_cast<const uint4*>(x + ((long long)_min_26 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_27 = reinterpret_cast<uint4*>(&xs[44]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_27[_blk] = _vptr_27[_blk];
            }
        }
        int _min_28 = ((row0 + 3) < (T - 1) ? (row0 + 3) : (T - 1));
        {
            const uint4* _vptr_29 = reinterpret_cast<const uint4*>(x + ((long long)_min_28 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_29 = reinterpret_cast<uint4*>(&xs[48]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_29[_blk] = _vptr_29[_blk];
            }
        }
        int _min_30 = ((row0 + 3) < (T - 1) ? (row0 + 3) : (T - 1));
        {
            const uint4* _vptr_31 = reinterpret_cast<const uint4*>(x + ((long long)_min_30 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_31 = reinterpret_cast<uint4*>(&xs[52]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_31[_blk] = _vptr_31[_blk];
            }
        }
        int _min_32 = ((row0 + 3) < (T - 1) ? (row0 + 3) : (T - 1));
        {
            const uint4* _vptr_33 = reinterpret_cast<const uint4*>(x + ((long long)_min_32 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_33 = reinterpret_cast<uint4*>(&xs[56]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_33[_blk] = _vptr_33[_blk];
            }
        }
        int _min_34 = ((row0 + 3) < (T - 1) ? (row0 + 3) : (T - 1));
        {
            const uint4* _vptr_35 = reinterpret_cast<const uint4*>(x + ((long long)_min_34 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_35 = reinterpret_cast<uint4*>(&xs[60]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_35[_blk] = _vptr_35[_blk];
            }
        }
        float tot[4];
        float sum_sq_0 = 0.0f;
        float _dot_36;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_36) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_36;
        float _dot_37;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_37) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_37;
        float _dot_38;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_38) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_38;
        float _dot_39;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_39) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_39;
        float _dot_40;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_40) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_0));
        sum_sq_0 = _dot_40;
        float _dot_41;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_41) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_0));
        sum_sq_0 = _dot_41;
        float _dot_42;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_42) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_0));
        sum_sq_0 = _dot_42;
        float _dot_43;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_43) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_0));
        sum_sq_0 = _dot_43;
        float _dot_44;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_44) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_0));
        sum_sq_0 = _dot_44;
        float _dot_45;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_45) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_0));
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
            : "=f"(_dot_46) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_0));
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
            : "=f"(_dot_47) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_0));
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
            : "=f"(_dot_48) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_0));
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
            : "=f"(_dot_49) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_0));
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
            : "=f"(_dot_50) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_0));
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
            : "=f"(_dot_51) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_0));
        sum_sq_0 = _dot_51;
        tot[0] = sum_sq_0;
        float sum_sq_1 = 0.0f;
        float _dot_52;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_52) : "r"(xs[16]), "r"(xs[16]), "f"(sum_sq_1));
        sum_sq_1 = _dot_52;
        float _dot_53;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_53) : "r"(xs[17]), "r"(xs[17]), "f"(sum_sq_1));
        sum_sq_1 = _dot_53;
        float _dot_54;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_54) : "r"(xs[18]), "r"(xs[18]), "f"(sum_sq_1));
        sum_sq_1 = _dot_54;
        float _dot_55;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_55) : "r"(xs[19]), "r"(xs[19]), "f"(sum_sq_1));
        sum_sq_1 = _dot_55;
        float _dot_56;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_56) : "r"(xs[20]), "r"(xs[20]), "f"(sum_sq_1));
        sum_sq_1 = _dot_56;
        float _dot_57;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_57) : "r"(xs[21]), "r"(xs[21]), "f"(sum_sq_1));
        sum_sq_1 = _dot_57;
        float _dot_58;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_58) : "r"(xs[22]), "r"(xs[22]), "f"(sum_sq_1));
        sum_sq_1 = _dot_58;
        float _dot_59;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_59) : "r"(xs[23]), "r"(xs[23]), "f"(sum_sq_1));
        sum_sq_1 = _dot_59;
        float _dot_60;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_60) : "r"(xs[24]), "r"(xs[24]), "f"(sum_sq_1));
        sum_sq_1 = _dot_60;
        float _dot_61;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_61) : "r"(xs[25]), "r"(xs[25]), "f"(sum_sq_1));
        sum_sq_1 = _dot_61;
        float _dot_62;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_62) : "r"(xs[26]), "r"(xs[26]), "f"(sum_sq_1));
        sum_sq_1 = _dot_62;
        float _dot_63;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_63) : "r"(xs[27]), "r"(xs[27]), "f"(sum_sq_1));
        sum_sq_1 = _dot_63;
        float _dot_64;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_64) : "r"(xs[28]), "r"(xs[28]), "f"(sum_sq_1));
        sum_sq_1 = _dot_64;
        float _dot_65;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_65) : "r"(xs[29]), "r"(xs[29]), "f"(sum_sq_1));
        sum_sq_1 = _dot_65;
        float _dot_66;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_66) : "r"(xs[30]), "r"(xs[30]), "f"(sum_sq_1));
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
            : "=f"(_dot_67) : "r"(xs[31]), "r"(xs[31]), "f"(sum_sq_1));
        sum_sq_1 = _dot_67;
        tot[1] = sum_sq_1;
        float sum_sq_2 = 0.0f;
        float _dot_68;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_68) : "r"(xs[32]), "r"(xs[32]), "f"(sum_sq_2));
        sum_sq_2 = _dot_68;
        float _dot_69;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_69) : "r"(xs[33]), "r"(xs[33]), "f"(sum_sq_2));
        sum_sq_2 = _dot_69;
        float _dot_70;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_70) : "r"(xs[34]), "r"(xs[34]), "f"(sum_sq_2));
        sum_sq_2 = _dot_70;
        float _dot_71;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_71) : "r"(xs[35]), "r"(xs[35]), "f"(sum_sq_2));
        sum_sq_2 = _dot_71;
        float _dot_72;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_72) : "r"(xs[36]), "r"(xs[36]), "f"(sum_sq_2));
        sum_sq_2 = _dot_72;
        float _dot_73;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_73) : "r"(xs[37]), "r"(xs[37]), "f"(sum_sq_2));
        sum_sq_2 = _dot_73;
        float _dot_74;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_74) : "r"(xs[38]), "r"(xs[38]), "f"(sum_sq_2));
        sum_sq_2 = _dot_74;
        float _dot_75;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_75) : "r"(xs[39]), "r"(xs[39]), "f"(sum_sq_2));
        sum_sq_2 = _dot_75;
        float _dot_76;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_76) : "r"(xs[40]), "r"(xs[40]), "f"(sum_sq_2));
        sum_sq_2 = _dot_76;
        float _dot_77;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_77) : "r"(xs[41]), "r"(xs[41]), "f"(sum_sq_2));
        sum_sq_2 = _dot_77;
        float _dot_78;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_78) : "r"(xs[42]), "r"(xs[42]), "f"(sum_sq_2));
        sum_sq_2 = _dot_78;
        float _dot_79;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_79) : "r"(xs[43]), "r"(xs[43]), "f"(sum_sq_2));
        sum_sq_2 = _dot_79;
        float _dot_80;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_80) : "r"(xs[44]), "r"(xs[44]), "f"(sum_sq_2));
        sum_sq_2 = _dot_80;
        float _dot_81;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_81) : "r"(xs[45]), "r"(xs[45]), "f"(sum_sq_2));
        sum_sq_2 = _dot_81;
        float _dot_82;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_82) : "r"(xs[46]), "r"(xs[46]), "f"(sum_sq_2));
        sum_sq_2 = _dot_82;
        float _dot_83;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_83) : "r"(xs[47]), "r"(xs[47]), "f"(sum_sq_2));
        sum_sq_2 = _dot_83;
        tot[2] = sum_sq_2;
        float sum_sq_3 = 0.0f;
        float _dot_84;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_84) : "r"(xs[48]), "r"(xs[48]), "f"(sum_sq_3));
        sum_sq_3 = _dot_84;
        float _dot_85;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_85) : "r"(xs[49]), "r"(xs[49]), "f"(sum_sq_3));
        sum_sq_3 = _dot_85;
        float _dot_86;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_86) : "r"(xs[50]), "r"(xs[50]), "f"(sum_sq_3));
        sum_sq_3 = _dot_86;
        float _dot_87;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_87) : "r"(xs[51]), "r"(xs[51]), "f"(sum_sq_3));
        sum_sq_3 = _dot_87;
        float _dot_88;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_88) : "r"(xs[52]), "r"(xs[52]), "f"(sum_sq_3));
        sum_sq_3 = _dot_88;
        float _dot_89;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_89) : "r"(xs[53]), "r"(xs[53]), "f"(sum_sq_3));
        sum_sq_3 = _dot_89;
        float _dot_90;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_90) : "r"(xs[54]), "r"(xs[54]), "f"(sum_sq_3));
        sum_sq_3 = _dot_90;
        float _dot_91;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_91) : "r"(xs[55]), "r"(xs[55]), "f"(sum_sq_3));
        sum_sq_3 = _dot_91;
        float _dot_92;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_92) : "r"(xs[56]), "r"(xs[56]), "f"(sum_sq_3));
        sum_sq_3 = _dot_92;
        float _dot_93;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_93) : "r"(xs[57]), "r"(xs[57]), "f"(sum_sq_3));
        sum_sq_3 = _dot_93;
        float _dot_94;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_94) : "r"(xs[58]), "r"(xs[58]), "f"(sum_sq_3));
        sum_sq_3 = _dot_94;
        float _dot_95;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_95) : "r"(xs[59]), "r"(xs[59]), "f"(sum_sq_3));
        sum_sq_3 = _dot_95;
        float _dot_96;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_96) : "r"(xs[60]), "r"(xs[60]), "f"(sum_sq_3));
        sum_sq_3 = _dot_96;
        float _dot_97;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_97) : "r"(xs[61]), "r"(xs[61]), "f"(sum_sq_3));
        sum_sq_3 = _dot_97;
        float _dot_98;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_98) : "r"(xs[62]), "r"(xs[62]), "f"(sum_sq_3));
        sum_sq_3 = _dot_98;
        float _dot_99;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_99) : "r"(xs[63]), "r"(xs[63]), "f"(sum_sq_3));
        sum_sq_3 = _dot_99;
        tot[3] = sum_sq_3;
        int h16 = lane >> 4 & 1;
        float l16[2];
        float keep16_0 = ((h16 != 0) ? tot[2] : tot[0]);
        float send16_0 = ((h16 != 0) ? tot[0] : tot[2]);
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, send16_0, 16);
        l16[0] = keep16_0 + _shfl_xor_sync_100;
        float keep16_1 = ((h16 != 0) ? tot[3] : tot[1]);
        float send16_1 = ((h16 != 0) ? tot[1] : tot[3]);
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, send16_1, 16);
        l16[1] = keep16_1 + _shfl_xor_sync_101;
        int h8 = lane >> 3 & 1;
        float l8[1];
        float keep8_0 = ((h8 != 0) ? l16[1] : l16[0]);
        float send8_0 = ((h8 != 0) ? l16[0] : l16[1]);
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, send8_0, 8);
        l8[0] = keep8_0 + _shfl_xor_sync_102;
        float own = l8[0];
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, own, 4);
        own = own + _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, own, 2);
        own = own + _shfl_xor_sync_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, own, 1);
        own = own + _shfl_xor_sync_105;
        float mean = own * 0.0009765625f;
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_106;
        asm("neg.f32 %0, %1;" : "=f"(_neg_106) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_106));
        float _neg_107;
        asm("neg.f32 %0, %1;" : "=f"(_neg_107) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_107), "f"(rs_y), "f"(1.0f));
        float _neg_109;
        asm("neg.f32 %0, %1;" : "=f"(_neg_109) : "f"(rs_tl));
        float _fma_108;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(_neg_109), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_108;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_110;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_110) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_110), "f"(rs_h), "f"(rs_y));
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_111), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_113;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_113) : "f"(rs_a));
            float _rcp_112 = approx_rcp(_sqrt_113);
            rs_y = _rcp_112;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_115;
            asm("neg.f32 %0, %1;" : "=f"(_neg_115) : "f"(rs_t));
            float _fma_114;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(rs_a), "f"(rs_y), "f"(_neg_115));
            rs_tl = _fma_114;
            float _neg_117;
            asm("neg.f32 %0, %1;" : "=f"(_neg_117) : "f"(rs_t));
            float _fma_116;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(_neg_117), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_116;
            float _neg_119;
            asm("neg.f32 %0, %1;" : "=f"(_neg_119) : "f"(rs_tl));
            float _fma_118;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(_neg_119), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_118;
            float _fma_120;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_120);
            unsigned long long rs_m = (unsigned long long)((rs_fb & 8388607) << 1 | 16777217);
            unsigned int rs_an = ((rs_ab < 8388608) ? __as_u32((float)rs_ab) : rs_ab);
            unsigned long long rs_p = rs_m * rs_m * (unsigned long long)(rs_an & 8388607 | 8388608);
            rs_r = ((rs_ab - 1 < 2139095039) ? __uint_as_float(rs_fb + (unsigned int)(rs_p >> 63ull)) : rs_y);
        }
        float rstd = rs_r;
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, 0);
        float out_0[8];
        float2 rk2_0 = make_float2(rk_0, rk_0);
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[0]));
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(xv_0, rk2_0);
        float2 _f32x2_mul_b_122 = make_float2(wreg[0], wreg[1]);
        float2 o2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
        out_0[0] = o2_0.x;
        out_0[1] = o2_0.y;
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(xv_1, rk2_0);
        float2 _f32x2_mul_b_124 = make_float2(wreg[2], wreg[3]);
        float2 o2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_124));
        out_0[2] = o2_1.x;
        out_0[3] = o2_1.y;
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        float2 _f32x2_mul_a_125 = mul_f32x2_noftz(xv_2, rk2_0);
        float2 _f32x2_mul_b_126 = make_float2(wreg[4], wreg[5]);
        float2 o2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_126));
        out_0[4] = o2_2.x;
        out_0[5] = o2_2.y;
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
        float2 _f32x2_mul_a_127 = mul_f32x2_noftz(xv_3, rk2_0);
        float2 _f32x2_mul_b_128 = make_float2(wreg[6], wreg[7]);
        float2 o2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&_f32x2_mul_b_128));
        out_0[6] = o2_3.x;
        out_0[7] = o2_3.y;
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
        float2 rk2_1 = make_float2(rk_0, rk_0);
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(xv_4, rk2_1);
        float2 _f32x2_mul_b_130 = make_float2(wreg[8], wreg[9]);
        float2 o2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&_f32x2_mul_b_130));
        out_1[0] = o2_4.x;
        out_1[1] = o2_4.y;
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        float2 _f32x2_mul_a_131 = mul_f32x2_noftz(xv_5, rk2_1);
        float2 _f32x2_mul_b_132 = make_float2(wreg[10], wreg[11]);
        float2 o2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_131), "l"(*(const unsigned long long*)&_f32x2_mul_b_132));
        out_1[2] = o2_5.x;
        out_1[3] = o2_5.y;
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(xv_6, rk2_1);
        float2 _f32x2_mul_b_134 = make_float2(wreg[12], wreg[13]);
        float2 o2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&_f32x2_mul_b_134));
        out_1[4] = o2_6.x;
        out_1[5] = o2_6.y;
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
        float2 _f32x2_mul_a_135 = mul_f32x2_noftz(xv_7, rk2_1);
        float2 _f32x2_mul_b_136 = make_float2(wreg[14], wreg[15]);
        float2 o2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&_f32x2_mul_b_136));
        out_1[6] = o2_7.x;
        out_1[7] = o2_7.y;
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
        float2 rk2_2 = make_float2(rk_0, rk_0);
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        float2 _f32x2_mul_a_137 = mul_f32x2_noftz(xv_8, rk2_2);
        float2 _f32x2_mul_b_138 = make_float2(wreg[16], wreg[17]);
        float2 o2_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
        out_2[0] = o2_8.x;
        out_2[1] = o2_8.y;
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(xv_9, rk2_2);
        float2 _f32x2_mul_b_140 = make_float2(wreg[18], wreg[19]);
        float2 o2_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
        out_2[2] = o2_9.x;
        out_2[3] = o2_9.y;
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        float2 _f32x2_mul_a_141 = mul_f32x2_noftz(xv_10, rk2_2);
        float2 _f32x2_mul_b_142 = make_float2(wreg[20], wreg[21]);
        float2 o2_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&_f32x2_mul_b_142));
        out_2[4] = o2_10.x;
        out_2[5] = o2_10.y;
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
        float2 _f32x2_mul_a_143 = mul_f32x2_noftz(xv_11, rk2_2);
        float2 _f32x2_mul_b_144 = make_float2(wreg[22], wreg[23]);
        float2 o2_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&_f32x2_mul_b_144));
        out_2[6] = o2_11.x;
        out_2[7] = o2_11.y;
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
        float2 rk2_3 = make_float2(rk_0, rk_0);
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        float2 _f32x2_mul_a_145 = mul_f32x2_noftz(xv_12, rk2_3);
        float2 _f32x2_mul_b_146 = make_float2(wreg[24], wreg[25]);
        float2 o2_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&_f32x2_mul_b_146));
        out_3[0] = o2_12.x;
        out_3[1] = o2_12.y;
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        float2 _f32x2_mul_a_147 = mul_f32x2_noftz(xv_13, rk2_3);
        float2 _f32x2_mul_b_148 = make_float2(wreg[26], wreg[27]);
        float2 o2_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_147), "l"(*(const unsigned long long*)&_f32x2_mul_b_148));
        out_3[2] = o2_13.x;
        out_3[3] = o2_13.y;
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        float2 _f32x2_mul_a_149 = mul_f32x2_noftz(xv_14, rk2_3);
        float2 _f32x2_mul_b_150 = make_float2(wreg[28], wreg[29]);
        float2 o2_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&_f32x2_mul_b_150));
        out_3[4] = o2_14.x;
        out_3[5] = o2_14.y;
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
        float2 _f32x2_mul_a_151 = mul_f32x2_noftz(xv_15, rk2_3);
        float2 _f32x2_mul_b_152 = make_float2(wreg[30], wreg[31]);
        float2 o2_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&_f32x2_mul_b_152));
        out_3[6] = o2_15.x;
        out_3[7] = o2_15.y;
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
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, 8);
        float out_4[8];
        float2 rk2_4 = make_float2(rk_1, rk_1);
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[16]));
        float2 _f32x2_mul_a_153 = mul_f32x2_noftz(xv_16, rk2_4);
        float2 _f32x2_mul_b_154 = make_float2(wreg[0], wreg[1]);
        float2 o2_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
        out_4[0] = o2_16.x;
        out_4[1] = o2_16.y;
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[17]));
        float2 _f32x2_mul_a_155 = mul_f32x2_noftz(xv_17, rk2_4);
        float2 _f32x2_mul_b_156 = make_float2(wreg[2], wreg[3]);
        float2 o2_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&_f32x2_mul_b_156));
        out_4[2] = o2_17.x;
        out_4[3] = o2_17.y;
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[18]));
        float2 _f32x2_mul_a_157 = mul_f32x2_noftz(xv_18, rk2_4);
        float2 _f32x2_mul_b_158 = make_float2(wreg[4], wreg[5]);
        float2 o2_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&_f32x2_mul_b_158));
        out_4[4] = o2_18.x;
        out_4[5] = o2_18.y;
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[19]));
        float2 _f32x2_mul_a_159 = mul_f32x2_noftz(xv_19, rk2_4);
        float2 _f32x2_mul_b_160 = make_float2(wreg[6], wreg[7]);
        float2 o2_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&_f32x2_mul_b_160));
        out_4[6] = o2_19.x;
        out_4[7] = o2_19.y;
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
        float2 rk2_5 = make_float2(rk_1, rk_1);
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[20]));
        float2 _f32x2_mul_a_161 = mul_f32x2_noftz(xv_20, rk2_5);
        float2 _f32x2_mul_b_162 = make_float2(wreg[8], wreg[9]);
        float2 o2_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_161), "l"(*(const unsigned long long*)&_f32x2_mul_b_162));
        out_5[0] = o2_20.x;
        out_5[1] = o2_20.y;
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[21]));
        float2 _f32x2_mul_a_163 = mul_f32x2_noftz(xv_21, rk2_5);
        float2 _f32x2_mul_b_164 = make_float2(wreg[10], wreg[11]);
        float2 o2_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&_f32x2_mul_b_164));
        out_5[2] = o2_21.x;
        out_5[3] = o2_21.y;
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[22]));
        float2 _f32x2_mul_a_165 = mul_f32x2_noftz(xv_22, rk2_5);
        float2 _f32x2_mul_b_166 = make_float2(wreg[12], wreg[13]);
        float2 o2_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&_f32x2_mul_b_166));
        out_5[4] = o2_22.x;
        out_5[5] = o2_22.y;
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[23]));
        float2 _f32x2_mul_a_167 = mul_f32x2_noftz(xv_23, rk2_5);
        float2 _f32x2_mul_b_168 = make_float2(wreg[14], wreg[15]);
        float2 o2_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_167), "l"(*(const unsigned long long*)&_f32x2_mul_b_168));
        out_5[6] = o2_23.x;
        out_5[7] = o2_23.y;
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
        float2 rk2_6 = make_float2(rk_1, rk_1);
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[24]));
        float2 _f32x2_mul_a_169 = mul_f32x2_noftz(xv_24, rk2_6);
        float2 _f32x2_mul_b_170 = make_float2(wreg[16], wreg[17]);
        float2 o2_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&_f32x2_mul_b_170));
        out_6[0] = o2_24.x;
        out_6[1] = o2_24.y;
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[25]));
        float2 _f32x2_mul_a_171 = mul_f32x2_noftz(xv_25, rk2_6);
        float2 _f32x2_mul_b_172 = make_float2(wreg[18], wreg[19]);
        float2 o2_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&_f32x2_mul_b_172));
        out_6[2] = o2_25.x;
        out_6[3] = o2_25.y;
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[26]));
        float2 _f32x2_mul_a_173 = mul_f32x2_noftz(xv_26, rk2_6);
        float2 _f32x2_mul_b_174 = make_float2(wreg[20], wreg[21]);
        float2 o2_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&_f32x2_mul_b_174));
        out_6[4] = o2_26.x;
        out_6[5] = o2_26.y;
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[27]));
        float2 _f32x2_mul_a_175 = mul_f32x2_noftz(xv_27, rk2_6);
        float2 _f32x2_mul_b_176 = make_float2(wreg[22], wreg[23]);
        float2 o2_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&_f32x2_mul_b_176));
        out_6[6] = o2_27.x;
        out_6[7] = o2_27.y;
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
        float2 rk2_7 = make_float2(rk_1, rk_1);
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[28]));
        float2 _f32x2_mul_a_177 = mul_f32x2_noftz(xv_28, rk2_7);
        float2 _f32x2_mul_b_178 = make_float2(wreg[24], wreg[25]);
        float2 o2_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
        out_7[0] = o2_28.x;
        out_7[1] = o2_28.y;
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[29]));
        float2 _f32x2_mul_a_179 = mul_f32x2_noftz(xv_29, rk2_7);
        float2 _f32x2_mul_b_180 = make_float2(wreg[26], wreg[27]);
        float2 o2_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&_f32x2_mul_b_180));
        out_7[2] = o2_29.x;
        out_7[3] = o2_29.y;
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[30]));
        float2 _f32x2_mul_a_181 = mul_f32x2_noftz(xv_30, rk2_7);
        float2 _f32x2_mul_b_182 = make_float2(wreg[28], wreg[29]);
        float2 o2_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
        out_7[4] = o2_30.x;
        out_7[5] = o2_30.y;
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[31]));
        float2 _f32x2_mul_a_183 = mul_f32x2_noftz(xv_31, rk2_7);
        float2 _f32x2_mul_b_184 = make_float2(wreg[30], wreg[31]);
        float2 o2_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&_f32x2_mul_b_184));
        out_7[6] = o2_31.x;
        out_7[7] = o2_31.y;
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
        float rk_2 = __shfl_sync(0xFFFFFFFF, rstd, 16);
        float out_8[8];
        float2 rk2_8 = make_float2(rk_2, rk_2);
        float2 xv_32 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[32]));
        float2 _f32x2_mul_a_185 = mul_f32x2_noftz(xv_32, rk2_8);
        float2 _f32x2_mul_b_186 = make_float2(wreg[0], wreg[1]);
        float2 o2_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&_f32x2_mul_b_186));
        out_8[0] = o2_32.x;
        out_8[1] = o2_32.y;
        float2 xv_33 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[33]));
        float2 _f32x2_mul_a_187 = mul_f32x2_noftz(xv_33, rk2_8);
        float2 _f32x2_mul_b_188 = make_float2(wreg[2], wreg[3]);
        float2 o2_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
        out_8[2] = o2_33.x;
        out_8[3] = o2_33.y;
        float2 xv_34 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[34]));
        float2 _f32x2_mul_a_189 = mul_f32x2_noftz(xv_34, rk2_8);
        float2 _f32x2_mul_b_190 = make_float2(wreg[4], wreg[5]);
        float2 o2_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_189), "l"(*(const unsigned long long*)&_f32x2_mul_b_190));
        out_8[4] = o2_34.x;
        out_8[5] = o2_34.y;
        float2 xv_35 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[35]));
        float2 _f32x2_mul_a_191 = mul_f32x2_noftz(xv_35, rk2_8);
        float2 _f32x2_mul_b_192 = make_float2(wreg[6], wreg[7]);
        float2 o2_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&_f32x2_mul_b_192));
        out_8[6] = o2_35.x;
        out_8[7] = o2_35.y;
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
        if (row0 + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 2) * 1024ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_8)[0];
        }
        float out_9[8];
        float2 rk2_9 = make_float2(rk_2, rk_2);
        float2 xv_36 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[36]));
        float2 _f32x2_mul_a_193 = mul_f32x2_noftz(xv_36, rk2_9);
        float2 _f32x2_mul_b_194 = make_float2(wreg[8], wreg[9]);
        float2 o2_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&_f32x2_mul_b_194));
        out_9[0] = o2_36.x;
        out_9[1] = o2_36.y;
        float2 xv_37 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[37]));
        float2 _f32x2_mul_a_195 = mul_f32x2_noftz(xv_37, rk2_9);
        float2 _f32x2_mul_b_196 = make_float2(wreg[10], wreg[11]);
        float2 o2_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_195), "l"(*(const unsigned long long*)&_f32x2_mul_b_196));
        out_9[2] = o2_37.x;
        out_9[3] = o2_37.y;
        float2 xv_38 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[38]));
        float2 _f32x2_mul_a_197 = mul_f32x2_noftz(xv_38, rk2_9);
        float2 _f32x2_mul_b_198 = make_float2(wreg[12], wreg[13]);
        float2 o2_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&_f32x2_mul_b_198));
        out_9[4] = o2_38.x;
        out_9[5] = o2_38.y;
        float2 xv_39 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[39]));
        float2 _f32x2_mul_a_199 = mul_f32x2_noftz(xv_39, rk2_9);
        float2 _f32x2_mul_b_200 = make_float2(wreg[14], wreg[15]);
        float2 o2_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_199), "l"(*(const unsigned long long*)&_f32x2_mul_b_200));
        out_9[6] = o2_39.x;
        out_9[7] = o2_39.y;
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
        if (row0 + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 2) * 1024ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_9)[0];
        }
        float out_10[8];
        float2 rk2_10 = make_float2(rk_2, rk_2);
        float2 xv_40 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[40]));
        float2 _f32x2_mul_a_201 = mul_f32x2_noftz(xv_40, rk2_10);
        float2 _f32x2_mul_b_202 = make_float2(wreg[16], wreg[17]);
        float2 o2_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_201), "l"(*(const unsigned long long*)&_f32x2_mul_b_202));
        out_10[0] = o2_40.x;
        out_10[1] = o2_40.y;
        float2 xv_41 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[41]));
        float2 _f32x2_mul_a_203 = mul_f32x2_noftz(xv_41, rk2_10);
        float2 _f32x2_mul_b_204 = make_float2(wreg[18], wreg[19]);
        float2 o2_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_203), "l"(*(const unsigned long long*)&_f32x2_mul_b_204));
        out_10[2] = o2_41.x;
        out_10[3] = o2_41.y;
        float2 xv_42 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[42]));
        float2 _f32x2_mul_a_205 = mul_f32x2_noftz(xv_42, rk2_10);
        float2 _f32x2_mul_b_206 = make_float2(wreg[20], wreg[21]);
        float2 o2_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_205), "l"(*(const unsigned long long*)&_f32x2_mul_b_206));
        out_10[4] = o2_42.x;
        out_10[5] = o2_42.y;
        float2 xv_43 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[43]));
        float2 _f32x2_mul_a_207 = mul_f32x2_noftz(xv_43, rk2_10);
        float2 _f32x2_mul_b_208 = make_float2(wreg[22], wreg[23]);
        float2 o2_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_207), "l"(*(const unsigned long long*)&_f32x2_mul_b_208));
        out_10[6] = o2_43.x;
        out_10[7] = o2_43.y;
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
        if (row0 + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 2) * 1024ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_10)[0];
        }
        float out_11[8];
        float2 rk2_11 = make_float2(rk_2, rk_2);
        float2 xv_44 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[44]));
        float2 _f32x2_mul_a_209 = mul_f32x2_noftz(xv_44, rk2_11);
        float2 _f32x2_mul_b_210 = make_float2(wreg[24], wreg[25]);
        float2 o2_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_209), "l"(*(const unsigned long long*)&_f32x2_mul_b_210));
        out_11[0] = o2_44.x;
        out_11[1] = o2_44.y;
        float2 xv_45 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[45]));
        float2 _f32x2_mul_a_211 = mul_f32x2_noftz(xv_45, rk2_11);
        float2 _f32x2_mul_b_212 = make_float2(wreg[26], wreg[27]);
        float2 o2_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_211), "l"(*(const unsigned long long*)&_f32x2_mul_b_212));
        out_11[2] = o2_45.x;
        out_11[3] = o2_45.y;
        float2 xv_46 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[46]));
        float2 _f32x2_mul_a_213 = mul_f32x2_noftz(xv_46, rk2_11);
        float2 _f32x2_mul_b_214 = make_float2(wreg[28], wreg[29]);
        float2 o2_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&_f32x2_mul_b_214));
        out_11[4] = o2_46.x;
        out_11[5] = o2_46.y;
        float2 xv_47 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[47]));
        float2 _f32x2_mul_a_215 = mul_f32x2_noftz(xv_47, rk2_11);
        float2 _f32x2_mul_b_216 = make_float2(wreg[30], wreg[31]);
        float2 o2_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_215), "l"(*(const unsigned long long*)&_f32x2_mul_b_216));
        out_11[6] = o2_47.x;
        out_11[7] = o2_47.y;
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
        if (row0 + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 2) * 1024ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_11)[0];
        }
        float rk_3 = __shfl_sync(0xFFFFFFFF, rstd, 24);
        float out_12[8];
        float2 rk2_12 = make_float2(rk_3, rk_3);
        float2 xv_48 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[48]));
        float2 _f32x2_mul_a_217 = mul_f32x2_noftz(xv_48, rk2_12);
        float2 _f32x2_mul_b_218 = make_float2(wreg[0], wreg[1]);
        float2 o2_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_217), "l"(*(const unsigned long long*)&_f32x2_mul_b_218));
        out_12[0] = o2_48.x;
        out_12[1] = o2_48.y;
        float2 xv_49 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[49]));
        float2 _f32x2_mul_a_219 = mul_f32x2_noftz(xv_49, rk2_12);
        float2 _f32x2_mul_b_220 = make_float2(wreg[2], wreg[3]);
        float2 o2_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_219), "l"(*(const unsigned long long*)&_f32x2_mul_b_220));
        out_12[2] = o2_49.x;
        out_12[3] = o2_49.y;
        float2 xv_50 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[50]));
        float2 _f32x2_mul_a_221 = mul_f32x2_noftz(xv_50, rk2_12);
        float2 _f32x2_mul_b_222 = make_float2(wreg[4], wreg[5]);
        float2 o2_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_221), "l"(*(const unsigned long long*)&_f32x2_mul_b_222));
        out_12[4] = o2_50.x;
        out_12[5] = o2_50.y;
        float2 xv_51 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[51]));
        float2 _f32x2_mul_a_223 = mul_f32x2_noftz(xv_51, rk2_12);
        float2 _f32x2_mul_b_224 = make_float2(wreg[6], wreg[7]);
        float2 o2_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&_f32x2_mul_b_224));
        out_12[6] = o2_51.x;
        out_12[7] = o2_51.y;
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
        if (row0 + 3 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 3) * 1024ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_12)[0];
        }
        float out_13[8];
        float2 rk2_13 = make_float2(rk_3, rk_3);
        float2 xv_52 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[52]));
        float2 _f32x2_mul_a_225 = mul_f32x2_noftz(xv_52, rk2_13);
        float2 _f32x2_mul_b_226 = make_float2(wreg[8], wreg[9]);
        float2 o2_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_225), "l"(*(const unsigned long long*)&_f32x2_mul_b_226));
        out_13[0] = o2_52.x;
        out_13[1] = o2_52.y;
        float2 xv_53 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[53]));
        float2 _f32x2_mul_a_227 = mul_f32x2_noftz(xv_53, rk2_13);
        float2 _f32x2_mul_b_228 = make_float2(wreg[10], wreg[11]);
        float2 o2_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_227), "l"(*(const unsigned long long*)&_f32x2_mul_b_228));
        out_13[2] = o2_53.x;
        out_13[3] = o2_53.y;
        float2 xv_54 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[54]));
        float2 _f32x2_mul_a_229 = mul_f32x2_noftz(xv_54, rk2_13);
        float2 _f32x2_mul_b_230 = make_float2(wreg[12], wreg[13]);
        float2 o2_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
        out_13[4] = o2_54.x;
        out_13[5] = o2_54.y;
        float2 xv_55 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[55]));
        float2 _f32x2_mul_a_231 = mul_f32x2_noftz(xv_55, rk2_13);
        float2 _f32x2_mul_b_232 = make_float2(wreg[14], wreg[15]);
        float2 o2_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&_f32x2_mul_b_232));
        out_13[6] = o2_55.x;
        out_13[7] = o2_55.y;
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
        if (row0 + 3 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 3) * 1024ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_13)[0];
        }
        float out_14[8];
        float2 rk2_14 = make_float2(rk_3, rk_3);
        float2 xv_56 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[56]));
        float2 _f32x2_mul_a_233 = mul_f32x2_noftz(xv_56, rk2_14);
        float2 _f32x2_mul_b_234 = make_float2(wreg[16], wreg[17]);
        float2 o2_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_233), "l"(*(const unsigned long long*)&_f32x2_mul_b_234));
        out_14[0] = o2_56.x;
        out_14[1] = o2_56.y;
        float2 xv_57 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[57]));
        float2 _f32x2_mul_a_235 = mul_f32x2_noftz(xv_57, rk2_14);
        float2 _f32x2_mul_b_236 = make_float2(wreg[18], wreg[19]);
        float2 o2_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_235), "l"(*(const unsigned long long*)&_f32x2_mul_b_236));
        out_14[2] = o2_57.x;
        out_14[3] = o2_57.y;
        float2 xv_58 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[58]));
        float2 _f32x2_mul_a_237 = mul_f32x2_noftz(xv_58, rk2_14);
        float2 _f32x2_mul_b_238 = make_float2(wreg[20], wreg[21]);
        float2 o2_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_237), "l"(*(const unsigned long long*)&_f32x2_mul_b_238));
        out_14[4] = o2_58.x;
        out_14[5] = o2_58.y;
        float2 xv_59 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[59]));
        float2 _f32x2_mul_a_239 = mul_f32x2_noftz(xv_59, rk2_14);
        float2 _f32x2_mul_b_240 = make_float2(wreg[22], wreg[23]);
        float2 o2_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_239), "l"(*(const unsigned long long*)&_f32x2_mul_b_240));
        out_14[6] = o2_59.x;
        out_14[7] = o2_59.y;
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
        if (row0 + 3 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 3) * 1024ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_14)[0];
        }
        float out_15[8];
        float2 rk2_15 = make_float2(rk_3, rk_3);
        float2 xv_60 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[60]));
        float2 _f32x2_mul_a_241 = mul_f32x2_noftz(xv_60, rk2_15);
        float2 _f32x2_mul_b_242 = make_float2(wreg[24], wreg[25]);
        float2 o2_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_241), "l"(*(const unsigned long long*)&_f32x2_mul_b_242));
        out_15[0] = o2_60.x;
        out_15[1] = o2_60.y;
        float2 xv_61 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[61]));
        float2 _f32x2_mul_a_243 = mul_f32x2_noftz(xv_61, rk2_15);
        float2 _f32x2_mul_b_244 = make_float2(wreg[26], wreg[27]);
        float2 o2_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_243), "l"(*(const unsigned long long*)&_f32x2_mul_b_244));
        out_15[2] = o2_61.x;
        out_15[3] = o2_61.y;
        float2 xv_62 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[62]));
        float2 _f32x2_mul_a_245 = mul_f32x2_noftz(xv_62, rk2_15);
        float2 _f32x2_mul_b_246 = make_float2(wreg[28], wreg[29]);
        float2 o2_62;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_245), "l"(*(const unsigned long long*)&_f32x2_mul_b_246));
        out_15[4] = o2_62.x;
        out_15[5] = o2_62.y;
        float2 xv_63 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[63]));
        float2 _f32x2_mul_a_247 = mul_f32x2_noftz(xv_63, rk2_15);
        float2 _f32x2_mul_b_248 = make_float2(wreg[30], wreg[31]);
        float2 o2_63;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_247), "l"(*(const unsigned long long*)&_f32x2_mul_b_248));
        out_15[6] = o2_63.x;
        out_15[7] = o2_63.y;
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
        if (row0 + 3 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 3) * 1024ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_15)[0];
        }
        if ((lane & 7) == 0) {
            if (lane >> 3 < 4) {
                if (row0 + (lane >> 3) < T) {
                    r[row0 + (lane >> 3)] = rstd;
                }
            }
        }
    }
}

} // extern "C"
