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
kernel_cake_rmsnorm_det_train_39b9b41b16769252153d(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = lane * 8;
    unsigned int wreg[20];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4* _vdst_0 = reinterpret_cast<uint4*>(&wreg[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
        }
    }
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (col + 256) + 0);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&wreg[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)) : "memory");
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (col + 512) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&wreg[8]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)) : "memory");
        }
    }
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(w + (col + 768) + 0);
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&wreg[12]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)) : "memory");
        }
    }
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(w + (col + 1024) + 0);
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&wreg[16]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)) : "memory");
        }
    }
    int nblocks = (T + 7) / 8;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 8 + warp * 2;
        unsigned int xs[40];
        int _min_5 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + ((long long)_min_5 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&xs[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
        int _min_7 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + ((long long)_min_7 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&xs[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
        int _min_9 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + ((long long)_min_9 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&xs[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
        int _min_11 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + ((long long)_min_11 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_12 = reinterpret_cast<uint4*>(&xs[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_12[_blk] = _vptr_12[_blk];
            }
        }
        int _min_13 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + ((long long)_min_13 * x_stride_p0 + (long long)(col + 1024)) + 0);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&xs[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
        int _min_15 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(x + ((long long)_min_15 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_16 = reinterpret_cast<uint4*>(&xs[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_16[_blk] = _vptr_16[_blk];
            }
        }
        int _min_17 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(x + ((long long)_min_17 * x_stride_p0 + (long long)(col + 256)) + 0);
            uint4* _vdst_18 = reinterpret_cast<uint4*>(&xs[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_18[_blk] = _vptr_18[_blk];
            }
        }
        int _min_19 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(x + ((long long)_min_19 * x_stride_p0 + (long long)(col + 512)) + 0);
            uint4* _vdst_20 = reinterpret_cast<uint4*>(&xs[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_20[_blk] = _vptr_20[_blk];
            }
        }
        int _min_21 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(x + ((long long)_min_21 * x_stride_p0 + (long long)(col + 768)) + 0);
            uint4* _vdst_22 = reinterpret_cast<uint4*>(&xs[32]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_22[_blk] = _vptr_22[_blk];
            }
        }
        int _min_23 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(x + ((long long)_min_23 * x_stride_p0 + (long long)(col + 1024)) + 0);
            uint4* _vdst_24 = reinterpret_cast<uint4*>(&xs[36]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_24[_blk] = _vptr_24[_blk];
            }
        }
        float tot[2];
        float sum_sq_0 = 0.0f;
        float _dot_25;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_25) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
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
            : "=f"(_dot_26) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
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
            : "=f"(_dot_27) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
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
            : "=f"(_dot_28) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
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
            : "=f"(_dot_29) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_0));
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
            : "=f"(_dot_30) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_0));
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
            : "=f"(_dot_31) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_0));
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
            : "=f"(_dot_32) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_0));
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
            : "=f"(_dot_33) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_0));
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
            : "=f"(_dot_34) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_0));
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
            : "=f"(_dot_35) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_0));
        sum_sq_0 = _dot_35;
        float _dot_36;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_36) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_0));
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
            : "=f"(_dot_37) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_0));
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
            : "=f"(_dot_38) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_0));
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
            : "=f"(_dot_39) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_0));
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
            : "=f"(_dot_40) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_0));
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
            : "=f"(_dot_41) : "r"(xs[16]), "r"(xs[16]), "f"(sum_sq_0));
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
            : "=f"(_dot_42) : "r"(xs[17]), "r"(xs[17]), "f"(sum_sq_0));
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
            : "=f"(_dot_43) : "r"(xs[18]), "r"(xs[18]), "f"(sum_sq_0));
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
            : "=f"(_dot_44) : "r"(xs[19]), "r"(xs[19]), "f"(sum_sq_0));
        sum_sq_0 = _dot_44;
        tot[0] = sum_sq_0;
        float sum_sq_1 = 0.0f;
        float _dot_45;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_45) : "r"(xs[20]), "r"(xs[20]), "f"(sum_sq_1));
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
            : "=f"(_dot_46) : "r"(xs[21]), "r"(xs[21]), "f"(sum_sq_1));
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
            : "=f"(_dot_47) : "r"(xs[22]), "r"(xs[22]), "f"(sum_sq_1));
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
            : "=f"(_dot_48) : "r"(xs[23]), "r"(xs[23]), "f"(sum_sq_1));
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
            : "=f"(_dot_49) : "r"(xs[24]), "r"(xs[24]), "f"(sum_sq_1));
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
            : "=f"(_dot_50) : "r"(xs[25]), "r"(xs[25]), "f"(sum_sq_1));
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
            : "=f"(_dot_51) : "r"(xs[26]), "r"(xs[26]), "f"(sum_sq_1));
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
            : "=f"(_dot_52) : "r"(xs[27]), "r"(xs[27]), "f"(sum_sq_1));
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
            : "=f"(_dot_53) : "r"(xs[28]), "r"(xs[28]), "f"(sum_sq_1));
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
            : "=f"(_dot_54) : "r"(xs[29]), "r"(xs[29]), "f"(sum_sq_1));
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
            : "=f"(_dot_55) : "r"(xs[30]), "r"(xs[30]), "f"(sum_sq_1));
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
            : "=f"(_dot_56) : "r"(xs[31]), "r"(xs[31]), "f"(sum_sq_1));
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
            : "=f"(_dot_57) : "r"(xs[32]), "r"(xs[32]), "f"(sum_sq_1));
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
            : "=f"(_dot_58) : "r"(xs[33]), "r"(xs[33]), "f"(sum_sq_1));
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
            : "=f"(_dot_59) : "r"(xs[34]), "r"(xs[34]), "f"(sum_sq_1));
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
            : "=f"(_dot_60) : "r"(xs[35]), "r"(xs[35]), "f"(sum_sq_1));
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
            : "=f"(_dot_61) : "r"(xs[36]), "r"(xs[36]), "f"(sum_sq_1));
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
            : "=f"(_dot_62) : "r"(xs[37]), "r"(xs[37]), "f"(sum_sq_1));
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
            : "=f"(_dot_63) : "r"(xs[38]), "r"(xs[38]), "f"(sum_sq_1));
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
            : "=f"(_dot_64) : "r"(xs[39]), "r"(xs[39]), "f"(sum_sq_1));
        sum_sq_1 = _dot_64;
        tot[1] = sum_sq_1;
        int h16 = lane >> 4 & 1;
        float l16[1];
        float keep16_0 = ((h16 != 0) ? tot[1] : tot[0]);
        float send16_0 = ((h16 != 0) ? tot[0] : tot[1]);
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, send16_0, 16);
        l16[0] = keep16_0 + _shfl_xor_sync_65;
        float own = l16[0];
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, own, 8);
        own = own + _shfl_xor_sync_66;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, own, 4);
        own = own + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, own, 2);
        own = own + _shfl_xor_sync_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, own, 1);
        own = own + _shfl_xor_sync_69;
        float mean = __fdiv_rn(own, 1280.0f);
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_70;
        asm("neg.f32 %0, %1;" : "=f"(_neg_70) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_70));
        float _neg_71;
        asm("neg.f32 %0, %1;" : "=f"(_neg_71) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_71), "f"(rs_y), "f"(1.0f));
        float _neg_73;
        asm("neg.f32 %0, %1;" : "=f"(_neg_73) : "f"(rs_tl));
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(_neg_73), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_72;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_74;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_74), "f"(rs_h), "f"(rs_y));
        float _add_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_75), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_77;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_77) : "f"(rs_a));
            float _rcp_76 = approx_rcp(_sqrt_77);
            rs_y = _rcp_76;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_79;
            asm("neg.f32 %0, %1;" : "=f"(_neg_79) : "f"(rs_t));
            float _fma_78;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(rs_a), "f"(rs_y), "f"(_neg_79));
            rs_tl = _fma_78;
            float _neg_81;
            asm("neg.f32 %0, %1;" : "=f"(_neg_81) : "f"(rs_t));
            float _fma_80;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(_neg_81), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_80;
            float _neg_83;
            asm("neg.f32 %0, %1;" : "=f"(_neg_83) : "f"(rs_tl));
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(_neg_83), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_82;
            float _fma_84;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_84);
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
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(xv_0, rk2_0);
        float2 _f32x2_mul_b_86 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
        out_0[0] = o2_0.x;
        out_0[1] = o2_0.y;
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(xv_1, rk2_0);
        float2 _f32x2_mul_b_88 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
        out_0[2] = o2_1.x;
        out_0[3] = o2_1.y;
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        float2 _f32x2_mul_a_89 = mul_f32x2_noftz(xv_2, rk2_0);
        float2 _f32x2_mul_b_90 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
        out_0[4] = o2_2.x;
        out_0[5] = o2_2.y;
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
        float2 _f32x2_mul_a_91 = mul_f32x2_noftz(xv_3, rk2_0);
        float2 _f32x2_mul_b_92 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
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
            reinterpret_cast<int4*>(y + ((long long)row0 * 1280ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        float out_1[8];
        float2 rk2_1 = make_float2(rk_0, rk_0);
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(xv_4, rk2_1);
        float2 _f32x2_mul_b_94 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[4]));
        float2 o2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
        out_1[0] = o2_4.x;
        out_1[1] = o2_4.y;
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        float2 _f32x2_mul_a_95 = mul_f32x2_noftz(xv_5, rk2_1);
        float2 _f32x2_mul_b_96 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[5]));
        float2 o2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_96));
        out_1[2] = o2_5.x;
        out_1[3] = o2_5.y;
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        float2 _f32x2_mul_a_97 = mul_f32x2_noftz(xv_6, rk2_1);
        float2 _f32x2_mul_b_98 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[6]));
        float2 o2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
        out_1[4] = o2_6.x;
        out_1[5] = o2_6.y;
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(xv_7, rk2_1);
        float2 _f32x2_mul_b_100 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[7]));
        float2 o2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_100));
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
            reinterpret_cast<int4*>(y + ((long long)row0 * 1280ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        float out_2[8];
        float2 rk2_2 = make_float2(rk_0, rk_0);
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        float2 _f32x2_mul_a_101 = mul_f32x2_noftz(xv_8, rk2_2);
        float2 _f32x2_mul_b_102 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[8]));
        float2 o2_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
        out_2[0] = o2_8.x;
        out_2[1] = o2_8.y;
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(xv_9, rk2_2);
        float2 _f32x2_mul_b_104 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[9]));
        float2 o2_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_104));
        out_2[2] = o2_9.x;
        out_2[3] = o2_9.y;
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(xv_10, rk2_2);
        float2 _f32x2_mul_b_106 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[10]));
        float2 o2_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
        out_2[4] = o2_10.x;
        out_2[5] = o2_10.y;
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
        float2 _f32x2_mul_a_107 = mul_f32x2_noftz(xv_11, rk2_2);
        float2 _f32x2_mul_b_108 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[11]));
        float2 o2_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_108));
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
            reinterpret_cast<int4*>(y + ((long long)row0 * 1280ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_2)[0];
        }
        float out_3[8];
        float2 rk2_3 = make_float2(rk_0, rk_0);
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(xv_12, rk2_3);
        float2 _f32x2_mul_b_110 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[12]));
        float2 o2_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
        out_3[0] = o2_12.x;
        out_3[1] = o2_12.y;
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(xv_13, rk2_3);
        float2 _f32x2_mul_b_112 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[13]));
        float2 o2_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_112));
        out_3[2] = o2_13.x;
        out_3[3] = o2_13.y;
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(xv_14, rk2_3);
        float2 _f32x2_mul_b_114 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[14]));
        float2 o2_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
        out_3[4] = o2_14.x;
        out_3[5] = o2_14.y;
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
        float2 _f32x2_mul_a_115 = mul_f32x2_noftz(xv_15, rk2_3);
        float2 _f32x2_mul_b_116 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[15]));
        float2 o2_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
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
            reinterpret_cast<int4*>(y + ((long long)row0 * 1280ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        float out_4[8];
        float2 rk2_4 = make_float2(rk_0, rk_0);
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[16]));
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(xv_16, rk2_4);
        float2 _f32x2_mul_b_118 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[16]));
        float2 o2_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
        out_4[0] = o2_16.x;
        out_4[1] = o2_16.y;
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[17]));
        float2 _f32x2_mul_a_119 = mul_f32x2_noftz(xv_17, rk2_4);
        float2 _f32x2_mul_b_120 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[17]));
        float2 o2_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_120));
        out_4[2] = o2_17.x;
        out_4[3] = o2_17.y;
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[18]));
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(xv_18, rk2_4);
        float2 _f32x2_mul_b_122 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[18]));
        float2 o2_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
        out_4[4] = o2_18.x;
        out_4[5] = o2_18.y;
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[19]));
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(xv_19, rk2_4);
        float2 _f32x2_mul_b_124 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[19]));
        float2 o2_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_124));
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
        if (row0 < T) {
            reinterpret_cast<int4*>(y + ((long long)row0 * 1280ll + (long long)(col + 1024)))[0] = reinterpret_cast<int4*>(packed_4)[0];
        }
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, 16);
        float out_5[8];
        float2 rk2_5 = make_float2(rk_1, rk_1);
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[20]));
        float2 _f32x2_mul_a_125 = mul_f32x2_noftz(xv_20, rk2_5);
        float2 _f32x2_mul_b_126 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_126));
        out_5[0] = o2_20.x;
        out_5[1] = o2_20.y;
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[21]));
        float2 _f32x2_mul_a_127 = mul_f32x2_noftz(xv_21, rk2_5);
        float2 _f32x2_mul_b_128 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&_f32x2_mul_b_128));
        out_5[2] = o2_21.x;
        out_5[3] = o2_21.y;
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[22]));
        float2 _f32x2_mul_a_129 = mul_f32x2_noftz(xv_22, rk2_5);
        float2 _f32x2_mul_b_130 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&_f32x2_mul_b_130));
        out_5[4] = o2_22.x;
        out_5[5] = o2_22.y;
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[23]));
        float2 _f32x2_mul_a_131 = mul_f32x2_noftz(xv_23, rk2_5);
        float2 _f32x2_mul_b_132 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_131), "l"(*(const unsigned long long*)&_f32x2_mul_b_132));
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
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1280ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_5)[0];
        }
        float out_6[8];
        float2 rk2_6 = make_float2(rk_1, rk_1);
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[24]));
        float2 _f32x2_mul_a_133 = mul_f32x2_noftz(xv_24, rk2_6);
        float2 _f32x2_mul_b_134 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[4]));
        float2 o2_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&_f32x2_mul_b_134));
        out_6[0] = o2_24.x;
        out_6[1] = o2_24.y;
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[25]));
        float2 _f32x2_mul_a_135 = mul_f32x2_noftz(xv_25, rk2_6);
        float2 _f32x2_mul_b_136 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[5]));
        float2 o2_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&_f32x2_mul_b_136));
        out_6[2] = o2_25.x;
        out_6[3] = o2_25.y;
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[26]));
        float2 _f32x2_mul_a_137 = mul_f32x2_noftz(xv_26, rk2_6);
        float2 _f32x2_mul_b_138 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[6]));
        float2 o2_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
        out_6[4] = o2_26.x;
        out_6[5] = o2_26.y;
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[27]));
        float2 _f32x2_mul_a_139 = mul_f32x2_noftz(xv_27, rk2_6);
        float2 _f32x2_mul_b_140 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[7]));
        float2 o2_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
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
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1280ll + (long long)(col + 256)))[0] = reinterpret_cast<int4*>(packed_6)[0];
        }
        float out_7[8];
        float2 rk2_7 = make_float2(rk_1, rk_1);
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[28]));
        float2 _f32x2_mul_a_141 = mul_f32x2_noftz(xv_28, rk2_7);
        float2 _f32x2_mul_b_142 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[8]));
        float2 o2_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&_f32x2_mul_b_142));
        out_7[0] = o2_28.x;
        out_7[1] = o2_28.y;
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[29]));
        float2 _f32x2_mul_a_143 = mul_f32x2_noftz(xv_29, rk2_7);
        float2 _f32x2_mul_b_144 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[9]));
        float2 o2_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&_f32x2_mul_b_144));
        out_7[2] = o2_29.x;
        out_7[3] = o2_29.y;
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[30]));
        float2 _f32x2_mul_a_145 = mul_f32x2_noftz(xv_30, rk2_7);
        float2 _f32x2_mul_b_146 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[10]));
        float2 o2_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&_f32x2_mul_b_146));
        out_7[4] = o2_30.x;
        out_7[5] = o2_30.y;
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[31]));
        float2 _f32x2_mul_a_147 = mul_f32x2_noftz(xv_31, rk2_7);
        float2 _f32x2_mul_b_148 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[11]));
        float2 o2_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_147), "l"(*(const unsigned long long*)&_f32x2_mul_b_148));
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
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1280ll + (long long)(col + 512)))[0] = reinterpret_cast<int4*>(packed_7)[0];
        }
        float out_8[8];
        float2 rk2_8 = make_float2(rk_1, rk_1);
        float2 xv_32 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[32]));
        float2 _f32x2_mul_a_149 = mul_f32x2_noftz(xv_32, rk2_8);
        float2 _f32x2_mul_b_150 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[12]));
        float2 o2_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&_f32x2_mul_b_150));
        out_8[0] = o2_32.x;
        out_8[1] = o2_32.y;
        float2 xv_33 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[33]));
        float2 _f32x2_mul_a_151 = mul_f32x2_noftz(xv_33, rk2_8);
        float2 _f32x2_mul_b_152 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[13]));
        float2 o2_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&_f32x2_mul_b_152));
        out_8[2] = o2_33.x;
        out_8[3] = o2_33.y;
        float2 xv_34 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[34]));
        float2 _f32x2_mul_a_153 = mul_f32x2_noftz(xv_34, rk2_8);
        float2 _f32x2_mul_b_154 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[14]));
        float2 o2_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
        out_8[4] = o2_34.x;
        out_8[5] = o2_34.y;
        float2 xv_35 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[35]));
        float2 _f32x2_mul_a_155 = mul_f32x2_noftz(xv_35, rk2_8);
        float2 _f32x2_mul_b_156 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[15]));
        float2 o2_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&_f32x2_mul_b_156));
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
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1280ll + (long long)(col + 768)))[0] = reinterpret_cast<int4*>(packed_8)[0];
        }
        float out_9[8];
        float2 rk2_9 = make_float2(rk_1, rk_1);
        float2 xv_36 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[36]));
        float2 _f32x2_mul_a_157 = mul_f32x2_noftz(xv_36, rk2_9);
        float2 _f32x2_mul_b_158 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[16]));
        float2 o2_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&_f32x2_mul_b_158));
        out_9[0] = o2_36.x;
        out_9[1] = o2_36.y;
        float2 xv_37 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[37]));
        float2 _f32x2_mul_a_159 = mul_f32x2_noftz(xv_37, rk2_9);
        float2 _f32x2_mul_b_160 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[17]));
        float2 o2_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&_f32x2_mul_b_160));
        out_9[2] = o2_37.x;
        out_9[3] = o2_37.y;
        float2 xv_38 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[38]));
        float2 _f32x2_mul_a_161 = mul_f32x2_noftz(xv_38, rk2_9);
        float2 _f32x2_mul_b_162 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[18]));
        float2 o2_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_161), "l"(*(const unsigned long long*)&_f32x2_mul_b_162));
        out_9[4] = o2_38.x;
        out_9[5] = o2_38.y;
        float2 xv_39 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[39]));
        float2 _f32x2_mul_a_163 = mul_f32x2_noftz(xv_39, rk2_9);
        float2 _f32x2_mul_b_164 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[19]));
        float2 o2_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&_f32x2_mul_b_164));
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
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 1280ll + (long long)(col + 1024)))[0] = reinterpret_cast<int4*>(packed_9)[0];
        }
        if ((lane & 15) == 0) {
            if (lane >> 4 < 2) {
                if (row0 + (lane >> 4) < T) {
                    r[row0 + (lane >> 4)] = rstd;
                }
            }
        }
    }
}

} // extern "C"
