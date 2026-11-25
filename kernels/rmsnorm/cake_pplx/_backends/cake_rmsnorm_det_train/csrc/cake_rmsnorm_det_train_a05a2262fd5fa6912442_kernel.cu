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

__global__ __launch_bounds__(64) void
kernel_cake_rmsnorm_det_train_a05a2262fd5fa6912442(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = (lane & 15) * 8;
    unsigned int wreg[4];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4* _vdst_0 = reinterpret_cast<uint4*>(&wreg[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
        }
    }
    int nblocks = (T + 31) / 32;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 32 + warp * 16;
        int lrow = row0 + (lane >> 4);
        unsigned int xs[32];
        int _min_1 = ((lrow) < (T - 1) ? (lrow) : (T - 1));
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + ((long long)_min_1 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&xs[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
        int _min_3 = ((lrow + 2) < (T - 1) ? (lrow + 2) : (T - 1));
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + ((long long)_min_3 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&xs[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
        int _min_5 = ((lrow + 4) < (T - 1) ? (lrow + 4) : (T - 1));
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + ((long long)_min_5 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&xs[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
        int _min_7 = ((lrow + 6) < (T - 1) ? (lrow + 6) : (T - 1));
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + ((long long)_min_7 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&xs[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
        int _min_9 = ((lrow + 8) < (T - 1) ? (lrow + 8) : (T - 1));
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + ((long long)_min_9 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&xs[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
        int _min_11 = ((lrow + 10) < (T - 1) ? (lrow + 10) : (T - 1));
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + ((long long)_min_11 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_12 = reinterpret_cast<uint4*>(&xs[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_12[_blk] = _vptr_12[_blk];
            }
        }
        int _min_13 = ((lrow + 12) < (T - 1) ? (lrow + 12) : (T - 1));
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + ((long long)_min_13 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&xs[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
        int _min_15 = ((lrow + 14) < (T - 1) ? (lrow + 14) : (T - 1));
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(x + ((long long)_min_15 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_16 = reinterpret_cast<uint4*>(&xs[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_16[_blk] = _vptr_16[_blk];
            }
        }
        float tot[8];
        float sum_sq_0 = 0.0f;
        float _dot_17;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_17) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_17;
        float _dot_18;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_18) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_18;
        float _dot_19;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_19) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_19;
        float _dot_20;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_20) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_20;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 8);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_21;
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 4);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_22;
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 2);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_23;
        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 1);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_24;
        tot[0] = sum_sq_0;
        float sum_sq_1 = 0.0f;
        float _dot_25;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_25) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_1));
        sum_sq_1 = _dot_25;
        float _dot_26;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_26) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_1));
        sum_sq_1 = _dot_26;
        float _dot_27;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_27) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_1));
        sum_sq_1 = _dot_27;
        float _dot_28;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_28) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_1));
        sum_sq_1 = _dot_28;
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 8);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_29;
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 4);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_30;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 2);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_31;
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 1);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_32;
        tot[1] = sum_sq_1;
        float sum_sq_2 = 0.0f;
        float _dot_33;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_33) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_2));
        sum_sq_2 = _dot_33;
        float _dot_34;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_34) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_2));
        sum_sq_2 = _dot_34;
        float _dot_35;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_35) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_2));
        sum_sq_2 = _dot_35;
        float _dot_36;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_36) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_2));
        sum_sq_2 = _dot_36;
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 8);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_37;
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 4);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_38;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 2);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_39;
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 1);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_40;
        tot[2] = sum_sq_2;
        float sum_sq_3 = 0.0f;
        float _dot_41;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_41) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_3));
        sum_sq_3 = _dot_41;
        float _dot_42;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_42) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_3));
        sum_sq_3 = _dot_42;
        float _dot_43;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_43) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_3));
        sum_sq_3 = _dot_43;
        float _dot_44;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_44) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_3));
        sum_sq_3 = _dot_44;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 8);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_45;
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 4);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_46;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 2);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_47;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 1);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_48;
        tot[3] = sum_sq_3;
        float sum_sq_4 = 0.0f;
        float _dot_49;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_49) : "r"(xs[16]), "r"(xs[16]), "f"(sum_sq_4));
        sum_sq_4 = _dot_49;
        float _dot_50;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_50) : "r"(xs[17]), "r"(xs[17]), "f"(sum_sq_4));
        sum_sq_4 = _dot_50;
        float _dot_51;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_51) : "r"(xs[18]), "r"(xs[18]), "f"(sum_sq_4));
        sum_sq_4 = _dot_51;
        float _dot_52;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_52) : "r"(xs[19]), "r"(xs[19]), "f"(sum_sq_4));
        sum_sq_4 = _dot_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_4, 8);
        sum_sq_4 = sum_sq_4 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_4, 4);
        sum_sq_4 = sum_sq_4 + _shfl_xor_sync_54;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_4, 2);
        sum_sq_4 = sum_sq_4 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_4, 1);
        sum_sq_4 = sum_sq_4 + _shfl_xor_sync_56;
        tot[4] = sum_sq_4;
        float sum_sq_5 = 0.0f;
        float _dot_57;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_57) : "r"(xs[20]), "r"(xs[20]), "f"(sum_sq_5));
        sum_sq_5 = _dot_57;
        float _dot_58;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_58) : "r"(xs[21]), "r"(xs[21]), "f"(sum_sq_5));
        sum_sq_5 = _dot_58;
        float _dot_59;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_59) : "r"(xs[22]), "r"(xs[22]), "f"(sum_sq_5));
        sum_sq_5 = _dot_59;
        float _dot_60;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_60) : "r"(xs[23]), "r"(xs[23]), "f"(sum_sq_5));
        sum_sq_5 = _dot_60;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_5, 8);
        sum_sq_5 = sum_sq_5 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_5, 4);
        sum_sq_5 = sum_sq_5 + _shfl_xor_sync_62;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_5, 2);
        sum_sq_5 = sum_sq_5 + _shfl_xor_sync_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_5, 1);
        sum_sq_5 = sum_sq_5 + _shfl_xor_sync_64;
        tot[5] = sum_sq_5;
        float sum_sq_6 = 0.0f;
        float _dot_65;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_65) : "r"(xs[24]), "r"(xs[24]), "f"(sum_sq_6));
        sum_sq_6 = _dot_65;
        float _dot_66;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_66) : "r"(xs[25]), "r"(xs[25]), "f"(sum_sq_6));
        sum_sq_6 = _dot_66;
        float _dot_67;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_67) : "r"(xs[26]), "r"(xs[26]), "f"(sum_sq_6));
        sum_sq_6 = _dot_67;
        float _dot_68;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_68) : "r"(xs[27]), "r"(xs[27]), "f"(sum_sq_6));
        sum_sq_6 = _dot_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_6, 8);
        sum_sq_6 = sum_sq_6 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_6, 4);
        sum_sq_6 = sum_sq_6 + _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_6, 2);
        sum_sq_6 = sum_sq_6 + _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_6, 1);
        sum_sq_6 = sum_sq_6 + _shfl_xor_sync_72;
        tot[6] = sum_sq_6;
        float sum_sq_7 = 0.0f;
        float _dot_73;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_73) : "r"(xs[28]), "r"(xs[28]), "f"(sum_sq_7));
        sum_sq_7 = _dot_73;
        float _dot_74;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_74) : "r"(xs[29]), "r"(xs[29]), "f"(sum_sq_7));
        sum_sq_7 = _dot_74;
        float _dot_75;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_75) : "r"(xs[30]), "r"(xs[30]), "f"(sum_sq_7));
        sum_sq_7 = _dot_75;
        float _dot_76;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_76) : "r"(xs[31]), "r"(xs[31]), "f"(sum_sq_7));
        sum_sq_7 = _dot_76;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_7, 8);
        sum_sq_7 = sum_sq_7 + _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_7, 4);
        sum_sq_7 = sum_sq_7 + _shfl_xor_sync_78;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_7, 2);
        sum_sq_7 = sum_sq_7 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_7, 1);
        sum_sq_7 = sum_sq_7 + _shfl_xor_sync_80;
        tot[7] = sum_sq_7;
        float own = tot[0];
        if ((lane & 15) == 1) {
            own = tot[1];
        }
        if ((lane & 15) == 2) {
            own = tot[2];
        }
        if ((lane & 15) == 3) {
            own = tot[3];
        }
        if ((lane & 15) == 4) {
            own = tot[4];
        }
        if ((lane & 15) == 5) {
            own = tot[5];
        }
        if ((lane & 15) == 6) {
            own = tot[6];
        }
        if ((lane & 15) == 7) {
            own = tot[7];
        }
        float mean = own * 0.0078125f;
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_81;
        asm("neg.f32 %0, %1;" : "=f"(_neg_81) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_81));
        float _neg_82;
        asm("neg.f32 %0, %1;" : "=f"(_neg_82) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_82), "f"(rs_y), "f"(1.0f));
        float _neg_84;
        asm("neg.f32 %0, %1;" : "=f"(_neg_84) : "f"(rs_tl));
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(_neg_84), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_83;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_85;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_85), "f"(rs_h), "f"(rs_y));
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_86), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_88;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_88) : "f"(rs_a));
            float _rcp_87 = approx_rcp(_sqrt_88);
            rs_y = _rcp_87;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_90;
            asm("neg.f32 %0, %1;" : "=f"(_neg_90) : "f"(rs_t));
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(rs_a), "f"(rs_y), "f"(_neg_90));
            rs_tl = _fma_89;
            float _neg_92;
            asm("neg.f32 %0, %1;" : "=f"(_neg_92) : "f"(rs_t));
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(_neg_92), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_91;
            float _neg_94;
            asm("neg.f32 %0, %1;" : "=f"(_neg_94) : "f"(rs_tl));
            float _fma_93;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(_neg_94), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_93;
            float _fma_95;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_95);
            unsigned long long rs_m = (unsigned long long)((rs_fb & 8388607) << 1 | 16777217);
            unsigned int rs_an = ((rs_ab < 8388608) ? __as_u32((float)rs_ab) : rs_ab);
            unsigned long long rs_p = rs_m * rs_m * (unsigned long long)(rs_an & 8388607 | 8388608);
            rs_r = ((rs_ab - 1 < 2139095039) ? __uint_as_float(rs_fb + (unsigned int)(rs_p >> 63ull)) : rs_y);
        }
        float rstd = rs_r;
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, lane & 16);
        float out_0[8];
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[0]));
        float2 wv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_0[0] = xv_0.x * rk_0 * wv_0.x;
        out_0[1] = xv_0.y * rk_0 * wv_0.y;
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        float2 wv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_0[2] = xv_1.x * rk_0 * wv_1.x;
        out_0[3] = xv_1.y * rk_0 * wv_1.y;
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        float2 wv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_0[4] = xv_2.x * rk_0 * wv_2.x;
        out_0[5] = xv_2.y * rk_0 * wv_2.y;
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
        float2 wv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_0[6] = xv_3.x * rk_0 * wv_3.x;
        out_0[7] = xv_3.y * rk_0 * wv_3.y;
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
        if (lrow < T) {
            reinterpret_cast<int4*>(y + ((long long)lrow * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 1);
        float out_1[8];
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        float2 wv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_1[0] = xv_4.x * rk_1 * wv_4.x;
        out_1[1] = xv_4.y * rk_1 * wv_4.y;
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        float2 wv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_1[2] = xv_5.x * rk_1 * wv_5.x;
        out_1[3] = xv_5.y * rk_1 * wv_5.y;
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        float2 wv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_1[4] = xv_6.x * rk_1 * wv_6.x;
        out_1[5] = xv_6.y * rk_1 * wv_6.y;
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
        float2 wv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_1[6] = xv_7.x * rk_1 * wv_7.x;
        out_1[7] = xv_7.y * rk_1 * wv_7.y;
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
        if (lrow + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 2) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        float rk_2 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 2);
        float out_2[8];
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        float2 wv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_2[0] = xv_8.x * rk_2 * wv_8.x;
        out_2[1] = xv_8.y * rk_2 * wv_8.y;
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        float2 wv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_2[2] = xv_9.x * rk_2 * wv_9.x;
        out_2[3] = xv_9.y * rk_2 * wv_9.y;
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        float2 wv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_2[4] = xv_10.x * rk_2 * wv_10.x;
        out_2[5] = xv_10.y * rk_2 * wv_10.y;
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
        float2 wv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_2[6] = xv_11.x * rk_2 * wv_11.x;
        out_2[7] = xv_11.y * rk_2 * wv_11.y;
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
        if (lrow + 4 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 4) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
        }
        float rk_3 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 3);
        float out_3[8];
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        float2 wv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_3[0] = xv_12.x * rk_3 * wv_12.x;
        out_3[1] = xv_12.y * rk_3 * wv_12.y;
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        float2 wv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_3[2] = xv_13.x * rk_3 * wv_13.x;
        out_3[3] = xv_13.y * rk_3 * wv_13.y;
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        float2 wv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_3[4] = xv_14.x * rk_3 * wv_14.x;
        out_3[5] = xv_14.y * rk_3 * wv_14.y;
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
        float2 wv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_3[6] = xv_15.x * rk_3 * wv_15.x;
        out_3[7] = xv_15.y * rk_3 * wv_15.y;
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
        if (lrow + 6 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 6) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        float rk_4 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 4);
        float out_4[8];
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[16]));
        float2 wv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_4[0] = xv_16.x * rk_4 * wv_16.x;
        out_4[1] = xv_16.y * rk_4 * wv_16.y;
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[17]));
        float2 wv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_4[2] = xv_17.x * rk_4 * wv_17.x;
        out_4[3] = xv_17.y * rk_4 * wv_17.y;
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[18]));
        float2 wv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_4[4] = xv_18.x * rk_4 * wv_18.x;
        out_4[5] = xv_18.y * rk_4 * wv_18.y;
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[19]));
        float2 wv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_4[6] = xv_19.x * rk_4 * wv_19.x;
        out_4[7] = xv_19.y * rk_4 * wv_19.y;
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
        if (lrow + 8 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 8) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_4)[0];
        }
        float rk_5 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 5);
        float out_5[8];
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[20]));
        float2 wv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_5[0] = xv_20.x * rk_5 * wv_20.x;
        out_5[1] = xv_20.y * rk_5 * wv_20.y;
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[21]));
        float2 wv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_5[2] = xv_21.x * rk_5 * wv_21.x;
        out_5[3] = xv_21.y * rk_5 * wv_21.y;
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[22]));
        float2 wv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_5[4] = xv_22.x * rk_5 * wv_22.x;
        out_5[5] = xv_22.y * rk_5 * wv_22.y;
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[23]));
        float2 wv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_5[6] = xv_23.x * rk_5 * wv_23.x;
        out_5[7] = xv_23.y * rk_5 * wv_23.y;
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
        if (lrow + 10 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 10) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_5)[0];
        }
        float rk_6 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 6);
        float out_6[8];
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[24]));
        float2 wv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_6[0] = xv_24.x * rk_6 * wv_24.x;
        out_6[1] = xv_24.y * rk_6 * wv_24.y;
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[25]));
        float2 wv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_6[2] = xv_25.x * rk_6 * wv_25.x;
        out_6[3] = xv_25.y * rk_6 * wv_25.y;
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[26]));
        float2 wv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_6[4] = xv_26.x * rk_6 * wv_26.x;
        out_6[5] = xv_26.y * rk_6 * wv_26.y;
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[27]));
        float2 wv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_6[6] = xv_27.x * rk_6 * wv_27.x;
        out_6[7] = xv_27.y * rk_6 * wv_27.y;
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
        if (lrow + 12 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 12) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_6)[0];
        }
        float rk_7 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 7);
        float out_7[8];
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[28]));
        float2 wv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        out_7[0] = xv_28.x * rk_7 * wv_28.x;
        out_7[1] = xv_28.y * rk_7 * wv_28.y;
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[29]));
        float2 wv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        out_7[2] = xv_29.x * rk_7 * wv_29.x;
        out_7[3] = xv_29.y * rk_7 * wv_29.y;
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[30]));
        float2 wv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        out_7[4] = xv_30.x * rk_7 * wv_30.x;
        out_7[5] = xv_30.y * rk_7 * wv_30.y;
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[31]));
        float2 wv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        out_7[6] = xv_31.x * rk_7 * wv_31.x;
        out_7[7] = xv_31.y * rk_7 * wv_31.y;
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
        if (lrow + 14 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 14) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_7)[0];
        }
        if ((lane & 0) == 0) {
            if ((lane & 15) < 8) {
                if (lrow + 2 * (lane & 15) < T) {
                    r[lrow + 2 * (lane & 15)] = rstd;
                }
            }
        }
    }
}

} // extern "C"
