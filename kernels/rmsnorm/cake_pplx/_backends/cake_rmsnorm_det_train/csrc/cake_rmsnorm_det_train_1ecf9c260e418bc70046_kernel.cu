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
kernel_cake_rmsnorm_det_train_1ecf9c260e418bc70046(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = (lane & 15) * 8;
    float wreg[8];
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
    int nblocks = (T + 31) / 32;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 32 + warp * 8;
        int lrow = row0 + (lane >> 4);
        unsigned int xs[16];
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
        float tot[4];
        float sum_sq_0 = 0.0f;
        float _dot_9;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_9) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_9;
        float _dot_10;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_10) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_10;
        float _dot_11;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_11) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_11;
        float _dot_12;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_12) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_12;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 8);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_13;
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 4);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_14;
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 2);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_15;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_0, 1);
        sum_sq_0 = sum_sq_0 + _shfl_xor_sync_16;
        tot[0] = sum_sq_0;
        float sum_sq_1 = 0.0f;
        float _dot_17;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_17) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_1));
        sum_sq_1 = _dot_17;
        float _dot_18;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_18) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_1));
        sum_sq_1 = _dot_18;
        float _dot_19;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_19) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_1));
        sum_sq_1 = _dot_19;
        float _dot_20;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_20) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_1));
        sum_sq_1 = _dot_20;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 8);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_21;
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 4);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_22;
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 2);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_23;
        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_1, 1);
        sum_sq_1 = sum_sq_1 + _shfl_xor_sync_24;
        tot[1] = sum_sq_1;
        float sum_sq_2 = 0.0f;
        float _dot_25;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_25) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_2));
        sum_sq_2 = _dot_25;
        float _dot_26;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_26) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_2));
        sum_sq_2 = _dot_26;
        float _dot_27;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_27) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_2));
        sum_sq_2 = _dot_27;
        float _dot_28;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_28) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_2));
        sum_sq_2 = _dot_28;
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 8);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_29;
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 4);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_30;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 2);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_31;
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_2, 1);
        sum_sq_2 = sum_sq_2 + _shfl_xor_sync_32;
        tot[2] = sum_sq_2;
        float sum_sq_3 = 0.0f;
        float _dot_33;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_33) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_3));
        sum_sq_3 = _dot_33;
        float _dot_34;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_34) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_3));
        sum_sq_3 = _dot_34;
        float _dot_35;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_35) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_3));
        sum_sq_3 = _dot_35;
        float _dot_36;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_36) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_3));
        sum_sq_3 = _dot_36;
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 8);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_37;
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 4);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_38;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 2);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_39;
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, sum_sq_3, 1);
        sum_sq_3 = sum_sq_3 + _shfl_xor_sync_40;
        tot[3] = sum_sq_3;
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
        float mean = own * 0.0078125f;
        float rs_a = mean + eps;
        unsigned int rs_ab = __as_u32(rs_a);
        float rs_y;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rs_y) : "f"(rs_a));
        float rs_t;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_t) : "f"(rs_a), "f"(rs_y));
        float _neg_41;
        asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(rs_t));
        float rs_tl;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_tl) : "f"(rs_a), "f"(rs_y), "f"(_neg_41));
        float _neg_42;
        asm("neg.f32 %0, %1;" : "=f"(_neg_42) : "f"(rs_t));
        float rs_e;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_e) : "f"(_neg_42), "f"(rs_y), "f"(1.0f));
        float _neg_44;
        asm("neg.f32 %0, %1;" : "=f"(_neg_44) : "f"(rs_tl));
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(_neg_44), "f"(rs_y), "f"(rs_e));
        rs_e = _fma_43;
        float rs_h;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rs_h) : "f"(rs_y), "f"(0.5f));
        float _sub_45;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_lo;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_lo) : "f"(_sub_45), "f"(rs_h), "f"(rs_y));
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(rs_e), "f"(3.637979e-12f));
        float rs_hi;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rs_hi) : "f"(_add_46), "f"(rs_h), "f"(rs_y));
        float rs_r = rs_lo;
        unsigned int rs_slow = ((rs_ab - 8388608 >= 2130706432) ? 1 : 0);
        if (__as_u32(rs_lo) != __as_u32(rs_hi)) {
            rs_slow = 1;
        }
        if (rs_slow != 0) {
            float _sqrt_48;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_48) : "f"(rs_a));
            float _rcp_47 = approx_rcp(_sqrt_48);
            rs_y = _rcp_47;
            rs_t = mul_f32_rn_noftz(rs_a, rs_y);
            float _neg_50;
            asm("neg.f32 %0, %1;" : "=f"(_neg_50) : "f"(rs_t));
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(rs_a), "f"(rs_y), "f"(_neg_50));
            rs_tl = _fma_49;
            float _neg_52;
            asm("neg.f32 %0, %1;" : "=f"(_neg_52) : "f"(rs_t));
            float _fma_51;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(_neg_52), "f"(rs_y), "f"(1.0f));
            rs_e = _fma_51;
            float _neg_54;
            asm("neg.f32 %0, %1;" : "=f"(_neg_54) : "f"(rs_tl));
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(_neg_54), "f"(rs_y), "f"(rs_e));
            rs_e = _fma_53;
            float _fma_55;
            asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(rs_e), "f"(mul_f32_rn_noftz(rs_y, 0.5f)), "f"(rs_y));
            unsigned int rs_fb = __as_u32(_fma_55);
            unsigned long long rs_m = (unsigned long long)((rs_fb & 8388607) << 1 | 16777217);
            unsigned int rs_an = ((rs_ab < 8388608) ? __as_u32((float)rs_ab) : rs_ab);
            unsigned long long rs_p = rs_m * rs_m * (unsigned long long)(rs_an & 8388607 | 8388608);
            rs_r = ((rs_ab - 1 < 2139095039) ? __uint_as_float(rs_fb + (unsigned int)(rs_p >> 63ull)) : rs_y);
        }
        float rstd = rs_r;
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, lane & 16);
        float out_0[8];
        float2 rk2_0 = make_float2(rk_0, rk_0);
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[0]));
        float2 _f32x2_mul_a_56 = mul_f32x2_noftz(xv_0, rk2_0);
        float2 _f32x2_mul_b_57 = make_float2(wreg[0], wreg[1]);
        float2 o2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
        out_0[0] = o2_0.x;
        out_0[1] = o2_0.y;
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        float2 _f32x2_mul_a_58 = mul_f32x2_noftz(xv_1, rk2_0);
        float2 _f32x2_mul_b_59 = make_float2(wreg[2], wreg[3]);
        float2 o2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
        out_0[2] = o2_1.x;
        out_0[3] = o2_1.y;
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(xv_2, rk2_0);
        float2 _f32x2_mul_b_61 = make_float2(wreg[4], wreg[5]);
        float2 o2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
        out_0[4] = o2_2.x;
        out_0[5] = o2_2.y;
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(xv_3, rk2_0);
        float2 _f32x2_mul_b_63 = make_float2(wreg[6], wreg[7]);
        float2 o2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
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
        if (lrow < T) {
            reinterpret_cast<int4*>(y + ((long long)lrow * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 1);
        float out_1[8];
        float2 rk2_1 = make_float2(rk_1, rk_1);
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        float2 _f32x2_mul_a_64 = mul_f32x2_noftz(xv_4, rk2_1);
        float2 _f32x2_mul_b_65 = make_float2(wreg[0], wreg[1]);
        float2 o2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
        out_1[0] = o2_4.x;
        out_1[1] = o2_4.y;
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        float2 _f32x2_mul_a_66 = mul_f32x2_noftz(xv_5, rk2_1);
        float2 _f32x2_mul_b_67 = make_float2(wreg[2], wreg[3]);
        float2 o2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
        out_1[2] = o2_5.x;
        out_1[3] = o2_5.y;
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(xv_6, rk2_1);
        float2 _f32x2_mul_b_69 = make_float2(wreg[4], wreg[5]);
        float2 o2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
        out_1[4] = o2_6.x;
        out_1[5] = o2_6.y;
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(xv_7, rk2_1);
        float2 _f32x2_mul_b_71 = make_float2(wreg[6], wreg[7]);
        float2 o2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
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
        if (lrow + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 2) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        float rk_2 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 2);
        float out_2[8];
        float2 rk2_2 = make_float2(rk_2, rk_2);
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(xv_8, rk2_2);
        float2 _f32x2_mul_b_73 = make_float2(wreg[0], wreg[1]);
        float2 o2_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
        out_2[0] = o2_8.x;
        out_2[1] = o2_8.y;
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        float2 _f32x2_mul_a_74 = mul_f32x2_noftz(xv_9, rk2_2);
        float2 _f32x2_mul_b_75 = make_float2(wreg[2], wreg[3]);
        float2 o2_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
        out_2[2] = o2_9.x;
        out_2[3] = o2_9.y;
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        float2 _f32x2_mul_a_76 = mul_f32x2_noftz(xv_10, rk2_2);
        float2 _f32x2_mul_b_77 = make_float2(wreg[4], wreg[5]);
        float2 o2_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
        out_2[4] = o2_10.x;
        out_2[5] = o2_10.y;
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(xv_11, rk2_2);
        float2 _f32x2_mul_b_79 = make_float2(wreg[6], wreg[7]);
        float2 o2_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
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
        if (lrow + 4 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 4) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
        }
        float rk_3 = __shfl_sync(0xFFFFFFFF, rstd, (lane & 16) + 3);
        float out_3[8];
        float2 rk2_3 = make_float2(rk_3, rk_3);
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        float2 _f32x2_mul_a_80 = mul_f32x2_noftz(xv_12, rk2_3);
        float2 _f32x2_mul_b_81 = make_float2(wreg[0], wreg[1]);
        float2 o2_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
        out_3[0] = o2_12.x;
        out_3[1] = o2_12.y;
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        float2 _f32x2_mul_a_82 = mul_f32x2_noftz(xv_13, rk2_3);
        float2 _f32x2_mul_b_83 = make_float2(wreg[2], wreg[3]);
        float2 o2_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
        out_3[2] = o2_13.x;
        out_3[3] = o2_13.y;
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        float2 _f32x2_mul_a_84 = mul_f32x2_noftz(xv_14, rk2_3);
        float2 _f32x2_mul_b_85 = make_float2(wreg[4], wreg[5]);
        float2 o2_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
        out_3[4] = o2_14.x;
        out_3[5] = o2_14.y;
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
        float2 _f32x2_mul_a_86 = mul_f32x2_noftz(xv_15, rk2_3);
        float2 _f32x2_mul_b_87 = make_float2(wreg[6], wreg[7]);
        float2 o2_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
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
        if (lrow + 6 < T) {
            reinterpret_cast<int4*>(y + ((long long)(lrow + 6) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        if ((lane & 0) == 0) {
            if ((lane & 15) < 4) {
                if (lrow + 2 * (lane & 15) < T) {
                    r[lrow + 2 * (lane & 15)] = rstd;
                }
            }
        }
    }
}

} // extern "C"
