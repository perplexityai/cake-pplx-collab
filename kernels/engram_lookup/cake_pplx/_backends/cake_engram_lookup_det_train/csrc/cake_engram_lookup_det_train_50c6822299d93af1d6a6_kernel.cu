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

#if __CUDA_ARCH__ == 1070
#define LAUNCH_MIN_BLOCKS 4
#else
#define LAUNCH_MIN_BLOCKS 8
#endif

#include <math_constants.h>

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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_engram_lookup_det_train_50c6822299d93af1d6a6(uint8_t* __restrict__ w, uint8_t* __restrict__ s, long long* __restrict__ row_ids, __nv_bfloat16* __restrict__ out, int P, long long w_size_p0, long long w_stride_p0, long long s_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned long long rows = (unsigned long long)w_size_p0;
    unsigned long long w_stride = (unsigned long long)w_stride_p0;
    unsigned long long s_stride = (unsigned long long)s_stride_p0;
    int lane_1 = lane;
    unsigned long long code_at = (unsigned long long)(lane_1 * 8);
    unsigned long long scale_at = (unsigned long long)(lane_1 >> 2);
    int last = P - 1;
    int steps = P + 3 >> 2;
    for (int unit = bid * 8 + warp; unit < steps; unit += gridDim.x * 8) {
        int first = unit * 4;
        unsigned long long rid[4];
        int _min_0 = ((first) < (last) ? (first) : (last));
        rid[0] = (unsigned long long)row_ids[_min_0];
        int _min_1 = ((first + 1) < (last) ? (first + 1) : (last));
        rid[1] = (unsigned long long)row_ids[_min_1];
        int _min_2 = ((first + 2) < (last) ? (first + 2) : (last));
        rid[2] = (unsigned long long)row_ids[_min_2];
        int _min_3 = ((first + 3) < (last) ? (first + 3) : (last));
        rid[3] = (unsigned long long)row_ids[_min_3];
        unsigned long long top = rid[0];
        unsigned long long _max_4 = ((top) > (rid[1]) ? (top) : (rid[1]));
        top = _max_4;
        unsigned long long _max_5 = ((top) > (rid[2]) ? (top) : (rid[2]));
        top = _max_5;
        unsigned long long _max_6 = ((top) > (rid[3]) ? (top) : (rid[3]));
        top = _max_6;
        if (top >= rows) {
            asm volatile("trap;" ::: "memory");
        }
        unsigned int codes[8];
        unsigned int scales[4];
        if (s_stride == w_stride) {
            {
                const uint2* _ivptr_7 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[0] * w_stride + code_at)) + 0);
                uint2 _ivld_7;
                asm volatile("ld.global.nc.L1::evict_normal.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_7.x), "=r"(_ivld_7.y) : "l"((const void*)(_ivptr_7)) : "memory");
                codes[0 + 0] = _ivld_7.x;
                codes[0 + 1] = _ivld_7.y;
            }
            {
                const uint2* _ivptr_8 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[1] * w_stride + code_at)) + 0);
                uint2 _ivld_8;
                asm volatile("ld.global.nc.L1::evict_normal.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_8.x), "=r"(_ivld_8.y) : "l"((const void*)(_ivptr_8)) : "memory");
                codes[2 + 0] = _ivld_8.x;
                codes[2 + 1] = _ivld_8.y;
            }
            {
                const uint2* _ivptr_9 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[2] * w_stride + code_at)) + 0);
                uint2 _ivld_9;
                asm volatile("ld.global.nc.L1::evict_normal.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_9.x), "=r"(_ivld_9.y) : "l"((const void*)(_ivptr_9)) : "memory");
                codes[4 + 0] = _ivld_9.x;
                codes[4 + 1] = _ivld_9.y;
            }
            {
                const uint2* _ivptr_10 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[3] * w_stride + code_at)) + 0);
                uint2 _ivld_10;
                asm volatile("ld.global.nc.L1::evict_normal.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_10.x), "=r"(_ivld_10.y) : "l"((const void*)(_ivptr_10)) : "memory");
                codes[6 + 0] = _ivld_10.x;
                codes[6 + 1] = _ivld_10.y;
            }
        } else {
            {
                const uint2* _ivptr_11 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[0] * w_stride + code_at)) + 0);
                uint2 _ivld_11;
                asm volatile("ld.global.nc.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_11.x), "=r"(_ivld_11.y) : "l"((const void*)(_ivptr_11)) : "memory");
                codes[0 + 0] = _ivld_11.x;
                codes[0 + 1] = _ivld_11.y;
            }
            {
                const uint2* _ivptr_12 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[1] * w_stride + code_at)) + 0);
                uint2 _ivld_12;
                asm volatile("ld.global.nc.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_12.x), "=r"(_ivld_12.y) : "l"((const void*)(_ivptr_12)) : "memory");
                codes[2 + 0] = _ivld_12.x;
                codes[2 + 1] = _ivld_12.y;
            }
            {
                const uint2* _ivptr_13 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[2] * w_stride + code_at)) + 0);
                uint2 _ivld_13;
                asm volatile("ld.global.nc.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_13.x), "=r"(_ivld_13.y) : "l"((const void*)(_ivptr_13)) : "memory");
                codes[4 + 0] = _ivld_13.x;
                codes[4 + 1] = _ivld_13.y;
            }
            {
                const uint2* _ivptr_14 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + (rid[3] * w_stride + code_at)) + 0);
                uint2 _ivld_14;
                asm volatile("ld.global.nc.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                    : "=r"(_ivld_14.x), "=r"(_ivld_14.y) : "l"((const void*)(_ivptr_14)) : "memory");
                codes[6 + 0] = _ivld_14.x;
                codes[6 + 1] = _ivld_14.y;
            }
        }
        scales[0] = (unsigned int)s[rid[0] * s_stride + scale_at];
        scales[1] = (unsigned int)s[rid[1] * s_stride + scale_at];
        scales[2] = (unsigned int)s[rid[2] * s_stride + scale_at];
        scales[3] = (unsigned int)s[rid[3] * s_stride + scale_at];
        unsigned int vals[16];
        unsigned int b_0 = scales[0];
        unsigned int bits_0 = ((b_0 == 255) ? 2143289344 : ((b_0 == 0) ? 4194304 : b_0 << 23));
        float sc_0 = __uint_as_float(bits_0);
        float2 pair_scale_0 = make_float2(sc_0, sc_0);
        uint16_t code_pair_0 = (uint16_t)(codes[0] & 65535);
        float2 _fp8_e4m3x2_decode_15;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_15.x), "=f"(_fp8_e4m3x2_decode_15.y) : "h"(code_pair_0));
        vals[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_15, pair_scale_0)));
        uint16_t code_pair_1 = (uint16_t)(codes[0] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_16;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_16.x), "=f"(_fp8_e4m3x2_decode_16.y) : "h"(code_pair_1));
        vals[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_16, pair_scale_0)));
        uint16_t code_pair_2 = (uint16_t)(codes[1] & 65535);
        float2 _fp8_e4m3x2_decode_17;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_17.x), "=f"(_fp8_e4m3x2_decode_17.y) : "h"(code_pair_2));
        vals[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_17, pair_scale_0)));
        uint16_t code_pair_3 = (uint16_t)(codes[1] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_18;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_18.x), "=f"(_fp8_e4m3x2_decode_18.y) : "h"(code_pair_3));
        vals[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_18, pair_scale_0)));
        if (first < P) {
            asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out + ((unsigned long long)first * 256ull + code_at)), "r"(vals[(0) + 0]), "r"(vals[(0) + 1]), "r"(vals[(0) + 2]), "r"(vals[(0) + 3]) : "memory");
        }
        unsigned int b_1 = scales[1];
        unsigned int bits_1 = ((b_1 == 255) ? 2143289344 : ((b_1 == 0) ? 4194304 : b_1 << 23));
        float sc_1 = __uint_as_float(bits_1);
        float2 pair_scale_1 = make_float2(sc_1, sc_1);
        uint16_t code_pair_4 = (uint16_t)(codes[2] & 65535);
        float2 _fp8_e4m3x2_decode_19;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_19.x), "=f"(_fp8_e4m3x2_decode_19.y) : "h"(code_pair_4));
        vals[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_19, pair_scale_1)));
        uint16_t code_pair_5 = (uint16_t)(codes[2] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_20;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_20.x), "=f"(_fp8_e4m3x2_decode_20.y) : "h"(code_pair_5));
        vals[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_20, pair_scale_1)));
        uint16_t code_pair_6 = (uint16_t)(codes[3] & 65535);
        float2 _fp8_e4m3x2_decode_21;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_21.x), "=f"(_fp8_e4m3x2_decode_21.y) : "h"(code_pair_6));
        vals[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_21, pair_scale_1)));
        uint16_t code_pair_7 = (uint16_t)(codes[3] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_22;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_22.x), "=f"(_fp8_e4m3x2_decode_22.y) : "h"(code_pair_7));
        vals[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_22, pair_scale_1)));
        if (first + 1 < P) {
            asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out + ((unsigned long long)(first + 1) * 256ull + code_at)), "r"(vals[(4) + 0]), "r"(vals[(4) + 1]), "r"(vals[(4) + 2]), "r"(vals[(4) + 3]) : "memory");
        }
        unsigned int b_2 = scales[2];
        unsigned int bits_2 = ((b_2 == 255) ? 2143289344 : ((b_2 == 0) ? 4194304 : b_2 << 23));
        float sc_2 = __uint_as_float(bits_2);
        float2 pair_scale_2 = make_float2(sc_2, sc_2);
        uint16_t code_pair_8 = (uint16_t)(codes[4] & 65535);
        float2 _fp8_e4m3x2_decode_23;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_23.x), "=f"(_fp8_e4m3x2_decode_23.y) : "h"(code_pair_8));
        vals[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_23, pair_scale_2)));
        uint16_t code_pair_9 = (uint16_t)(codes[4] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_24;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_24.x), "=f"(_fp8_e4m3x2_decode_24.y) : "h"(code_pair_9));
        vals[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_24, pair_scale_2)));
        uint16_t code_pair_10 = (uint16_t)(codes[5] & 65535);
        float2 _fp8_e4m3x2_decode_25;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_25.x), "=f"(_fp8_e4m3x2_decode_25.y) : "h"(code_pair_10));
        vals[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_25, pair_scale_2)));
        uint16_t code_pair_11 = (uint16_t)(codes[5] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_26;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_26.x), "=f"(_fp8_e4m3x2_decode_26.y) : "h"(code_pair_11));
        vals[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_26, pair_scale_2)));
        if (first + 2 < P) {
            asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out + ((unsigned long long)(first + 2) * 256ull + code_at)), "r"(vals[(8) + 0]), "r"(vals[(8) + 1]), "r"(vals[(8) + 2]), "r"(vals[(8) + 3]) : "memory");
        }
        unsigned int b_3 = scales[3];
        unsigned int bits_3 = ((b_3 == 255) ? 2143289344 : ((b_3 == 0) ? 4194304 : b_3 << 23));
        float sc_3 = __uint_as_float(bits_3);
        float2 pair_scale_3 = make_float2(sc_3, sc_3);
        uint16_t code_pair_12 = (uint16_t)(codes[6] & 65535);
        float2 _fp8_e4m3x2_decode_27;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_27.x), "=f"(_fp8_e4m3x2_decode_27.y) : "h"(code_pair_12));
        vals[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_27, pair_scale_3)));
        uint16_t code_pair_13 = (uint16_t)(codes[6] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_28;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_28.x), "=f"(_fp8_e4m3x2_decode_28.y) : "h"(code_pair_13));
        vals[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_28, pair_scale_3)));
        uint16_t code_pair_14 = (uint16_t)(codes[7] & 65535);
        float2 _fp8_e4m3x2_decode_29;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_29.x), "=f"(_fp8_e4m3x2_decode_29.y) : "h"(code_pair_14));
        vals[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_29, pair_scale_3)));
        uint16_t code_pair_15 = (uint16_t)(codes[7] >> 16 & 65535);
        float2 _fp8_e4m3x2_decode_30;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8_e4m3x2_decode_30.x), "=f"(_fp8_e4m3x2_decode_30.y) : "h"(code_pair_15));
        vals[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(_fp8_e4m3x2_decode_30, pair_scale_3)));
        if (first + 3 < P) {
            asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out + ((unsigned long long)(first + 3) * 256ull + code_at)), "r"(vals[(12) + 0]), "r"(vals[(12) + 1]), "r"(vals[(12) + 2]), "r"(vals[(12) + 3]) : "memory");
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
