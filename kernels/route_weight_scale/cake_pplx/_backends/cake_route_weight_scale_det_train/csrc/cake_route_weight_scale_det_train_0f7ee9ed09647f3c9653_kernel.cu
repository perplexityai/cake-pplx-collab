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

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_route_weight_scale_det_train_0f7ee9ed09647f3c9653(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ dy, float* __restrict__ w, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, int rows, long long x_stride, long long dy_stride, long long dx_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int lane_1 = tid & 31;
    int nblocks = (rows + 3) / 4;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row = blk * 4 + (tid >> 5);
        int src = ((row) < (rows - 1) ? (row) : (rows - 1));
        long long xoff = (long long)src * x_stride;
        long long doff = (long long)src * dy_stride;
        long long ooff = (long long)row * dx_stride;
        float wr = w[src];
        float2 wr2 = make_float2(wr, wr);
        unsigned int xw[28];
        unsigned int yw[28];
        long long col_0 = (long long)(lane_1 * 8);
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(x + (xoff + col_0) + 0);
            xw[0 + 0] = _uv4_0.x;
            xw[0 + 1] = _uv4_0.y;
            xw[0 + 2] = _uv4_0.z;
            xw[0 + 3] = _uv4_0.w;
        }
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(dy + (doff + col_0) + 0);
            yw[0 + 0] = _uv4_1.x;
            yw[0 + 1] = _uv4_1.y;
            yw[0 + 2] = _uv4_1.z;
            yw[0 + 3] = _uv4_1.w;
        }
        long long col_1 = (long long)((32 + lane_1) * 8);
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(x + (xoff + col_1) + 0);
            xw[4 + 0] = _uv4_2.x;
            xw[4 + 1] = _uv4_2.y;
            xw[4 + 2] = _uv4_2.z;
            xw[4 + 3] = _uv4_2.w;
        }
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(dy + (doff + col_1) + 0);
            yw[4 + 0] = _uv4_3.x;
            yw[4 + 1] = _uv4_3.y;
            yw[4 + 2] = _uv4_3.z;
            yw[4 + 3] = _uv4_3.w;
        }
        long long col_2 = (long long)((64 + lane_1) * 8);
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(x + (xoff + col_2) + 0);
            xw[8 + 0] = _uv4_4.x;
            xw[8 + 1] = _uv4_4.y;
            xw[8 + 2] = _uv4_4.z;
            xw[8 + 3] = _uv4_4.w;
        }
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(dy + (doff + col_2) + 0);
            yw[8 + 0] = _uv4_5.x;
            yw[8 + 1] = _uv4_5.y;
            yw[8 + 2] = _uv4_5.z;
            yw[8 + 3] = _uv4_5.w;
        }
        long long col_3 = (long long)((96 + lane_1) * 8);
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(x + (xoff + col_3) + 0);
            xw[12 + 0] = _uv4_6.x;
            xw[12 + 1] = _uv4_6.y;
            xw[12 + 2] = _uv4_6.z;
            xw[12 + 3] = _uv4_6.w;
        }
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(dy + (doff + col_3) + 0);
            yw[12 + 0] = _uv4_7.x;
            yw[12 + 1] = _uv4_7.y;
            yw[12 + 2] = _uv4_7.z;
            yw[12 + 3] = _uv4_7.w;
        }
        long long col_4 = (long long)((128 + lane_1) * 8);
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(x + (xoff + col_4) + 0);
            xw[16 + 0] = _uv4_8.x;
            xw[16 + 1] = _uv4_8.y;
            xw[16 + 2] = _uv4_8.z;
            xw[16 + 3] = _uv4_8.w;
        }
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(dy + (doff + col_4) + 0);
            yw[16 + 0] = _uv4_9.x;
            yw[16 + 1] = _uv4_9.y;
            yw[16 + 2] = _uv4_9.z;
            yw[16 + 3] = _uv4_9.w;
        }
        long long col_5 = (long long)((160 + lane_1) * 8);
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(x + (xoff + col_5) + 0);
            xw[20 + 0] = _uv4_10.x;
            xw[20 + 1] = _uv4_10.y;
            xw[20 + 2] = _uv4_10.z;
            xw[20 + 3] = _uv4_10.w;
        }
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(dy + (doff + col_5) + 0);
            yw[20 + 0] = _uv4_11.x;
            yw[20 + 1] = _uv4_11.y;
            yw[20 + 2] = _uv4_11.z;
            yw[20 + 3] = _uv4_11.w;
        }
        long long col_6 = (long long)((192 + lane_1) * 8);
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(x + (xoff + col_6) + 0);
            xw[24 + 0] = _uv4_12.x;
            xw[24 + 1] = _uv4_12.y;
            xw[24 + 2] = _uv4_12.z;
            xw[24 + 3] = _uv4_12.w;
        }
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(dy + (doff + col_6) + 0);
            yw[24 + 0] = _uv4_13.x;
            yw[24 + 1] = _uv4_13.y;
            yw[24 + 2] = _uv4_13.z;
            yw[24 + 3] = _uv4_13.w;
        }
        unsigned int gw[28];
        float2 d_0 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[0]));
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[0]));
        gw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_0)));
        float2 d_1 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[1]));
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[1]));
        gw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_1)));
        float2 d_2 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[2]));
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[2]));
        gw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_2)));
        float2 d_3 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[3]));
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[3]));
        gw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_3)));
        float2 lo_0 = fma_f32x2_rn_noftz(d_1, xv_1, mul_f32x2_noftz(d_0, xv_0));
        float2 hi_0 = fma_f32x2_rn_noftz(d_3, xv_3, mul_f32x2_noftz(d_2, xv_2));
        float2 piece_0 = add_f32x2_noftz(lo_0, hi_0);
        float2 d_4 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[4]));
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4]));
        gw[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_4)));
        float2 d_5 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[5]));
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[5]));
        gw[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_5)));
        float2 d_6 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[6]));
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[6]));
        gw[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_6)));
        float2 d_7 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[7]));
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[7]));
        gw[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_7)));
        float2 lo_1 = fma_f32x2_rn_noftz(d_5, xv_5, mul_f32x2_noftz(d_4, xv_4));
        float2 hi_1 = fma_f32x2_rn_noftz(d_7, xv_7, mul_f32x2_noftz(d_6, xv_6));
        float2 piece_1 = add_f32x2_noftz(lo_1, hi_1);
        float2 joined_0 = add_f32x2_noftz(piece_0, piece_1);
        float2 d_8 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[8]));
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[8]));
        gw[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_8)));
        float2 d_9 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[9]));
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[9]));
        gw[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_9)));
        float2 d_10 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[10]));
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[10]));
        gw[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_10)));
        float2 d_11 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[11]));
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[11]));
        gw[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_11)));
        float2 lo_2 = fma_f32x2_rn_noftz(d_9, xv_9, mul_f32x2_noftz(d_8, xv_8));
        float2 hi_2 = fma_f32x2_rn_noftz(d_11, xv_11, mul_f32x2_noftz(d_10, xv_10));
        float2 piece_2 = add_f32x2_noftz(lo_2, hi_2);
        float2 d_12 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[12]));
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[12]));
        gw[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_12)));
        float2 d_13 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[13]));
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[13]));
        gw[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_13)));
        float2 d_14 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[14]));
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[14]));
        gw[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_14)));
        float2 d_15 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[15]));
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[15]));
        gw[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_15)));
        float2 lo_3 = fma_f32x2_rn_noftz(d_13, xv_13, mul_f32x2_noftz(d_12, xv_12));
        float2 hi_3 = fma_f32x2_rn_noftz(d_15, xv_15, mul_f32x2_noftz(d_14, xv_14));
        float2 piece_3 = add_f32x2_noftz(lo_3, hi_3);
        float2 joined_1 = add_f32x2_noftz(piece_2, piece_3);
        float2 joined_2 = add_f32x2_noftz(joined_0, joined_1);
        float2 d_16 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[16]));
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[16]));
        gw[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_16)));
        float2 d_17 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[17]));
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[17]));
        gw[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_17)));
        float2 d_18 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[18]));
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[18]));
        gw[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_18)));
        float2 d_19 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[19]));
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[19]));
        gw[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_19)));
        float2 lo_4 = fma_f32x2_rn_noftz(d_17, xv_17, mul_f32x2_noftz(d_16, xv_16));
        float2 hi_4 = fma_f32x2_rn_noftz(d_19, xv_19, mul_f32x2_noftz(d_18, xv_18));
        float2 piece_4 = add_f32x2_noftz(lo_4, hi_4);
        float2 d_20 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[20]));
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[20]));
        gw[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_20)));
        float2 d_21 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[21]));
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[21]));
        gw[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_21)));
        float2 d_22 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[22]));
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[22]));
        gw[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_22)));
        float2 d_23 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[23]));
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[23]));
        gw[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_23)));
        float2 lo_5 = fma_f32x2_rn_noftz(d_21, xv_21, mul_f32x2_noftz(d_20, xv_20));
        float2 hi_5 = fma_f32x2_rn_noftz(d_23, xv_23, mul_f32x2_noftz(d_22, xv_22));
        float2 piece_5 = add_f32x2_noftz(lo_5, hi_5);
        float2 joined_3 = add_f32x2_noftz(piece_4, piece_5);
        float2 d_24 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[24]));
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[24]));
        gw[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_24)));
        float2 d_25 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[25]));
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[25]));
        gw[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_25)));
        float2 d_26 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[26]));
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[26]));
        gw[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_26)));
        float2 d_27 = cake_bf16x2_to_f32x2(__as_bf16x2(yw[27]));
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[27]));
        gw[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_27)));
        float2 lo_6 = fma_f32x2_rn_noftz(d_25, xv_25, mul_f32x2_noftz(d_24, xv_24));
        float2 hi_6 = fma_f32x2_rn_noftz(d_27, xv_27, mul_f32x2_noftz(d_26, xv_26));
        float2 piece_6 = add_f32x2_noftz(lo_6, hi_6);
        if (row < rows) {
            long long col_7 = (long long)(lane_1 * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_7))[0] = reinterpret_cast<int4*>(gw)[0];
            long long col_8 = (long long)((32 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_8))[0] = reinterpret_cast<int4*>(gw + (4))[0];
            long long col_9 = (long long)((64 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_9))[0] = reinterpret_cast<int4*>(gw + (8))[0];
            long long col_10 = (long long)((96 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_10))[0] = reinterpret_cast<int4*>(gw + (12))[0];
            long long col_11 = (long long)((128 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_11))[0] = reinterpret_cast<int4*>(gw + (16))[0];
            long long col_12 = (long long)((160 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_12))[0] = reinterpret_cast<int4*>(gw + (20))[0];
            long long col_13 = (long long)((192 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_13))[0] = reinterpret_cast<int4*>(gw + (24))[0];
        }
        unsigned int xw_0[28];
        unsigned int yw_0[28];
        long long col_14 = (long long)((224 + lane_1) * 8);
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(x + (xoff + col_14) + 0);
            xw_0[0 + 0] = _uv4_14.x;
            xw_0[0 + 1] = _uv4_14.y;
            xw_0[0 + 2] = _uv4_14.z;
            xw_0[0 + 3] = _uv4_14.w;
        }
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(dy + (doff + col_14) + 0);
            yw_0[0 + 0] = _uv4_15.x;
            yw_0[0 + 1] = _uv4_15.y;
            yw_0[0 + 2] = _uv4_15.z;
            yw_0[0 + 3] = _uv4_15.w;
        }
        long long col_15 = (long long)((256 + lane_1) * 8);
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(x + (xoff + col_15) + 0);
            xw_0[4 + 0] = _uv4_16.x;
            xw_0[4 + 1] = _uv4_16.y;
            xw_0[4 + 2] = _uv4_16.z;
            xw_0[4 + 3] = _uv4_16.w;
        }
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(dy + (doff + col_15) + 0);
            yw_0[4 + 0] = _uv4_17.x;
            yw_0[4 + 1] = _uv4_17.y;
            yw_0[4 + 2] = _uv4_17.z;
            yw_0[4 + 3] = _uv4_17.w;
        }
        long long col_16 = (long long)((288 + lane_1) * 8);
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(x + (xoff + col_16) + 0);
            xw_0[8 + 0] = _uv4_18.x;
            xw_0[8 + 1] = _uv4_18.y;
            xw_0[8 + 2] = _uv4_18.z;
            xw_0[8 + 3] = _uv4_18.w;
        }
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(dy + (doff + col_16) + 0);
            yw_0[8 + 0] = _uv4_19.x;
            yw_0[8 + 1] = _uv4_19.y;
            yw_0[8 + 2] = _uv4_19.z;
            yw_0[8 + 3] = _uv4_19.w;
        }
        long long col_17 = (long long)((320 + lane_1) * 8);
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(x + (xoff + col_17) + 0);
            xw_0[12 + 0] = _uv4_20.x;
            xw_0[12 + 1] = _uv4_20.y;
            xw_0[12 + 2] = _uv4_20.z;
            xw_0[12 + 3] = _uv4_20.w;
        }
        {
            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(dy + (doff + col_17) + 0);
            yw_0[12 + 0] = _uv4_21.x;
            yw_0[12 + 1] = _uv4_21.y;
            yw_0[12 + 2] = _uv4_21.z;
            yw_0[12 + 3] = _uv4_21.w;
        }
        long long col_18 = (long long)((352 + lane_1) * 8);
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(x + (xoff + col_18) + 0);
            xw_0[16 + 0] = _uv4_22.x;
            xw_0[16 + 1] = _uv4_22.y;
            xw_0[16 + 2] = _uv4_22.z;
            xw_0[16 + 3] = _uv4_22.w;
        }
        {
            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(dy + (doff + col_18) + 0);
            yw_0[16 + 0] = _uv4_23.x;
            yw_0[16 + 1] = _uv4_23.y;
            yw_0[16 + 2] = _uv4_23.z;
            yw_0[16 + 3] = _uv4_23.w;
        }
        long long col_19 = (long long)((384 + lane_1) * 8);
        {
            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(x + (xoff + col_19) + 0);
            xw_0[20 + 0] = _uv4_24.x;
            xw_0[20 + 1] = _uv4_24.y;
            xw_0[20 + 2] = _uv4_24.z;
            xw_0[20 + 3] = _uv4_24.w;
        }
        {
            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(dy + (doff + col_19) + 0);
            yw_0[20 + 0] = _uv4_25.x;
            yw_0[20 + 1] = _uv4_25.y;
            yw_0[20 + 2] = _uv4_25.z;
            yw_0[20 + 3] = _uv4_25.w;
        }
        long long col_20 = (long long)((416 + lane_1) * 8);
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(x + (xoff + col_20) + 0);
            xw_0[24 + 0] = _uv4_26.x;
            xw_0[24 + 1] = _uv4_26.y;
            xw_0[24 + 2] = _uv4_26.z;
            xw_0[24 + 3] = _uv4_26.w;
        }
        {
            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(dy + (doff + col_20) + 0);
            yw_0[24 + 0] = _uv4_27.x;
            yw_0[24 + 1] = _uv4_27.y;
            yw_0[24 + 2] = _uv4_27.z;
            yw_0[24 + 3] = _uv4_27.w;
        }
        unsigned int gw_0[28];
        float2 d_28 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[0]));
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[0]));
        gw_0[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_28)));
        float2 d_29 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[1]));
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[1]));
        gw_0[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_29)));
        float2 d_30 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[2]));
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[2]));
        gw_0[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_30)));
        float2 d_31 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[3]));
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[3]));
        gw_0[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_31)));
        float2 lo_7 = fma_f32x2_rn_noftz(d_29, xv_29, mul_f32x2_noftz(d_28, xv_28));
        float2 hi_7 = fma_f32x2_rn_noftz(d_31, xv_31, mul_f32x2_noftz(d_30, xv_30));
        float2 piece_7 = add_f32x2_noftz(lo_7, hi_7);
        float2 joined_4 = add_f32x2_noftz(piece_6, piece_7);
        float2 joined_5 = add_f32x2_noftz(joined_3, joined_4);
        float2 joined_6 = add_f32x2_noftz(joined_2, joined_5);
        float2 d_32 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[4]));
        float2 xv_32 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[4]));
        gw_0[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_32)));
        float2 d_33 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[5]));
        float2 xv_33 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[5]));
        gw_0[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_33)));
        float2 d_34 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[6]));
        float2 xv_34 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[6]));
        gw_0[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_34)));
        float2 d_35 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[7]));
        float2 xv_35 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[7]));
        gw_0[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_35)));
        float2 lo_8 = fma_f32x2_rn_noftz(d_33, xv_33, mul_f32x2_noftz(d_32, xv_32));
        float2 hi_8 = fma_f32x2_rn_noftz(d_35, xv_35, mul_f32x2_noftz(d_34, xv_34));
        float2 piece_8 = add_f32x2_noftz(lo_8, hi_8);
        float2 d_36 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[8]));
        float2 xv_36 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[8]));
        gw_0[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_36)));
        float2 d_37 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[9]));
        float2 xv_37 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[9]));
        gw_0[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_37)));
        float2 d_38 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[10]));
        float2 xv_38 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[10]));
        gw_0[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_38)));
        float2 d_39 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[11]));
        float2 xv_39 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[11]));
        gw_0[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_39)));
        float2 lo_9 = fma_f32x2_rn_noftz(d_37, xv_37, mul_f32x2_noftz(d_36, xv_36));
        float2 hi_9 = fma_f32x2_rn_noftz(d_39, xv_39, mul_f32x2_noftz(d_38, xv_38));
        float2 piece_9 = add_f32x2_noftz(lo_9, hi_9);
        float2 joined_7 = add_f32x2_noftz(piece_8, piece_9);
        float2 d_40 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[12]));
        float2 xv_40 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[12]));
        gw_0[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_40)));
        float2 d_41 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[13]));
        float2 xv_41 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[13]));
        gw_0[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_41)));
        float2 d_42 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[14]));
        float2 xv_42 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[14]));
        gw_0[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_42)));
        float2 d_43 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[15]));
        float2 xv_43 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[15]));
        gw_0[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_43)));
        float2 lo_10 = fma_f32x2_rn_noftz(d_41, xv_41, mul_f32x2_noftz(d_40, xv_40));
        float2 hi_10 = fma_f32x2_rn_noftz(d_43, xv_43, mul_f32x2_noftz(d_42, xv_42));
        float2 piece_10 = add_f32x2_noftz(lo_10, hi_10);
        float2 d_44 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[16]));
        float2 xv_44 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[16]));
        gw_0[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_44)));
        float2 d_45 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[17]));
        float2 xv_45 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[17]));
        gw_0[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_45)));
        float2 d_46 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[18]));
        float2 xv_46 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[18]));
        gw_0[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_46)));
        float2 d_47 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[19]));
        float2 xv_47 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[19]));
        gw_0[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_47)));
        float2 lo_11 = fma_f32x2_rn_noftz(d_45, xv_45, mul_f32x2_noftz(d_44, xv_44));
        float2 hi_11 = fma_f32x2_rn_noftz(d_47, xv_47, mul_f32x2_noftz(d_46, xv_46));
        float2 piece_11 = add_f32x2_noftz(lo_11, hi_11);
        float2 joined_8 = add_f32x2_noftz(piece_10, piece_11);
        float2 joined_9 = add_f32x2_noftz(joined_7, joined_8);
        float2 d_48 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[20]));
        float2 xv_48 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[20]));
        gw_0[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_48)));
        float2 d_49 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[21]));
        float2 xv_49 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[21]));
        gw_0[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_49)));
        float2 d_50 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[22]));
        float2 xv_50 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[22]));
        gw_0[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_50)));
        float2 d_51 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[23]));
        float2 xv_51 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[23]));
        gw_0[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_51)));
        float2 lo_12 = fma_f32x2_rn_noftz(d_49, xv_49, mul_f32x2_noftz(d_48, xv_48));
        float2 hi_12 = fma_f32x2_rn_noftz(d_51, xv_51, mul_f32x2_noftz(d_50, xv_50));
        float2 piece_12 = add_f32x2_noftz(lo_12, hi_12);
        float2 d_52 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[24]));
        float2 xv_52 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[24]));
        gw_0[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_52)));
        float2 d_53 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[25]));
        float2 xv_53 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[25]));
        gw_0[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_53)));
        float2 d_54 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[26]));
        float2 xv_54 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[26]));
        gw_0[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_54)));
        float2 d_55 = cake_bf16x2_to_f32x2(__as_bf16x2(yw_0[27]));
        float2 xv_55 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_0[27]));
        gw_0[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, d_55)));
        float2 lo_13 = fma_f32x2_rn_noftz(d_53, xv_53, mul_f32x2_noftz(d_52, xv_52));
        float2 hi_13 = fma_f32x2_rn_noftz(d_55, xv_55, mul_f32x2_noftz(d_54, xv_54));
        float2 piece_13 = add_f32x2_noftz(lo_13, hi_13);
        float2 joined_10 = add_f32x2_noftz(piece_12, piece_13);
        float2 joined_11 = add_f32x2_noftz(joined_9, joined_10);
        float2 joined_12 = add_f32x2_noftz(joined_6, joined_11);
        if (row < rows) {
            long long col_21 = (long long)((224 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_21))[0] = reinterpret_cast<int4*>(gw_0)[0];
            long long col_22 = (long long)((256 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_22))[0] = reinterpret_cast<int4*>(gw_0 + (4))[0];
            long long col_23 = (long long)((288 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_23))[0] = reinterpret_cast<int4*>(gw_0 + (8))[0];
            long long col_24 = (long long)((320 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_24))[0] = reinterpret_cast<int4*>(gw_0 + (12))[0];
            long long col_25 = (long long)((352 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_25))[0] = reinterpret_cast<int4*>(gw_0 + (16))[0];
            long long col_26 = (long long)((384 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_26))[0] = reinterpret_cast<int4*>(gw_0 + (20))[0];
            long long col_27 = (long long)((416 + lane_1) * 8);
            reinterpret_cast<int4*>(dx + (ooff + col_27))[0] = reinterpret_cast<int4*>(gw_0 + (24))[0];
        }
        float total;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(total) : "f"(joined_12.x), "f"(joined_12.y));
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, total, 16);
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(total), "f"(_shfl_xor_sync_29));
        total = _add_28;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(total), "f"(_shfl_xor_sync_31));
        total = _add_30;
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(total), "f"(_shfl_xor_sync_33));
        total = _add_32;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(total), "f"(_shfl_xor_sync_35));
        total = _add_34;
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(total), "f"(_shfl_xor_sync_37));
        total = _add_36;
        if (row < rows) {
            if (lane_1 == 0) {
                *(reinterpret_cast<float*>(dw + row) + (0)) = total;
            }
        }
    }
}

} // extern "C"
