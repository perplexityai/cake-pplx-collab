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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_route_weight_scale_det_train_bd43b17800f7121c6434(__nv_bfloat16* __restrict__ x, float* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, long long x_stride, long long y_stride)
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
        long long yoff = (long long)row * y_stride;
        float wr = w[src];
        float2 wr2 = make_float2(wr, wr);
        #pragma unroll 1
        for (int run = 0; run < 1; run++) {
            unsigned int xw[64];
            unsigned int yw[64];
            long long col_0 = (long long)((run * 16 * 32 + lane_1) * 8);
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(x + (xoff + col_0) + 0);
                xw[0 + 0] = _uv4_0.x;
                xw[0 + 1] = _uv4_0.y;
                xw[0 + 2] = _uv4_0.z;
                xw[0 + 3] = _uv4_0.w;
            }
            long long col_1 = (long long)(((run * 16 + 1) * 32 + lane_1) * 8);
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(x + (xoff + col_1) + 0);
                xw[4 + 0] = _uv4_1.x;
                xw[4 + 1] = _uv4_1.y;
                xw[4 + 2] = _uv4_1.z;
                xw[4 + 3] = _uv4_1.w;
            }
            long long col_2 = (long long)(((run * 16 + 2) * 32 + lane_1) * 8);
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(x + (xoff + col_2) + 0);
                xw[8 + 0] = _uv4_2.x;
                xw[8 + 1] = _uv4_2.y;
                xw[8 + 2] = _uv4_2.z;
                xw[8 + 3] = _uv4_2.w;
            }
            long long col_3 = (long long)(((run * 16 + 3) * 32 + lane_1) * 8);
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(x + (xoff + col_3) + 0);
                xw[12 + 0] = _uv4_3.x;
                xw[12 + 1] = _uv4_3.y;
                xw[12 + 2] = _uv4_3.z;
                xw[12 + 3] = _uv4_3.w;
            }
            long long col_4 = (long long)(((run * 16 + 4) * 32 + lane_1) * 8);
            {
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(x + (xoff + col_4) + 0);
                xw[16 + 0] = _uv4_4.x;
                xw[16 + 1] = _uv4_4.y;
                xw[16 + 2] = _uv4_4.z;
                xw[16 + 3] = _uv4_4.w;
            }
            long long col_5 = (long long)(((run * 16 + 5) * 32 + lane_1) * 8);
            {
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(x + (xoff + col_5) + 0);
                xw[20 + 0] = _uv4_5.x;
                xw[20 + 1] = _uv4_5.y;
                xw[20 + 2] = _uv4_5.z;
                xw[20 + 3] = _uv4_5.w;
            }
            long long col_6 = (long long)(((run * 16 + 6) * 32 + lane_1) * 8);
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(x + (xoff + col_6) + 0);
                xw[24 + 0] = _uv4_6.x;
                xw[24 + 1] = _uv4_6.y;
                xw[24 + 2] = _uv4_6.z;
                xw[24 + 3] = _uv4_6.w;
            }
            long long col_7 = (long long)(((run * 16 + 7) * 32 + lane_1) * 8);
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(x + (xoff + col_7) + 0);
                xw[28 + 0] = _uv4_7.x;
                xw[28 + 1] = _uv4_7.y;
                xw[28 + 2] = _uv4_7.z;
                xw[28 + 3] = _uv4_7.w;
            }
            long long col_8 = (long long)(((run * 16 + 8) * 32 + lane_1) * 8);
            {
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(x + (xoff + col_8) + 0);
                xw[32 + 0] = _uv4_8.x;
                xw[32 + 1] = _uv4_8.y;
                xw[32 + 2] = _uv4_8.z;
                xw[32 + 3] = _uv4_8.w;
            }
            long long col_9 = (long long)(((run * 16 + 9) * 32 + lane_1) * 8);
            {
                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(x + (xoff + col_9) + 0);
                xw[36 + 0] = _uv4_9.x;
                xw[36 + 1] = _uv4_9.y;
                xw[36 + 2] = _uv4_9.z;
                xw[36 + 3] = _uv4_9.w;
            }
            long long col_10 = (long long)(((run * 16 + 10) * 32 + lane_1) * 8);
            {
                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(x + (xoff + col_10) + 0);
                xw[40 + 0] = _uv4_10.x;
                xw[40 + 1] = _uv4_10.y;
                xw[40 + 2] = _uv4_10.z;
                xw[40 + 3] = _uv4_10.w;
            }
            long long col_11 = (long long)(((run * 16 + 11) * 32 + lane_1) * 8);
            {
                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(x + (xoff + col_11) + 0);
                xw[44 + 0] = _uv4_11.x;
                xw[44 + 1] = _uv4_11.y;
                xw[44 + 2] = _uv4_11.z;
                xw[44 + 3] = _uv4_11.w;
            }
            long long col_12 = (long long)(((run * 16 + 12) * 32 + lane_1) * 8);
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(x + (xoff + col_12) + 0);
                xw[48 + 0] = _uv4_12.x;
                xw[48 + 1] = _uv4_12.y;
                xw[48 + 2] = _uv4_12.z;
                xw[48 + 3] = _uv4_12.w;
            }
            long long col_13 = (long long)(((run * 16 + 13) * 32 + lane_1) * 8);
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(x + (xoff + col_13) + 0);
                xw[52 + 0] = _uv4_13.x;
                xw[52 + 1] = _uv4_13.y;
                xw[52 + 2] = _uv4_13.z;
                xw[52 + 3] = _uv4_13.w;
            }
            long long col_14 = (long long)(((run * 16 + 14) * 32 + lane_1) * 8);
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(x + (xoff + col_14) + 0);
                xw[56 + 0] = _uv4_14.x;
                xw[56 + 1] = _uv4_14.y;
                xw[56 + 2] = _uv4_14.z;
                xw[56 + 3] = _uv4_14.w;
            }
            long long col_15 = (long long)(((run * 16 + 15) * 32 + lane_1) * 8);
            {
                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(x + (xoff + col_15) + 0);
                xw[60 + 0] = _uv4_15.x;
                xw[60 + 1] = _uv4_15.y;
                xw[60 + 2] = _uv4_15.z;
                xw[60 + 3] = _uv4_15.w;
            }
            float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[0]));
            yw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_0)));
            float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[1]));
            yw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_1)));
            float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[2]));
            yw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_2)));
            float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[3]));
            yw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_3)));
            float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4]));
            yw[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_4)));
            float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[5]));
            yw[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_5)));
            float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[6]));
            yw[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_6)));
            float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[7]));
            yw[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_7)));
            float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[8]));
            yw[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_8)));
            float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[9]));
            yw[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_9)));
            float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[10]));
            yw[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_10)));
            float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[11]));
            yw[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_11)));
            float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[12]));
            yw[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_12)));
            float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[13]));
            yw[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_13)));
            float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[14]));
            yw[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_14)));
            float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[15]));
            yw[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_15)));
            float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[16]));
            yw[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_16)));
            float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[17]));
            yw[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_17)));
            float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[18]));
            yw[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_18)));
            float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[19]));
            yw[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_19)));
            float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[20]));
            yw[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_20)));
            float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[21]));
            yw[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_21)));
            float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[22]));
            yw[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_22)));
            float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[23]));
            yw[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_23)));
            float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[24]));
            yw[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_24)));
            float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[25]));
            yw[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_25)));
            float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[26]));
            yw[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_26)));
            float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[27]));
            yw[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_27)));
            float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[28]));
            yw[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_28)));
            float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[29]));
            yw[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_29)));
            float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[30]));
            yw[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_30)));
            float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[31]));
            yw[31] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_31)));
            float2 xv_32 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[32]));
            yw[32] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_32)));
            float2 xv_33 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[33]));
            yw[33] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_33)));
            float2 xv_34 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[34]));
            yw[34] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_34)));
            float2 xv_35 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[35]));
            yw[35] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_35)));
            float2 xv_36 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[36]));
            yw[36] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_36)));
            float2 xv_37 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[37]));
            yw[37] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_37)));
            float2 xv_38 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[38]));
            yw[38] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_38)));
            float2 xv_39 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[39]));
            yw[39] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_39)));
            float2 xv_40 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[40]));
            yw[40] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_40)));
            float2 xv_41 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[41]));
            yw[41] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_41)));
            float2 xv_42 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[42]));
            yw[42] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_42)));
            float2 xv_43 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[43]));
            yw[43] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_43)));
            float2 xv_44 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[44]));
            yw[44] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_44)));
            float2 xv_45 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[45]));
            yw[45] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_45)));
            float2 xv_46 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[46]));
            yw[46] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_46)));
            float2 xv_47 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[47]));
            yw[47] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_47)));
            float2 xv_48 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[48]));
            yw[48] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_48)));
            float2 xv_49 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[49]));
            yw[49] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_49)));
            float2 xv_50 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[50]));
            yw[50] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_50)));
            float2 xv_51 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[51]));
            yw[51] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_51)));
            float2 xv_52 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[52]));
            yw[52] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_52)));
            float2 xv_53 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[53]));
            yw[53] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_53)));
            float2 xv_54 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[54]));
            yw[54] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_54)));
            float2 xv_55 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[55]));
            yw[55] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_55)));
            float2 xv_56 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[56]));
            yw[56] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_56)));
            float2 xv_57 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[57]));
            yw[57] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_57)));
            float2 xv_58 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[58]));
            yw[58] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_58)));
            float2 xv_59 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[59]));
            yw[59] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_59)));
            float2 xv_60 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[60]));
            yw[60] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_60)));
            float2 xv_61 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[61]));
            yw[61] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_61)));
            float2 xv_62 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[62]));
            yw[62] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_62)));
            float2 xv_63 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[63]));
            yw[63] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2, xv_63)));
            if (row < rows) {
                long long col_16 = (long long)((run * 16 * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_16))[0] = reinterpret_cast<int4*>(yw)[0];
                long long col_17 = (long long)(((run * 16 + 1) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_17))[0] = reinterpret_cast<int4*>(yw + (4))[0];
                long long col_18 = (long long)(((run * 16 + 2) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_18))[0] = reinterpret_cast<int4*>(yw + (8))[0];
                long long col_19 = (long long)(((run * 16 + 3) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_19))[0] = reinterpret_cast<int4*>(yw + (12))[0];
                long long col_20 = (long long)(((run * 16 + 4) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_20))[0] = reinterpret_cast<int4*>(yw + (16))[0];
                long long col_21 = (long long)(((run * 16 + 5) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_21))[0] = reinterpret_cast<int4*>(yw + (20))[0];
                long long col_22 = (long long)(((run * 16 + 6) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_22))[0] = reinterpret_cast<int4*>(yw + (24))[0];
                long long col_23 = (long long)(((run * 16 + 7) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_23))[0] = reinterpret_cast<int4*>(yw + (28))[0];
                long long col_24 = (long long)(((run * 16 + 8) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_24))[0] = reinterpret_cast<int4*>(yw + (32))[0];
                long long col_25 = (long long)(((run * 16 + 9) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_25))[0] = reinterpret_cast<int4*>(yw + (36))[0];
                long long col_26 = (long long)(((run * 16 + 10) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_26))[0] = reinterpret_cast<int4*>(yw + (40))[0];
                long long col_27 = (long long)(((run * 16 + 11) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_27))[0] = reinterpret_cast<int4*>(yw + (44))[0];
                long long col_28 = (long long)(((run * 16 + 12) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_28))[0] = reinterpret_cast<int4*>(yw + (48))[0];
                long long col_29 = (long long)(((run * 16 + 13) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_29))[0] = reinterpret_cast<int4*>(yw + (52))[0];
                long long col_30 = (long long)(((run * 16 + 14) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_30))[0] = reinterpret_cast<int4*>(yw + (56))[0];
                long long col_31 = (long long)(((run * 16 + 15) * 32 + lane_1) * 8);
                reinterpret_cast<int4*>(y + (yoff + col_31))[0] = reinterpret_cast<int4*>(yw + (60))[0];
            }
        }
    }
}

} // extern "C"
