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
kernel_cake_route_weight_scale_det_train_76225253b414f853a74d(__nv_bfloat16* __restrict__ x, float* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, long long x_stride, long long y_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int nchunks = (int)(((long long)rows * 3584ll + 2047ll) / 2048ll);
    long long _max_0 = ((x_stride) > (y_stride) ? (x_stride) : (y_stride));
    long long bound = (((long long)rows * 3584ll + 2048ll) > ((long long)rows * _max_0 + 3584ll) ? ((long long)rows * 3584ll + 2048ll) : ((long long)rows * _max_0 + 3584ll));
    if (bound <= 2147483647ll) {
        int total = (int)rows * 3584;
        int last = total - 8;
        int xs = (int)x_stride;
        int ys = (int)y_stride;
        #pragma unroll 1
        for (int blk = bid; blk < nchunks; blk += num_bids) {
            unsigned int xw[8];
            unsigned int yw[8];
            int e_0 = (int)blk * 2048 + (int)(tid * 8);
            int ec_0 = ((e_0) < (last) ? (e_0) : (last));
            int r_0 = ec_0 / 3584;
            int c_0 = ec_0 - r_0 * 3584;
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(x + (r_0 * xs + c_0) + 0);
                xw[0 + 0] = _uv4_1.x;
                xw[0 + 1] = _uv4_1.y;
                xw[0 + 2] = _uv4_1.z;
                xw[0 + 3] = _uv4_1.w;
            }
            int e_1 = (int)blk * 2048 + (int)((128 + tid) * 8);
            int ec_1 = ((e_1) < (last) ? (e_1) : (last));
            int r_1 = ec_1 / 3584;
            int c_1 = ec_1 - r_1 * 3584;
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(x + (r_1 * xs + c_1) + 0);
                xw[4 + 0] = _uv4_2.x;
                xw[4 + 1] = _uv4_2.y;
                xw[4 + 2] = _uv4_2.z;
                xw[4 + 3] = _uv4_2.w;
            }
            int e_2 = (int)blk * 2048 + (int)(tid * 8);
            int ec_2 = ((e_2) < (last) ? (e_2) : (last));
            int r_2 = ec_2 / 3584;
            int c_2 = ec_2 - r_2 * 3584;
            float wr_0 = w[(int)r_2];
            float2 wr2_0 = make_float2(wr_0, wr_0);
            float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[0]));
            yw[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_0, xv_0)));
            float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[1]));
            yw[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_0, xv_1)));
            float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[2]));
            yw[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_0, xv_2)));
            float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[3]));
            yw[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_0, xv_3)));
            if (e_2 < total) {
                reinterpret_cast<int4*>(y + (r_2 * ys + c_2))[0] = reinterpret_cast<int4*>(yw)[0];
            }
            int e_3 = (int)blk * 2048 + (int)((128 + tid) * 8);
            int ec_3 = ((e_3) < (last) ? (e_3) : (last));
            int r_3 = ec_3 / 3584;
            int c_3 = ec_3 - r_3 * 3584;
            float wr_1 = w[(int)r_3];
            float2 wr2_1 = make_float2(wr_1, wr_1);
            float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[4]));
            yw[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_1, xv_4)));
            float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[5]));
            yw[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_1, xv_5)));
            float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[6]));
            yw[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_1, xv_6)));
            float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[7]));
            yw[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_1, xv_7)));
            if (e_3 < total) {
                reinterpret_cast<int4*>(y + (r_3 * ys + c_3))[0] = reinterpret_cast<int4*>(yw + (4))[0];
            }
        }
    }
    if (bound > 2147483647ll) {
        long long total_1 = (long long)rows * 3584ll;
        long long last_1 = total_1 - 8ll;
        long long xs_1 = (long long)x_stride;
        long long ys_1 = (long long)y_stride;
        #pragma unroll 1
        for (int blk_1 = bid; blk_1 < nchunks; blk_1 += num_bids) {
            unsigned int xw_1[8];
            unsigned int yw_1[8];
            long long e_4 = (long long)blk_1 * 2048ll + (long long)(tid * 8);
            long long ec_4 = ((e_4) < (last_1) ? (e_4) : (last_1));
            long long r_4 = ec_4 / 3584ll;
            long long c_4 = ec_4 - r_4 * 3584ll;
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(x + (r_4 * xs_1 + c_4) + 0);
                xw_1[0 + 0] = _uv4_3.x;
                xw_1[0 + 1] = _uv4_3.y;
                xw_1[0 + 2] = _uv4_3.z;
                xw_1[0 + 3] = _uv4_3.w;
            }
            long long e_5 = (long long)blk_1 * 2048ll + (long long)((128 + tid) * 8);
            long long ec_5 = ((e_5) < (last_1) ? (e_5) : (last_1));
            long long r_5 = ec_5 / 3584ll;
            long long c_5 = ec_5 - r_5 * 3584ll;
            {
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(x + (r_5 * xs_1 + c_5) + 0);
                xw_1[4 + 0] = _uv4_4.x;
                xw_1[4 + 1] = _uv4_4.y;
                xw_1[4 + 2] = _uv4_4.z;
                xw_1[4 + 3] = _uv4_4.w;
            }
            long long e_6 = (long long)blk_1 * 2048ll + (long long)(tid * 8);
            long long ec_6 = ((e_6) < (last_1) ? (e_6) : (last_1));
            long long r_6 = ec_6 / 3584ll;
            long long c_6 = ec_6 - r_6 * 3584ll;
            float wr_2 = w[(int)r_6];
            float2 wr2_2 = make_float2(wr_2, wr_2);
            float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[0]));
            yw_1[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_2, xv_8)));
            float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[1]));
            yw_1[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_2, xv_9)));
            float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[2]));
            yw_1[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_2, xv_10)));
            float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[3]));
            yw_1[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_2, xv_11)));
            if (e_6 < total_1) {
                reinterpret_cast<int4*>(y + (r_6 * ys_1 + c_6))[0] = reinterpret_cast<int4*>(yw_1)[0];
            }
            long long e_7 = (long long)blk_1 * 2048ll + (long long)((128 + tid) * 8);
            long long ec_7 = ((e_7) < (last_1) ? (e_7) : (last_1));
            long long r_7 = ec_7 / 3584ll;
            long long c_7 = ec_7 - r_7 * 3584ll;
            float wr_3 = w[(int)r_7];
            float2 wr2_3 = make_float2(wr_3, wr_3);
            float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[4]));
            yw_1[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_3, xv_12)));
            float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[5]));
            yw_1[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_3, xv_13)));
            float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[6]));
            yw_1[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_3, xv_14)));
            float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xw_1[7]));
            yw_1[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(wr2_3, xv_15)));
            if (e_7 < total_1) {
                reinterpret_cast<int4*>(y + (r_7 * ys_1 + c_7))[0] = reinterpret_cast<int4*>(yw_1 + (4))[0];
            }
        }
    }
}

} // extern "C"
