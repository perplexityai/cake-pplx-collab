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

#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>


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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_collapse_bwd__vjpwd__h5120(__nv_bfloat16* __restrict__ x, float* __restrict__ pre_in, __nv_bfloat16* __restrict__ du, __nv_bfloat16* __restrict__ dx, float* __restrict__ dpre_in, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* dpre_part = reinterpret_cast<float*>(smem_raw + 0);
    const int dpre_part_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = (int)warp;
    int lane_1 = (int)lane;
    int lcs = lane_1 & 3;
    long long wcol = (long long)(warp_0 * 640) + (long long)(lane_1 * 8);
    long long col_off = (long long)tid * 20ll;
    int gi[1];
    gi[0] = 0;
    #pragma unroll 1
    for (int item = bid; item < T; item += num_bids) {
        int row = item;
        int s0 = (item & 0) * 4;
        long long t64 = (long long)row;
        int cur_base = (gi[0] & 1) * 32;
        float pin[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(pre_in + (t64 * 4ll + (long long)s0) + 0);
            pin[0 + 0] = _v4.x;
            pin[0 + 1] = _v4.y;
            pin[0 + 2] = _v4.z;
            pin[0 + 3] = _v4.w;
        }
        float du_row[20];
        unsigned int du_w[12];
        if (lane_1 < 80) {
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol)) + 0);
                du_w[0 + 0] = _uv4_1.x;
                du_w[0 + 1] = _uv4_1.y;
                du_w[0 + 2] = _uv4_1.z;
                du_w[0 + 3] = _uv4_1.w;
            }
        }
        if (lane_1 + 32 < 80) {
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol + 256ll)) + 0);
                du_w[4 + 0] = _uv4_2.x;
                du_w[4 + 1] = _uv4_2.y;
                du_w[4 + 2] = _uv4_2.z;
                du_w[4 + 3] = _uv4_2.w;
            }
        }
        if (lane_1 + 64 < 80) {
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol + 512ll)) + 0);
                du_w[8 + 0] = _uv4_3.x;
                du_w[8 + 1] = _uv4_3.y;
                du_w[8 + 2] = _uv4_3.z;
                du_w[8 + 3] = _uv4_3.w;
            }
        }
        unsigned int x_w[40];
        {
            const uint2* _ivptr_4 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off)) + 0);
            uint2 _ivld_4;
            _ivld_4 = *_ivptr_4;
            x_w[0 + 0] = _ivld_4.x;
            x_w[0 + 1] = _ivld_4.y;
        }
        {
            const uint2* _ivptr_5 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_5;
            _ivld_5 = *_ivptr_5;
            x_w[2 + 0] = _ivld_5.x;
            x_w[2 + 1] = _ivld_5.y;
        }
        {
            const uint2* _ivptr_6 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_6;
            _ivld_6 = *_ivptr_6;
            x_w[4 + 0] = _ivld_6.x;
            x_w[4 + 1] = _ivld_6.y;
        }
        {
            const uint2* _ivptr_7 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_7;
            _ivld_7 = *_ivptr_7;
            x_w[6 + 0] = _ivld_7.x;
            x_w[6 + 1] = _ivld_7.y;
        }
        {
            const uint2* _ivptr_8 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_8;
            _ivld_8 = *_ivptr_8;
            x_w[8 + 0] = _ivld_8.x;
            x_w[8 + 1] = _ivld_8.y;
        }
        {
            const uint2* _ivptr_9 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + col_off)) + 0);
            uint2 _ivld_9;
            _ivld_9 = *_ivptr_9;
            x_w[10 + 0] = _ivld_9.x;
            x_w[10 + 1] = _ivld_9.y;
        }
        {
            const uint2* _ivptr_10 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_10;
            _ivld_10 = *_ivptr_10;
            x_w[12 + 0] = _ivld_10.x;
            x_w[12 + 1] = _ivld_10.y;
        }
        {
            const uint2* _ivptr_11 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_11;
            _ivld_11 = *_ivptr_11;
            x_w[14 + 0] = _ivld_11.x;
            x_w[14 + 1] = _ivld_11.y;
        }
        {
            const uint2* _ivptr_12 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_12;
            _ivld_12 = *_ivptr_12;
            x_w[16 + 0] = _ivld_12.x;
            x_w[16 + 1] = _ivld_12.y;
        }
        {
            const uint2* _ivptr_13 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_13;
            _ivld_13 = *_ivptr_13;
            x_w[18 + 0] = _ivld_13.x;
            x_w[18 + 1] = _ivld_13.y;
        }
        {
            const uint2* _ivptr_14 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + col_off)) + 0);
            uint2 _ivld_14;
            _ivld_14 = *_ivptr_14;
            x_w[20 + 0] = _ivld_14.x;
            x_w[20 + 1] = _ivld_14.y;
        }
        {
            const uint2* _ivptr_15 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_15;
            _ivld_15 = *_ivptr_15;
            x_w[22 + 0] = _ivld_15.x;
            x_w[22 + 1] = _ivld_15.y;
        }
        {
            const uint2* _ivptr_16 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_16;
            _ivld_16 = *_ivptr_16;
            x_w[24 + 0] = _ivld_16.x;
            x_w[24 + 1] = _ivld_16.y;
        }
        {
            const uint2* _ivptr_17 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_17;
            _ivld_17 = *_ivptr_17;
            x_w[26 + 0] = _ivld_17.x;
            x_w[26 + 1] = _ivld_17.y;
        }
        {
            const uint2* _ivptr_18 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_18;
            _ivld_18 = *_ivptr_18;
            x_w[28 + 0] = _ivld_18.x;
            x_w[28 + 1] = _ivld_18.y;
        }
        {
            const uint2* _ivptr_19 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + col_off)) + 0);
            uint2 _ivld_19;
            _ivld_19 = *_ivptr_19;
            x_w[30 + 0] = _ivld_19.x;
            x_w[30 + 1] = _ivld_19.y;
        }
        {
            const uint2* _ivptr_20 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_20;
            _ivld_20 = *_ivptr_20;
            x_w[32 + 0] = _ivld_20.x;
            x_w[32 + 1] = _ivld_20.y;
        }
        {
            const uint2* _ivptr_21 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_21;
            _ivld_21 = *_ivptr_21;
            x_w[34 + 0] = _ivld_21.x;
            x_w[34 + 1] = _ivld_21.y;
        }
        {
            const uint2* _ivptr_22 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_22;
            _ivld_22 = *_ivptr_22;
            x_w[36 + 0] = _ivld_22.x;
            x_w[36 + 1] = _ivld_22.y;
        }
        {
            const uint2* _ivptr_23 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_23;
            _ivld_23 = *_ivptr_23;
            x_w[38 + 0] = _ivld_23.x;
            x_w[38 + 1] = _ivld_23.y;
        }
        unsigned int own[10];
        {
            const uint2* _ivptr_24 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off)) + 0);
            uint2 _ivld_24;
            _ivld_24 = *_ivptr_24;
            own[0 + 0] = _ivld_24.x;
            own[0 + 1] = _ivld_24.y;
        }
        {
            const uint2* _ivptr_25 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_25;
            _ivld_25 = *_ivptr_25;
            own[2 + 0] = _ivld_25.x;
            own[2 + 1] = _ivld_25.y;
        }
        {
            const uint2* _ivptr_26 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_26;
            _ivld_26 = *_ivptr_26;
            own[4 + 0] = _ivld_26.x;
            own[4 + 1] = _ivld_26.y;
        }
        {
            const uint2* _ivptr_27 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_27;
            _ivld_27 = *_ivptr_27;
            own[6 + 0] = _ivld_27.x;
            own[6 + 1] = _ivld_27.y;
        }
        {
            const uint2* _ivptr_28 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_28;
            _ivld_28 = *_ivptr_28;
            own[8 + 0] = _ivld_28.x;
            own[8 + 1] = _ivld_28.y;
        }
        if (lane_1 < 80) {
            float dua_0[8];
            dua_0[0] = __uint_as_float(du_w[0] << 16);
            dua_0[1] = __uint_as_float(du_w[0] & 4294901760u);
            dua_0[2] = __uint_as_float(du_w[1] << 16);
            dua_0[3] = __uint_as_float(du_w[1] & 4294901760u);
            dua_0[4] = __uint_as_float(du_w[2] << 16);
            dua_0[5] = __uint_as_float(du_w[2] & 4294901760u);
            dua_0[6] = __uint_as_float(du_w[3] << 16);
            dua_0[7] = __uint_as_float(du_w[3] & 4294901760u);
            unsigned int dw4_0[4];
            dw4_0[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[0]), mul_f32_rn_noftz(pin[0], dua_0[1]))));
            dw4_0[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[2]), mul_f32_rn_noftz(pin[0], dua_0[3]))));
            dw4_0[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[4]), mul_f32_rn_noftz(pin[0], dua_0[5]))));
            dw4_0[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[6]), mul_f32_rn_noftz(pin[0], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)s0) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_0)[0];
            unsigned int dw4_1[4];
            dw4_1[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[0]), mul_f32_rn_noftz(pin[1], dua_0[1]))));
            dw4_1[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[2]), mul_f32_rn_noftz(pin[1], dua_0[3]))));
            dw4_1[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[4]), mul_f32_rn_noftz(pin[1], dua_0[5]))));
            dw4_1[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[6]), mul_f32_rn_noftz(pin[1], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_1)[0];
            unsigned int dw4_2[4];
            dw4_2[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[0]), mul_f32_rn_noftz(pin[2], dua_0[1]))));
            dw4_2[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[2]), mul_f32_rn_noftz(pin[2], dua_0[3]))));
            dw4_2[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[4]), mul_f32_rn_noftz(pin[2], dua_0[5]))));
            dw4_2[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[6]), mul_f32_rn_noftz(pin[2], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_2)[0];
            unsigned int dw4_3[4];
            dw4_3[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[0]), mul_f32_rn_noftz(pin[3], dua_0[1]))));
            dw4_3[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[2]), mul_f32_rn_noftz(pin[3], dua_0[3]))));
            dw4_3[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[4]), mul_f32_rn_noftz(pin[3], dua_0[5]))));
            dw4_3[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[6]), mul_f32_rn_noftz(pin[3], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_3)[0];
        }
        if (lane_1 + 32 < 80) {
            float dua_1[8];
            dua_1[0] = __uint_as_float(du_w[4] << 16);
            dua_1[1] = __uint_as_float(du_w[4] & 4294901760u);
            dua_1[2] = __uint_as_float(du_w[5] << 16);
            dua_1[3] = __uint_as_float(du_w[5] & 4294901760u);
            dua_1[4] = __uint_as_float(du_w[6] << 16);
            dua_1[5] = __uint_as_float(du_w[6] & 4294901760u);
            dua_1[6] = __uint_as_float(du_w[7] << 16);
            dua_1[7] = __uint_as_float(du_w[7] & 4294901760u);
            unsigned int dw4_4[4];
            dw4_4[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[0]), mul_f32_rn_noftz(pin[0], dua_1[1]))));
            dw4_4[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[2]), mul_f32_rn_noftz(pin[0], dua_1[3]))));
            dw4_4[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[4]), mul_f32_rn_noftz(pin[0], dua_1[5]))));
            dw4_4[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[6]), mul_f32_rn_noftz(pin[0], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)s0) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_4)[0];
            unsigned int dw4_5[4];
            dw4_5[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[0]), mul_f32_rn_noftz(pin[1], dua_1[1]))));
            dw4_5[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[2]), mul_f32_rn_noftz(pin[1], dua_1[3]))));
            dw4_5[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[4]), mul_f32_rn_noftz(pin[1], dua_1[5]))));
            dw4_5[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[6]), mul_f32_rn_noftz(pin[1], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_5)[0];
            unsigned int dw4_6[4];
            dw4_6[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[0]), mul_f32_rn_noftz(pin[2], dua_1[1]))));
            dw4_6[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[2]), mul_f32_rn_noftz(pin[2], dua_1[3]))));
            dw4_6[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[4]), mul_f32_rn_noftz(pin[2], dua_1[5]))));
            dw4_6[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[6]), mul_f32_rn_noftz(pin[2], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_6)[0];
            unsigned int dw4_7[4];
            dw4_7[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[0]), mul_f32_rn_noftz(pin[3], dua_1[1]))));
            dw4_7[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[2]), mul_f32_rn_noftz(pin[3], dua_1[3]))));
            dw4_7[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[4]), mul_f32_rn_noftz(pin[3], dua_1[5]))));
            dw4_7[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[6]), mul_f32_rn_noftz(pin[3], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_7)[0];
        }
        if (lane_1 + 64 < 80) {
            float dua_2[8];
            dua_2[0] = __uint_as_float(du_w[8] << 16);
            dua_2[1] = __uint_as_float(du_w[8] & 4294901760u);
            dua_2[2] = __uint_as_float(du_w[9] << 16);
            dua_2[3] = __uint_as_float(du_w[9] & 4294901760u);
            dua_2[4] = __uint_as_float(du_w[10] << 16);
            dua_2[5] = __uint_as_float(du_w[10] & 4294901760u);
            dua_2[6] = __uint_as_float(du_w[11] << 16);
            dua_2[7] = __uint_as_float(du_w[11] & 4294901760u);
            unsigned int dw4_8[4];
            dw4_8[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[0]), mul_f32_rn_noftz(pin[0], dua_2[1]))));
            dw4_8[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[2]), mul_f32_rn_noftz(pin[0], dua_2[3]))));
            dw4_8[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[4]), mul_f32_rn_noftz(pin[0], dua_2[5]))));
            dw4_8[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[6]), mul_f32_rn_noftz(pin[0], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)s0) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_8)[0];
            unsigned int dw4_9[4];
            dw4_9[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[0]), mul_f32_rn_noftz(pin[1], dua_2[1]))));
            dw4_9[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[2]), mul_f32_rn_noftz(pin[1], dua_2[3]))));
            dw4_9[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[4]), mul_f32_rn_noftz(pin[1], dua_2[5]))));
            dw4_9[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[6]), mul_f32_rn_noftz(pin[1], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 1)) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_9)[0];
            unsigned int dw4_10[4];
            dw4_10[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[0]), mul_f32_rn_noftz(pin[2], dua_2[1]))));
            dw4_10[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[2]), mul_f32_rn_noftz(pin[2], dua_2[3]))));
            dw4_10[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[4]), mul_f32_rn_noftz(pin[2], dua_2[5]))));
            dw4_10[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[6]), mul_f32_rn_noftz(pin[2], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 2)) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_10)[0];
            unsigned int dw4_11[4];
            dw4_11[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[0]), mul_f32_rn_noftz(pin[3], dua_2[1]))));
            dw4_11[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[2]), mul_f32_rn_noftz(pin[3], dua_2[3]))));
            dw4_11[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[4]), mul_f32_rn_noftz(pin[3], dua_2[5]))));
            dw4_11[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[6]), mul_f32_rn_noftz(pin[3], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)(s0 + 3)) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_11)[0];
        }
        du_row[0] = __uint_as_float(own[0] << 16);
        du_row[1] = __uint_as_float(own[0] & 4294901760u);
        du_row[2] = __uint_as_float(own[1] << 16);
        du_row[3] = __uint_as_float(own[1] & 4294901760u);
        du_row[4] = __uint_as_float(own[2] << 16);
        du_row[5] = __uint_as_float(own[2] & 4294901760u);
        du_row[6] = __uint_as_float(own[3] << 16);
        du_row[7] = __uint_as_float(own[3] & 4294901760u);
        du_row[8] = __uint_as_float(own[4] << 16);
        du_row[9] = __uint_as_float(own[4] & 4294901760u);
        du_row[10] = __uint_as_float(own[5] << 16);
        du_row[11] = __uint_as_float(own[5] & 4294901760u);
        du_row[12] = __uint_as_float(own[6] << 16);
        du_row[13] = __uint_as_float(own[6] & 4294901760u);
        du_row[14] = __uint_as_float(own[7] << 16);
        du_row[15] = __uint_as_float(own[7] & 4294901760u);
        du_row[16] = __uint_as_float(own[8] << 16);
        du_row[17] = __uint_as_float(own[8] & 4294901760u);
        du_row[18] = __uint_as_float(own[9] << 16);
        du_row[19] = __uint_as_float(own[9] & 4294901760u);
        float dots_p[4];
        float x_row[20];
        x_row[0] = __uint_as_float(x_w[0] << 16);
        x_row[1] = __uint_as_float(x_w[0] & 4294901760u);
        x_row[2] = __uint_as_float(x_w[1] << 16);
        x_row[3] = __uint_as_float(x_w[1] & 4294901760u);
        x_row[4] = __uint_as_float(x_w[2] << 16);
        x_row[5] = __uint_as_float(x_w[2] & 4294901760u);
        x_row[6] = __uint_as_float(x_w[3] << 16);
        x_row[7] = __uint_as_float(x_w[3] & 4294901760u);
        x_row[8] = __uint_as_float(x_w[4] << 16);
        x_row[9] = __uint_as_float(x_w[4] & 4294901760u);
        x_row[10] = __uint_as_float(x_w[5] << 16);
        x_row[11] = __uint_as_float(x_w[5] & 4294901760u);
        x_row[12] = __uint_as_float(x_w[6] << 16);
        x_row[13] = __uint_as_float(x_w[6] & 4294901760u);
        x_row[14] = __uint_as_float(x_w[7] << 16);
        x_row[15] = __uint_as_float(x_w[7] & 4294901760u);
        x_row[16] = __uint_as_float(x_w[8] << 16);
        x_row[17] = __uint_as_float(x_w[8] & 4294901760u);
        x_row[18] = __uint_as_float(x_w[9] << 16);
        x_row[19] = __uint_as_float(x_w[9] & 4294901760u);
        dots_p[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[0]));
        dots_p[0] = _fma_29;
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[0]));
        dots_p[0] = _fma_30;
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[0]));
        dots_p[0] = _fma_31;
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[0]));
        dots_p[0] = _fma_32;
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[0]));
        dots_p[0] = _fma_33;
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[0]));
        dots_p[0] = _fma_34;
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[0]));
        dots_p[0] = _fma_35;
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[0]));
        dots_p[0] = _fma_36;
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[0]));
        dots_p[0] = _fma_37;
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[0]));
        dots_p[0] = _fma_38;
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[0]));
        dots_p[0] = _fma_39;
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[0]));
        dots_p[0] = _fma_40;
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[0]));
        dots_p[0] = _fma_41;
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[0]));
        dots_p[0] = _fma_42;
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[0]));
        dots_p[0] = _fma_43;
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[0]));
        dots_p[0] = _fma_44;
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[0]));
        dots_p[0] = _fma_45;
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[0]));
        dots_p[0] = _fma_46;
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[0]));
        dots_p[0] = _fma_47;
        x_row[0] = __uint_as_float(x_w[10] << 16);
        x_row[1] = __uint_as_float(x_w[10] & 4294901760u);
        x_row[2] = __uint_as_float(x_w[11] << 16);
        x_row[3] = __uint_as_float(x_w[11] & 4294901760u);
        x_row[4] = __uint_as_float(x_w[12] << 16);
        x_row[5] = __uint_as_float(x_w[12] & 4294901760u);
        x_row[6] = __uint_as_float(x_w[13] << 16);
        x_row[7] = __uint_as_float(x_w[13] & 4294901760u);
        x_row[8] = __uint_as_float(x_w[14] << 16);
        x_row[9] = __uint_as_float(x_w[14] & 4294901760u);
        x_row[10] = __uint_as_float(x_w[15] << 16);
        x_row[11] = __uint_as_float(x_w[15] & 4294901760u);
        x_row[12] = __uint_as_float(x_w[16] << 16);
        x_row[13] = __uint_as_float(x_w[16] & 4294901760u);
        x_row[14] = __uint_as_float(x_w[17] << 16);
        x_row[15] = __uint_as_float(x_w[17] & 4294901760u);
        x_row[16] = __uint_as_float(x_w[18] << 16);
        x_row[17] = __uint_as_float(x_w[18] & 4294901760u);
        x_row[18] = __uint_as_float(x_w[19] << 16);
        x_row[19] = __uint_as_float(x_w[19] & 4294901760u);
        dots_p[1] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[1]));
        dots_p[1] = _fma_48;
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[1]));
        dots_p[1] = _fma_49;
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[1]));
        dots_p[1] = _fma_50;
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[1]));
        dots_p[1] = _fma_51;
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[1]));
        dots_p[1] = _fma_52;
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[1]));
        dots_p[1] = _fma_53;
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[1]));
        dots_p[1] = _fma_54;
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[1]));
        dots_p[1] = _fma_55;
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[1]));
        dots_p[1] = _fma_56;
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[1]));
        dots_p[1] = _fma_57;
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[1]));
        dots_p[1] = _fma_58;
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[1]));
        dots_p[1] = _fma_59;
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[1]));
        dots_p[1] = _fma_60;
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[1]));
        dots_p[1] = _fma_61;
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[1]));
        dots_p[1] = _fma_62;
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[1]));
        dots_p[1] = _fma_63;
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[1]));
        dots_p[1] = _fma_64;
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[1]));
        dots_p[1] = _fma_65;
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[1]));
        dots_p[1] = _fma_66;
        x_row[0] = __uint_as_float(x_w[20] << 16);
        x_row[1] = __uint_as_float(x_w[20] & 4294901760u);
        x_row[2] = __uint_as_float(x_w[21] << 16);
        x_row[3] = __uint_as_float(x_w[21] & 4294901760u);
        x_row[4] = __uint_as_float(x_w[22] << 16);
        x_row[5] = __uint_as_float(x_w[22] & 4294901760u);
        x_row[6] = __uint_as_float(x_w[23] << 16);
        x_row[7] = __uint_as_float(x_w[23] & 4294901760u);
        x_row[8] = __uint_as_float(x_w[24] << 16);
        x_row[9] = __uint_as_float(x_w[24] & 4294901760u);
        x_row[10] = __uint_as_float(x_w[25] << 16);
        x_row[11] = __uint_as_float(x_w[25] & 4294901760u);
        x_row[12] = __uint_as_float(x_w[26] << 16);
        x_row[13] = __uint_as_float(x_w[26] & 4294901760u);
        x_row[14] = __uint_as_float(x_w[27] << 16);
        x_row[15] = __uint_as_float(x_w[27] & 4294901760u);
        x_row[16] = __uint_as_float(x_w[28] << 16);
        x_row[17] = __uint_as_float(x_w[28] & 4294901760u);
        x_row[18] = __uint_as_float(x_w[29] << 16);
        x_row[19] = __uint_as_float(x_w[29] & 4294901760u);
        dots_p[2] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[2]));
        dots_p[2] = _fma_67;
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[2]));
        dots_p[2] = _fma_68;
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[2]));
        dots_p[2] = _fma_69;
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[2]));
        dots_p[2] = _fma_70;
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[2]));
        dots_p[2] = _fma_71;
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[2]));
        dots_p[2] = _fma_72;
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[2]));
        dots_p[2] = _fma_73;
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[2]));
        dots_p[2] = _fma_74;
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[2]));
        dots_p[2] = _fma_75;
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[2]));
        dots_p[2] = _fma_76;
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[2]));
        dots_p[2] = _fma_77;
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[2]));
        dots_p[2] = _fma_78;
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[2]));
        dots_p[2] = _fma_79;
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[2]));
        dots_p[2] = _fma_80;
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[2]));
        dots_p[2] = _fma_81;
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[2]));
        dots_p[2] = _fma_82;
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[2]));
        dots_p[2] = _fma_83;
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[2]));
        dots_p[2] = _fma_84;
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[2]));
        dots_p[2] = _fma_85;
        x_row[0] = __uint_as_float(x_w[30] << 16);
        x_row[1] = __uint_as_float(x_w[30] & 4294901760u);
        x_row[2] = __uint_as_float(x_w[31] << 16);
        x_row[3] = __uint_as_float(x_w[31] & 4294901760u);
        x_row[4] = __uint_as_float(x_w[32] << 16);
        x_row[5] = __uint_as_float(x_w[32] & 4294901760u);
        x_row[6] = __uint_as_float(x_w[33] << 16);
        x_row[7] = __uint_as_float(x_w[33] & 4294901760u);
        x_row[8] = __uint_as_float(x_w[34] << 16);
        x_row[9] = __uint_as_float(x_w[34] & 4294901760u);
        x_row[10] = __uint_as_float(x_w[35] << 16);
        x_row[11] = __uint_as_float(x_w[35] & 4294901760u);
        x_row[12] = __uint_as_float(x_w[36] << 16);
        x_row[13] = __uint_as_float(x_w[36] & 4294901760u);
        x_row[14] = __uint_as_float(x_w[37] << 16);
        x_row[15] = __uint_as_float(x_w[37] & 4294901760u);
        x_row[16] = __uint_as_float(x_w[38] << 16);
        x_row[17] = __uint_as_float(x_w[38] & 4294901760u);
        x_row[18] = __uint_as_float(x_w[39] << 16);
        x_row[19] = __uint_as_float(x_w[39] & 4294901760u);
        dots_p[3] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[3]));
        dots_p[3] = _fma_86;
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[3]));
        dots_p[3] = _fma_87;
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[3]));
        dots_p[3] = _fma_88;
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[3]));
        dots_p[3] = _fma_89;
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[3]));
        dots_p[3] = _fma_90;
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[3]));
        dots_p[3] = _fma_91;
        float _fma_92;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[3]));
        dots_p[3] = _fma_92;
        float _fma_93;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[3]));
        dots_p[3] = _fma_93;
        float _fma_94;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[3]));
        dots_p[3] = _fma_94;
        float _fma_95;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[3]));
        dots_p[3] = _fma_95;
        float _fma_96;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[3]));
        dots_p[3] = _fma_96;
        float _fma_97;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[3]));
        dots_p[3] = _fma_97;
        float _fma_98;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[3]));
        dots_p[3] = _fma_98;
        float _fma_99;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[3]));
        dots_p[3] = _fma_99;
        float _fma_100;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[3]));
        dots_p[3] = _fma_100;
        float _fma_101;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[3]));
        dots_p[3] = _fma_101;
        float _fma_102;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[3]));
        dots_p[3] = _fma_102;
        float _fma_103;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[3]));
        dots_p[3] = _fma_103;
        float _fma_104;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[3]));
        dots_p[3] = _fma_104;
        int lane8 = lane_1 & 7;
        int lane_hi = lane_1 >> 3;
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(keep), "f"(_shfl_xor_sync_105));
        float send_0 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_106));
        int upper_2 = lane_1 & 8;
        float send_3 = ((upper_2 != 0) ? add : add_1);
        float keep_4 = ((upper_2 != 0) ? add_1 : add);
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_4), "f"(_shfl_xor_sync_107));
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_108));
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(add_3), "f"(_shfl_xor_sync_109));
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, add_4, 1);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_4), "f"(_shfl_xor_sync_110));
        if (lane8 == 0) {
            dpre_part[cur_base + warp_0 * 4 + lane_hi] = add_5;
        }
        __syncthreads();
        if (warp_0 == 0) {
            float dp[1];
            dp[0] = dpre_part[cur_base + lcs];
            float _add_111;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lcs]));
            dp[0] = _add_111;
            float _add_112;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lcs]));
            dp[0] = _add_112;
            float _add_113;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lcs]));
            dp[0] = _add_113;
            float _add_114;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lcs]));
            dp[0] = _add_114;
            float _add_115;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lcs]));
            dp[0] = _add_115;
            float _add_116;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lcs]));
            dp[0] = _add_116;
            float _add_117;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lcs]));
            dp[0] = _add_117;
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dpre_in[t64 * 4ll + (long long)(s0 + lane_1)])), "f"(dp[0]), "r"((unsigned int)(lane_1 < 4)) : "memory");
        }
        gi[0] = gi[0] + 1;
    }
}

} // extern "C"
