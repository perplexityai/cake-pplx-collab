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

#define LAUNCH_MIN_BLOCKS 4

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
kernel_cake_mhc_train_collapse_bwd__vjpwd_sp4_mb4__h5120(__nv_bfloat16* __restrict__ x, float* __restrict__ pre_in, __nv_bfloat16* __restrict__ du, __nv_bfloat16* __restrict__ dx, float* __restrict__ dpre_in, int T)
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
    int lcs = lane_1 & 0;
    long long wcol = (long long)(warp_0 * 640) + (long long)(lane_1 * 8);
    long long col_off = (long long)tid * 20ll;
    int gi[1];
    gi[0] = 0;
    #pragma unroll 1
    for (int item = bid; item < T * 4; item += num_bids) {
        int row = item >> 2;
        int s0 = item & 3;
        long long t64 = (long long)row;
        int cur_base = (gi[0] & 1) * 32;
        float pin[1];
        {
            pin[0] = *reinterpret_cast<const float*>(pre_in + (t64 * 4ll + (long long)s0) + 0);
        }
        float du_row[20];
        unsigned int du_w[12];
        if (lane_1 < 80) {
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol)) + 0);
                du_w[0 + 0] = _uv4_0.x;
                du_w[0 + 1] = _uv4_0.y;
                du_w[0 + 2] = _uv4_0.z;
                du_w[0 + 3] = _uv4_0.w;
            }
        }
        if (lane_1 + 32 < 80) {
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol + 256ll)) + 0);
                du_w[4 + 0] = _uv4_1.x;
                du_w[4 + 1] = _uv4_1.y;
                du_w[4 + 2] = _uv4_1.z;
                du_w[4 + 3] = _uv4_1.w;
            }
        }
        if (lane_1 + 64 < 80) {
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol + 512ll)) + 0);
                du_w[8 + 0] = _uv4_2.x;
                du_w[8 + 1] = _uv4_2.y;
                du_w[8 + 2] = _uv4_2.z;
                du_w[8 + 3] = _uv4_2.w;
            }
        }
        unsigned int x_w[10];
        {
            const uint2* _ivptr_3 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off)) + 0);
            uint2 _ivld_3;
            _ivld_3 = *_ivptr_3;
            x_w[0 + 0] = _ivld_3.x;
            x_w[0 + 1] = _ivld_3.y;
        }
        {
            const uint2* _ivptr_4 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_4;
            _ivld_4 = *_ivptr_4;
            x_w[2 + 0] = _ivld_4.x;
            x_w[2 + 1] = _ivld_4.y;
        }
        {
            const uint2* _ivptr_5 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_5;
            _ivld_5 = *_ivptr_5;
            x_w[4 + 0] = _ivld_5.x;
            x_w[4 + 1] = _ivld_5.y;
        }
        {
            const uint2* _ivptr_6 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_6;
            _ivld_6 = *_ivptr_6;
            x_w[6 + 0] = _ivld_6.x;
            x_w[6 + 1] = _ivld_6.y;
        }
        {
            const uint2* _ivptr_7 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + (long long)s0) * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_7;
            _ivld_7 = *_ivptr_7;
            x_w[8 + 0] = _ivld_7.x;
            x_w[8 + 1] = _ivld_7.y;
        }
        unsigned int own[10];
        {
            const uint2* _ivptr_8 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off)) + 0);
            uint2 _ivld_8;
            _ivld_8 = *_ivptr_8;
            own[0 + 0] = _ivld_8.x;
            own[0 + 1] = _ivld_8.y;
        }
        {
            const uint2* _ivptr_9 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 4ll)) + 0);
            uint2 _ivld_9;
            _ivld_9 = *_ivptr_9;
            own[2 + 0] = _ivld_9.x;
            own[2 + 1] = _ivld_9.y;
        }
        {
            const uint2* _ivptr_10 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 8ll)) + 0);
            uint2 _ivld_10;
            _ivld_10 = *_ivptr_10;
            own[4 + 0] = _ivld_10.x;
            own[4 + 1] = _ivld_10.y;
        }
        {
            const uint2* _ivptr_11 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 12ll)) + 0);
            uint2 _ivld_11;
            _ivld_11 = *_ivptr_11;
            own[6 + 0] = _ivld_11.x;
            own[6 + 1] = _ivld_11.y;
        }
        {
            const uint2* _ivptr_12 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + col_off + 16ll)) + 0);
            uint2 _ivld_12;
            _ivld_12 = *_ivptr_12;
            own[8 + 0] = _ivld_12.x;
            own[8 + 1] = _ivld_12.y;
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
            unsigned int dw4_1[4];
            dw4_1[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[0]), mul_f32_rn_noftz(pin[0], dua_1[1]))));
            dw4_1[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[2]), mul_f32_rn_noftz(pin[0], dua_1[3]))));
            dw4_1[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[4]), mul_f32_rn_noftz(pin[0], dua_1[5]))));
            dw4_1[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[6]), mul_f32_rn_noftz(pin[0], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)s0) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_1)[0];
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
            unsigned int dw4_2[4];
            dw4_2[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[0]), mul_f32_rn_noftz(pin[0], dua_2[1]))));
            dw4_2[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[2]), mul_f32_rn_noftz(pin[0], dua_2[3]))));
            dw4_2[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[4]), mul_f32_rn_noftz(pin[0], dua_2[5]))));
            dw4_2[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[6]), mul_f32_rn_noftz(pin[0], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + (long long)s0) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_2)[0];
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
        float dots_p[1];
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
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[0]));
        dots_p[0] = _fma_13;
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[0]));
        dots_p[0] = _fma_14;
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[0]));
        dots_p[0] = _fma_15;
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[0]));
        dots_p[0] = _fma_16;
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[0]));
        dots_p[0] = _fma_17;
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[0]));
        dots_p[0] = _fma_18;
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[0]));
        dots_p[0] = _fma_19;
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[0]));
        dots_p[0] = _fma_20;
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[0]));
        dots_p[0] = _fma_21;
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[0]));
        dots_p[0] = _fma_22;
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[0]));
        dots_p[0] = _fma_23;
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[0]));
        dots_p[0] = _fma_24;
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[0]));
        dots_p[0] = _fma_25;
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[0]));
        dots_p[0] = _fma_26;
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[0]));
        dots_p[0] = _fma_27;
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[0]));
        dots_p[0] = _fma_28;
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[0]));
        dots_p[0] = _fma_29;
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[0]));
        dots_p[0] = _fma_30;
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[0]));
        dots_p[0] = _fma_31;
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, dots_p[0], 16);
        float b16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(b16) : "f"(dots_p[0]), "f"(_shfl_xor_sync_32));
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, b16, 8);
        float b8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(b8) : "f"(b16), "f"(_shfl_xor_sync_33));
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, b8, 4);
        float b4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(b4) : "f"(b8), "f"(_shfl_xor_sync_34));
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, b4, 2);
        float b2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(b2) : "f"(b4), "f"(_shfl_xor_sync_35));
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, b2, 1);
        float b1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(b1) : "f"(b2), "f"(_shfl_xor_sync_36));
        if (lane_1 == 0) {
            dpre_part[cur_base + warp_0 * 4] = b1;
        }
        __syncthreads();
        if (warp_0 == 0) {
            float dp[1];
            dp[0] = dpre_part[cur_base + lcs];
            float _add_37;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lcs]));
            dp[0] = _add_37;
            float _add_38;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lcs]));
            dp[0] = _add_38;
            float _add_39;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lcs]));
            dp[0] = _add_39;
            float _add_40;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lcs]));
            dp[0] = _add_40;
            float _add_41;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lcs]));
            dp[0] = _add_41;
            float _add_42;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lcs]));
            dp[0] = _add_42;
            float _add_43;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lcs]));
            dp[0] = _add_43;
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dpre_in[t64 * 4ll + (long long)(s0 + lane_1)])), "f"(dp[0]), "r"((unsigned int)(lane_1 < 1)) : "memory");
        }
        gi[0] = gi[0] + 1;
    }
}

} // extern "C"
