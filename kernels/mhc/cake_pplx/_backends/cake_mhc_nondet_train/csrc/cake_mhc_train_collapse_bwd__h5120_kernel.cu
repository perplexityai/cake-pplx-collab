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
kernel_cake_mhc_train_collapse_bwd__h5120(__nv_bfloat16* __restrict__ x, float* __restrict__ pre_in, __nv_bfloat16* __restrict__ du, __nv_bfloat16* __restrict__ dx, float* __restrict__ dpre_in, int T)
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
    unsigned int* stage = reinterpret_cast<unsigned int*>(smem_raw + 256);
    const int stage_addr = smem + 256;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = (int)warp;
    int lane_1 = (int)lane;
    int lane8 = lane_1 & 7;
    int lane_hi = lane_1 >> 3;
    int lc4 = ((lane_1 < 4) ? lane_1 : 3);
    long long col_off = (long long)tid * 20ll;
    unsigned int wst = stage_addr + (unsigned int)(warp_0 * 6400);
    long long wcol = (long long)(warp_0 * 640) + (long long)(lane_1 * 8);
    unsigned int own_b = (unsigned int)(lane_1 * 20 * 2);
    int gi[1];
    gi[0] = 0;
    #pragma unroll 1
    for (int row = bid; row < T; row += num_bids) {
        long long t64 = (long long)row;
        int cur_base = (gi[0] & 1) * 32;
        float pin[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(pre_in + t64 * 4ll + 0);
            pin[0 + 0] = _v4.x;
            pin[0 + 1] = _v4.y;
            pin[0 + 2] = _v4.z;
            pin[0 + 3] = _v4.w;
        }
        float du_row[20];
        unsigned int du_w[12];
        unsigned int x_w[48];
        if (lane_1 < 80) {
            {
                uint32_t _uv4_1_0;
                uint32_t _uv4_1_1;
                uint32_t _uv4_1_2;
                uint32_t _uv4_1_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                du_w[0 + 0] = _uv4_1_0;
                du_w[0 + 1] = _uv4_1_1;
                du_w[0 + 2] = _uv4_1_2;
                du_w[0 + 3] = _uv4_1_3;
            }
            {
                uint32_t _uv4_2_0;
                uint32_t _uv4_2_1;
                uint32_t _uv4_2_2;
                uint32_t _uv4_2_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (t64 * 4ll * 5120ll + wcol)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[0 + 0] = _uv4_2_0;
                x_w[0 + 1] = _uv4_2_1;
                x_w[0 + 2] = _uv4_2_2;
                x_w[0 + 3] = _uv4_2_3;
            }
            {
                uint32_t _uv4_3_0;
                uint32_t _uv4_3_1;
                uint32_t _uv4_3_2;
                uint32_t _uv4_3_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 1ll) * 5120ll + wcol)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[4 + 0] = _uv4_3_0;
                x_w[4 + 1] = _uv4_3_1;
                x_w[4 + 2] = _uv4_3_2;
                x_w[4 + 3] = _uv4_3_3;
            }
            {
                uint32_t _uv4_4_0;
                uint32_t _uv4_4_1;
                uint32_t _uv4_4_2;
                uint32_t _uv4_4_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_4_0), "=r"(_uv4_4_1), "=r"(_uv4_4_2), "=r"(_uv4_4_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 2ll) * 5120ll + wcol)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[8 + 0] = _uv4_4_0;
                x_w[8 + 1] = _uv4_4_1;
                x_w[8 + 2] = _uv4_4_2;
                x_w[8 + 3] = _uv4_4_3;
            }
            {
                uint32_t _uv4_5_0;
                uint32_t _uv4_5_1;
                uint32_t _uv4_5_2;
                uint32_t _uv4_5_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_5_0), "=r"(_uv4_5_1), "=r"(_uv4_5_2), "=r"(_uv4_5_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 3ll) * 5120ll + wcol)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[12 + 0] = _uv4_5_0;
                x_w[12 + 1] = _uv4_5_1;
                x_w[12 + 2] = _uv4_5_2;
                x_w[12 + 3] = _uv4_5_3;
            }
        }
        if (lane_1 + 32 < 80) {
            {
                uint32_t _uv4_6_0;
                uint32_t _uv4_6_1;
                uint32_t _uv4_6_2;
                uint32_t _uv4_6_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_6_0), "=r"(_uv4_6_1), "=r"(_uv4_6_2), "=r"(_uv4_6_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol + 256ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                du_w[4 + 0] = _uv4_6_0;
                du_w[4 + 1] = _uv4_6_1;
                du_w[4 + 2] = _uv4_6_2;
                du_w[4 + 3] = _uv4_6_3;
            }
            {
                uint32_t _uv4_7_0;
                uint32_t _uv4_7_1;
                uint32_t _uv4_7_2;
                uint32_t _uv4_7_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_7_0), "=r"(_uv4_7_1), "=r"(_uv4_7_2), "=r"(_uv4_7_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (t64 * 4ll * 5120ll + wcol + 256ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[16 + 0] = _uv4_7_0;
                x_w[16 + 1] = _uv4_7_1;
                x_w[16 + 2] = _uv4_7_2;
                x_w[16 + 3] = _uv4_7_3;
            }
            {
                uint32_t _uv4_8_0;
                uint32_t _uv4_8_1;
                uint32_t _uv4_8_2;
                uint32_t _uv4_8_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_8_0), "=r"(_uv4_8_1), "=r"(_uv4_8_2), "=r"(_uv4_8_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 1ll) * 5120ll + wcol + 256ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[20 + 0] = _uv4_8_0;
                x_w[20 + 1] = _uv4_8_1;
                x_w[20 + 2] = _uv4_8_2;
                x_w[20 + 3] = _uv4_8_3;
            }
            {
                uint32_t _uv4_9_0;
                uint32_t _uv4_9_1;
                uint32_t _uv4_9_2;
                uint32_t _uv4_9_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_9_0), "=r"(_uv4_9_1), "=r"(_uv4_9_2), "=r"(_uv4_9_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 2ll) * 5120ll + wcol + 256ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[24 + 0] = _uv4_9_0;
                x_w[24 + 1] = _uv4_9_1;
                x_w[24 + 2] = _uv4_9_2;
                x_w[24 + 3] = _uv4_9_3;
            }
            {
                uint32_t _uv4_10_0;
                uint32_t _uv4_10_1;
                uint32_t _uv4_10_2;
                uint32_t _uv4_10_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_10_0), "=r"(_uv4_10_1), "=r"(_uv4_10_2), "=r"(_uv4_10_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 3ll) * 5120ll + wcol + 256ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[28 + 0] = _uv4_10_0;
                x_w[28 + 1] = _uv4_10_1;
                x_w[28 + 2] = _uv4_10_2;
                x_w[28 + 3] = _uv4_10_3;
            }
        }
        if (lane_1 + 64 < 80) {
            {
                uint32_t _uv4_11_0;
                uint32_t _uv4_11_1;
                uint32_t _uv4_11_2;
                uint32_t _uv4_11_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_11_0), "=r"(_uv4_11_1), "=r"(_uv4_11_2), "=r"(_uv4_11_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(du + (t64 * 5120ll + wcol + 512ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                du_w[8 + 0] = _uv4_11_0;
                du_w[8 + 1] = _uv4_11_1;
                du_w[8 + 2] = _uv4_11_2;
                du_w[8 + 3] = _uv4_11_3;
            }
            {
                uint32_t _uv4_12_0;
                uint32_t _uv4_12_1;
                uint32_t _uv4_12_2;
                uint32_t _uv4_12_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_12_0), "=r"(_uv4_12_1), "=r"(_uv4_12_2), "=r"(_uv4_12_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (t64 * 4ll * 5120ll + wcol + 512ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[32 + 0] = _uv4_12_0;
                x_w[32 + 1] = _uv4_12_1;
                x_w[32 + 2] = _uv4_12_2;
                x_w[32 + 3] = _uv4_12_3;
            }
            {
                uint32_t _uv4_13_0;
                uint32_t _uv4_13_1;
                uint32_t _uv4_13_2;
                uint32_t _uv4_13_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_13_0), "=r"(_uv4_13_1), "=r"(_uv4_13_2), "=r"(_uv4_13_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 1ll) * 5120ll + wcol + 512ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[36 + 0] = _uv4_13_0;
                x_w[36 + 1] = _uv4_13_1;
                x_w[36 + 2] = _uv4_13_2;
                x_w[36 + 3] = _uv4_13_3;
            }
            {
                uint32_t _uv4_14_0;
                uint32_t _uv4_14_1;
                uint32_t _uv4_14_2;
                uint32_t _uv4_14_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_14_0), "=r"(_uv4_14_1), "=r"(_uv4_14_2), "=r"(_uv4_14_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 2ll) * 5120ll + wcol + 512ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[40 + 0] = _uv4_14_0;
                x_w[40 + 1] = _uv4_14_1;
                x_w[40 + 2] = _uv4_14_2;
                x_w[40 + 3] = _uv4_14_3;
            }
            {
                uint32_t _uv4_15_0;
                uint32_t _uv4_15_1;
                uint32_t _uv4_15_2;
                uint32_t _uv4_15_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_15_0), "=r"(_uv4_15_1), "=r"(_uv4_15_2), "=r"(_uv4_15_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + ((t64 * 4ll + 3ll) * 5120ll + wcol + 512ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                x_w[44 + 0] = _uv4_15_0;
                x_w[44 + 1] = _uv4_15_1;
                x_w[44 + 2] = _uv4_15_2;
                x_w[44 + 3] = _uv4_15_3;
            }
        }
        if (lane_1 < 80) {
            float duf_0[8];
            duf_0[0] = __uint_as_float(du_w[0] << 16);
            duf_0[1] = __uint_as_float(du_w[0] & 4294901760u);
            duf_0[2] = __uint_as_float(du_w[1] << 16);
            duf_0[3] = __uint_as_float(du_w[1] & 4294901760u);
            duf_0[4] = __uint_as_float(du_w[2] << 16);
            duf_0[5] = __uint_as_float(du_w[2] & 4294901760u);
            duf_0[6] = __uint_as_float(du_w[3] << 16);
            duf_0[7] = __uint_as_float(du_w[3] & 4294901760u);
            unsigned int dw4_0[4];
            dw4_0[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_0[0]), mul_f32_rn_noftz(pin[0], duf_0[1]))));
            dw4_0[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_0[2]), mul_f32_rn_noftz(pin[0], duf_0[3]))));
            dw4_0[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_0[4]), mul_f32_rn_noftz(pin[0], duf_0[5]))));
            dw4_0[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_0[6]), mul_f32_rn_noftz(pin[0], duf_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + (t64 * 4ll * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_0)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(16 * lane_1)), "r"(x_w[0]), "r"(x_w[1]), "r"(x_w[2]), "r"(x_w[3]) : "memory");
            unsigned int dw4_1[4];
            dw4_1[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_0[0]), mul_f32_rn_noftz(pin[1], duf_0[1]))));
            dw4_1[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_0[2]), mul_f32_rn_noftz(pin[1], duf_0[3]))));
            dw4_1[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_0[4]), mul_f32_rn_noftz(pin[1], duf_0[5]))));
            dw4_1[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_0[6]), mul_f32_rn_noftz(pin[1], duf_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 1ll) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_1)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(1280 + 16 * lane_1)), "r"(x_w[4]), "r"(x_w[5]), "r"(x_w[6]), "r"(x_w[7]) : "memory");
            unsigned int dw4_2[4];
            dw4_2[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_0[0]), mul_f32_rn_noftz(pin[2], duf_0[1]))));
            dw4_2[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_0[2]), mul_f32_rn_noftz(pin[2], duf_0[3]))));
            dw4_2[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_0[4]), mul_f32_rn_noftz(pin[2], duf_0[5]))));
            dw4_2[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_0[6]), mul_f32_rn_noftz(pin[2], duf_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 2ll) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_2)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(2560 + 16 * lane_1)), "r"(x_w[8]), "r"(x_w[9]), "r"(x_w[10]), "r"(x_w[11]) : "memory");
            unsigned int dw4_3[4];
            dw4_3[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_0[0]), mul_f32_rn_noftz(pin[3], duf_0[1]))));
            dw4_3[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_0[2]), mul_f32_rn_noftz(pin[3], duf_0[3]))));
            dw4_3[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_0[4]), mul_f32_rn_noftz(pin[3], duf_0[5]))));
            dw4_3[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_0[6]), mul_f32_rn_noftz(pin[3], duf_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 3ll) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_3)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(3840 + 16 * lane_1)), "r"(x_w[12]), "r"(x_w[13]), "r"(x_w[14]), "r"(x_w[15]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(5120 + 16 * lane_1)), "r"(du_w[0]), "r"(du_w[1]), "r"(du_w[2]), "r"(du_w[3]) : "memory");
        }
        if (lane_1 + 32 < 80) {
            float duf_1[8];
            duf_1[0] = __uint_as_float(du_w[4] << 16);
            duf_1[1] = __uint_as_float(du_w[4] & 4294901760u);
            duf_1[2] = __uint_as_float(du_w[5] << 16);
            duf_1[3] = __uint_as_float(du_w[5] & 4294901760u);
            duf_1[4] = __uint_as_float(du_w[6] << 16);
            duf_1[5] = __uint_as_float(du_w[6] & 4294901760u);
            duf_1[6] = __uint_as_float(du_w[7] << 16);
            duf_1[7] = __uint_as_float(du_w[7] & 4294901760u);
            unsigned int dw4_4[4];
            dw4_4[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_1[0]), mul_f32_rn_noftz(pin[0], duf_1[1]))));
            dw4_4[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_1[2]), mul_f32_rn_noftz(pin[0], duf_1[3]))));
            dw4_4[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_1[4]), mul_f32_rn_noftz(pin[0], duf_1[5]))));
            dw4_4[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_1[6]), mul_f32_rn_noftz(pin[0], duf_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + (t64 * 4ll * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_4)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(512 + 16 * lane_1)), "r"(x_w[16]), "r"(x_w[17]), "r"(x_w[18]), "r"(x_w[19]) : "memory");
            unsigned int dw4_5[4];
            dw4_5[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_1[0]), mul_f32_rn_noftz(pin[1], duf_1[1]))));
            dw4_5[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_1[2]), mul_f32_rn_noftz(pin[1], duf_1[3]))));
            dw4_5[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_1[4]), mul_f32_rn_noftz(pin[1], duf_1[5]))));
            dw4_5[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_1[6]), mul_f32_rn_noftz(pin[1], duf_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 1ll) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_5)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(1792 + 16 * lane_1)), "r"(x_w[20]), "r"(x_w[21]), "r"(x_w[22]), "r"(x_w[23]) : "memory");
            unsigned int dw4_6[4];
            dw4_6[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_1[0]), mul_f32_rn_noftz(pin[2], duf_1[1]))));
            dw4_6[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_1[2]), mul_f32_rn_noftz(pin[2], duf_1[3]))));
            dw4_6[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_1[4]), mul_f32_rn_noftz(pin[2], duf_1[5]))));
            dw4_6[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_1[6]), mul_f32_rn_noftz(pin[2], duf_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 2ll) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_6)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(3072 + 16 * lane_1)), "r"(x_w[24]), "r"(x_w[25]), "r"(x_w[26]), "r"(x_w[27]) : "memory");
            unsigned int dw4_7[4];
            dw4_7[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_1[0]), mul_f32_rn_noftz(pin[3], duf_1[1]))));
            dw4_7[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_1[2]), mul_f32_rn_noftz(pin[3], duf_1[3]))));
            dw4_7[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_1[4]), mul_f32_rn_noftz(pin[3], duf_1[5]))));
            dw4_7[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_1[6]), mul_f32_rn_noftz(pin[3], duf_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 3ll) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_7)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(4352 + 16 * lane_1)), "r"(x_w[28]), "r"(x_w[29]), "r"(x_w[30]), "r"(x_w[31]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(5632 + 16 * lane_1)), "r"(du_w[4]), "r"(du_w[5]), "r"(du_w[6]), "r"(du_w[7]) : "memory");
        }
        if (lane_1 + 64 < 80) {
            float duf_2[8];
            duf_2[0] = __uint_as_float(du_w[8] << 16);
            duf_2[1] = __uint_as_float(du_w[8] & 4294901760u);
            duf_2[2] = __uint_as_float(du_w[9] << 16);
            duf_2[3] = __uint_as_float(du_w[9] & 4294901760u);
            duf_2[4] = __uint_as_float(du_w[10] << 16);
            duf_2[5] = __uint_as_float(du_w[10] & 4294901760u);
            duf_2[6] = __uint_as_float(du_w[11] << 16);
            duf_2[7] = __uint_as_float(du_w[11] & 4294901760u);
            unsigned int dw4_8[4];
            dw4_8[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_2[0]), mul_f32_rn_noftz(pin[0], duf_2[1]))));
            dw4_8[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_2[2]), mul_f32_rn_noftz(pin[0], duf_2[3]))));
            dw4_8[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_2[4]), mul_f32_rn_noftz(pin[0], duf_2[5]))));
            dw4_8[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], duf_2[6]), mul_f32_rn_noftz(pin[0], duf_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + (t64 * 4ll * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_8)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(1024 + 16 * lane_1)), "r"(x_w[32]), "r"(x_w[33]), "r"(x_w[34]), "r"(x_w[35]) : "memory");
            unsigned int dw4_9[4];
            dw4_9[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_2[0]), mul_f32_rn_noftz(pin[1], duf_2[1]))));
            dw4_9[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_2[2]), mul_f32_rn_noftz(pin[1], duf_2[3]))));
            dw4_9[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_2[4]), mul_f32_rn_noftz(pin[1], duf_2[5]))));
            dw4_9[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], duf_2[6]), mul_f32_rn_noftz(pin[1], duf_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 1ll) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_9)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(2304 + 16 * lane_1)), "r"(x_w[36]), "r"(x_w[37]), "r"(x_w[38]), "r"(x_w[39]) : "memory");
            unsigned int dw4_10[4];
            dw4_10[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_2[0]), mul_f32_rn_noftz(pin[2], duf_2[1]))));
            dw4_10[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_2[2]), mul_f32_rn_noftz(pin[2], duf_2[3]))));
            dw4_10[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_2[4]), mul_f32_rn_noftz(pin[2], duf_2[5]))));
            dw4_10[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], duf_2[6]), mul_f32_rn_noftz(pin[2], duf_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 2ll) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_10)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(3584 + 16 * lane_1)), "r"(x_w[40]), "r"(x_w[41]), "r"(x_w[42]), "r"(x_w[43]) : "memory");
            unsigned int dw4_11[4];
            dw4_11[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_2[0]), mul_f32_rn_noftz(pin[3], duf_2[1]))));
            dw4_11[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_2[2]), mul_f32_rn_noftz(pin[3], duf_2[3]))));
            dw4_11[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_2[4]), mul_f32_rn_noftz(pin[3], duf_2[5]))));
            dw4_11[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], duf_2[6]), mul_f32_rn_noftz(pin[3], duf_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 3ll) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_11)[0];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(4864 + 16 * lane_1)), "r"(x_w[44]), "r"(x_w[45]), "r"(x_w[46]), "r"(x_w[47]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wst + (unsigned int)(6144 + 16 * lane_1)), "r"(du_w[8]), "r"(du_w[9]), "r"(du_w[10]), "r"(du_w[11]) : "memory");
        }
        __syncwarp();
        unsigned int own[10];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1]))
            : "r"(wst + 5120 + own_b + 0));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&own[2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(2) + 1]))
            : "r"(wst + 5120 + own_b + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&own[4])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 1]))
            : "r"(wst + 5120 + own_b + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&own[6])), "=r"(*reinterpret_cast<uint32_t*>(&own[(6) + 1]))
            : "r"(wst + 5120 + own_b + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&own[8])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 1]))
            : "r"(wst + 5120 + own_b + 32));
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
        long long xrow_0 = t64 * 4ll * 5120ll + col_off;
        unsigned int ownx_0[10];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[(0) + 1]))
            : "r"(wst + 0 + own_b + 0));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[(2) + 1]))
            : "r"(wst + 0 + own_b + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[(4) + 1]))
            : "r"(wst + 0 + own_b + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[(6) + 1]))
            : "r"(wst + 0 + own_b + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_0[(8) + 1]))
            : "r"(wst + 0 + own_b + 32));
        x_row[0] = __uint_as_float(ownx_0[0] << 16);
        x_row[1] = __uint_as_float(ownx_0[0] & 4294901760u);
        x_row[2] = __uint_as_float(ownx_0[1] << 16);
        x_row[3] = __uint_as_float(ownx_0[1] & 4294901760u);
        x_row[4] = __uint_as_float(ownx_0[2] << 16);
        x_row[5] = __uint_as_float(ownx_0[2] & 4294901760u);
        x_row[6] = __uint_as_float(ownx_0[3] << 16);
        x_row[7] = __uint_as_float(ownx_0[3] & 4294901760u);
        x_row[8] = __uint_as_float(ownx_0[4] << 16);
        x_row[9] = __uint_as_float(ownx_0[4] & 4294901760u);
        x_row[10] = __uint_as_float(ownx_0[5] << 16);
        x_row[11] = __uint_as_float(ownx_0[5] & 4294901760u);
        x_row[12] = __uint_as_float(ownx_0[6] << 16);
        x_row[13] = __uint_as_float(ownx_0[6] & 4294901760u);
        x_row[14] = __uint_as_float(ownx_0[7] << 16);
        x_row[15] = __uint_as_float(ownx_0[7] & 4294901760u);
        x_row[16] = __uint_as_float(ownx_0[8] << 16);
        x_row[17] = __uint_as_float(ownx_0[8] & 4294901760u);
        x_row[18] = __uint_as_float(ownx_0[9] << 16);
        x_row[19] = __uint_as_float(ownx_0[9] & 4294901760u);
        dots_p[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[0]));
        dots_p[0] = _fma_16;
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[0]));
        dots_p[0] = _fma_17;
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[0]));
        dots_p[0] = _fma_18;
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[0]));
        dots_p[0] = _fma_19;
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[0]));
        dots_p[0] = _fma_20;
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[0]));
        dots_p[0] = _fma_21;
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[0]));
        dots_p[0] = _fma_22;
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[0]));
        dots_p[0] = _fma_23;
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[0]));
        dots_p[0] = _fma_24;
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[0]));
        dots_p[0] = _fma_25;
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[0]));
        dots_p[0] = _fma_26;
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[0]));
        dots_p[0] = _fma_27;
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[0]));
        dots_p[0] = _fma_28;
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[0]));
        dots_p[0] = _fma_29;
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[0]));
        dots_p[0] = _fma_30;
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[0]));
        dots_p[0] = _fma_31;
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[0]));
        dots_p[0] = _fma_32;
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[0]));
        dots_p[0] = _fma_33;
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[0]));
        dots_p[0] = _fma_34;
        long long xrow_1 = (t64 * 4ll + 1ll) * 5120ll + col_off;
        unsigned int ownx_1[10];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[(0) + 1]))
            : "r"(wst + 1280 + own_b + 0));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[(2) + 1]))
            : "r"(wst + 1280 + own_b + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[(4) + 1]))
            : "r"(wst + 1280 + own_b + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[(6) + 1]))
            : "r"(wst + 1280 + own_b + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_1[(8) + 1]))
            : "r"(wst + 1280 + own_b + 32));
        x_row[0] = __uint_as_float(ownx_1[0] << 16);
        x_row[1] = __uint_as_float(ownx_1[0] & 4294901760u);
        x_row[2] = __uint_as_float(ownx_1[1] << 16);
        x_row[3] = __uint_as_float(ownx_1[1] & 4294901760u);
        x_row[4] = __uint_as_float(ownx_1[2] << 16);
        x_row[5] = __uint_as_float(ownx_1[2] & 4294901760u);
        x_row[6] = __uint_as_float(ownx_1[3] << 16);
        x_row[7] = __uint_as_float(ownx_1[3] & 4294901760u);
        x_row[8] = __uint_as_float(ownx_1[4] << 16);
        x_row[9] = __uint_as_float(ownx_1[4] & 4294901760u);
        x_row[10] = __uint_as_float(ownx_1[5] << 16);
        x_row[11] = __uint_as_float(ownx_1[5] & 4294901760u);
        x_row[12] = __uint_as_float(ownx_1[6] << 16);
        x_row[13] = __uint_as_float(ownx_1[6] & 4294901760u);
        x_row[14] = __uint_as_float(ownx_1[7] << 16);
        x_row[15] = __uint_as_float(ownx_1[7] & 4294901760u);
        x_row[16] = __uint_as_float(ownx_1[8] << 16);
        x_row[17] = __uint_as_float(ownx_1[8] & 4294901760u);
        x_row[18] = __uint_as_float(ownx_1[9] << 16);
        x_row[19] = __uint_as_float(ownx_1[9] & 4294901760u);
        dots_p[1] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[1]));
        dots_p[1] = _fma_35;
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[1]));
        dots_p[1] = _fma_36;
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[1]));
        dots_p[1] = _fma_37;
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[1]));
        dots_p[1] = _fma_38;
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[1]));
        dots_p[1] = _fma_39;
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[1]));
        dots_p[1] = _fma_40;
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[1]));
        dots_p[1] = _fma_41;
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[1]));
        dots_p[1] = _fma_42;
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[1]));
        dots_p[1] = _fma_43;
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[1]));
        dots_p[1] = _fma_44;
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[1]));
        dots_p[1] = _fma_45;
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[1]));
        dots_p[1] = _fma_46;
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[1]));
        dots_p[1] = _fma_47;
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[1]));
        dots_p[1] = _fma_48;
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[1]));
        dots_p[1] = _fma_49;
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[1]));
        dots_p[1] = _fma_50;
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[1]));
        dots_p[1] = _fma_51;
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[1]));
        dots_p[1] = _fma_52;
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[1]));
        dots_p[1] = _fma_53;
        long long xrow_2 = (t64 * 4ll + 2ll) * 5120ll + col_off;
        unsigned int ownx_2[10];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[(0) + 1]))
            : "r"(wst + 2560 + own_b + 0));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[(2) + 1]))
            : "r"(wst + 2560 + own_b + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[(4) + 1]))
            : "r"(wst + 2560 + own_b + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[(6) + 1]))
            : "r"(wst + 2560 + own_b + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_2[(8) + 1]))
            : "r"(wst + 2560 + own_b + 32));
        x_row[0] = __uint_as_float(ownx_2[0] << 16);
        x_row[1] = __uint_as_float(ownx_2[0] & 4294901760u);
        x_row[2] = __uint_as_float(ownx_2[1] << 16);
        x_row[3] = __uint_as_float(ownx_2[1] & 4294901760u);
        x_row[4] = __uint_as_float(ownx_2[2] << 16);
        x_row[5] = __uint_as_float(ownx_2[2] & 4294901760u);
        x_row[6] = __uint_as_float(ownx_2[3] << 16);
        x_row[7] = __uint_as_float(ownx_2[3] & 4294901760u);
        x_row[8] = __uint_as_float(ownx_2[4] << 16);
        x_row[9] = __uint_as_float(ownx_2[4] & 4294901760u);
        x_row[10] = __uint_as_float(ownx_2[5] << 16);
        x_row[11] = __uint_as_float(ownx_2[5] & 4294901760u);
        x_row[12] = __uint_as_float(ownx_2[6] << 16);
        x_row[13] = __uint_as_float(ownx_2[6] & 4294901760u);
        x_row[14] = __uint_as_float(ownx_2[7] << 16);
        x_row[15] = __uint_as_float(ownx_2[7] & 4294901760u);
        x_row[16] = __uint_as_float(ownx_2[8] << 16);
        x_row[17] = __uint_as_float(ownx_2[8] & 4294901760u);
        x_row[18] = __uint_as_float(ownx_2[9] << 16);
        x_row[19] = __uint_as_float(ownx_2[9] & 4294901760u);
        dots_p[2] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[2]));
        dots_p[2] = _fma_54;
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[2]));
        dots_p[2] = _fma_55;
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[2]));
        dots_p[2] = _fma_56;
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[2]));
        dots_p[2] = _fma_57;
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[2]));
        dots_p[2] = _fma_58;
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[2]));
        dots_p[2] = _fma_59;
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[2]));
        dots_p[2] = _fma_60;
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[2]));
        dots_p[2] = _fma_61;
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[2]));
        dots_p[2] = _fma_62;
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[2]));
        dots_p[2] = _fma_63;
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[2]));
        dots_p[2] = _fma_64;
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[2]));
        dots_p[2] = _fma_65;
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[2]));
        dots_p[2] = _fma_66;
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[2]));
        dots_p[2] = _fma_67;
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[2]));
        dots_p[2] = _fma_68;
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[2]));
        dots_p[2] = _fma_69;
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[2]));
        dots_p[2] = _fma_70;
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[2]));
        dots_p[2] = _fma_71;
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[2]));
        dots_p[2] = _fma_72;
        long long xrow_3 = (t64 * 4ll + 3ll) * 5120ll + col_off;
        unsigned int ownx_3[10];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[(0) + 1]))
            : "r"(wst + 3840 + own_b + 0));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[(2) + 1]))
            : "r"(wst + 3840 + own_b + 8));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[(4) + 1]))
            : "r"(wst + 3840 + own_b + 16));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[(6) + 1]))
            : "r"(wst + 3840 + own_b + 24));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ownx_3[(8) + 1]))
            : "r"(wst + 3840 + own_b + 32));
        x_row[0] = __uint_as_float(ownx_3[0] << 16);
        x_row[1] = __uint_as_float(ownx_3[0] & 4294901760u);
        x_row[2] = __uint_as_float(ownx_3[1] << 16);
        x_row[3] = __uint_as_float(ownx_3[1] & 4294901760u);
        x_row[4] = __uint_as_float(ownx_3[2] << 16);
        x_row[5] = __uint_as_float(ownx_3[2] & 4294901760u);
        x_row[6] = __uint_as_float(ownx_3[3] << 16);
        x_row[7] = __uint_as_float(ownx_3[3] & 4294901760u);
        x_row[8] = __uint_as_float(ownx_3[4] << 16);
        x_row[9] = __uint_as_float(ownx_3[4] & 4294901760u);
        x_row[10] = __uint_as_float(ownx_3[5] << 16);
        x_row[11] = __uint_as_float(ownx_3[5] & 4294901760u);
        x_row[12] = __uint_as_float(ownx_3[6] << 16);
        x_row[13] = __uint_as_float(ownx_3[6] & 4294901760u);
        x_row[14] = __uint_as_float(ownx_3[7] << 16);
        x_row[15] = __uint_as_float(ownx_3[7] & 4294901760u);
        x_row[16] = __uint_as_float(ownx_3[8] << 16);
        x_row[17] = __uint_as_float(ownx_3[8] & 4294901760u);
        x_row[18] = __uint_as_float(ownx_3[9] << 16);
        x_row[19] = __uint_as_float(ownx_3[9] & 4294901760u);
        dots_p[3] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[3]));
        dots_p[3] = _fma_73;
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[3]));
        dots_p[3] = _fma_74;
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[3]));
        dots_p[3] = _fma_75;
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[3]));
        dots_p[3] = _fma_76;
        float _fma_77;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[3]));
        dots_p[3] = _fma_77;
        float _fma_78;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[3]));
        dots_p[3] = _fma_78;
        float _fma_79;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[3]));
        dots_p[3] = _fma_79;
        float _fma_80;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[3]));
        dots_p[3] = _fma_80;
        float _fma_81;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[3]));
        dots_p[3] = _fma_81;
        float _fma_82;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[3]));
        dots_p[3] = _fma_82;
        float _fma_83;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[3]));
        dots_p[3] = _fma_83;
        float _fma_84;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[3]));
        dots_p[3] = _fma_84;
        float _fma_85;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[3]));
        dots_p[3] = _fma_85;
        float _fma_86;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[3]));
        dots_p[3] = _fma_86;
        float _fma_87;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[3]));
        dots_p[3] = _fma_87;
        float _fma_88;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[3]));
        dots_p[3] = _fma_88;
        float _fma_89;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[3]));
        dots_p[3] = _fma_89;
        float _fma_90;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[3]));
        dots_p[3] = _fma_90;
        float _fma_91;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[3]));
        dots_p[3] = _fma_91;
        __syncwarp();
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(keep), "f"(_shfl_xor_sync_92));
        float send_0 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_93));
        int upper_2 = lane_1 & 8;
        float send_3 = ((upper_2 != 0) ? add : add_1);
        float keep_4 = ((upper_2 != 0) ? add_1 : add);
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_4), "f"(_shfl_xor_sync_94));
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_95));
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(add_3), "f"(_shfl_xor_sync_96));
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, add_4, 1);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_4), "f"(_shfl_xor_sync_97));
        if (lane8 == 0) {
            dpre_part[cur_base + warp_0 * 4 + lane_hi] = add_5;
        }
        __syncthreads();
        if (warp_0 == 0) {
            float dp[1];
            dp[0] = dpre_part[cur_base + lc4];
            float _add_98;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lc4]));
            dp[0] = _add_98;
            float _add_99;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lc4]));
            dp[0] = _add_99;
            float _add_100;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lc4]));
            dp[0] = _add_100;
            float _add_101;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lc4]));
            dp[0] = _add_101;
            float _add_102;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lc4]));
            dp[0] = _add_102;
            float _add_103;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lc4]));
            dp[0] = _add_103;
            float _add_104;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lc4]));
            dp[0] = _add_104;
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dpre_in[t64 * 4ll + (long long)lane_1])), "f"(dp[0]), "r"((unsigned int)(lane_1 < 4)) : "memory");
        }
        gi[0] = gi[0] + 1;
    }
}

} // extern "C"
