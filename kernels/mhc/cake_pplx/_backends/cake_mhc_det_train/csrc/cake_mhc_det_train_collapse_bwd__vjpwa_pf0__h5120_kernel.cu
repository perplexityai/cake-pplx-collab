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
kernel_cake_mhc_det_train_collapse_bwd__vjpwa_pf0__h5120(__nv_bfloat16* __restrict__ x, float* __restrict__ pre_in, __nv_bfloat16* __restrict__ du, __nv_bfloat16* __restrict__ dx, float* __restrict__ dpre_in, int T)
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
        long long tp64 = (long long)row;
        int pl_0 = tid;
        if (row < T && pl_0 < 320) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + (tp64 * 20480ll + (long long)(pl_0 * 64))))); }
        if (row < T && pl_0 >= 320 && pl_0 < 400) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(du + (tp64 * 5120ll + (long long)((pl_0 - 320) * 64))))); }
        int pl_1 = tid + 256;
        if (row < T && pl_1 < 320) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + (tp64 * 20480ll + (long long)(pl_1 * 64))))); }
        if (row < T && pl_1 >= 320 && pl_1 < 400) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(du + (tp64 * 5120ll + (long long)((pl_1 - 320) * 64))))); }
        float pin[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(pre_in + t64 * 4ll + 0);
            pin[0 + 0] = _v4.x;
            pin[0 + 1] = _v4.y;
            pin[0 + 2] = _v4.z;
            pin[0 + 3] = _v4.w;
        }
        float du_row[20];
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 < 80) ? 1 : 0), "r"(wst + (unsigned int)(5120 + 16 * lane_1)), "l"(du + (t64 * 5120ll + wcol)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 32 < 80) ? 1 : 0), "r"(wst + (unsigned int)(5632 + 16 * lane_1)), "l"(du + (t64 * 5120ll + wcol + 256ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 64 < 80) ? 1 : 0), "r"(wst + (unsigned int)(6144 + 16 * lane_1)), "l"(du + (t64 * 5120ll + wcol + 512ll)), "l"(0x12F0000000000000ULL));
        asm volatile("cp.async.commit_group;");
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 < 80) ? 1 : 0), "r"(wst + (unsigned int)(16 * lane_1)), "l"(x + (t64 * 4ll * 5120ll + wcol)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 < 80) ? 1 : 0), "r"(wst + (unsigned int)(1280 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 1ll) * 5120ll + wcol)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 < 80) ? 1 : 0), "r"(wst + (unsigned int)(2560 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 2ll) * 5120ll + wcol)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 < 80) ? 1 : 0), "r"(wst + (unsigned int)(3840 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 3ll) * 5120ll + wcol)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 32 < 80) ? 1 : 0), "r"(wst + (unsigned int)(512 + 16 * lane_1)), "l"(x + (t64 * 4ll * 5120ll + wcol + 256ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 32 < 80) ? 1 : 0), "r"(wst + (unsigned int)(1792 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 1ll) * 5120ll + wcol + 256ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 32 < 80) ? 1 : 0), "r"(wst + (unsigned int)(3072 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 2ll) * 5120ll + wcol + 256ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 32 < 80) ? 1 : 0), "r"(wst + (unsigned int)(4352 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 3ll) * 5120ll + wcol + 256ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 64 < 80) ? 1 : 0), "r"(wst + (unsigned int)(1024 + 16 * lane_1)), "l"(x + (t64 * 4ll * 5120ll + wcol + 512ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 64 < 80) ? 1 : 0), "r"(wst + (unsigned int)(2304 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 1ll) * 5120ll + wcol + 512ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 64 < 80) ? 1 : 0), "r"(wst + (unsigned int)(3584 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 2ll) * 5120ll + wcol + 512ll)), "l"(0x12F0000000000000ULL));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
            "}"
            :: "r"((lane_1 + 64 < 80) ? 1 : 0), "r"(wst + (unsigned int)(4864 + 16 * lane_1)), "l"(x + ((t64 * 4ll + 3ll) * 5120ll + wcol + 512ll)), "l"(0x12F0000000000000ULL));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        if (lane_1 < 80) {
            unsigned int dwa_0[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_0[(0) + 3]))
                : "r"(wst + (unsigned int)(5120 + 16 * lane_1)));
            float dua_0[8];
            dua_0[0] = __uint_as_float(dwa_0[0] << 16);
            dua_0[1] = __uint_as_float(dwa_0[0] & 4294901760u);
            dua_0[2] = __uint_as_float(dwa_0[1] << 16);
            dua_0[3] = __uint_as_float(dwa_0[1] & 4294901760u);
            dua_0[4] = __uint_as_float(dwa_0[2] << 16);
            dua_0[5] = __uint_as_float(dwa_0[2] & 4294901760u);
            dua_0[6] = __uint_as_float(dwa_0[3] << 16);
            dua_0[7] = __uint_as_float(dwa_0[3] & 4294901760u);
            unsigned int dw4_0[4];
            dw4_0[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[0]), mul_f32_rn_noftz(pin[0], dua_0[1]))));
            dw4_0[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[2]), mul_f32_rn_noftz(pin[0], dua_0[3]))));
            dw4_0[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[4]), mul_f32_rn_noftz(pin[0], dua_0[5]))));
            dw4_0[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_0[6]), mul_f32_rn_noftz(pin[0], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + (t64 * 4ll * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_0)[0];
            unsigned int dw4_1[4];
            dw4_1[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[0]), mul_f32_rn_noftz(pin[1], dua_0[1]))));
            dw4_1[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[2]), mul_f32_rn_noftz(pin[1], dua_0[3]))));
            dw4_1[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[4]), mul_f32_rn_noftz(pin[1], dua_0[5]))));
            dw4_1[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_0[6]), mul_f32_rn_noftz(pin[1], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 1ll) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_1)[0];
            unsigned int dw4_2[4];
            dw4_2[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[0]), mul_f32_rn_noftz(pin[2], dua_0[1]))));
            dw4_2[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[2]), mul_f32_rn_noftz(pin[2], dua_0[3]))));
            dw4_2[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[4]), mul_f32_rn_noftz(pin[2], dua_0[5]))));
            dw4_2[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_0[6]), mul_f32_rn_noftz(pin[2], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 2ll) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_2)[0];
            unsigned int dw4_3[4];
            dw4_3[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[0]), mul_f32_rn_noftz(pin[3], dua_0[1]))));
            dw4_3[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[2]), mul_f32_rn_noftz(pin[3], dua_0[3]))));
            dw4_3[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[4]), mul_f32_rn_noftz(pin[3], dua_0[5]))));
            dw4_3[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_0[6]), mul_f32_rn_noftz(pin[3], dua_0[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 3ll) * 5120ll + wcol)))[0] = reinterpret_cast<int4*>(dw4_3)[0];
        }
        if (lane_1 + 32 < 80) {
            unsigned int dwa_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_1[(0) + 3]))
                : "r"(wst + (unsigned int)(5632 + 16 * lane_1)));
            float dua_1[8];
            dua_1[0] = __uint_as_float(dwa_1[0] << 16);
            dua_1[1] = __uint_as_float(dwa_1[0] & 4294901760u);
            dua_1[2] = __uint_as_float(dwa_1[1] << 16);
            dua_1[3] = __uint_as_float(dwa_1[1] & 4294901760u);
            dua_1[4] = __uint_as_float(dwa_1[2] << 16);
            dua_1[5] = __uint_as_float(dwa_1[2] & 4294901760u);
            dua_1[6] = __uint_as_float(dwa_1[3] << 16);
            dua_1[7] = __uint_as_float(dwa_1[3] & 4294901760u);
            unsigned int dw4_4[4];
            dw4_4[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[0]), mul_f32_rn_noftz(pin[0], dua_1[1]))));
            dw4_4[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[2]), mul_f32_rn_noftz(pin[0], dua_1[3]))));
            dw4_4[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[4]), mul_f32_rn_noftz(pin[0], dua_1[5]))));
            dw4_4[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_1[6]), mul_f32_rn_noftz(pin[0], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + (t64 * 4ll * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_4)[0];
            unsigned int dw4_5[4];
            dw4_5[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[0]), mul_f32_rn_noftz(pin[1], dua_1[1]))));
            dw4_5[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[2]), mul_f32_rn_noftz(pin[1], dua_1[3]))));
            dw4_5[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[4]), mul_f32_rn_noftz(pin[1], dua_1[5]))));
            dw4_5[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_1[6]), mul_f32_rn_noftz(pin[1], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 1ll) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_5)[0];
            unsigned int dw4_6[4];
            dw4_6[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[0]), mul_f32_rn_noftz(pin[2], dua_1[1]))));
            dw4_6[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[2]), mul_f32_rn_noftz(pin[2], dua_1[3]))));
            dw4_6[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[4]), mul_f32_rn_noftz(pin[2], dua_1[5]))));
            dw4_6[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_1[6]), mul_f32_rn_noftz(pin[2], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 2ll) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_6)[0];
            unsigned int dw4_7[4];
            dw4_7[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[0]), mul_f32_rn_noftz(pin[3], dua_1[1]))));
            dw4_7[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[2]), mul_f32_rn_noftz(pin[3], dua_1[3]))));
            dw4_7[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[4]), mul_f32_rn_noftz(pin[3], dua_1[5]))));
            dw4_7[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_1[6]), mul_f32_rn_noftz(pin[3], dua_1[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 3ll) * 5120ll + wcol + 256ll)))[0] = reinterpret_cast<int4*>(dw4_7)[0];
        }
        if (lane_1 + 64 < 80) {
            unsigned int dwa_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dwa_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dwa_2[(0) + 3]))
                : "r"(wst + (unsigned int)(6144 + 16 * lane_1)));
            float dua_2[8];
            dua_2[0] = __uint_as_float(dwa_2[0] << 16);
            dua_2[1] = __uint_as_float(dwa_2[0] & 4294901760u);
            dua_2[2] = __uint_as_float(dwa_2[1] << 16);
            dua_2[3] = __uint_as_float(dwa_2[1] & 4294901760u);
            dua_2[4] = __uint_as_float(dwa_2[2] << 16);
            dua_2[5] = __uint_as_float(dwa_2[2] & 4294901760u);
            dua_2[6] = __uint_as_float(dwa_2[3] << 16);
            dua_2[7] = __uint_as_float(dwa_2[3] & 4294901760u);
            unsigned int dw4_8[4];
            dw4_8[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[0]), mul_f32_rn_noftz(pin[0], dua_2[1]))));
            dw4_8[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[2]), mul_f32_rn_noftz(pin[0], dua_2[3]))));
            dw4_8[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[4]), mul_f32_rn_noftz(pin[0], dua_2[5]))));
            dw4_8[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], dua_2[6]), mul_f32_rn_noftz(pin[0], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + (t64 * 4ll * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_8)[0];
            unsigned int dw4_9[4];
            dw4_9[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[0]), mul_f32_rn_noftz(pin[1], dua_2[1]))));
            dw4_9[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[2]), mul_f32_rn_noftz(pin[1], dua_2[3]))));
            dw4_9[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[4]), mul_f32_rn_noftz(pin[1], dua_2[5]))));
            dw4_9[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], dua_2[6]), mul_f32_rn_noftz(pin[1], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 1ll) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_9)[0];
            unsigned int dw4_10[4];
            dw4_10[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[0]), mul_f32_rn_noftz(pin[2], dua_2[1]))));
            dw4_10[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[2]), mul_f32_rn_noftz(pin[2], dua_2[3]))));
            dw4_10[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[4]), mul_f32_rn_noftz(pin[2], dua_2[5]))));
            dw4_10[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], dua_2[6]), mul_f32_rn_noftz(pin[2], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 2ll) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_10)[0];
            unsigned int dw4_11[4];
            dw4_11[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[0]), mul_f32_rn_noftz(pin[3], dua_2[1]))));
            dw4_11[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[2]), mul_f32_rn_noftz(pin[3], dua_2[3]))));
            dw4_11[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[4]), mul_f32_rn_noftz(pin[3], dua_2[5]))));
            dw4_11[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], dua_2[6]), mul_f32_rn_noftz(pin[3], dua_2[7]))));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(dx + ((t64 * 4ll + 3ll) * 5120ll + wcol + 512ll)))[0] = reinterpret_cast<int4*>(dw4_11)[0];
        }
        asm volatile("cp.async.wait_group 0;");
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
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[0]));
        dots_p[0] = _fma_1;
        float _fma_2;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[0]));
        dots_p[0] = _fma_2;
        float _fma_3;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[0]));
        dots_p[0] = _fma_3;
        float _fma_4;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[0]));
        dots_p[0] = _fma_4;
        float _fma_5;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[0]));
        dots_p[0] = _fma_5;
        float _fma_6;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[0]));
        dots_p[0] = _fma_6;
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[0]));
        dots_p[0] = _fma_7;
        float _fma_8;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[0]));
        dots_p[0] = _fma_8;
        float _fma_9;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[0]));
        dots_p[0] = _fma_9;
        float _fma_10;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[0]));
        dots_p[0] = _fma_10;
        float _fma_11;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[0]));
        dots_p[0] = _fma_11;
        float _fma_12;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[0]));
        dots_p[0] = _fma_12;
        float _fma_13;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[0]));
        dots_p[0] = _fma_13;
        float _fma_14;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[0]));
        dots_p[0] = _fma_14;
        float _fma_15;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[0]));
        dots_p[0] = _fma_15;
        float _fma_16;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[0]));
        dots_p[0] = _fma_16;
        float _fma_17;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[0]));
        dots_p[0] = _fma_17;
        float _fma_18;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[0]));
        dots_p[0] = _fma_18;
        float _fma_19;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[0]));
        dots_p[0] = _fma_19;
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
        float _fma_20;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[1]));
        dots_p[1] = _fma_20;
        float _fma_21;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[1]));
        dots_p[1] = _fma_21;
        float _fma_22;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[1]));
        dots_p[1] = _fma_22;
        float _fma_23;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[1]));
        dots_p[1] = _fma_23;
        float _fma_24;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[1]));
        dots_p[1] = _fma_24;
        float _fma_25;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[1]));
        dots_p[1] = _fma_25;
        float _fma_26;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[1]));
        dots_p[1] = _fma_26;
        float _fma_27;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[1]));
        dots_p[1] = _fma_27;
        float _fma_28;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[1]));
        dots_p[1] = _fma_28;
        float _fma_29;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[1]));
        dots_p[1] = _fma_29;
        float _fma_30;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[1]));
        dots_p[1] = _fma_30;
        float _fma_31;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[1]));
        dots_p[1] = _fma_31;
        float _fma_32;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[1]));
        dots_p[1] = _fma_32;
        float _fma_33;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[1]));
        dots_p[1] = _fma_33;
        float _fma_34;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[1]));
        dots_p[1] = _fma_34;
        float _fma_35;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[1]));
        dots_p[1] = _fma_35;
        float _fma_36;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[1]));
        dots_p[1] = _fma_36;
        float _fma_37;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[1]));
        dots_p[1] = _fma_37;
        float _fma_38;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[1]));
        dots_p[1] = _fma_38;
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
        float _fma_39;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[2]));
        dots_p[2] = _fma_39;
        float _fma_40;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[2]));
        dots_p[2] = _fma_40;
        float _fma_41;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[2]));
        dots_p[2] = _fma_41;
        float _fma_42;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[2]));
        dots_p[2] = _fma_42;
        float _fma_43;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[2]));
        dots_p[2] = _fma_43;
        float _fma_44;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[2]));
        dots_p[2] = _fma_44;
        float _fma_45;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[2]));
        dots_p[2] = _fma_45;
        float _fma_46;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[2]));
        dots_p[2] = _fma_46;
        float _fma_47;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[2]));
        dots_p[2] = _fma_47;
        float _fma_48;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[2]));
        dots_p[2] = _fma_48;
        float _fma_49;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[2]));
        dots_p[2] = _fma_49;
        float _fma_50;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[2]));
        dots_p[2] = _fma_50;
        float _fma_51;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[2]));
        dots_p[2] = _fma_51;
        float _fma_52;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[2]));
        dots_p[2] = _fma_52;
        float _fma_53;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[2]));
        dots_p[2] = _fma_53;
        float _fma_54;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[2]));
        dots_p[2] = _fma_54;
        float _fma_55;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[2]));
        dots_p[2] = _fma_55;
        float _fma_56;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[2]));
        dots_p[2] = _fma_56;
        float _fma_57;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[2]));
        dots_p[2] = _fma_57;
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
        float _fma_58;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[3]));
        dots_p[3] = _fma_58;
        float _fma_59;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[3]));
        dots_p[3] = _fma_59;
        float _fma_60;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[3]));
        dots_p[3] = _fma_60;
        float _fma_61;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[3]));
        dots_p[3] = _fma_61;
        float _fma_62;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[3]));
        dots_p[3] = _fma_62;
        float _fma_63;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[3]));
        dots_p[3] = _fma_63;
        float _fma_64;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[3]));
        dots_p[3] = _fma_64;
        float _fma_65;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[3]));
        dots_p[3] = _fma_65;
        float _fma_66;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[3]));
        dots_p[3] = _fma_66;
        float _fma_67;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[3]));
        dots_p[3] = _fma_67;
        float _fma_68;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[3]));
        dots_p[3] = _fma_68;
        float _fma_69;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[3]));
        dots_p[3] = _fma_69;
        float _fma_70;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[3]));
        dots_p[3] = _fma_70;
        float _fma_71;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[3]));
        dots_p[3] = _fma_71;
        float _fma_72;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[3]));
        dots_p[3] = _fma_72;
        float _fma_73;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[3]));
        dots_p[3] = _fma_73;
        float _fma_74;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[3]));
        dots_p[3] = _fma_74;
        float _fma_75;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[3]));
        dots_p[3] = _fma_75;
        float _fma_76;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[3]));
        dots_p[3] = _fma_76;
        __syncwarp();
        int upper = lane_1 & 16;
        float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
        float add;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(keep), "f"(_shfl_xor_sync_77));
        float send_0 = ((upper != 0) ? dots_p[1] : dots_p[3]);
        float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_78));
        int upper_2 = lane_1 & 8;
        float send_3 = ((upper_2 != 0) ? add : add_1);
        float keep_4 = ((upper_2 != 0) ? add_1 : add);
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_4), "f"(_shfl_xor_sync_79));
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_80));
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(add_3), "f"(_shfl_xor_sync_81));
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, add_4, 1);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_4), "f"(_shfl_xor_sync_82));
        if (lane8 == 0) {
            dpre_part[cur_base + warp_0 * 4 + lane_hi] = add_5;
        }
        __syncthreads();
        if (warp_0 == 0) {
            float dp[1];
            dp[0] = dpre_part[cur_base + lc4];
            float _add_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(dp[0]), "f"(dpre_part[cur_base + 4 + lc4]));
            dp[0] = _add_83;
            float _add_84;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(dp[0]), "f"(dpre_part[cur_base + 8 + lc4]));
            dp[0] = _add_84;
            float _add_85;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(dp[0]), "f"(dpre_part[cur_base + 12 + lc4]));
            dp[0] = _add_85;
            float _add_86;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(dp[0]), "f"(dpre_part[cur_base + 16 + lc4]));
            dp[0] = _add_86;
            float _add_87;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(dp[0]), "f"(dpre_part[cur_base + 20 + lc4]));
            dp[0] = _add_87;
            float _add_88;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(dp[0]), "f"(dpre_part[cur_base + 24 + lc4]));
            dp[0] = _add_88;
            float _add_89;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(dp[0]), "f"(dpre_part[cur_base + 28 + lc4]));
            dp[0] = _add_89;
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dpre_in[t64 * 4ll + (long long)lane_1])), "f"(dp[0]), "r"((unsigned int)(lane_1 < 4)) : "memory");
        }
        gi[0] = gi[0] + 1;
    }
}

} // extern "C"
