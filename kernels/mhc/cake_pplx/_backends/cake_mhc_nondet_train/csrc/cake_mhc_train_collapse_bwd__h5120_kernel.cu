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
    float* dpre_comp_s = reinterpret_cast<float*>(smem_raw + 256);
    const int dpre_comp_s_addr = smem + 256;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = (int)warp;
    int lane_1 = (int)lane;
    int lane8 = lane_1 & 7;
    int lane_hi = lane_1 >> 3;
    int lc4 = ((lane_1 < 4) ? lane_1 : 3);
    long long col_off = (long long)tid * 20ll;
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
        {
            uint2 _vld_1;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(du + (t64 * 5120ll + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&du_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&du_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_2;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_2.x), "=r"(_vld_2.y) : "l"((const void*)(du + (t64 * 5120ll + col_off + 4ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&du_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&du_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_3;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_3.x), "=r"(_vld_3.y) : "l"((const void*)(du + (t64 * 5120ll + col_off + 8ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&du_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&du_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_4;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_4.x), "=r"(_vld_4.y) : "l"((const void*)(du + (t64 * 5120ll + col_off + 12ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&du_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&du_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_5;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_5.x), "=r"(_vld_5.y) : "l"((const void*)(du + (t64 * 5120ll + col_off + 16ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&du_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&du_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
        float dots_p[8];
        float x_row[20];
        long long xrow_0 = t64 * 4ll * 5120ll + col_off;
        {
            uint2 _vld_6;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_6.x), "=r"(_vld_6.y) : "l"((const void*)(x + xrow_0 + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_7;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_7.x), "=r"(_vld_7.y) : "l"((const void*)(x + (xrow_0 + 4ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_8;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_8.x), "=r"(_vld_8.y) : "l"((const void*)(x + (xrow_0 + 8ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_9;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_9.x), "=r"(_vld_9.y) : "l"((const void*)(x + (xrow_0 + 12ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_10;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_10.x), "=r"(_vld_10.y) : "l"((const void*)(x + (xrow_0 + 16ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
            }
        }
        float vacc_0[2];
        vacc_0[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        vacc_0[1] = 0.0f;
        float vprod_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_0) : "f"(x_row[1]), "f"(du_row[1]));
        float ts;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts) : "f"(vacc_0[0]), "f"(vprod_0));
        float tz;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz) : "f"(ts), "f"(vacc_0[0]));
        float _sub_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(ts), "f"(tz));
        float _sub_13;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(vacc_0[0]), "f"(_sub_14));
        float _sub_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(vprod_0), "f"(tz));
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_sub_13), "f"(_sub_15));
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(vacc_0[1]), "f"(_add_12));
        vacc_0[1] = _add_11;
        vacc_0[0] = ts;
        float vprod_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_1) : "f"(x_row[2]), "f"(du_row[2]));
        float ts_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_0) : "f"(vacc_0[0]), "f"(vprod_1));
        float tz_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_0) : "f"(ts_0), "f"(vacc_0[0]));
        float _sub_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(ts_0), "f"(tz_0));
        float _sub_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(vacc_0[0]), "f"(_sub_19));
        float _sub_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(vprod_1), "f"(tz_0));
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_sub_18), "f"(_sub_20));
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(vacc_0[1]), "f"(_add_17));
        vacc_0[1] = _add_16;
        vacc_0[0] = ts_0;
        float vprod_2;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_2) : "f"(x_row[3]), "f"(du_row[3]));
        float ts_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_1) : "f"(vacc_0[0]), "f"(vprod_2));
        float tz_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_1) : "f"(ts_1), "f"(vacc_0[0]));
        float _sub_24;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(ts_1), "f"(tz_1));
        float _sub_23;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(vacc_0[0]), "f"(_sub_24));
        float _sub_25;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(vprod_2), "f"(tz_1));
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_sub_23), "f"(_sub_25));
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(vacc_0[1]), "f"(_add_22));
        vacc_0[1] = _add_21;
        vacc_0[0] = ts_1;
        float vprod_3;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_3) : "f"(x_row[4]), "f"(du_row[4]));
        float ts_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_2) : "f"(vacc_0[0]), "f"(vprod_3));
        float tz_2;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_2) : "f"(ts_2), "f"(vacc_0[0]));
        float _sub_29;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(ts_2), "f"(tz_2));
        float _sub_28;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(vacc_0[0]), "f"(_sub_29));
        float _sub_30;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(vprod_3), "f"(tz_2));
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_sub_28), "f"(_sub_30));
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(vacc_0[1]), "f"(_add_27));
        vacc_0[1] = _add_26;
        vacc_0[0] = ts_2;
        float vprod_4;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_4) : "f"(x_row[5]), "f"(du_row[5]));
        float ts_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_3) : "f"(vacc_0[0]), "f"(vprod_4));
        float tz_3;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_3) : "f"(ts_3), "f"(vacc_0[0]));
        float _sub_34;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(ts_3), "f"(tz_3));
        float _sub_33;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(vacc_0[0]), "f"(_sub_34));
        float _sub_35;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(vprod_4), "f"(tz_3));
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_sub_33), "f"(_sub_35));
        float _add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(vacc_0[1]), "f"(_add_32));
        vacc_0[1] = _add_31;
        vacc_0[0] = ts_3;
        float vprod_5;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_5) : "f"(x_row[6]), "f"(du_row[6]));
        float ts_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_4) : "f"(vacc_0[0]), "f"(vprod_5));
        float tz_4;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_4) : "f"(ts_4), "f"(vacc_0[0]));
        float _sub_39;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(ts_4), "f"(tz_4));
        float _sub_38;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(vacc_0[0]), "f"(_sub_39));
        float _sub_40;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(vprod_5), "f"(tz_4));
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_sub_38), "f"(_sub_40));
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(vacc_0[1]), "f"(_add_37));
        vacc_0[1] = _add_36;
        vacc_0[0] = ts_4;
        float vprod_6;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_6) : "f"(x_row[7]), "f"(du_row[7]));
        float ts_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_5) : "f"(vacc_0[0]), "f"(vprod_6));
        float tz_5;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_5) : "f"(ts_5), "f"(vacc_0[0]));
        float _sub_44;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(ts_5), "f"(tz_5));
        float _sub_43;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(vacc_0[0]), "f"(_sub_44));
        float _sub_45;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(vprod_6), "f"(tz_5));
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_sub_43), "f"(_sub_45));
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(vacc_0[1]), "f"(_add_42));
        vacc_0[1] = _add_41;
        vacc_0[0] = ts_5;
        float vprod_7;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_7) : "f"(x_row[8]), "f"(du_row[8]));
        float ts_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_6) : "f"(vacc_0[0]), "f"(vprod_7));
        float tz_6;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_6) : "f"(ts_6), "f"(vacc_0[0]));
        float _sub_49;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(ts_6), "f"(tz_6));
        float _sub_48;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(vacc_0[0]), "f"(_sub_49));
        float _sub_50;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(vprod_7), "f"(tz_6));
        float _add_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_sub_48), "f"(_sub_50));
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(vacc_0[1]), "f"(_add_47));
        vacc_0[1] = _add_46;
        vacc_0[0] = ts_6;
        float vprod_8;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_8) : "f"(x_row[9]), "f"(du_row[9]));
        float ts_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_7) : "f"(vacc_0[0]), "f"(vprod_8));
        float tz_7;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_7) : "f"(ts_7), "f"(vacc_0[0]));
        float _sub_54;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(ts_7), "f"(tz_7));
        float _sub_53;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(vacc_0[0]), "f"(_sub_54));
        float _sub_55;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(vprod_8), "f"(tz_7));
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_sub_53), "f"(_sub_55));
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(vacc_0[1]), "f"(_add_52));
        vacc_0[1] = _add_51;
        vacc_0[0] = ts_7;
        float vprod_9;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_9) : "f"(x_row[10]), "f"(du_row[10]));
        float ts_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_8) : "f"(vacc_0[0]), "f"(vprod_9));
        float tz_8;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_8) : "f"(ts_8), "f"(vacc_0[0]));
        float _sub_59;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(ts_8), "f"(tz_8));
        float _sub_58;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(vacc_0[0]), "f"(_sub_59));
        float _sub_60;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(vprod_9), "f"(tz_8));
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_sub_58), "f"(_sub_60));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(vacc_0[1]), "f"(_add_57));
        vacc_0[1] = _add_56;
        vacc_0[0] = ts_8;
        float vprod_10;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_10) : "f"(x_row[11]), "f"(du_row[11]));
        float ts_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_9) : "f"(vacc_0[0]), "f"(vprod_10));
        float tz_9;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_9) : "f"(ts_9), "f"(vacc_0[0]));
        float _sub_64;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(ts_9), "f"(tz_9));
        float _sub_63;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(vacc_0[0]), "f"(_sub_64));
        float _sub_65;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(vprod_10), "f"(tz_9));
        float _add_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_sub_63), "f"(_sub_65));
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(vacc_0[1]), "f"(_add_62));
        vacc_0[1] = _add_61;
        vacc_0[0] = ts_9;
        float vprod_11;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_11) : "f"(x_row[12]), "f"(du_row[12]));
        float ts_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_10) : "f"(vacc_0[0]), "f"(vprod_11));
        float tz_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_10) : "f"(ts_10), "f"(vacc_0[0]));
        float _sub_69;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(ts_10), "f"(tz_10));
        float _sub_68;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(vacc_0[0]), "f"(_sub_69));
        float _sub_70;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(vprod_11), "f"(tz_10));
        float _add_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_sub_68), "f"(_sub_70));
        float _add_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(vacc_0[1]), "f"(_add_67));
        vacc_0[1] = _add_66;
        vacc_0[0] = ts_10;
        float vprod_12;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_12) : "f"(x_row[13]), "f"(du_row[13]));
        float ts_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_11) : "f"(vacc_0[0]), "f"(vprod_12));
        float tz_11;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_11) : "f"(ts_11), "f"(vacc_0[0]));
        float _sub_74;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(ts_11), "f"(tz_11));
        float _sub_73;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(vacc_0[0]), "f"(_sub_74));
        float _sub_75;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(vprod_12), "f"(tz_11));
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_sub_73), "f"(_sub_75));
        float _add_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(vacc_0[1]), "f"(_add_72));
        vacc_0[1] = _add_71;
        vacc_0[0] = ts_11;
        float vprod_13;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_13) : "f"(x_row[14]), "f"(du_row[14]));
        float ts_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_12) : "f"(vacc_0[0]), "f"(vprod_13));
        float tz_12;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_12) : "f"(ts_12), "f"(vacc_0[0]));
        float _sub_79;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(ts_12), "f"(tz_12));
        float _sub_78;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(vacc_0[0]), "f"(_sub_79));
        float _sub_80;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(vprod_13), "f"(tz_12));
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_sub_78), "f"(_sub_80));
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(vacc_0[1]), "f"(_add_77));
        vacc_0[1] = _add_76;
        vacc_0[0] = ts_12;
        float vprod_14;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_14) : "f"(x_row[15]), "f"(du_row[15]));
        float ts_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_13) : "f"(vacc_0[0]), "f"(vprod_14));
        float tz_13;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_13) : "f"(ts_13), "f"(vacc_0[0]));
        float _sub_84;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(ts_13), "f"(tz_13));
        float _sub_83;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(vacc_0[0]), "f"(_sub_84));
        float _sub_85;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(vprod_14), "f"(tz_13));
        float _add_82;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_sub_83), "f"(_sub_85));
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(vacc_0[1]), "f"(_add_82));
        vacc_0[1] = _add_81;
        vacc_0[0] = ts_13;
        float vprod_15;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_15) : "f"(x_row[16]), "f"(du_row[16]));
        float ts_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_14) : "f"(vacc_0[0]), "f"(vprod_15));
        float tz_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_14) : "f"(ts_14), "f"(vacc_0[0]));
        float _sub_89;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(ts_14), "f"(tz_14));
        float _sub_88;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(vacc_0[0]), "f"(_sub_89));
        float _sub_90;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(vprod_15), "f"(tz_14));
        float _add_87;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_sub_88), "f"(_sub_90));
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(vacc_0[1]), "f"(_add_87));
        vacc_0[1] = _add_86;
        vacc_0[0] = ts_14;
        float vprod_16;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_16) : "f"(x_row[17]), "f"(du_row[17]));
        float ts_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_15) : "f"(vacc_0[0]), "f"(vprod_16));
        float tz_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_15) : "f"(ts_15), "f"(vacc_0[0]));
        float _sub_94;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(ts_15), "f"(tz_15));
        float _sub_93;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(vacc_0[0]), "f"(_sub_94));
        float _sub_95;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(vprod_16), "f"(tz_15));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_sub_93), "f"(_sub_95));
        float _add_91;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(vacc_0[1]), "f"(_add_92));
        vacc_0[1] = _add_91;
        vacc_0[0] = ts_15;
        float vprod_17;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_17) : "f"(x_row[18]), "f"(du_row[18]));
        float ts_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_16) : "f"(vacc_0[0]), "f"(vprod_17));
        float tz_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_16) : "f"(ts_16), "f"(vacc_0[0]));
        float _sub_99;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(ts_16), "f"(tz_16));
        float _sub_98;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_98) : "f"(vacc_0[0]), "f"(_sub_99));
        float _sub_100;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(vprod_17), "f"(tz_16));
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_sub_98), "f"(_sub_100));
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(vacc_0[1]), "f"(_add_97));
        vacc_0[1] = _add_96;
        vacc_0[0] = ts_16;
        float vprod_18;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_18) : "f"(x_row[19]), "f"(du_row[19]));
        float ts_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_17) : "f"(vacc_0[0]), "f"(vprod_18));
        float tz_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_17) : "f"(ts_17), "f"(vacc_0[0]));
        float _sub_104;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(ts_17), "f"(tz_17));
        float _sub_103;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(vacc_0[0]), "f"(_sub_104));
        float _sub_105;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(vprod_18), "f"(tz_17));
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_sub_103), "f"(_sub_105));
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(vacc_0[1]), "f"(_add_102));
        vacc_0[1] = _add_101;
        vacc_0[0] = ts_17;
        dots_p[0] = vacc_0[0];
        dots_p[4] = vacc_0[1];
        unsigned int dw_0[2];
        dw_0[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[0]), mul_f32_rn_noftz(pin[0], du_row[1]))));
        dw_0[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[2]), mul_f32_rn_noftz(pin[0], du_row[3]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + xrow_0))[0] = reinterpret_cast<int2*>(dw_0)[0];
        unsigned int dw_1[2];
        dw_1[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[4]), mul_f32_rn_noftz(pin[0], du_row[5]))));
        dw_1[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[6]), mul_f32_rn_noftz(pin[0], du_row[7]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_0 + 4ll)))[0] = reinterpret_cast<int2*>(dw_1)[0];
        unsigned int dw_2[2];
        dw_2[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[8]), mul_f32_rn_noftz(pin[0], du_row[9]))));
        dw_2[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[10]), mul_f32_rn_noftz(pin[0], du_row[11]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_0 + 8ll)))[0] = reinterpret_cast<int2*>(dw_2)[0];
        unsigned int dw_3[2];
        dw_3[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[12]), mul_f32_rn_noftz(pin[0], du_row[13]))));
        dw_3[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[14]), mul_f32_rn_noftz(pin[0], du_row[15]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_0 + 12ll)))[0] = reinterpret_cast<int2*>(dw_3)[0];
        unsigned int dw_4[2];
        dw_4[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[16]), mul_f32_rn_noftz(pin[0], du_row[17]))));
        dw_4[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[0], du_row[18]), mul_f32_rn_noftz(pin[0], du_row[19]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_0 + 16ll)))[0] = reinterpret_cast<int2*>(dw_4)[0];
        long long xrow_1 = (t64 * 4ll + 1ll) * 5120ll + col_off;
        {
            uint2 _vld_106;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_106.x), "=r"(_vld_106.y) : "l"((const void*)(x + xrow_1 + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_106 = reinterpret_cast<uint32_t*>(&_vld_106);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_106[_pair]) << 16);
                (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_106[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_107;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_107.x), "=r"(_vld_107.y) : "l"((const void*)(x + (xrow_1 + 4ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_107 = reinterpret_cast<uint32_t*>(&_vld_107);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_107[_pair]) << 16);
                (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_107[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_108;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_108.x), "=r"(_vld_108.y) : "l"((const void*)(x + (xrow_1 + 8ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_108 = reinterpret_cast<uint32_t*>(&_vld_108);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_108[_pair]) << 16);
                (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_108[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_109;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_109.x), "=r"(_vld_109.y) : "l"((const void*)(x + (xrow_1 + 12ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_109 = reinterpret_cast<uint32_t*>(&_vld_109);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) << 16);
                (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_110;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_110.x), "=r"(_vld_110.y) : "l"((const void*)(x + (xrow_1 + 16ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_110 = reinterpret_cast<uint32_t*>(&_vld_110);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_110[_pair]) << 16);
                (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_110[_pair]) & 0xffff0000u);
            }
        }
        float vacc_1[2];
        vacc_1[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        vacc_1[1] = 0.0f;
        float vprod_19;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_19) : "f"(x_row[1]), "f"(du_row[1]));
        float ts_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_18) : "f"(vacc_1[0]), "f"(vprod_19));
        float tz_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_18) : "f"(ts_18), "f"(vacc_1[0]));
        float _sub_114;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(ts_18), "f"(tz_18));
        float _sub_113;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(vacc_1[0]), "f"(_sub_114));
        float _sub_115;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_115) : "f"(vprod_19), "f"(tz_18));
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_sub_113), "f"(_sub_115));
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(vacc_1[1]), "f"(_add_112));
        vacc_1[1] = _add_111;
        vacc_1[0] = ts_18;
        float vprod_20;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_20) : "f"(x_row[2]), "f"(du_row[2]));
        float ts_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_19) : "f"(vacc_1[0]), "f"(vprod_20));
        float tz_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_19) : "f"(ts_19), "f"(vacc_1[0]));
        float _sub_119;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(ts_19), "f"(tz_19));
        float _sub_118;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(vacc_1[0]), "f"(_sub_119));
        float _sub_120;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(vprod_20), "f"(tz_19));
        float _add_117;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(_sub_118), "f"(_sub_120));
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(vacc_1[1]), "f"(_add_117));
        vacc_1[1] = _add_116;
        vacc_1[0] = ts_19;
        float vprod_21;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_21) : "f"(x_row[3]), "f"(du_row[3]));
        float ts_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_20) : "f"(vacc_1[0]), "f"(vprod_21));
        float tz_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_20) : "f"(ts_20), "f"(vacc_1[0]));
        float _sub_124;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_124) : "f"(ts_20), "f"(tz_20));
        float _sub_123;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_123) : "f"(vacc_1[0]), "f"(_sub_124));
        float _sub_125;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_125) : "f"(vprod_21), "f"(tz_20));
        float _add_122;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(_sub_123), "f"(_sub_125));
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(vacc_1[1]), "f"(_add_122));
        vacc_1[1] = _add_121;
        vacc_1[0] = ts_20;
        float vprod_22;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_22) : "f"(x_row[4]), "f"(du_row[4]));
        float ts_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_21) : "f"(vacc_1[0]), "f"(vprod_22));
        float tz_21;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_21) : "f"(ts_21), "f"(vacc_1[0]));
        float _sub_129;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(ts_21), "f"(tz_21));
        float _sub_128;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_128) : "f"(vacc_1[0]), "f"(_sub_129));
        float _sub_130;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_130) : "f"(vprod_22), "f"(tz_21));
        float _add_127;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_sub_128), "f"(_sub_130));
        float _add_126;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(vacc_1[1]), "f"(_add_127));
        vacc_1[1] = _add_126;
        vacc_1[0] = ts_21;
        float vprod_23;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_23) : "f"(x_row[5]), "f"(du_row[5]));
        float ts_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_22) : "f"(vacc_1[0]), "f"(vprod_23));
        float tz_22;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_22) : "f"(ts_22), "f"(vacc_1[0]));
        float _sub_134;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_134) : "f"(ts_22), "f"(tz_22));
        float _sub_133;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_133) : "f"(vacc_1[0]), "f"(_sub_134));
        float _sub_135;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_135) : "f"(vprod_23), "f"(tz_22));
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_sub_133), "f"(_sub_135));
        float _add_131;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(vacc_1[1]), "f"(_add_132));
        vacc_1[1] = _add_131;
        vacc_1[0] = ts_22;
        float vprod_24;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_24) : "f"(x_row[6]), "f"(du_row[6]));
        float ts_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_23) : "f"(vacc_1[0]), "f"(vprod_24));
        float tz_23;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_23) : "f"(ts_23), "f"(vacc_1[0]));
        float _sub_139;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_139) : "f"(ts_23), "f"(tz_23));
        float _sub_138;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(vacc_1[0]), "f"(_sub_139));
        float _sub_140;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_140) : "f"(vprod_24), "f"(tz_23));
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_sub_138), "f"(_sub_140));
        float _add_136;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(vacc_1[1]), "f"(_add_137));
        vacc_1[1] = _add_136;
        vacc_1[0] = ts_23;
        float vprod_25;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_25) : "f"(x_row[7]), "f"(du_row[7]));
        float ts_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_24) : "f"(vacc_1[0]), "f"(vprod_25));
        float tz_24;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_24) : "f"(ts_24), "f"(vacc_1[0]));
        float _sub_144;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_144) : "f"(ts_24), "f"(tz_24));
        float _sub_143;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_143) : "f"(vacc_1[0]), "f"(_sub_144));
        float _sub_145;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_145) : "f"(vprod_25), "f"(tz_24));
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(_sub_143), "f"(_sub_145));
        float _add_141;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(vacc_1[1]), "f"(_add_142));
        vacc_1[1] = _add_141;
        vacc_1[0] = ts_24;
        float vprod_26;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_26) : "f"(x_row[8]), "f"(du_row[8]));
        float ts_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_25) : "f"(vacc_1[0]), "f"(vprod_26));
        float tz_25;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_25) : "f"(ts_25), "f"(vacc_1[0]));
        float _sub_149;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(ts_25), "f"(tz_25));
        float _sub_148;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_148) : "f"(vacc_1[0]), "f"(_sub_149));
        float _sub_150;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(vprod_26), "f"(tz_25));
        float _add_147;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(_sub_148), "f"(_sub_150));
        float _add_146;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(vacc_1[1]), "f"(_add_147));
        vacc_1[1] = _add_146;
        vacc_1[0] = ts_25;
        float vprod_27;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_27) : "f"(x_row[9]), "f"(du_row[9]));
        float ts_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_26) : "f"(vacc_1[0]), "f"(vprod_27));
        float tz_26;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_26) : "f"(ts_26), "f"(vacc_1[0]));
        float _sub_154;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_154) : "f"(ts_26), "f"(tz_26));
        float _sub_153;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_153) : "f"(vacc_1[0]), "f"(_sub_154));
        float _sub_155;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_155) : "f"(vprod_27), "f"(tz_26));
        float _add_152;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(_sub_153), "f"(_sub_155));
        float _add_151;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(vacc_1[1]), "f"(_add_152));
        vacc_1[1] = _add_151;
        vacc_1[0] = ts_26;
        float vprod_28;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_28) : "f"(x_row[10]), "f"(du_row[10]));
        float ts_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_27) : "f"(vacc_1[0]), "f"(vprod_28));
        float tz_27;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_27) : "f"(ts_27), "f"(vacc_1[0]));
        float _sub_159;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_159) : "f"(ts_27), "f"(tz_27));
        float _sub_158;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_158) : "f"(vacc_1[0]), "f"(_sub_159));
        float _sub_160;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_160) : "f"(vprod_28), "f"(tz_27));
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(_sub_158), "f"(_sub_160));
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(vacc_1[1]), "f"(_add_157));
        vacc_1[1] = _add_156;
        vacc_1[0] = ts_27;
        float vprod_29;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_29) : "f"(x_row[11]), "f"(du_row[11]));
        float ts_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_28) : "f"(vacc_1[0]), "f"(vprod_29));
        float tz_28;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_28) : "f"(ts_28), "f"(vacc_1[0]));
        float _sub_164;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_164) : "f"(ts_28), "f"(tz_28));
        float _sub_163;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_163) : "f"(vacc_1[0]), "f"(_sub_164));
        float _sub_165;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_165) : "f"(vprod_29), "f"(tz_28));
        float _add_162;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(_sub_163), "f"(_sub_165));
        float _add_161;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(vacc_1[1]), "f"(_add_162));
        vacc_1[1] = _add_161;
        vacc_1[0] = ts_28;
        float vprod_30;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_30) : "f"(x_row[12]), "f"(du_row[12]));
        float ts_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_29) : "f"(vacc_1[0]), "f"(vprod_30));
        float tz_29;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_29) : "f"(ts_29), "f"(vacc_1[0]));
        float _sub_169;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_169) : "f"(ts_29), "f"(tz_29));
        float _sub_168;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(vacc_1[0]), "f"(_sub_169));
        float _sub_170;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(vprod_30), "f"(tz_29));
        float _add_167;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_sub_168), "f"(_sub_170));
        float _add_166;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(vacc_1[1]), "f"(_add_167));
        vacc_1[1] = _add_166;
        vacc_1[0] = ts_29;
        float vprod_31;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_31) : "f"(x_row[13]), "f"(du_row[13]));
        float ts_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_30) : "f"(vacc_1[0]), "f"(vprod_31));
        float tz_30;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_30) : "f"(ts_30), "f"(vacc_1[0]));
        float _sub_174;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_174) : "f"(ts_30), "f"(tz_30));
        float _sub_173;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_173) : "f"(vacc_1[0]), "f"(_sub_174));
        float _sub_175;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_175) : "f"(vprod_31), "f"(tz_30));
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(_sub_173), "f"(_sub_175));
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(vacc_1[1]), "f"(_add_172));
        vacc_1[1] = _add_171;
        vacc_1[0] = ts_30;
        float vprod_32;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_32) : "f"(x_row[14]), "f"(du_row[14]));
        float ts_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_31) : "f"(vacc_1[0]), "f"(vprod_32));
        float tz_31;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_31) : "f"(ts_31), "f"(vacc_1[0]));
        float _sub_179;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_179) : "f"(ts_31), "f"(tz_31));
        float _sub_178;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_178) : "f"(vacc_1[0]), "f"(_sub_179));
        float _sub_180;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_180) : "f"(vprod_32), "f"(tz_31));
        float _add_177;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(_sub_178), "f"(_sub_180));
        float _add_176;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(vacc_1[1]), "f"(_add_177));
        vacc_1[1] = _add_176;
        vacc_1[0] = ts_31;
        float vprod_33;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_33) : "f"(x_row[15]), "f"(du_row[15]));
        float ts_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_32) : "f"(vacc_1[0]), "f"(vprod_33));
        float tz_32;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_32) : "f"(ts_32), "f"(vacc_1[0]));
        float _sub_184;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_184) : "f"(ts_32), "f"(tz_32));
        float _sub_183;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_183) : "f"(vacc_1[0]), "f"(_sub_184));
        float _sub_185;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_185) : "f"(vprod_33), "f"(tz_32));
        float _add_182;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(_sub_183), "f"(_sub_185));
        float _add_181;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(vacc_1[1]), "f"(_add_182));
        vacc_1[1] = _add_181;
        vacc_1[0] = ts_32;
        float vprod_34;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_34) : "f"(x_row[16]), "f"(du_row[16]));
        float ts_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_33) : "f"(vacc_1[0]), "f"(vprod_34));
        float tz_33;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_33) : "f"(ts_33), "f"(vacc_1[0]));
        float _sub_189;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_189) : "f"(ts_33), "f"(tz_33));
        float _sub_188;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_188) : "f"(vacc_1[0]), "f"(_sub_189));
        float _sub_190;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_190) : "f"(vprod_34), "f"(tz_33));
        float _add_187;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(_sub_188), "f"(_sub_190));
        float _add_186;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(vacc_1[1]), "f"(_add_187));
        vacc_1[1] = _add_186;
        vacc_1[0] = ts_33;
        float vprod_35;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_35) : "f"(x_row[17]), "f"(du_row[17]));
        float ts_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_34) : "f"(vacc_1[0]), "f"(vprod_35));
        float tz_34;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_34) : "f"(ts_34), "f"(vacc_1[0]));
        float _sub_194;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_194) : "f"(ts_34), "f"(tz_34));
        float _sub_193;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_193) : "f"(vacc_1[0]), "f"(_sub_194));
        float _sub_195;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_195) : "f"(vprod_35), "f"(tz_34));
        float _add_192;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(_sub_193), "f"(_sub_195));
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(vacc_1[1]), "f"(_add_192));
        vacc_1[1] = _add_191;
        vacc_1[0] = ts_34;
        float vprod_36;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_36) : "f"(x_row[18]), "f"(du_row[18]));
        float ts_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_35) : "f"(vacc_1[0]), "f"(vprod_36));
        float tz_35;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_35) : "f"(ts_35), "f"(vacc_1[0]));
        float _sub_199;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_199) : "f"(ts_35), "f"(tz_35));
        float _sub_198;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_198) : "f"(vacc_1[0]), "f"(_sub_199));
        float _sub_200;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_200) : "f"(vprod_36), "f"(tz_35));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_sub_198), "f"(_sub_200));
        float _add_196;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(vacc_1[1]), "f"(_add_197));
        vacc_1[1] = _add_196;
        vacc_1[0] = ts_35;
        float vprod_37;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_37) : "f"(x_row[19]), "f"(du_row[19]));
        float ts_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_36) : "f"(vacc_1[0]), "f"(vprod_37));
        float tz_36;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_36) : "f"(ts_36), "f"(vacc_1[0]));
        float _sub_204;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_204) : "f"(ts_36), "f"(tz_36));
        float _sub_203;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_203) : "f"(vacc_1[0]), "f"(_sub_204));
        float _sub_205;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_205) : "f"(vprod_37), "f"(tz_36));
        float _add_202;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(_sub_203), "f"(_sub_205));
        float _add_201;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(vacc_1[1]), "f"(_add_202));
        vacc_1[1] = _add_201;
        vacc_1[0] = ts_36;
        dots_p[1] = vacc_1[0];
        dots_p[5] = vacc_1[1];
        unsigned int dw_5[2];
        dw_5[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[0]), mul_f32_rn_noftz(pin[1], du_row[1]))));
        dw_5[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[2]), mul_f32_rn_noftz(pin[1], du_row[3]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + xrow_1))[0] = reinterpret_cast<int2*>(dw_5)[0];
        unsigned int dw_6[2];
        dw_6[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[4]), mul_f32_rn_noftz(pin[1], du_row[5]))));
        dw_6[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[6]), mul_f32_rn_noftz(pin[1], du_row[7]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_1 + 4ll)))[0] = reinterpret_cast<int2*>(dw_6)[0];
        unsigned int dw_7[2];
        dw_7[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[8]), mul_f32_rn_noftz(pin[1], du_row[9]))));
        dw_7[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[10]), mul_f32_rn_noftz(pin[1], du_row[11]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_1 + 8ll)))[0] = reinterpret_cast<int2*>(dw_7)[0];
        unsigned int dw_8[2];
        dw_8[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[12]), mul_f32_rn_noftz(pin[1], du_row[13]))));
        dw_8[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[14]), mul_f32_rn_noftz(pin[1], du_row[15]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_1 + 12ll)))[0] = reinterpret_cast<int2*>(dw_8)[0];
        unsigned int dw_9[2];
        dw_9[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[16]), mul_f32_rn_noftz(pin[1], du_row[17]))));
        dw_9[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[1], du_row[18]), mul_f32_rn_noftz(pin[1], du_row[19]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_1 + 16ll)))[0] = reinterpret_cast<int2*>(dw_9)[0];
        long long xrow_2 = (t64 * 4ll + 2ll) * 5120ll + col_off;
        {
            uint2 _vld_206;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_206.x), "=r"(_vld_206.y) : "l"((const void*)(x + xrow_2 + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_206 = reinterpret_cast<uint32_t*>(&_vld_206);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_206[_pair]) << 16);
                (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_206[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_207;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_207.x), "=r"(_vld_207.y) : "l"((const void*)(x + (xrow_2 + 4ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_207 = reinterpret_cast<uint32_t*>(&_vld_207);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_207[_pair]) << 16);
                (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_207[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_208;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_208.x), "=r"(_vld_208.y) : "l"((const void*)(x + (xrow_2 + 8ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_208 = reinterpret_cast<uint32_t*>(&_vld_208);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_208[_pair]) << 16);
                (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_208[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_209;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_209.x), "=r"(_vld_209.y) : "l"((const void*)(x + (xrow_2 + 12ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_209 = reinterpret_cast<uint32_t*>(&_vld_209);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_209[_pair]) << 16);
                (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_209[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_210;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_210.x), "=r"(_vld_210.y) : "l"((const void*)(x + (xrow_2 + 16ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_210 = reinterpret_cast<uint32_t*>(&_vld_210);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_210[_pair]) << 16);
                (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_210[_pair]) & 0xffff0000u);
            }
        }
        float vacc_2[2];
        vacc_2[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        vacc_2[1] = 0.0f;
        float vprod_38;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_38) : "f"(x_row[1]), "f"(du_row[1]));
        float ts_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_37) : "f"(vacc_2[0]), "f"(vprod_38));
        float tz_37;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_37) : "f"(ts_37), "f"(vacc_2[0]));
        float _sub_214;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_214) : "f"(ts_37), "f"(tz_37));
        float _sub_213;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(vacc_2[0]), "f"(_sub_214));
        float _sub_215;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_215) : "f"(vprod_38), "f"(tz_37));
        float _add_212;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(_sub_213), "f"(_sub_215));
        float _add_211;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(vacc_2[1]), "f"(_add_212));
        vacc_2[1] = _add_211;
        vacc_2[0] = ts_37;
        float vprod_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_39) : "f"(x_row[2]), "f"(du_row[2]));
        float ts_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_38) : "f"(vacc_2[0]), "f"(vprod_39));
        float tz_38;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_38) : "f"(ts_38), "f"(vacc_2[0]));
        float _sub_219;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_219) : "f"(ts_38), "f"(tz_38));
        float _sub_218;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_218) : "f"(vacc_2[0]), "f"(_sub_219));
        float _sub_220;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_220) : "f"(vprod_39), "f"(tz_38));
        float _add_217;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(_sub_218), "f"(_sub_220));
        float _add_216;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(vacc_2[1]), "f"(_add_217));
        vacc_2[1] = _add_216;
        vacc_2[0] = ts_38;
        float vprod_40;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_40) : "f"(x_row[3]), "f"(du_row[3]));
        float ts_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_39) : "f"(vacc_2[0]), "f"(vprod_40));
        float tz_39;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_39) : "f"(ts_39), "f"(vacc_2[0]));
        float _sub_224;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_224) : "f"(ts_39), "f"(tz_39));
        float _sub_223;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_223) : "f"(vacc_2[0]), "f"(_sub_224));
        float _sub_225;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_225) : "f"(vprod_40), "f"(tz_39));
        float _add_222;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(_sub_223), "f"(_sub_225));
        float _add_221;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(vacc_2[1]), "f"(_add_222));
        vacc_2[1] = _add_221;
        vacc_2[0] = ts_39;
        float vprod_41;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_41) : "f"(x_row[4]), "f"(du_row[4]));
        float ts_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_40) : "f"(vacc_2[0]), "f"(vprod_41));
        float tz_40;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_40) : "f"(ts_40), "f"(vacc_2[0]));
        float _sub_229;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_229) : "f"(ts_40), "f"(tz_40));
        float _sub_228;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_228) : "f"(vacc_2[0]), "f"(_sub_229));
        float _sub_230;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_230) : "f"(vprod_41), "f"(tz_40));
        float _add_227;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(_sub_228), "f"(_sub_230));
        float _add_226;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(vacc_2[1]), "f"(_add_227));
        vacc_2[1] = _add_226;
        vacc_2[0] = ts_40;
        float vprod_42;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_42) : "f"(x_row[5]), "f"(du_row[5]));
        float ts_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_41) : "f"(vacc_2[0]), "f"(vprod_42));
        float tz_41;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_41) : "f"(ts_41), "f"(vacc_2[0]));
        float _sub_234;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_234) : "f"(ts_41), "f"(tz_41));
        float _sub_233;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_233) : "f"(vacc_2[0]), "f"(_sub_234));
        float _sub_235;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_235) : "f"(vprod_42), "f"(tz_41));
        float _add_232;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(_sub_233), "f"(_sub_235));
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(vacc_2[1]), "f"(_add_232));
        vacc_2[1] = _add_231;
        vacc_2[0] = ts_41;
        float vprod_43;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_43) : "f"(x_row[6]), "f"(du_row[6]));
        float ts_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_42) : "f"(vacc_2[0]), "f"(vprod_43));
        float tz_42;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_42) : "f"(ts_42), "f"(vacc_2[0]));
        float _sub_239;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_239) : "f"(ts_42), "f"(tz_42));
        float _sub_238;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_238) : "f"(vacc_2[0]), "f"(_sub_239));
        float _sub_240;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_240) : "f"(vprod_43), "f"(tz_42));
        float _add_237;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(_sub_238), "f"(_sub_240));
        float _add_236;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(vacc_2[1]), "f"(_add_237));
        vacc_2[1] = _add_236;
        vacc_2[0] = ts_42;
        float vprod_44;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_44) : "f"(x_row[7]), "f"(du_row[7]));
        float ts_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_43) : "f"(vacc_2[0]), "f"(vprod_44));
        float tz_43;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_43) : "f"(ts_43), "f"(vacc_2[0]));
        float _sub_244;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_244) : "f"(ts_43), "f"(tz_43));
        float _sub_243;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_243) : "f"(vacc_2[0]), "f"(_sub_244));
        float _sub_245;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_245) : "f"(vprod_44), "f"(tz_43));
        float _add_242;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(_sub_243), "f"(_sub_245));
        float _add_241;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(vacc_2[1]), "f"(_add_242));
        vacc_2[1] = _add_241;
        vacc_2[0] = ts_43;
        float vprod_45;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_45) : "f"(x_row[8]), "f"(du_row[8]));
        float ts_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_44) : "f"(vacc_2[0]), "f"(vprod_45));
        float tz_44;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_44) : "f"(ts_44), "f"(vacc_2[0]));
        float _sub_249;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_249) : "f"(ts_44), "f"(tz_44));
        float _sub_248;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_248) : "f"(vacc_2[0]), "f"(_sub_249));
        float _sub_250;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_250) : "f"(vprod_45), "f"(tz_44));
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(_sub_248), "f"(_sub_250));
        float _add_246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(vacc_2[1]), "f"(_add_247));
        vacc_2[1] = _add_246;
        vacc_2[0] = ts_44;
        float vprod_46;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_46) : "f"(x_row[9]), "f"(du_row[9]));
        float ts_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_45) : "f"(vacc_2[0]), "f"(vprod_46));
        float tz_45;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_45) : "f"(ts_45), "f"(vacc_2[0]));
        float _sub_254;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_254) : "f"(ts_45), "f"(tz_45));
        float _sub_253;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_253) : "f"(vacc_2[0]), "f"(_sub_254));
        float _sub_255;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_255) : "f"(vprod_46), "f"(tz_45));
        float _add_252;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(_sub_253), "f"(_sub_255));
        float _add_251;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(vacc_2[1]), "f"(_add_252));
        vacc_2[1] = _add_251;
        vacc_2[0] = ts_45;
        float vprod_47;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_47) : "f"(x_row[10]), "f"(du_row[10]));
        float ts_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_46) : "f"(vacc_2[0]), "f"(vprod_47));
        float tz_46;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_46) : "f"(ts_46), "f"(vacc_2[0]));
        float _sub_259;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_259) : "f"(ts_46), "f"(tz_46));
        float _sub_258;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_258) : "f"(vacc_2[0]), "f"(_sub_259));
        float _sub_260;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_260) : "f"(vprod_47), "f"(tz_46));
        float _add_257;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(_sub_258), "f"(_sub_260));
        float _add_256;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(vacc_2[1]), "f"(_add_257));
        vacc_2[1] = _add_256;
        vacc_2[0] = ts_46;
        float vprod_48;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_48) : "f"(x_row[11]), "f"(du_row[11]));
        float ts_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_47) : "f"(vacc_2[0]), "f"(vprod_48));
        float tz_47;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_47) : "f"(ts_47), "f"(vacc_2[0]));
        float _sub_264;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_264) : "f"(ts_47), "f"(tz_47));
        float _sub_263;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_263) : "f"(vacc_2[0]), "f"(_sub_264));
        float _sub_265;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_265) : "f"(vprod_48), "f"(tz_47));
        float _add_262;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(_sub_263), "f"(_sub_265));
        float _add_261;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(vacc_2[1]), "f"(_add_262));
        vacc_2[1] = _add_261;
        vacc_2[0] = ts_47;
        float vprod_49;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_49) : "f"(x_row[12]), "f"(du_row[12]));
        float ts_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_48) : "f"(vacc_2[0]), "f"(vprod_49));
        float tz_48;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_48) : "f"(ts_48), "f"(vacc_2[0]));
        float _sub_269;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_269) : "f"(ts_48), "f"(tz_48));
        float _sub_268;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_268) : "f"(vacc_2[0]), "f"(_sub_269));
        float _sub_270;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_270) : "f"(vprod_49), "f"(tz_48));
        float _add_267;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(_sub_268), "f"(_sub_270));
        float _add_266;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(vacc_2[1]), "f"(_add_267));
        vacc_2[1] = _add_266;
        vacc_2[0] = ts_48;
        float vprod_50;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_50) : "f"(x_row[13]), "f"(du_row[13]));
        float ts_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_49) : "f"(vacc_2[0]), "f"(vprod_50));
        float tz_49;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_49) : "f"(ts_49), "f"(vacc_2[0]));
        float _sub_274;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_274) : "f"(ts_49), "f"(tz_49));
        float _sub_273;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_273) : "f"(vacc_2[0]), "f"(_sub_274));
        float _sub_275;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_275) : "f"(vprod_50), "f"(tz_49));
        float _add_272;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(_sub_273), "f"(_sub_275));
        float _add_271;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(vacc_2[1]), "f"(_add_272));
        vacc_2[1] = _add_271;
        vacc_2[0] = ts_49;
        float vprod_51;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_51) : "f"(x_row[14]), "f"(du_row[14]));
        float ts_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_50) : "f"(vacc_2[0]), "f"(vprod_51));
        float tz_50;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_50) : "f"(ts_50), "f"(vacc_2[0]));
        float _sub_279;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_279) : "f"(ts_50), "f"(tz_50));
        float _sub_278;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_278) : "f"(vacc_2[0]), "f"(_sub_279));
        float _sub_280;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_280) : "f"(vprod_51), "f"(tz_50));
        float _add_277;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(_sub_278), "f"(_sub_280));
        float _add_276;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(vacc_2[1]), "f"(_add_277));
        vacc_2[1] = _add_276;
        vacc_2[0] = ts_50;
        float vprod_52;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_52) : "f"(x_row[15]), "f"(du_row[15]));
        float ts_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_51) : "f"(vacc_2[0]), "f"(vprod_52));
        float tz_51;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_51) : "f"(ts_51), "f"(vacc_2[0]));
        float _sub_284;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(ts_51), "f"(tz_51));
        float _sub_283;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_283) : "f"(vacc_2[0]), "f"(_sub_284));
        float _sub_285;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_285) : "f"(vprod_52), "f"(tz_51));
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(_sub_283), "f"(_sub_285));
        float _add_281;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(vacc_2[1]), "f"(_add_282));
        vacc_2[1] = _add_281;
        vacc_2[0] = ts_51;
        float vprod_53;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_53) : "f"(x_row[16]), "f"(du_row[16]));
        float ts_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_52) : "f"(vacc_2[0]), "f"(vprod_53));
        float tz_52;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_52) : "f"(ts_52), "f"(vacc_2[0]));
        float _sub_289;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(ts_52), "f"(tz_52));
        float _sub_288;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_288) : "f"(vacc_2[0]), "f"(_sub_289));
        float _sub_290;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_290) : "f"(vprod_53), "f"(tz_52));
        float _add_287;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(_sub_288), "f"(_sub_290));
        float _add_286;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(vacc_2[1]), "f"(_add_287));
        vacc_2[1] = _add_286;
        vacc_2[0] = ts_52;
        float vprod_54;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_54) : "f"(x_row[17]), "f"(du_row[17]));
        float ts_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_53) : "f"(vacc_2[0]), "f"(vprod_54));
        float tz_53;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_53) : "f"(ts_53), "f"(vacc_2[0]));
        float _sub_294;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_294) : "f"(ts_53), "f"(tz_53));
        float _sub_293;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(vacc_2[0]), "f"(_sub_294));
        float _sub_295;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(vprod_54), "f"(tz_53));
        float _add_292;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_292) : "f"(_sub_293), "f"(_sub_295));
        float _add_291;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(vacc_2[1]), "f"(_add_292));
        vacc_2[1] = _add_291;
        vacc_2[0] = ts_53;
        float vprod_55;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_55) : "f"(x_row[18]), "f"(du_row[18]));
        float ts_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_54) : "f"(vacc_2[0]), "f"(vprod_55));
        float tz_54;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_54) : "f"(ts_54), "f"(vacc_2[0]));
        float _sub_299;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_299) : "f"(ts_54), "f"(tz_54));
        float _sub_298;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(vacc_2[0]), "f"(_sub_299));
        float _sub_300;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_300) : "f"(vprod_55), "f"(tz_54));
        float _add_297;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_297) : "f"(_sub_298), "f"(_sub_300));
        float _add_296;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_296) : "f"(vacc_2[1]), "f"(_add_297));
        vacc_2[1] = _add_296;
        vacc_2[0] = ts_54;
        float vprod_56;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_56) : "f"(x_row[19]), "f"(du_row[19]));
        float ts_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_55) : "f"(vacc_2[0]), "f"(vprod_56));
        float tz_55;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_55) : "f"(ts_55), "f"(vacc_2[0]));
        float _sub_304;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_304) : "f"(ts_55), "f"(tz_55));
        float _sub_303;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_303) : "f"(vacc_2[0]), "f"(_sub_304));
        float _sub_305;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_305) : "f"(vprod_56), "f"(tz_55));
        float _add_302;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_302) : "f"(_sub_303), "f"(_sub_305));
        float _add_301;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_301) : "f"(vacc_2[1]), "f"(_add_302));
        vacc_2[1] = _add_301;
        vacc_2[0] = ts_55;
        dots_p[2] = vacc_2[0];
        dots_p[6] = vacc_2[1];
        unsigned int dw_10[2];
        dw_10[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[0]), mul_f32_rn_noftz(pin[2], du_row[1]))));
        dw_10[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[2]), mul_f32_rn_noftz(pin[2], du_row[3]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + xrow_2))[0] = reinterpret_cast<int2*>(dw_10)[0];
        unsigned int dw_11[2];
        dw_11[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[4]), mul_f32_rn_noftz(pin[2], du_row[5]))));
        dw_11[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[6]), mul_f32_rn_noftz(pin[2], du_row[7]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_2 + 4ll)))[0] = reinterpret_cast<int2*>(dw_11)[0];
        unsigned int dw_12[2];
        dw_12[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[8]), mul_f32_rn_noftz(pin[2], du_row[9]))));
        dw_12[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[10]), mul_f32_rn_noftz(pin[2], du_row[11]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_2 + 8ll)))[0] = reinterpret_cast<int2*>(dw_12)[0];
        unsigned int dw_13[2];
        dw_13[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[12]), mul_f32_rn_noftz(pin[2], du_row[13]))));
        dw_13[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[14]), mul_f32_rn_noftz(pin[2], du_row[15]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_2 + 12ll)))[0] = reinterpret_cast<int2*>(dw_13)[0];
        unsigned int dw_14[2];
        dw_14[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[16]), mul_f32_rn_noftz(pin[2], du_row[17]))));
        dw_14[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[2], du_row[18]), mul_f32_rn_noftz(pin[2], du_row[19]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_2 + 16ll)))[0] = reinterpret_cast<int2*>(dw_14)[0];
        long long xrow_3 = (t64 * 4ll + 3ll) * 5120ll + col_off;
        {
            uint2 _vld_306;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_306.x), "=r"(_vld_306.y) : "l"((const void*)(x + xrow_3 + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_306 = reinterpret_cast<uint32_t*>(&_vld_306);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_306[_pair]) << 16);
                (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_306[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_307;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_307.x), "=r"(_vld_307.y) : "l"((const void*)(x + (xrow_3 + 4ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_307 = reinterpret_cast<uint32_t*>(&_vld_307);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_307[_pair]) << 16);
                (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_307[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_308;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_308.x), "=r"(_vld_308.y) : "l"((const void*)(x + (xrow_3 + 8ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_308 = reinterpret_cast<uint32_t*>(&_vld_308);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_308[_pair]) << 16);
                (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_308[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_309;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_309.x), "=r"(_vld_309.y) : "l"((const void*)(x + (xrow_3 + 12ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_309 = reinterpret_cast<uint32_t*>(&_vld_309);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_309[_pair]) << 16);
                (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_309[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_310;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_vld_310.x), "=r"(_vld_310.y) : "l"((const void*)(x + (xrow_3 + 16ll) + 0)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_310 = reinterpret_cast<uint32_t*>(&_vld_310);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_310[_pair]) << 16);
                (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_310[_pair]) & 0xffff0000u);
            }
        }
        float vacc_3[2];
        vacc_3[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
        vacc_3[1] = 0.0f;
        float vprod_57;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_57) : "f"(x_row[1]), "f"(du_row[1]));
        float ts_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_56) : "f"(vacc_3[0]), "f"(vprod_57));
        float tz_56;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_56) : "f"(ts_56), "f"(vacc_3[0]));
        float _sub_314;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_314) : "f"(ts_56), "f"(tz_56));
        float _sub_313;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_313) : "f"(vacc_3[0]), "f"(_sub_314));
        float _sub_315;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_315) : "f"(vprod_57), "f"(tz_56));
        float _add_312;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(_sub_313), "f"(_sub_315));
        float _add_311;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_311) : "f"(vacc_3[1]), "f"(_add_312));
        vacc_3[1] = _add_311;
        vacc_3[0] = ts_56;
        float vprod_58;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_58) : "f"(x_row[2]), "f"(du_row[2]));
        float ts_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_57) : "f"(vacc_3[0]), "f"(vprod_58));
        float tz_57;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_57) : "f"(ts_57), "f"(vacc_3[0]));
        float _sub_319;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(ts_57), "f"(tz_57));
        float _sub_318;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_318) : "f"(vacc_3[0]), "f"(_sub_319));
        float _sub_320;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_320) : "f"(vprod_58), "f"(tz_57));
        float _add_317;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_317) : "f"(_sub_318), "f"(_sub_320));
        float _add_316;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_316) : "f"(vacc_3[1]), "f"(_add_317));
        vacc_3[1] = _add_316;
        vacc_3[0] = ts_57;
        float vprod_59;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_59) : "f"(x_row[3]), "f"(du_row[3]));
        float ts_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_58) : "f"(vacc_3[0]), "f"(vprod_59));
        float tz_58;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_58) : "f"(ts_58), "f"(vacc_3[0]));
        float _sub_324;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_324) : "f"(ts_58), "f"(tz_58));
        float _sub_323;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(vacc_3[0]), "f"(_sub_324));
        float _sub_325;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_325) : "f"(vprod_59), "f"(tz_58));
        float _add_322;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_322) : "f"(_sub_323), "f"(_sub_325));
        float _add_321;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_321) : "f"(vacc_3[1]), "f"(_add_322));
        vacc_3[1] = _add_321;
        vacc_3[0] = ts_58;
        float vprod_60;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_60) : "f"(x_row[4]), "f"(du_row[4]));
        float ts_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_59) : "f"(vacc_3[0]), "f"(vprod_60));
        float tz_59;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_59) : "f"(ts_59), "f"(vacc_3[0]));
        float _sub_329;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(ts_59), "f"(tz_59));
        float _sub_328;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_328) : "f"(vacc_3[0]), "f"(_sub_329));
        float _sub_330;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_330) : "f"(vprod_60), "f"(tz_59));
        float _add_327;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_327) : "f"(_sub_328), "f"(_sub_330));
        float _add_326;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_326) : "f"(vacc_3[1]), "f"(_add_327));
        vacc_3[1] = _add_326;
        vacc_3[0] = ts_59;
        float vprod_61;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_61) : "f"(x_row[5]), "f"(du_row[5]));
        float ts_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_60) : "f"(vacc_3[0]), "f"(vprod_61));
        float tz_60;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_60) : "f"(ts_60), "f"(vacc_3[0]));
        float _sub_334;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_334) : "f"(ts_60), "f"(tz_60));
        float _sub_333;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_333) : "f"(vacc_3[0]), "f"(_sub_334));
        float _sub_335;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_335) : "f"(vprod_61), "f"(tz_60));
        float _add_332;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_332) : "f"(_sub_333), "f"(_sub_335));
        float _add_331;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(vacc_3[1]), "f"(_add_332));
        vacc_3[1] = _add_331;
        vacc_3[0] = ts_60;
        float vprod_62;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_62) : "f"(x_row[6]), "f"(du_row[6]));
        float ts_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_61) : "f"(vacc_3[0]), "f"(vprod_62));
        float tz_61;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_61) : "f"(ts_61), "f"(vacc_3[0]));
        float _sub_339;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_339) : "f"(ts_61), "f"(tz_61));
        float _sub_338;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_338) : "f"(vacc_3[0]), "f"(_sub_339));
        float _sub_340;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_340) : "f"(vprod_62), "f"(tz_61));
        float _add_337;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_337) : "f"(_sub_338), "f"(_sub_340));
        float _add_336;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_336) : "f"(vacc_3[1]), "f"(_add_337));
        vacc_3[1] = _add_336;
        vacc_3[0] = ts_61;
        float vprod_63;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_63) : "f"(x_row[7]), "f"(du_row[7]));
        float ts_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_62) : "f"(vacc_3[0]), "f"(vprod_63));
        float tz_62;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_62) : "f"(ts_62), "f"(vacc_3[0]));
        float _sub_344;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_344) : "f"(ts_62), "f"(tz_62));
        float _sub_343;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_343) : "f"(vacc_3[0]), "f"(_sub_344));
        float _sub_345;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_345) : "f"(vprod_63), "f"(tz_62));
        float _add_342;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_342) : "f"(_sub_343), "f"(_sub_345));
        float _add_341;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_341) : "f"(vacc_3[1]), "f"(_add_342));
        vacc_3[1] = _add_341;
        vacc_3[0] = ts_62;
        float vprod_64;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_64) : "f"(x_row[8]), "f"(du_row[8]));
        float ts_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_63) : "f"(vacc_3[0]), "f"(vprod_64));
        float tz_63;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_63) : "f"(ts_63), "f"(vacc_3[0]));
        float _sub_349;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_349) : "f"(ts_63), "f"(tz_63));
        float _sub_348;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_348) : "f"(vacc_3[0]), "f"(_sub_349));
        float _sub_350;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_350) : "f"(vprod_64), "f"(tz_63));
        float _add_347;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_347) : "f"(_sub_348), "f"(_sub_350));
        float _add_346;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(vacc_3[1]), "f"(_add_347));
        vacc_3[1] = _add_346;
        vacc_3[0] = ts_63;
        float vprod_65;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_65) : "f"(x_row[9]), "f"(du_row[9]));
        float ts_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_64) : "f"(vacc_3[0]), "f"(vprod_65));
        float tz_64;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_64) : "f"(ts_64), "f"(vacc_3[0]));
        float _sub_354;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_354) : "f"(ts_64), "f"(tz_64));
        float _sub_353;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_353) : "f"(vacc_3[0]), "f"(_sub_354));
        float _sub_355;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_355) : "f"(vprod_65), "f"(tz_64));
        float _add_352;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(_sub_353), "f"(_sub_355));
        float _add_351;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_351) : "f"(vacc_3[1]), "f"(_add_352));
        vacc_3[1] = _add_351;
        vacc_3[0] = ts_64;
        float vprod_66;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_66) : "f"(x_row[10]), "f"(du_row[10]));
        float ts_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_65) : "f"(vacc_3[0]), "f"(vprod_66));
        float tz_65;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_65) : "f"(ts_65), "f"(vacc_3[0]));
        float _sub_359;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_359) : "f"(ts_65), "f"(tz_65));
        float _sub_358;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_358) : "f"(vacc_3[0]), "f"(_sub_359));
        float _sub_360;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_360) : "f"(vprod_66), "f"(tz_65));
        float _add_357;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_357) : "f"(_sub_358), "f"(_sub_360));
        float _add_356;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_356) : "f"(vacc_3[1]), "f"(_add_357));
        vacc_3[1] = _add_356;
        vacc_3[0] = ts_65;
        float vprod_67;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_67) : "f"(x_row[11]), "f"(du_row[11]));
        float ts_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_66) : "f"(vacc_3[0]), "f"(vprod_67));
        float tz_66;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_66) : "f"(ts_66), "f"(vacc_3[0]));
        float _sub_364;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_364) : "f"(ts_66), "f"(tz_66));
        float _sub_363;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_363) : "f"(vacc_3[0]), "f"(_sub_364));
        float _sub_365;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(vprod_67), "f"(tz_66));
        float _add_362;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_362) : "f"(_sub_363), "f"(_sub_365));
        float _add_361;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(vacc_3[1]), "f"(_add_362));
        vacc_3[1] = _add_361;
        vacc_3[0] = ts_66;
        float vprod_68;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_68) : "f"(x_row[12]), "f"(du_row[12]));
        float ts_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_67) : "f"(vacc_3[0]), "f"(vprod_68));
        float tz_67;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_67) : "f"(ts_67), "f"(vacc_3[0]));
        float _sub_369;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_369) : "f"(ts_67), "f"(tz_67));
        float _sub_368;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_368) : "f"(vacc_3[0]), "f"(_sub_369));
        float _sub_370;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_370) : "f"(vprod_68), "f"(tz_67));
        float _add_367;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_367) : "f"(_sub_368), "f"(_sub_370));
        float _add_366;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_366) : "f"(vacc_3[1]), "f"(_add_367));
        vacc_3[1] = _add_366;
        vacc_3[0] = ts_67;
        float vprod_69;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_69) : "f"(x_row[13]), "f"(du_row[13]));
        float ts_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_68) : "f"(vacc_3[0]), "f"(vprod_69));
        float tz_68;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_68) : "f"(ts_68), "f"(vacc_3[0]));
        float _sub_374;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_374) : "f"(ts_68), "f"(tz_68));
        float _sub_373;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_373) : "f"(vacc_3[0]), "f"(_sub_374));
        float _sub_375;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_375) : "f"(vprod_69), "f"(tz_68));
        float _add_372;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_372) : "f"(_sub_373), "f"(_sub_375));
        float _add_371;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_371) : "f"(vacc_3[1]), "f"(_add_372));
        vacc_3[1] = _add_371;
        vacc_3[0] = ts_68;
        float vprod_70;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_70) : "f"(x_row[14]), "f"(du_row[14]));
        float ts_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_69) : "f"(vacc_3[0]), "f"(vprod_70));
        float tz_69;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_69) : "f"(ts_69), "f"(vacc_3[0]));
        float _sub_379;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_379) : "f"(ts_69), "f"(tz_69));
        float _sub_378;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_378) : "f"(vacc_3[0]), "f"(_sub_379));
        float _sub_380;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_380) : "f"(vprod_70), "f"(tz_69));
        float _add_377;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_377) : "f"(_sub_378), "f"(_sub_380));
        float _add_376;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_376) : "f"(vacc_3[1]), "f"(_add_377));
        vacc_3[1] = _add_376;
        vacc_3[0] = ts_69;
        float vprod_71;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_71) : "f"(x_row[15]), "f"(du_row[15]));
        float ts_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_70) : "f"(vacc_3[0]), "f"(vprod_71));
        float tz_70;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_70) : "f"(ts_70), "f"(vacc_3[0]));
        float _sub_384;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_384) : "f"(ts_70), "f"(tz_70));
        float _sub_383;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_383) : "f"(vacc_3[0]), "f"(_sub_384));
        float _sub_385;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_385) : "f"(vprod_71), "f"(tz_70));
        float _add_382;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_382) : "f"(_sub_383), "f"(_sub_385));
        float _add_381;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_381) : "f"(vacc_3[1]), "f"(_add_382));
        vacc_3[1] = _add_381;
        vacc_3[0] = ts_70;
        float vprod_72;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_72) : "f"(x_row[16]), "f"(du_row[16]));
        float ts_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_71) : "f"(vacc_3[0]), "f"(vprod_72));
        float tz_71;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_71) : "f"(ts_71), "f"(vacc_3[0]));
        float _sub_389;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_389) : "f"(ts_71), "f"(tz_71));
        float _sub_388;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_388) : "f"(vacc_3[0]), "f"(_sub_389));
        float _sub_390;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_390) : "f"(vprod_72), "f"(tz_71));
        float _add_387;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_387) : "f"(_sub_388), "f"(_sub_390));
        float _add_386;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_386) : "f"(vacc_3[1]), "f"(_add_387));
        vacc_3[1] = _add_386;
        vacc_3[0] = ts_71;
        float vprod_73;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_73) : "f"(x_row[17]), "f"(du_row[17]));
        float ts_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_72) : "f"(vacc_3[0]), "f"(vprod_73));
        float tz_72;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_72) : "f"(ts_72), "f"(vacc_3[0]));
        float _sub_394;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_394) : "f"(ts_72), "f"(tz_72));
        float _sub_393;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_393) : "f"(vacc_3[0]), "f"(_sub_394));
        float _sub_395;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_395) : "f"(vprod_73), "f"(tz_72));
        float _add_392;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_392) : "f"(_sub_393), "f"(_sub_395));
        float _add_391;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_391) : "f"(vacc_3[1]), "f"(_add_392));
        vacc_3[1] = _add_391;
        vacc_3[0] = ts_72;
        float vprod_74;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_74) : "f"(x_row[18]), "f"(du_row[18]));
        float ts_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_73) : "f"(vacc_3[0]), "f"(vprod_74));
        float tz_73;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_73) : "f"(ts_73), "f"(vacc_3[0]));
        float _sub_399;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_399) : "f"(ts_73), "f"(tz_73));
        float _sub_398;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_398) : "f"(vacc_3[0]), "f"(_sub_399));
        float _sub_400;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_400) : "f"(vprod_74), "f"(tz_73));
        float _add_397;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_397) : "f"(_sub_398), "f"(_sub_400));
        float _add_396;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_396) : "f"(vacc_3[1]), "f"(_add_397));
        vacc_3[1] = _add_396;
        vacc_3[0] = ts_73;
        float vprod_75;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(vprod_75) : "f"(x_row[19]), "f"(du_row[19]));
        float ts_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_74) : "f"(vacc_3[0]), "f"(vprod_75));
        float tz_74;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_74) : "f"(ts_74), "f"(vacc_3[0]));
        float _sub_404;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_404) : "f"(ts_74), "f"(tz_74));
        float _sub_403;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_403) : "f"(vacc_3[0]), "f"(_sub_404));
        float _sub_405;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_405) : "f"(vprod_75), "f"(tz_74));
        float _add_402;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_402) : "f"(_sub_403), "f"(_sub_405));
        float _add_401;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_401) : "f"(vacc_3[1]), "f"(_add_402));
        vacc_3[1] = _add_401;
        vacc_3[0] = ts_74;
        dots_p[3] = vacc_3[0];
        dots_p[7] = vacc_3[1];
        unsigned int dw_15[2];
        dw_15[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[0]), mul_f32_rn_noftz(pin[3], du_row[1]))));
        dw_15[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[2]), mul_f32_rn_noftz(pin[3], du_row[3]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + xrow_3))[0] = reinterpret_cast<int2*>(dw_15)[0];
        unsigned int dw_16[2];
        dw_16[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[4]), mul_f32_rn_noftz(pin[3], du_row[5]))));
        dw_16[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[6]), mul_f32_rn_noftz(pin[3], du_row[7]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_3 + 4ll)))[0] = reinterpret_cast<int2*>(dw_16)[0];
        unsigned int dw_17[2];
        dw_17[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[8]), mul_f32_rn_noftz(pin[3], du_row[9]))));
        dw_17[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[10]), mul_f32_rn_noftz(pin[3], du_row[11]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_3 + 8ll)))[0] = reinterpret_cast<int2*>(dw_17)[0];
        unsigned int dw_18[2];
        dw_18[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[12]), mul_f32_rn_noftz(pin[3], du_row[13]))));
        dw_18[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[14]), mul_f32_rn_noftz(pin[3], du_row[15]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_3 + 12ll)))[0] = reinterpret_cast<int2*>(dw_18)[0];
        unsigned int dw_19[2];
        dw_19[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[16]), mul_f32_rn_noftz(pin[3], du_row[17]))));
        dw_19[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(pin[3], du_row[18]), mul_f32_rn_noftz(pin[3], du_row[19]))));
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xrow_3 + 16ll)))[0] = reinterpret_cast<int2*>(dw_19)[0];
        float pa_[2];
        float pb_[2];
        int upper = lane_1 & 16;
        float send_s = ((upper != 0) ? dots_p[0] : dots_p[2]);
        float send_c = ((upper != 0) ? dots_p[4] : dots_p[6]);
        float keep_s = ((upper != 0) ? dots_p[2] : dots_p[0]);
        float keep_c = ((upper != 0) ? dots_p[6] : dots_p[4]);
        float rs = __shfl_xor_sync(0xFFFFFFFF, send_s, 16);
        float rc = __shfl_xor_sync(0xFFFFFFFF, send_c, 16);
        pa_[0] = keep_s;
        float _add_406;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_406) : "f"(keep_c), "f"(rc));
        pa_[1] = _add_406;
        float ts_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_75) : "f"(pa_[0]), "f"(rs));
        float tz_75;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_75) : "f"(ts_75), "f"(pa_[0]));
        float _sub_410;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_410) : "f"(ts_75), "f"(tz_75));
        float _sub_409;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_409) : "f"(pa_[0]), "f"(_sub_410));
        float _sub_411;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_411) : "f"(rs), "f"(tz_75));
        float _add_408;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_408) : "f"(_sub_409), "f"(_sub_411));
        float _add_407;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_407) : "f"(pa_[1]), "f"(_add_408));
        pa_[1] = _add_407;
        pa_[0] = ts_75;
        int upper_0 = lane_1 & 16;
        float send_s_0 = ((upper_0 != 0) ? dots_p[1] : dots_p[3]);
        float send_c_0 = ((upper_0 != 0) ? dots_p[5] : dots_p[7]);
        float keep_s_0 = ((upper_0 != 0) ? dots_p[3] : dots_p[1]);
        float keep_c_0 = ((upper_0 != 0) ? dots_p[7] : dots_p[5]);
        float rs_0 = __shfl_xor_sync(0xFFFFFFFF, send_s_0, 16);
        float rc_0 = __shfl_xor_sync(0xFFFFFFFF, send_c_0, 16);
        pb_[0] = keep_s_0;
        float _add_412;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_412) : "f"(keep_c_0), "f"(rc_0));
        pb_[1] = _add_412;
        float ts_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_76) : "f"(pb_[0]), "f"(rs_0));
        float tz_76;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_76) : "f"(ts_76), "f"(pb_[0]));
        float _sub_416;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_416) : "f"(ts_76), "f"(tz_76));
        float _sub_415;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_415) : "f"(pb_[0]), "f"(_sub_416));
        float _sub_417;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_417) : "f"(rs_0), "f"(tz_76));
        float _add_414;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_414) : "f"(_sub_415), "f"(_sub_417));
        float _add_413;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_413) : "f"(pb_[1]), "f"(_add_414));
        pb_[1] = _add_413;
        pb_[0] = ts_76;
        int upper_1 = lane_1 & 8;
        float out[2];
        float send_s2 = ((upper_1 != 0) ? pa_[0] : pb_[0]);
        float send_c2 = ((upper_1 != 0) ? pa_[1] : pb_[1]);
        out[0] = ((upper_1 != 0) ? pb_[0] : pa_[0]);
        float keep_c2 = ((upper_1 != 0) ? pb_[1] : pa_[1]);
        float rs2 = __shfl_xor_sync(0xFFFFFFFF, send_s2, 8);
        float rc2 = __shfl_xor_sync(0xFFFFFFFF, send_c2, 8);
        float _add_418;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_418) : "f"(keep_c2), "f"(rc2));
        out[1] = _add_418;
        float ts_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_77) : "f"(out[0]), "f"(rs2));
        float tz_77;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_77) : "f"(ts_77), "f"(out[0]));
        float _sub_422;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_422) : "f"(ts_77), "f"(tz_77));
        float _sub_421;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_421) : "f"(out[0]), "f"(_sub_422));
        float _sub_423;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_423) : "f"(rs2), "f"(tz_77));
        float _add_420;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_420) : "f"(_sub_421), "f"(_sub_423));
        float _add_419;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(out[1]), "f"(_add_420));
        out[1] = _add_419;
        out[0] = ts_77;
        float rs_1 = __shfl_xor_sync(0xFFFFFFFF, out[0], 4);
        float rc_1 = __shfl_xor_sync(0xFFFFFFFF, out[1], 4);
        float _add_424;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_424) : "f"(out[1]), "f"(rc_1));
        out[1] = _add_424;
        float ts_78;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_78) : "f"(out[0]), "f"(rs_1));
        float tz_78;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_78) : "f"(ts_78), "f"(out[0]));
        float _sub_428;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_428) : "f"(ts_78), "f"(tz_78));
        float _sub_427;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_427) : "f"(out[0]), "f"(_sub_428));
        float _sub_429;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_429) : "f"(rs_1), "f"(tz_78));
        float _add_426;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_426) : "f"(_sub_427), "f"(_sub_429));
        float _add_425;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_425) : "f"(out[1]), "f"(_add_426));
        out[1] = _add_425;
        out[0] = ts_78;
        float rs_2 = __shfl_xor_sync(0xFFFFFFFF, out[0], 2);
        float rc_2 = __shfl_xor_sync(0xFFFFFFFF, out[1], 2);
        float _add_430;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_430) : "f"(out[1]), "f"(rc_2));
        out[1] = _add_430;
        float ts_79;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_79) : "f"(out[0]), "f"(rs_2));
        float tz_79;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_79) : "f"(ts_79), "f"(out[0]));
        float _sub_434;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_434) : "f"(ts_79), "f"(tz_79));
        float _sub_433;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_433) : "f"(out[0]), "f"(_sub_434));
        float _sub_435;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_435) : "f"(rs_2), "f"(tz_79));
        float _add_432;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_432) : "f"(_sub_433), "f"(_sub_435));
        float _add_431;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_431) : "f"(out[1]), "f"(_add_432));
        out[1] = _add_431;
        out[0] = ts_79;
        float rs_3 = __shfl_xor_sync(0xFFFFFFFF, out[0], 1);
        float rc_3 = __shfl_xor_sync(0xFFFFFFFF, out[1], 1);
        float _add_436;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_436) : "f"(out[1]), "f"(rc_3));
        out[1] = _add_436;
        float ts_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_80) : "f"(out[0]), "f"(rs_3));
        float tz_80;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_80) : "f"(ts_80), "f"(out[0]));
        float _sub_440;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_440) : "f"(ts_80), "f"(tz_80));
        float _sub_439;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_439) : "f"(out[0]), "f"(_sub_440));
        float _sub_441;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_441) : "f"(rs_3), "f"(tz_80));
        float _add_438;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_438) : "f"(_sub_439), "f"(_sub_441));
        float _add_437;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_437) : "f"(out[1]), "f"(_add_438));
        out[1] = _add_437;
        out[0] = ts_80;
        if (lane8 == 0) {
            dpre_part[cur_base + warp_0 * 4 + lane_hi] = out[0];
            dpre_comp_s[cur_base + warp_0 * 4 + lane_hi] = out[1];
        }
        __syncthreads();
        if (warp_0 == 0) {
            float dp[1];
            float facc[2];
            facc[0] = dpre_part[cur_base + lc4];
            facc[1] = dpre_comp_s[cur_base + lc4];
            float _add_442;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_442) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 4 + lc4]));
            facc[1] = _add_442;
            float fx_0 = dpre_part[cur_base + 4 + lc4];
            float ts_81;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_81) : "f"(facc[0]), "f"(fx_0));
            float tz_81;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_81) : "f"(ts_81), "f"(facc[0]));
            float _sub_446;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_446) : "f"(ts_81), "f"(tz_81));
            float _sub_445;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_445) : "f"(facc[0]), "f"(_sub_446));
            float _sub_447;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_447) : "f"(fx_0), "f"(tz_81));
            float _add_444;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_444) : "f"(_sub_445), "f"(_sub_447));
            float _add_443;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_443) : "f"(facc[1]), "f"(_add_444));
            facc[1] = _add_443;
            facc[0] = ts_81;
            float _add_448;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_448) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 8 + lc4]));
            facc[1] = _add_448;
            float fx_1 = dpre_part[cur_base + 8 + lc4];
            float ts_82;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_82) : "f"(facc[0]), "f"(fx_1));
            float tz_82;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_82) : "f"(ts_82), "f"(facc[0]));
            float _sub_452;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_452) : "f"(ts_82), "f"(tz_82));
            float _sub_451;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_451) : "f"(facc[0]), "f"(_sub_452));
            float _sub_453;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_453) : "f"(fx_1), "f"(tz_82));
            float _add_450;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_450) : "f"(_sub_451), "f"(_sub_453));
            float _add_449;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(facc[1]), "f"(_add_450));
            facc[1] = _add_449;
            facc[0] = ts_82;
            float _add_454;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_454) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 12 + lc4]));
            facc[1] = _add_454;
            float fx_2 = dpre_part[cur_base + 12 + lc4];
            float ts_83;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_83) : "f"(facc[0]), "f"(fx_2));
            float tz_83;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_83) : "f"(ts_83), "f"(facc[0]));
            float _sub_458;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_458) : "f"(ts_83), "f"(tz_83));
            float _sub_457;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_457) : "f"(facc[0]), "f"(_sub_458));
            float _sub_459;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_459) : "f"(fx_2), "f"(tz_83));
            float _add_456;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_456) : "f"(_sub_457), "f"(_sub_459));
            float _add_455;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_455) : "f"(facc[1]), "f"(_add_456));
            facc[1] = _add_455;
            facc[0] = ts_83;
            float _add_460;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_460) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 16 + lc4]));
            facc[1] = _add_460;
            float fx_3 = dpre_part[cur_base + 16 + lc4];
            float ts_84;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_84) : "f"(facc[0]), "f"(fx_3));
            float tz_84;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_84) : "f"(ts_84), "f"(facc[0]));
            float _sub_464;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_464) : "f"(ts_84), "f"(tz_84));
            float _sub_463;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_463) : "f"(facc[0]), "f"(_sub_464));
            float _sub_465;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_465) : "f"(fx_3), "f"(tz_84));
            float _add_462;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_462) : "f"(_sub_463), "f"(_sub_465));
            float _add_461;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_461) : "f"(facc[1]), "f"(_add_462));
            facc[1] = _add_461;
            facc[0] = ts_84;
            float _add_466;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_466) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 20 + lc4]));
            facc[1] = _add_466;
            float fx_4 = dpre_part[cur_base + 20 + lc4];
            float ts_85;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_85) : "f"(facc[0]), "f"(fx_4));
            float tz_85;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_85) : "f"(ts_85), "f"(facc[0]));
            float _sub_470;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_470) : "f"(ts_85), "f"(tz_85));
            float _sub_469;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_469) : "f"(facc[0]), "f"(_sub_470));
            float _sub_471;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_471) : "f"(fx_4), "f"(tz_85));
            float _add_468;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_468) : "f"(_sub_469), "f"(_sub_471));
            float _add_467;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_467) : "f"(facc[1]), "f"(_add_468));
            facc[1] = _add_467;
            facc[0] = ts_85;
            float _add_472;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_472) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 24 + lc4]));
            facc[1] = _add_472;
            float fx_5 = dpre_part[cur_base + 24 + lc4];
            float ts_86;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_86) : "f"(facc[0]), "f"(fx_5));
            float tz_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_86) : "f"(ts_86), "f"(facc[0]));
            float _sub_476;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_476) : "f"(ts_86), "f"(tz_86));
            float _sub_475;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_475) : "f"(facc[0]), "f"(_sub_476));
            float _sub_477;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_477) : "f"(fx_5), "f"(tz_86));
            float _add_474;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_474) : "f"(_sub_475), "f"(_sub_477));
            float _add_473;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_473) : "f"(facc[1]), "f"(_add_474));
            facc[1] = _add_473;
            facc[0] = ts_86;
            float _add_478;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_478) : "f"(facc[1]), "f"(dpre_comp_s[cur_base + 28 + lc4]));
            facc[1] = _add_478;
            float fx_6 = dpre_part[cur_base + 28 + lc4];
            float ts_87;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_87) : "f"(facc[0]), "f"(fx_6));
            float tz_87;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(tz_87) : "f"(ts_87), "f"(facc[0]));
            float _sub_482;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_482) : "f"(ts_87), "f"(tz_87));
            float _sub_481;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_481) : "f"(facc[0]), "f"(_sub_482));
            float _sub_483;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_483) : "f"(fx_6), "f"(tz_87));
            float _add_480;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_480) : "f"(_sub_481), "f"(_sub_483));
            float _add_479;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_479) : "f"(facc[1]), "f"(_add_480));
            facc[1] = _add_479;
            facc[0] = ts_87;
            float _add_484;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_484) : "f"(facc[0]), "f"(facc[1]));
            dp[0] = _add_484;
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dpre_in[t64 * 4ll + (long long)lane_1])), "f"(dp[0]), "r"((unsigned int)(lane_1 < 4)) : "memory");
        }
        gi[0] = gi[0] + 1;
    }
}

} // extern "C"
