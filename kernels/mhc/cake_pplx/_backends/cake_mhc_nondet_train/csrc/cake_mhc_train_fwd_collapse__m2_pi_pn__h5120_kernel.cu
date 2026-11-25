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
kernel_cake_mhc_train_fwd_collapse__m2_pi_pn__h5120(__nv_bfloat16* __restrict__ x, float* __restrict__ pre_in, float* __restrict__ stats, __nv_bfloat16* __restrict__ u, float* __restrict__ pre_next, int T)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int group = tid / 128;
    int gtid = tid - group * 128;
    long long col0 = (long long)(gtid * 8);
    int nsub = T * 5;
    int pn_gtid = ((gtid < 4) ? gtid : 3);
    long long pn_lane = (long long)pn_gtid;
    #pragma unroll 1
    for (int row0 = bid * 4; row0 < nsub; row0 += num_bids * 4) {
        unsigned int raw[32];
        float pre[8];
        float pnw[2];
        int sub_l_0 = row0 + group;
        if (sub_l_0 < nsub) {
            int row_l_0 = sub_l_0 / 5;
            int part_l_0 = sub_l_0 - row_l_0 * 5;
            long long r64_l_0 = (long long)row_l_0;
            long long xb_l_0 = r64_l_0 * 20480ll + (long long)(part_l_0 * 1024) + col0;
            {
                uint32_t _uv4_0_0;
                uint32_t _uv4_0_1;
                uint32_t _uv4_0_2;
                uint32_t _uv4_0_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_0 + 0ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[0 + 0] = _uv4_0_0;
                raw[0 + 1] = _uv4_0_1;
                raw[0 + 2] = _uv4_0_2;
                raw[0 + 3] = _uv4_0_3;
            }
            {
                uint32_t _uv4_1_0;
                uint32_t _uv4_1_1;
                uint32_t _uv4_1_2;
                uint32_t _uv4_1_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_0 + 5120ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[4 + 0] = _uv4_1_0;
                raw[4 + 1] = _uv4_1_1;
                raw[4 + 2] = _uv4_1_2;
                raw[4 + 3] = _uv4_1_3;
            }
            {
                uint32_t _uv4_2_0;
                uint32_t _uv4_2_1;
                uint32_t _uv4_2_2;
                uint32_t _uv4_2_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_0 + 10240ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[8 + 0] = _uv4_2_0;
                raw[8 + 1] = _uv4_2_1;
                raw[8 + 2] = _uv4_2_2;
                raw[8 + 3] = _uv4_2_3;
            }
            {
                uint32_t _uv4_3_0;
                uint32_t _uv4_3_1;
                uint32_t _uv4_3_2;
                uint32_t _uv4_3_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_0 + 15360ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[12 + 0] = _uv4_3_0;
                raw[12 + 1] = _uv4_3_1;
                raw[12 + 2] = _uv4_3_2;
                raw[12 + 3] = _uv4_3_3;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(pre_in + r64_l_0 * 4ll + 0);
                pre[0 + 0] = _v4.x;
                pre[0 + 1] = _v4.y;
                pre[0 + 2] = _v4.z;
                pre[0 + 3] = _v4.w;
            }
            pnw[0] = stats[r64_l_0 * 64ll + 25ll + pn_lane];
        }
        int sub_l_1 = row0 + 2 + group;
        if (sub_l_1 < nsub) {
            int row_l_1 = sub_l_1 / 5;
            int part_l_1 = sub_l_1 - row_l_1 * 5;
            long long r64_l_1 = (long long)row_l_1;
            long long xb_l_1 = r64_l_1 * 20480ll + (long long)(part_l_1 * 1024) + col0;
            {
                uint32_t _uv4_5_0;
                uint32_t _uv4_5_1;
                uint32_t _uv4_5_2;
                uint32_t _uv4_5_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_5_0), "=r"(_uv4_5_1), "=r"(_uv4_5_2), "=r"(_uv4_5_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_1 + 0ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[16 + 0] = _uv4_5_0;
                raw[16 + 1] = _uv4_5_1;
                raw[16 + 2] = _uv4_5_2;
                raw[16 + 3] = _uv4_5_3;
            }
            {
                uint32_t _uv4_6_0;
                uint32_t _uv4_6_1;
                uint32_t _uv4_6_2;
                uint32_t _uv4_6_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_6_0), "=r"(_uv4_6_1), "=r"(_uv4_6_2), "=r"(_uv4_6_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_1 + 5120ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[20 + 0] = _uv4_6_0;
                raw[20 + 1] = _uv4_6_1;
                raw[20 + 2] = _uv4_6_2;
                raw[20 + 3] = _uv4_6_3;
            }
            {
                uint32_t _uv4_7_0;
                uint32_t _uv4_7_1;
                uint32_t _uv4_7_2;
                uint32_t _uv4_7_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_7_0), "=r"(_uv4_7_1), "=r"(_uv4_7_2), "=r"(_uv4_7_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_1 + 10240ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[24 + 0] = _uv4_7_0;
                raw[24 + 1] = _uv4_7_1;
                raw[24 + 2] = _uv4_7_2;
                raw[24 + 3] = _uv4_7_3;
            }
            {
                uint32_t _uv4_8_0;
                uint32_t _uv4_8_1;
                uint32_t _uv4_8_2;
                uint32_t _uv4_8_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_uv4_8_0), "=r"(_uv4_8_1), "=r"(_uv4_8_2), "=r"(_uv4_8_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xb_l_1 + 15360ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                raw[28 + 0] = _uv4_8_0;
                raw[28 + 1] = _uv4_8_1;
                raw[28 + 2] = _uv4_8_2;
                raw[28 + 3] = _uv4_8_3;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(pre_in + r64_l_1 * 4ll + 0);
                pre[4 + 0] = _v4.x;
                pre[4 + 1] = _v4.y;
                pre[4 + 2] = _v4.z;
                pre[4 + 3] = _v4.w;
            }
            pnw[1] = stats[r64_l_1 * 64ll + 25ll + pn_lane];
        }
        int sub_c_0 = row0 + group;
        if (sub_c_0 < nsub) {
            int row_c_0 = sub_c_0 / 5;
            int part_c_0 = sub_c_0 - row_c_0 * 5;
            long long r64_c_0 = (long long)row_c_0;
            long long ub_c_0 = r64_c_0 * 5120ll + (long long)(part_c_0 * 1024) + col0;
            unsigned int uw_0[4];
            float _fma_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(pre[1]), "f"(__uint_as_float(raw[4] << 16)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[0] << 16))));
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(pre[2]), "f"(__uint_as_float(raw[8] << 16)), "f"(_fma_12));
            float _fma_10;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(pre[3]), "f"(__uint_as_float(raw[12] << 16)), "f"(_fma_11));
            float _fma_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(pre[1]), "f"(__uint_as_float(raw[4] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[0] & 4294901760u))));
            float _fma_14;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(pre[2]), "f"(__uint_as_float(raw[8] & 4294901760u)), "f"(_fma_15));
            float _fma_13;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(pre[3]), "f"(__uint_as_float(raw[12] & 4294901760u)), "f"(_fma_14));
            uw_0[0] = __as_u32(__float22bfloat162_rn(make_float2(_fma_10, _fma_13)));
            float _fma_18;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(pre[1]), "f"(__uint_as_float(raw[5] << 16)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[1] << 16))));
            float _fma_17;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(pre[2]), "f"(__uint_as_float(raw[9] << 16)), "f"(_fma_18));
            float _fma_16;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(pre[3]), "f"(__uint_as_float(raw[13] << 16)), "f"(_fma_17));
            float _fma_21;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(pre[1]), "f"(__uint_as_float(raw[5] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[1] & 4294901760u))));
            float _fma_20;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(pre[2]), "f"(__uint_as_float(raw[9] & 4294901760u)), "f"(_fma_21));
            float _fma_19;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(pre[3]), "f"(__uint_as_float(raw[13] & 4294901760u)), "f"(_fma_20));
            uw_0[1] = __as_u32(__float22bfloat162_rn(make_float2(_fma_16, _fma_19)));
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(pre[1]), "f"(__uint_as_float(raw[6] << 16)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[2] << 16))));
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(pre[2]), "f"(__uint_as_float(raw[10] << 16)), "f"(_fma_24));
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(pre[3]), "f"(__uint_as_float(raw[14] << 16)), "f"(_fma_23));
            float _fma_27;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(pre[1]), "f"(__uint_as_float(raw[6] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[2] & 4294901760u))));
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(pre[2]), "f"(__uint_as_float(raw[10] & 4294901760u)), "f"(_fma_27));
            float _fma_25;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(pre[3]), "f"(__uint_as_float(raw[14] & 4294901760u)), "f"(_fma_26));
            uw_0[2] = __as_u32(__float22bfloat162_rn(make_float2(_fma_22, _fma_25)));
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(pre[1]), "f"(__uint_as_float(raw[7] << 16)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[3] << 16))));
            float _fma_29;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(pre[2]), "f"(__uint_as_float(raw[11] << 16)), "f"(_fma_30));
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(pre[3]), "f"(__uint_as_float(raw[15] << 16)), "f"(_fma_29));
            float _fma_33;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(pre[1]), "f"(__uint_as_float(raw[7] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[0], __uint_as_float(raw[3] & 4294901760u))));
            float _fma_32;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(pre[2]), "f"(__uint_as_float(raw[11] & 4294901760u)), "f"(_fma_33));
            float _fma_31;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(pre[3]), "f"(__uint_as_float(raw[15] & 4294901760u)), "f"(_fma_32));
            uw_0[3] = __as_u32(__float22bfloat162_rn(make_float2(_fma_28, _fma_31)));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(u + ub_c_0))[0] = reinterpret_cast<int4*>(uw_0)[0];
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&pre_next[r64_c_0 * 4ll + pn_lane])), "f"(pnw[0]), "r"((unsigned int)(part_c_0 == 0 && gtid < 4)) : "memory");
        }
        int sub_c_1 = row0 + 2 + group;
        if (sub_c_1 < nsub) {
            int row_c_1 = sub_c_1 / 5;
            int part_c_1 = sub_c_1 - row_c_1 * 5;
            long long r64_c_1 = (long long)row_c_1;
            long long ub_c_1 = r64_c_1 * 5120ll + (long long)(part_c_1 * 1024) + col0;
            unsigned int uw_1[4];
            float _fma_36;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(pre[5]), "f"(__uint_as_float(raw[20] << 16)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[16] << 16))));
            float _fma_35;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(pre[6]), "f"(__uint_as_float(raw[24] << 16)), "f"(_fma_36));
            float _fma_34;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(pre[7]), "f"(__uint_as_float(raw[28] << 16)), "f"(_fma_35));
            float _fma_39;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(pre[5]), "f"(__uint_as_float(raw[20] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[16] & 4294901760u))));
            float _fma_38;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(pre[6]), "f"(__uint_as_float(raw[24] & 4294901760u)), "f"(_fma_39));
            float _fma_37;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(pre[7]), "f"(__uint_as_float(raw[28] & 4294901760u)), "f"(_fma_38));
            uw_1[0] = __as_u32(__float22bfloat162_rn(make_float2(_fma_34, _fma_37)));
            float _fma_42;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(pre[5]), "f"(__uint_as_float(raw[21] << 16)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[17] << 16))));
            float _fma_41;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(pre[6]), "f"(__uint_as_float(raw[25] << 16)), "f"(_fma_42));
            float _fma_40;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(pre[7]), "f"(__uint_as_float(raw[29] << 16)), "f"(_fma_41));
            float _fma_45;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(pre[5]), "f"(__uint_as_float(raw[21] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[17] & 4294901760u))));
            float _fma_44;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(pre[6]), "f"(__uint_as_float(raw[25] & 4294901760u)), "f"(_fma_45));
            float _fma_43;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(pre[7]), "f"(__uint_as_float(raw[29] & 4294901760u)), "f"(_fma_44));
            uw_1[1] = __as_u32(__float22bfloat162_rn(make_float2(_fma_40, _fma_43)));
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(pre[5]), "f"(__uint_as_float(raw[22] << 16)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[18] << 16))));
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(pre[6]), "f"(__uint_as_float(raw[26] << 16)), "f"(_fma_48));
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(pre[7]), "f"(__uint_as_float(raw[30] << 16)), "f"(_fma_47));
            float _fma_51;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(pre[5]), "f"(__uint_as_float(raw[22] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[18] & 4294901760u))));
            float _fma_50;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(pre[6]), "f"(__uint_as_float(raw[26] & 4294901760u)), "f"(_fma_51));
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(pre[7]), "f"(__uint_as_float(raw[30] & 4294901760u)), "f"(_fma_50));
            uw_1[2] = __as_u32(__float22bfloat162_rn(make_float2(_fma_46, _fma_49)));
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(pre[5]), "f"(__uint_as_float(raw[23] << 16)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[19] << 16))));
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(pre[6]), "f"(__uint_as_float(raw[27] << 16)), "f"(_fma_54));
            float _fma_52;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(pre[7]), "f"(__uint_as_float(raw[31] << 16)), "f"(_fma_53));
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(pre[5]), "f"(__uint_as_float(raw[23] & 4294901760u)), "f"(mul_f32_rn_noftz(pre[4], __uint_as_float(raw[19] & 4294901760u))));
            float _fma_56;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(pre[6]), "f"(__uint_as_float(raw[27] & 4294901760u)), "f"(_fma_57));
            float _fma_55;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(pre[7]), "f"(__uint_as_float(raw[31] & 4294901760u)), "f"(_fma_56));
            uw_1[3] = __as_u32(__float22bfloat162_rn(make_float2(_fma_52, _fma_55)));
            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(u + ub_c_1))[0] = reinterpret_cast<int4*>(uw_1)[0];
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&pre_next[r64_c_1 * 4ll + pn_lane])), "f"(pnw[1]), "r"((unsigned int)(part_c_1 == 0 && gtid < 4)) : "memory");
        }
    }
}

} // extern "C"
