/*
 * Copyright 2026 Cursor Research
 * Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from the Apache-2.0 Mixture of Kittens BF16 training kernels.
 * Modified: generated CUDA implementation and standalone TVM-FFI bindings.
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

__global__ __launch_bounds__(256) void
kernel_cake_mok_shared_scale_rows(__nv_bfloat16* __restrict__ rows, float* __restrict__ scales, __nv_bfloat16* __restrict__ scaled, int tokens, int width)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int chunks = (width + 4095) / 4096;
    #pragma unroll 1
    for (int token = bid; token < tokens; token += num_bids) {
        long long row = (long long)token * (long long)width;
        float scale = scales[token];
        #pragma unroll 1
        for (int chunk = 0; chunk < chunks; chunk++) {
            int first = chunk * 4096 + tid * 8;
            float values[16];
            if (first < width) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(rows + (row + (long long)first) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&values[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&values[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
            if (first + 2048 < width) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rows + (row + (long long)(first + 2048)) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&values[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&values[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
            if (first < width) {
                unsigned int packed_0[4];
                float low_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_0) : "f"(values[0]), "f"(scale));
                float high_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_0) : "f"(values[1]), "f"(scale));
                packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(low_0, high_0)));
                float low_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_1) : "f"(values[2]), "f"(scale));
                float high_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_1) : "f"(values[3]), "f"(scale));
                packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(low_1, high_1)));
                float low_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_2) : "f"(values[4]), "f"(scale));
                float high_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_2) : "f"(values[5]), "f"(scale));
                packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(low_2, high_2)));
                float low_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_3) : "f"(values[6]), "f"(scale));
                float high_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_3) : "f"(values[7]), "f"(scale));
                packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(low_3, high_3)));
                reinterpret_cast<int4*>(scaled + (row + (long long)first))[0] = reinterpret_cast<int4*>(packed_0)[0];
            }
            if (first + 2048 < width) {
                unsigned int packed_1[4];
                float low_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_4) : "f"(values[8]), "f"(scale));
                float high_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_4) : "f"(values[9]), "f"(scale));
                packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(low_4, high_4)));
                float low_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_5) : "f"(values[10]), "f"(scale));
                float high_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_5) : "f"(values[11]), "f"(scale));
                packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(low_5, high_5)));
                float low_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_6) : "f"(values[12]), "f"(scale));
                float high_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_6) : "f"(values[13]), "f"(scale));
                packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(low_6, high_6)));
                float low_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(low_7) : "f"(values[14]), "f"(scale));
                float high_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(high_7) : "f"(values[15]), "f"(scale));
                packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(low_7, high_7)));
                reinterpret_cast<int4*>(scaled + (row + (long long)(first + 2048)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        }
    }
}

} // extern "C"
