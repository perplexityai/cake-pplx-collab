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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mok_shared_scale_dot(__nv_bfloat16* __restrict__ left, __nv_bfloat16* __restrict__ right, float* __restrict__ dots, int tokens, int width)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int passes = (width + 255) / 256;
    #pragma unroll 1
    for (int token = bid * 8 + warp; token < tokens; token += num_bids * 8) {
        long long row = (long long)token * (long long)width;
        float acc[8];
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll 1
        for (int p = 0; p < passes; p++) {
            int col = p * 256 + lane * 8;
            if (col < width) {
                float a[8];
                float b[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(left + (row + (long long)col) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&a[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&a[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(right + (row + (long long)col) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&b[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&b[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float _add_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(acc[0]), "f"(mul_f32_rn_noftz(a[0], b[0])));
                acc[0] = _add_2;
                float _add_3;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(acc[1]), "f"(mul_f32_rn_noftz(a[1], b[1])));
                acc[1] = _add_3;
                float _add_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(acc[2]), "f"(mul_f32_rn_noftz(a[2], b[2])));
                acc[2] = _add_4;
                float _add_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(acc[3]), "f"(mul_f32_rn_noftz(a[3], b[3])));
                acc[3] = _add_5;
                float _add_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(acc[4]), "f"(mul_f32_rn_noftz(a[4], b[4])));
                acc[4] = _add_6;
                float _add_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(acc[5]), "f"(mul_f32_rn_noftz(a[5], b[5])));
                acc[5] = _add_7;
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(acc[6]), "f"(mul_f32_rn_noftz(a[6], b[6])));
                acc[6] = _add_8;
                float _add_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(acc[7]), "f"(mul_f32_rn_noftz(a[7], b[7])));
                acc[7] = _add_9;
            }
        }
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(acc[0]), "f"(acc[1]));
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(acc[2]), "f"(acc[3]));
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_add_11), "f"(_add_12));
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(acc[4]), "f"(acc[5]));
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(acc[6]), "f"(acc[7]));
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_add_14), "f"(_add_15));
        float total;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(total) : "f"(_add_10), "f"(_add_13));
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, total, 16);
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(total), "f"(_shfl_xor_sync_17));
        total = _add_16;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(total), "f"(_shfl_xor_sync_19));
        total = _add_18;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(total), "f"(_shfl_xor_sync_21));
        total = _add_20;
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(total), "f"(_shfl_xor_sync_23));
        total = _add_22;
        float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(total), "f"(_shfl_xor_sync_25));
        total = _add_24;
        if (lane == 0) {
            *(reinterpret_cast<float*>(dots + token) + (0)) = total;
        }
    }
}

} // extern "C"
