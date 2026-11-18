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

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_engram_gate_det_train_eec3759c50f56c38622b(float* __restrict__ dw_part, float* __restrict__ qw, float* __restrict__ kw, float* __restrict__ dq, float* __restrict__ dk, int NP)
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
    float* sums = reinterpret_cast<float*>(smem_raw + 0);
    const int sums_addr = smem + 0;

    // === Task calls (declaration order) ===
    int column = bid * 128 + lane * 4;
    float acc[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        acc[c] = -0.0f;
    }
    #pragma unroll 4
    for (int p = warp; p < NP; p += 8) {
        float row[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dw_part + ((long long)p * 20480ll + (long long)column) + 0);
            row[0 + 0] = _v4.x;
            row[0 + 1] = _v4.y;
            row[0 + 2] = _v4.z;
            row[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int c_1 = 0; c_1 < 4; c_1++) {
            float _add_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(acc[c_1]), "f"(row[c_1]));
            acc[c_1] = _add_1;
        }
    }
    #pragma unroll
    for (int c_2 = 0; c_2 < 4; c_2++) {
        sums[warp * 128 + lane * 4 + c_2] = acc[c_2];
    }
    __syncthreads();
    if (warp == 0) {
        float qv[4];
        float kv[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(qw + column + 0);
            qv[0 + 0] = _v4.x;
            qv[0 + 1] = _v4.y;
            qv[0 + 2] = _v4.z;
            qv[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(kw + column + 0);
            kv[0 + 0] = _v4.x;
            kv[0 + 1] = _v4.y;
            kv[0 + 2] = _v4.z;
            kv[0 + 3] = _v4.w;
        }
        float dqv[4];
        float dkv[4];
        #pragma unroll
        for (int c_3 = 0; c_3 < 4; c_3++) {
            float tree[8];
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                tree[g] = sums[g * 128 + lane * 4 + c_3];
            }
            #pragma unroll
            for (int g_0 = 0; g_0 < 8; g_0 += 2) {
                float _add_4;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(tree[g_0]), "f"(tree[g_0 + 1]));
                tree[g_0] = _add_4;
            }
            #pragma unroll
            for (int g_1 = 0; g_1 < 8; g_1 += 4) {
                float _add_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(tree[g_1]), "f"(tree[g_1 + 2]));
                tree[g_1] = _add_5;
            }
            #pragma unroll
            for (int g_2 = 0; g_2 < 8; g_2 += 8) {
                float _add_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(tree[g_2]), "f"(tree[g_2 + 4]));
                tree[g_2] = _add_6;
            }
            dqv[c_3] = mul_f32_rn_noftz(tree[0], kv[c_3]);
            dkv[c_3] = mul_f32_rn_noftz(tree[0], qv[c_3]);
        }
        {
            float4 _v4 = make_float4(dqv[0 + 0], dqv[0 + 1], dqv[0 + 2], dqv[0 + 3]);
            *reinterpret_cast<float4*>((dq + column) + 0) = _v4;
        }
        {
            float4 _v4 = make_float4(dkv[0 + 0], dkv[0 + 1], dkv[0 + 2], dkv[0 + 3]);
            *reinterpret_cast<float4*>((dk + column) + 0) = _v4;
        }
    }
}

} // extern "C"
