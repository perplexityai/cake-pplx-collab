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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_det_loss_e03bef31303e0675324f(float* __restrict__ term, float* __restrict__ loss_acc, float* __restrict__ loss_out, int rows_c, int first_chunk, int last_chunk, int mode, float loss_div)
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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (declaration order) ===
    float zero = 0.0f;
    float s = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < rows_c; i += 256) {
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(s), "f"(term[i]));
        s = _add_0;
    }
    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, s, 16);
    float _add_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(s), "f"(_shfl_xor_sync_2));
    s = _add_1;
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, s, 8);
    float _add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(s), "f"(_shfl_xor_sync_4));
    s = _add_3;
    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, s, 4);
    float _add_5;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(s), "f"(_shfl_xor_sync_6));
    s = _add_5;
    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, s, 2);
    float _add_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(s), "f"(_shfl_xor_sync_8));
    s = _add_7;
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, s, 1);
    float _add_9;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(s), "f"(_shfl_xor_sync_10));
    s = _add_9;
    if (lane == 0) {
        partials[warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
        float p = ((lane < 8) ? partials[lane] : zero);
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, p, 16);
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(p), "f"(_shfl_xor_sync_12));
        p = _add_11;
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, p, 8);
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(p), "f"(_shfl_xor_sync_14));
        p = _add_13;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, p, 4);
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(p), "f"(_shfl_xor_sync_16));
        p = _add_15;
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, p, 2);
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(p), "f"(_shfl_xor_sync_18));
        p = _add_17;
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, p, 1);
        float _add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(p), "f"(_shfl_xor_sync_20));
        p = _add_19;
        if (lane == 0) {
            if (first_chunk == 0) {
                float prev = loss_acc[0];
                float _add_21;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(prev), "f"(p));
                p = _add_21;
            }
            loss_acc[0] = p;
            if (last_chunk != 0) {
                float fin = p;
                if (mode == 0) {
                    float quo = __fdiv_rn(p, loss_div);
                    fin = quo;
                }
                float _sub_22;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(zero), "f"(fin));
                loss_out[0] = _sub_22;
            }
        }
    }
}

} // extern "C"
