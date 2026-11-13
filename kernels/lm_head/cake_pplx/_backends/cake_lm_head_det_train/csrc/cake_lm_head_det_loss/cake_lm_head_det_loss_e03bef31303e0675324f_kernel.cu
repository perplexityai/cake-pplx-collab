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
    #pragma unroll 1
    for (int b = 0; b < rows_c; b += 4096) {
        float v[16];
        v[0] = zero;
        if (b + tid < rows_c) {
            v[0] = term[b + tid];
        }
        v[1] = zero;
        if (b + 256 + tid < rows_c) {
            v[1] = term[b + 256 + tid];
        }
        v[2] = zero;
        if (b + 512 + tid < rows_c) {
            v[2] = term[b + 512 + tid];
        }
        v[3] = zero;
        if (b + 768 + tid < rows_c) {
            v[3] = term[b + 768 + tid];
        }
        v[4] = zero;
        if (b + 1024 + tid < rows_c) {
            v[4] = term[b + 1024 + tid];
        }
        v[5] = zero;
        if (b + 1280 + tid < rows_c) {
            v[5] = term[b + 1280 + tid];
        }
        v[6] = zero;
        if (b + 1536 + tid < rows_c) {
            v[6] = term[b + 1536 + tid];
        }
        v[7] = zero;
        if (b + 1792 + tid < rows_c) {
            v[7] = term[b + 1792 + tid];
        }
        v[8] = zero;
        if (b + 2048 + tid < rows_c) {
            v[8] = term[b + 2048 + tid];
        }
        v[9] = zero;
        if (b + 2304 + tid < rows_c) {
            v[9] = term[b + 2304 + tid];
        }
        v[10] = zero;
        if (b + 2560 + tid < rows_c) {
            v[10] = term[b + 2560 + tid];
        }
        v[11] = zero;
        if (b + 2816 + tid < rows_c) {
            v[11] = term[b + 2816 + tid];
        }
        v[12] = zero;
        if (b + 3072 + tid < rows_c) {
            v[12] = term[b + 3072 + tid];
        }
        v[13] = zero;
        if (b + 3328 + tid < rows_c) {
            v[13] = term[b + 3328 + tid];
        }
        v[14] = zero;
        if (b + 3584 + tid < rows_c) {
            v[14] = term[b + 3584 + tid];
        }
        v[15] = zero;
        if (b + 3840 + tid < rows_c) {
            v[15] = term[b + 3840 + tid];
        }
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(v[0]), "f"(v[8]));
        v[0] = _add_0;
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(v[1]), "f"(v[9]));
        v[1] = _add_1;
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(v[2]), "f"(v[10]));
        v[2] = _add_2;
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(v[3]), "f"(v[11]));
        v[3] = _add_3;
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(v[4]), "f"(v[12]));
        v[4] = _add_4;
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(v[5]), "f"(v[13]));
        v[5] = _add_5;
        float _add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(v[6]), "f"(v[14]));
        v[6] = _add_6;
        float _add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(v[7]), "f"(v[15]));
        v[7] = _add_7;
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(v[0]), "f"(v[4]));
        v[0] = _add_8;
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(v[1]), "f"(v[5]));
        v[1] = _add_9;
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(v[2]), "f"(v[6]));
        v[2] = _add_10;
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(v[3]), "f"(v[7]));
        v[3] = _add_11;
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(v[0]), "f"(v[2]));
        v[0] = _add_12;
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(v[1]), "f"(v[3]));
        v[1] = _add_13;
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(v[0]), "f"(v[1]));
        v[0] = _add_14;
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(s), "f"(v[0]));
        s = _add_15;
    }
    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, s, 16);
    float _add_16;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(s), "f"(_shfl_xor_sync_17));
    s = _add_16;
    float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, s, 8);
    float _add_18;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(s), "f"(_shfl_xor_sync_19));
    s = _add_18;
    float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, s, 4);
    float _add_20;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(s), "f"(_shfl_xor_sync_21));
    s = _add_20;
    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, s, 2);
    float _add_22;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(s), "f"(_shfl_xor_sync_23));
    s = _add_22;
    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, s, 1);
    float _add_24;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(s), "f"(_shfl_xor_sync_25));
    s = _add_24;
    if (lane == 0) {
        partials[warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
        float p = ((lane < 8) ? partials[lane] : zero);
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, p, 16);
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(p), "f"(_shfl_xor_sync_27));
        p = _add_26;
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, p, 8);
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(p), "f"(_shfl_xor_sync_29));
        p = _add_28;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, p, 4);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(p), "f"(_shfl_xor_sync_31));
        p = _add_30;
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, p, 2);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(p), "f"(_shfl_xor_sync_33));
        p = _add_32;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, p, 1);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(p), "f"(_shfl_xor_sync_35));
        p = _add_34;
        if (lane == 0) {
            int n = 0;
            if (first_chunk == 0) {
                n = (int)loss_acc[31];
            }
            int top = 0;
            #pragma unroll 1
            for (int j = 0; j < 31; j++) {
                if (n >> j == 0) {
                    break;
                }
                if ((n >> j & 1) == 0) {
                    if (last_chunk == 0) {
                        break;
                    }
                } else {
                    float lj = loss_acc[j];
                    float _add_36;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(lj), "f"(p));
                    p = _add_36;
                    top = j + 1;
                }
            }
            if (last_chunk == 0) {
                *(reinterpret_cast<float*>(loss_acc) + (top)) = p;
            } else {
                loss_acc[0] = p;
            }
            loss_acc[31] = (float)(n + 1);
            if (last_chunk != 0) {
                float fin = p;
                if (mode == 0) {
                    float quo = __fdiv_rn(p, loss_div);
                    fin = quo;
                }
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(zero), "f"(fin));
                loss_out[0] = _sub_37;
            }
        }
    }
}

} // extern "C"
