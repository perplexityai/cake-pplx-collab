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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_nope_train_36d862698eea1ea770ef(float* __restrict__ lse, float* __restrict__ delta, float* __restrict__ sink, float* __restrict__ dsink, float* __restrict__ partials, unsigned int* __restrict__ counter, int num_tokens)
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
    float* fold = reinterpret_cast<float*>(smem_raw + 0);
    const int fold_addr = smem + 0;
    int* last = reinterpret_cast<int*>(smem_raw + 1024);
    const int last_addr = smem + 1024;

    // === Task calls (declaration order) ===
    int lane_row = tid / 64;
    int head = tid % 64;
    float sink_h = sink[head];
    int grid = gridDim.x;
    int n_chunks = (num_tokens + 63) / 64;
    #pragma unroll 1
    for (int chunk = bid; chunk < n_chunks; chunk += grid) {
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int t = chunk * 64 + i * 4 + lane_row;
            if (t < num_tokens) {
                long long row = (long long)t * 64ll + (long long)head;
                float lse_v = lse[row];
                float delta_v = delta[row];
                float weight = 0.0f;
                if (sink_h != -CUDART_INF_F && lse_v != -CUDART_INF_F) {
                    float _exp_0 = expf(sink_h - lse_v);
                    weight = -_exp_0;
                }
                acc = __fmaf_rn(weight, delta_v, acc);
            }
        }
        fold[tid] = acc;
        __syncthreads();
        if (tid < 64) {
            float folded = fold[head];
            folded = folded + fold[64 + head];
            folded = folded + fold[128 + head];
            folded = folded + fold[192 + head];
            partials[(long long)chunk * 64ll + (long long)head] = folded;
        }
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        uint32_t drawn;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(drawn) : "l"(counter), "r"(static_cast<uint32_t>((unsigned int)(grid - 1))) : "memory");
        last[0] = 0;
        if (drawn == (unsigned int)(grid - 1)) {
            last[0] = 1;
        }
    }
    __syncthreads();
    int is_last = last[0];
    float total = 0.0f;
    if (is_last != 0) {
        __threadfence();
        #pragma unroll 4
        for (int done = lane_row; done < n_chunks; done += 4) {
            float value[1];
            {
                uint32_t _scalar_bits_1;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_1) : "l"((const void*)(partials + ((long long)done * 64ll + (long long)head) + (0))) : "memory");
                value[0] = __uint_as_float(_scalar_bits_1);
            }
            total = total + value[0];
        }
    }
    fold[tid] = total;
    __syncthreads();
    if (is_last != 0 && tid < 64) {
        float folded_1 = fold[head];
        folded_1 = folded_1 + fold[64 + head];
        folded_1 = folded_1 + fold[128 + head];
        folded_1 = folded_1 + fold[192 + head];
        dsink[head] = folded_1;
    }
}

} // extern "C"
