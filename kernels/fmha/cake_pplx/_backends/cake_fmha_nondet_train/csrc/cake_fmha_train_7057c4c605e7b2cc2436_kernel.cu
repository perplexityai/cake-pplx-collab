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

#define BLOCK_M 128
#define HEAD_DIM 64

#include <math_constants.h>

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_fmha_train_7057c4c605e7b2cc2436(float* __restrict__ dQ_accum, __nv_bfloat16* __restrict__ dQ, int seqlen_q, float softmax_scale, int num_heads)
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
    float* dq_accum_stage = reinterpret_cast<float*>(smem_raw + 0);
    const int dq_accum_stage_addr = smem + 0;
    __nv_bfloat16* dq_output_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int dq_output_stage_addr = smem + 0;

    // === Task calls (declaration order) ===
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int bh = batch_idx * num_heads + head_idx;
    int tile_base = (bh * seqlen_q + blockIdx.x * BLOCK_M) * HEAD_DIM;
    int dq_row_stride = num_heads * HEAD_DIM;
    int dq_base = ((batch_idx * seqlen_q + blockIdx.x * BLOCK_M) * num_heads + head_idx) * HEAD_DIM;
    #pragma unroll
    for (int copy_turn = 0; copy_turn < BLOCK_M * HEAD_DIM / 512; copy_turn++) {
        int accum_elem = tid * 4 + copy_turn * 128 * 4;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(dq_accum_stage_addr + (unsigned int)(accum_elem * 4)), "l"(dQ_accum + (tile_base + accum_elem)));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    float dq_vals[64];
    #pragma unroll
    for (int copy_turn_1 = 0; copy_turn_1 < BLOCK_M * HEAD_DIM / 512; copy_turn_1++) {
        int accum_elem_1 = tid * 4 + copy_turn_1 * 128 * 4;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[copy_turn_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[(copy_turn_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[(copy_turn_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[(copy_turn_1 * 4) + 3]))
            : "r"(dq_accum_stage_addr + (unsigned int)(accum_elem_1 * 4)));
    }
    const float2 _scale2_0 = {softmax_scale, softmax_scale};
    #pragma unroll
    for (int _ls = 0; _ls < 32; _ls++)
        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dq_vals)[_ls], _scale2_0);
    unsigned int dq_vals_bf16[32];
    #pragma unroll
    for (int _lp = 0; _lp < 32; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_vals[_lp*2 + 0], dq_vals[_lp*2+1 + 0]));
        dq_vals_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    __syncthreads();
    #pragma unroll
    for (int panel = 0; panel < HEAD_DIM / 64; panel++) {
        #pragma unroll
        for (int vec = 0; vec < 8; vec++) {
            int packed_base = panel * 32 + vec * 4;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dq_output_stage_addr + (unsigned int)(panel * 16384) + (unsigned int)(tid * 128 + vec * 16 ^ (tid * 128 + vec * 16 >> 7 & 7) << 4))), "r"(dq_vals_bf16[packed_base]), "r"(dq_vals_bf16[packed_base + 1]), "r"(dq_vals_bf16[packed_base + 2]), "r"(dq_vals_bf16[packed_base + 3]) : "memory");
        }
    }
    __syncthreads();
    int copy_col = tid % 8 * 8;
    int copy_panel = copy_col / 64;
    int copy_col_bytes = copy_col % 64 * 2;
    unsigned int dq_copy[32];
    #pragma unroll
    for (int row_turn = 0; row_turn < 8; row_turn++) {
        int copy_row = tid / 8 + row_turn * 16;
        int copy_addr = (dq_output_stage_addr + (unsigned int)(copy_panel * 16384) + (unsigned int)(copy_row * 128 + copy_col_bytes ^ (copy_row * 128 + copy_col_bytes >> 7 & 7) << 4));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[row_turn * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[(row_turn * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[(row_turn * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[(row_turn * 4) + 3]))
            : "r"(copy_addr));
    }
    #pragma unroll
    for (int row_turn_1 = 0; row_turn_1 < 8; row_turn_1++) {
        int copy_row_1 = tid / 8 + row_turn_1 * 16;
        int copy_offset = row_turn_1 * 4;
        reinterpret_cast<int4*>(dQ + (dq_base + copy_row_1 * dq_row_stride + copy_col))[0] = reinterpret_cast<int4*>(dq_copy + copy_offset)[0];
    }
}

} // extern "C"
