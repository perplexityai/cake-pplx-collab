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

#define BLOCK_M 128
#define HEAD_DIM 128

#include <math_constants.h>

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_fmha_train_d35767c0136e7222051a(float* __restrict__ dQ_accum, __nv_bfloat16* __restrict__ dQ, float* __restrict__ dK_part, float* __restrict__ dV_part, __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV, int seqlen_q, int seqlen_kv, float softmax_scale, int num_heads)
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
    if (blockIdx.x * BLOCK_M < seqlen_q) {
        int tile_base = (bh * seqlen_q + blockIdx.x * BLOCK_M) * HEAD_DIM;
        int dq_row_stride = num_heads * HEAD_DIM;
        int dq_base = ((batch_idx * seqlen_q + blockIdx.x * BLOCK_M) * num_heads + head_idx) * HEAD_DIM;
        #pragma unroll
        for (int copy_turn = 0; copy_turn < BLOCK_M * HEAD_DIM / 512; copy_turn++) {
            int accum_elem = (tid >> 6) * 8192 + (tid & 63) * 4 + copy_turn * 256;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dq_accum_stage_addr + (unsigned int)(accum_elem * 4)), "l"(dQ_accum + (tile_base + accum_elem)));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        float dq_vals[128];
        #pragma unroll
        for (int copy_turn_1 = 0; copy_turn_1 < BLOCK_M * HEAD_DIM / 512; copy_turn_1++) {
            int accum_elem_1 = (tid >> 6) * 8192 + (tid & 63) * 4 + copy_turn_1 * 256;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[copy_turn_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[(copy_turn_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[(copy_turn_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_vals[(copy_turn_1 * 4) + 3]))
                : "r"(dq_accum_stage_addr + (unsigned int)(accum_elem_1 * 4)));
        }
        const float2 _scale2_0 = {softmax_scale, softmax_scale};
        #pragma unroll
        for (int _ls = 0; _ls < 64; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dq_vals)[_ls], _scale2_0);
        unsigned int dq_vals_bf16[64];
        #pragma unroll
        for (int _lp = 0; _lp < 64; _lp++) {
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
        int copy_col = tid % 16 * 8;
        int copy_panel = copy_col / 64;
        int copy_col_bytes = copy_col % 64 * 2;
        unsigned int dq_copy[64];
        #pragma unroll
        for (int row_turn = 0; row_turn < 16; row_turn++) {
            int copy_row = tid / 16 + row_turn * 8;
            int copy_addr = (dq_output_stage_addr + (unsigned int)(copy_panel * 16384) + (unsigned int)(copy_row * 128 + copy_col_bytes ^ (copy_row * 128 + copy_col_bytes >> 7 & 7) << 4));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[row_turn * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[(row_turn * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[(row_turn * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_copy[(row_turn * 4) + 3]))
                : "r"(copy_addr));
        }
        #pragma unroll
        for (int row_turn_1 = 0; row_turn_1 < 16; row_turn_1++) {
            int copy_row_1 = tid / 16 + row_turn_1 * 8;
            int copy_offset = row_turn_1 * 4;
            reinterpret_cast<int4*>(dQ + (dq_base + copy_row_1 * dq_row_stride + copy_col))[0] = reinterpret_cast<int4*>(dq_copy + copy_offset)[0];
        }
    }
    if (blockIdx.x * BLOCK_M >= seqlen_q) {
        int fold_idx = blockIdx.x - seqlen_q / BLOCK_M;
        int fold_row0 = fold_idx / 4 * 128 + fold_idx % 4 * 32;
        int warp_row = tid / 32;
        int col = tid % 32 * 4;
        int part_row0 = batch_idx * 2 * seqlen_kv + fold_row0;
        int out_row0 = batch_idx * seqlen_kv + fold_row0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int stage_row = warp_row + k * 4;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dq_accum_stage_addr + (unsigned int)((stage_row * 128 + col) * 4)), "l"(dV_part + (((part_row0 + stage_row) * num_heads + head_idx) * 128 + col)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dq_accum_stage_addr + (unsigned int)(((32 + stage_row) * 128 + col) * 4)), "l"(dV_part + (((part_row0 + seqlen_kv + stage_row) * num_heads + head_idx) * 128 + col)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dq_accum_stage_addr + (unsigned int)(((64 + stage_row) * 128 + col) * 4)), "l"(dK_part + (((part_row0 + stage_row) * num_heads + head_idx) * 128 + col)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dq_accum_stage_addr + (unsigned int)(((96 + stage_row) * 128 + col) * 4)), "l"(dK_part + (((part_row0 + seqlen_kv + stage_row) * num_heads + head_idx) * 128 + col)));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        float acc[4];
        float plane[4];
        unsigned int out_bf16[2];
        #pragma unroll
        for (int k_1 = 0; k_1 < 8; k_1++) {
            int fold_row = warp_row + k_1 * 4;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(0) + 3]))
                : "r"(dq_accum_stage_addr + (unsigned int)((fold_row * 128 + col) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&plane[0])), "=r"(*reinterpret_cast<uint32_t*>(&plane[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&plane[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&plane[(0) + 3]))
                : "r"(dq_accum_stage_addr + (unsigned int)(((32 + fold_row) * 128 + col) * 4)));
            #pragma unroll
            for (int _la = 0; _la < 2; _la++) {
                reinterpret_cast<float2*>(acc)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(acc)[_la], reinterpret_cast<const float2*>(plane)[_la]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                out_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dV + (((out_row0 + fold_row) * num_heads + head_idx) * 128 + col))[0] = reinterpret_cast<int2*>(out_bf16)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[(0) + 3]))
                : "r"(dq_accum_stage_addr + (unsigned int)(((64 + fold_row) * 128 + col) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&plane[0])), "=r"(*reinterpret_cast<uint32_t*>(&plane[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&plane[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&plane[(0) + 3]))
                : "r"(dq_accum_stage_addr + (unsigned int)(((96 + fold_row) * 128 + col) * 4)));
            #pragma unroll
            for (int _la = 0; _la < 2; _la++) {
                reinterpret_cast<float2*>(acc)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(acc)[_la], reinterpret_cast<const float2*>(plane)[_la]);
            }
            const float2 _scale2_1 = {softmax_scale, softmax_scale};
            #pragma unroll
            for (int _ls = 0; _ls < 2; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc)[_ls], _scale2_1);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                out_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dK + (((out_row0 + fold_row) * num_heads + head_idx) * 128 + col))[0] = reinterpret_cast<int2*>(out_bf16)[0];
        }
    }
}

} // extern "C"
