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
#define HEAD_DIM 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_fmha_train_f0b639ee014a01d20db0(__nv_bfloat16* __restrict__ O, __nv_bfloat16* __restrict__ dO, float* __restrict__ LSE, float* __restrict__ LSE_log2, float* __restrict__ dPsum, float* __restrict__ dQ_accum, int seqlen_q, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int bh = batch_idx * num_heads + head_idx;
    int row_tile = blockIdx.x * BLOCK_M;
    int row_in_warp = lane / 16;
    int lane_in_row = lane % 16;
    int lane_col = lane_in_row * 8;
    float zeros[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        zeros[j] = 0.0f;
    }
    #pragma unroll
    for (int row_turn = 0; row_turn < BLOCK_M / 16; row_turn++) {
        int row = row_tile + row_turn * 8 * 2 + warp * 2 + row_in_warp;
        int row_base = ((batch_idx * seqlen_q + row) * num_heads + head_idx) * HEAD_DIM;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(O + row_base + lane_col);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dO + row_base + lane_col);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        float partial = 0.0f;
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            partial = __fmaf_rn(_vec_load_0[j_1], _vec_load_1[j_1], partial);
        }
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, partial, 8);
        partial = partial + _shfl_xor_sync_2;
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, partial, 4);
        partial = partial + _shfl_xor_sync_3;
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, partial, 2);
        partial = partial + _shfl_xor_sync_4;
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, partial, 1);
        partial = partial + _shfl_xor_sync_5;
        if (lane_in_row == 0) {
            *(reinterpret_cast<float*>(dPsum + (bh * seqlen_q + row)) + (0)) = partial;
        }
    }
    int tile_base = (bh * seqlen_q + row_tile) * HEAD_DIM;
    #pragma unroll
    for (int clear_turn = 0; clear_turn < BLOCK_M * HEAD_DIM / 1024; clear_turn++) {
        int elem = tile_base + clear_turn * 256 * 4 + tid * 4;
        {
            float4 _v4 = make_float4(zeros[0 + 0], zeros[0 + 1], zeros[0 + 2], zeros[0 + 3]);
            *reinterpret_cast<float4*>((dQ_accum + elem) + 0) = _v4;
        }
    }
    if (tid < BLOCK_M) {
        int row_1 = row_tile + tid;
        float lse = LSE[bh * seqlen_q + row_1];
        float lse_log2 = ((lse != -CUDART_INF_F) ? lse * 1.442695f : 0.0f);
        *(reinterpret_cast<float*>(LSE_log2 + (bh * seqlen_q + row_1)) + (0)) = lse_log2;
    }
}

} // extern "C"
