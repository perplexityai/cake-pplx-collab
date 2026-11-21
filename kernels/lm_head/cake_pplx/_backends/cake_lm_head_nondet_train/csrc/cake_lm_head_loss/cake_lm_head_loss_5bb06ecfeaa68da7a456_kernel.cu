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

#define LAUNCH_MIN_BLOCKS 5

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_lm_head_loss_5bb06ecfeaa68da7a456(float* __restrict__ dx, float* __restrict__ ws, int ws_slab, int num_vecs, int n_slabs, int row_vecs, long long* __restrict__ labels, float* __restrict__ rlab, __nv_bfloat16* __restrict__ W, int V, int lr, int* __restrict__ CNT)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int stride = gridDim.x * 256;
    unsigned long long slab64 = (unsigned long long)ws_slab;
    int vi0 = bid * 256 + tid;
    long long rr_y = -1ll;
    float rr_r = 0.0f;
    int rr_col = 0;
    if (lr != 0) {
        if (vi0 < num_vecs) {
            int rr_row = vi0 / row_vecs;
            rr_col = vi0 - rr_row * row_vecs;
            rr_y = labels[rr_row];
            rr_r = rlab[rr_row];
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int n_vecs = num_vecs;
    int live_rows = CNT[0];
    int live_vecs = (live_rows + 255) / 256 * 256 * row_vecs;
    n_vecs = ((live_vecs < num_vecs) ? live_vecs : num_vecs);
    if (lr == 0) {
        if (n_slabs > 0) {
            for (int vi = vi0; vi < n_vecs; vi += stride) {
                unsigned long long off = (unsigned long long)vi * 8ull;
                float o[8];
                float _vec_load_0[4];
                float _vec_load_1[4];
                if (n_slabs > 0) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(ws + off + 0);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(ws + (off + 4ull) + 0);
                        _vec_load_1[0 + 0] = _v4.x;
                        _vec_load_1[0 + 1] = _v4.y;
                        _vec_load_1[0 + 2] = _v4.z;
                        _vec_load_1[0 + 3] = _v4.w;
                    }
                } else {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dx + off + 0);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dx + (off + 4ull) + 0);
                        _vec_load_1[0 + 0] = _v4.x;
                        _vec_load_1[0 + 1] = _v4.y;
                        _vec_load_1[0 + 2] = _v4.z;
                        _vec_load_1[0 + 3] = _v4.w;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    o[j] = _vec_load_0[j];
                    o[j + 4] = _vec_load_1[j];
                }
                #pragma unroll 4
                for (int s = 1; s < n_slabs; s++) {
                    unsigned long long soff = off + (unsigned long long)s * slab64;
                    float _vec_load_2[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(ws + soff + 0);
                        _vec_load_2[0 + 0] = _v4.x;
                        _vec_load_2[0 + 1] = _v4.y;
                        _vec_load_2[0 + 2] = _v4.z;
                        _vec_load_2[0 + 3] = _v4.w;
                    }
                    float _vec_load_3[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(ws + (soff + 4ull) + 0);
                        _vec_load_3[0 + 0] = _v4.x;
                        _vec_load_3[0 + 1] = _v4.y;
                        _vec_load_3[0 + 2] = _v4.z;
                        _vec_load_3[0 + 3] = _v4.w;
                    }
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        o[j_1] = o[j_1] + _vec_load_2[j_1];
                        o[j_1 + 4] = o[j_1 + 4] + _vec_load_3[j_1];
                    }
                }
                {
                    float4 _v4 = make_float4(o[0 + 0], o[0 + 1], o[0 + 2], o[0 + 3]);
                    *reinterpret_cast<float4*>((dx + off) + 0) = _v4;
                }
                {
                    float4 _v4 = make_float4(o[4 + 0], o[4 + 1], o[4 + 2], o[4 + 3]);
                    *reinterpret_cast<float4*>((dx + (off + 4ull)) + 0) = _v4;
                }
            }
        }
    } else if (n_slabs <= 0) {
        if (vi0 < n_vecs) {
            int rl_live = (((unsigned long long)rr_y < (unsigned long long)V) ? 1 : 0);
            if (rr_r == 0.0f) {
                rl_live = 0;
            }
            if (rl_live != 0) {
                unsigned long long off0 = (unsigned long long)vi0 * 8ull;
                unsigned int words0[4];
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(W + ((unsigned long long)rr_y * (unsigned long long)row_vecs + (unsigned long long)rr_col) * 8ull + 0);
                    words0[0 + 0] = _uv4_6.x;
                    words0[0 + 1] = _uv4_6.y;
                    words0[0 + 2] = _uv4_6.z;
                    words0[0 + 3] = _uv4_6.w;
                }
                float rf_o[8];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dx + off0 + 0);
                    rf_o[0 + 0] = _v4.x;
                    rf_o[0 + 1] = _v4.y;
                    rf_o[0 + 2] = _v4.z;
                    rf_o[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dx + (off0 + 4ull) + 0);
                    rf_o[4 + 0] = _v4.x;
                    rf_o[4 + 1] = _v4.y;
                    rf_o[4 + 2] = _v4.z;
                    rf_o[4 + 3] = _v4.w;
                }
                float _fma_9;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(rr_r), "f"(__uint_as_float(words0[0] << 16)), "f"(rf_o[0]));
                rf_o[0] = _fma_9;
                float _fma_10;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(rr_r), "f"(__uint_as_float(words0[0] & 4294901760u)), "f"(rf_o[1]));
                rf_o[1] = _fma_10;
                float _fma_11;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(rr_r), "f"(__uint_as_float(words0[1] << 16)), "f"(rf_o[2]));
                rf_o[2] = _fma_11;
                float _fma_12;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(rr_r), "f"(__uint_as_float(words0[1] & 4294901760u)), "f"(rf_o[3]));
                rf_o[3] = _fma_12;
                float _fma_13;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(rr_r), "f"(__uint_as_float(words0[2] << 16)), "f"(rf_o[4]));
                rf_o[4] = _fma_13;
                float _fma_14;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(rr_r), "f"(__uint_as_float(words0[2] & 4294901760u)), "f"(rf_o[5]));
                rf_o[5] = _fma_14;
                float _fma_15;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(rr_r), "f"(__uint_as_float(words0[3] << 16)), "f"(rf_o[6]));
                rf_o[6] = _fma_15;
                float _fma_16;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(rr_r), "f"(__uint_as_float(words0[3] & 4294901760u)), "f"(rf_o[7]));
                rf_o[7] = _fma_16;
                {
                    float4 _v4 = make_float4(rf_o[0 + 0], rf_o[0 + 1], rf_o[0 + 2], rf_o[0 + 3]);
                    *reinterpret_cast<float4*>((dx + off0) + 0) = _v4;
                }
                {
                    float4 _v4 = make_float4(rf_o[4 + 0], rf_o[4 + 1], rf_o[4 + 2], rf_o[4 + 3]);
                    *reinterpret_cast<float4*>((dx + (off0 + 4ull)) + 0) = _v4;
                }
            }
        }
        #pragma unroll 2
        for (int vi_1 = vi0 + stride; vi_1 < n_vecs; vi_1 += stride) {
            long long rr_y_0 = -1ll;
            float rr_r_0 = 0.0f;
            int rr_col_0 = 0;
            if (lr != 0) {
                if (vi_1 < n_vecs) {
                    int rr_row_1 = vi_1 / row_vecs;
                    rr_col_0 = vi_1 - rr_row_1 * row_vecs;
                    rr_y_0 = labels[rr_row_1];
                    rr_r_0 = rlab[rr_row_1];
                }
            }
            int rl_live_1 = (((unsigned long long)rr_y_0 < (unsigned long long)V) ? 1 : 0);
            if (rr_r_0 == 0.0f) {
                rl_live_1 = 0;
            }
            if (rl_live_1 != 0) {
                unsigned long long off_1 = (unsigned long long)vi_1 * 8ull;
                unsigned int words[4];
                {
                    uint4 _uv4_17 = *reinterpret_cast<const uint4*>(W + ((unsigned long long)rr_y_0 * (unsigned long long)row_vecs + (unsigned long long)rr_col_0) * 8ull + 0);
                    words[0 + 0] = _uv4_17.x;
                    words[0 + 1] = _uv4_17.y;
                    words[0 + 2] = _uv4_17.z;
                    words[0 + 3] = _uv4_17.w;
                }
                float rf_o_1[8];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dx + off_1 + 0);
                    rf_o_1[0 + 0] = _v4.x;
                    rf_o_1[0 + 1] = _v4.y;
                    rf_o_1[0 + 2] = _v4.z;
                    rf_o_1[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dx + (off_1 + 4ull) + 0);
                    rf_o_1[4 + 0] = _v4.x;
                    rf_o_1[4 + 1] = _v4.y;
                    rf_o_1[4 + 2] = _v4.z;
                    rf_o_1[4 + 3] = _v4.w;
                }
                float _fma_20;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(rr_r_0), "f"(__uint_as_float(words[0] << 16)), "f"(rf_o_1[0]));
                rf_o_1[0] = _fma_20;
                float _fma_21;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(rr_r_0), "f"(__uint_as_float(words[0] & 4294901760u)), "f"(rf_o_1[1]));
                rf_o_1[1] = _fma_21;
                float _fma_22;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(rr_r_0), "f"(__uint_as_float(words[1] << 16)), "f"(rf_o_1[2]));
                rf_o_1[2] = _fma_22;
                float _fma_23;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(rr_r_0), "f"(__uint_as_float(words[1] & 4294901760u)), "f"(rf_o_1[3]));
                rf_o_1[3] = _fma_23;
                float _fma_24;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(rr_r_0), "f"(__uint_as_float(words[2] << 16)), "f"(rf_o_1[4]));
                rf_o_1[4] = _fma_24;
                float _fma_25;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(rr_r_0), "f"(__uint_as_float(words[2] & 4294901760u)), "f"(rf_o_1[5]));
                rf_o_1[5] = _fma_25;
                float _fma_26;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(rr_r_0), "f"(__uint_as_float(words[3] << 16)), "f"(rf_o_1[6]));
                rf_o_1[6] = _fma_26;
                float _fma_27;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(rr_r_0), "f"(__uint_as_float(words[3] & 4294901760u)), "f"(rf_o_1[7]));
                rf_o_1[7] = _fma_27;
                {
                    float4 _v4 = make_float4(rf_o_1[0 + 0], rf_o_1[0 + 1], rf_o_1[0 + 2], rf_o_1[0 + 3]);
                    *reinterpret_cast<float4*>((dx + off_1) + 0) = _v4;
                }
                {
                    float4 _v4 = make_float4(rf_o_1[4 + 0], rf_o_1[4 + 1], rf_o_1[4 + 2], rf_o_1[4 + 3]);
                    *reinterpret_cast<float4*>((dx + (off_1 + 4ull)) + 0) = _v4;
                }
            }
        }
    } else {
        long long y = rr_y;
        float r = rr_r;
        int col = rr_col;
        for (int vi_2 = vi0; vi_2 < n_vecs; vi_2 += stride) {
            int rl_live_2 = (((unsigned long long)y < (unsigned long long)V) ? 1 : 0);
            if (r == 0.0f) {
                rl_live_2 = 0;
            }
            long long rr_y_1 = -1ll;
            float rr_r_1 = 0.0f;
            int rr_col_1 = 0;
            if (lr != 0) {
                if (vi_2 + stride < n_vecs) {
                    int rr_row_2 = (vi_2 + stride) / row_vecs;
                    rr_col_1 = vi_2 + stride - rr_row_2 * row_vecs;
                    rr_y_1 = labels[rr_row_2];
                    rr_r_1 = rlab[rr_row_2];
                }
            }
            unsigned long long off_2 = (unsigned long long)vi_2 * 8ull;
            unsigned int words_1[4];
            if (rl_live_2 != 0) {
                {
                    uint4 _uv4_28 = *reinterpret_cast<const uint4*>(W + ((unsigned long long)y * (unsigned long long)row_vecs + (unsigned long long)col) * 8ull + 0);
                    words_1[0 + 0] = _uv4_28.x;
                    words_1[0 + 1] = _uv4_28.y;
                    words_1[0 + 2] = _uv4_28.z;
                    words_1[0 + 3] = _uv4_28.w;
                }
            }
            float o_1[8];
            float _vec_load_0_1[4];
            float _vec_load_1_1[4];
            if (n_slabs > 0) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + off_2 + 0);
                    _vec_load_0_1[0 + 0] = _v4.x;
                    _vec_load_0_1[0 + 1] = _v4.y;
                    _vec_load_0_1[0 + 2] = _v4.z;
                    _vec_load_0_1[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_2 + 4ull) + 0);
                    _vec_load_1_1[0 + 0] = _v4.x;
                    _vec_load_1_1[0 + 1] = _v4.y;
                    _vec_load_1_1[0 + 2] = _v4.z;
                    _vec_load_1_1[0 + 3] = _v4.w;
                }
            } else {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dx + off_2 + 0);
                    _vec_load_0_1[0 + 0] = _v4.x;
                    _vec_load_0_1[0 + 1] = _v4.y;
                    _vec_load_0_1[0 + 2] = _v4.z;
                    _vec_load_0_1[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dx + (off_2 + 4ull) + 0);
                    _vec_load_1_1[0 + 0] = _v4.x;
                    _vec_load_1_1[0 + 1] = _v4.y;
                    _vec_load_1_1[0 + 2] = _v4.z;
                    _vec_load_1_1[0 + 3] = _v4.w;
                }
            }
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                o_1[j_2] = _vec_load_0_1[j_2];
                o_1[j_2 + 4] = _vec_load_1_1[j_2];
            }
            for (int s_1 = 1; s_1 < n_slabs; s_1++) {
                unsigned long long soff_1 = off_2 + (unsigned long long)s_1 * slab64;
                float _vec_load_2_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + soff_1 + 0);
                    _vec_load_2_1[0 + 0] = _v4.x;
                    _vec_load_2_1[0 + 1] = _v4.y;
                    _vec_load_2_1[0 + 2] = _v4.z;
                    _vec_load_2_1[0 + 3] = _v4.w;
                }
                float _vec_load_3_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (soff_1 + 4ull) + 0);
                    _vec_load_3_1[0 + 0] = _v4.x;
                    _vec_load_3_1[0 + 1] = _v4.y;
                    _vec_load_3_1[0 + 2] = _v4.z;
                    _vec_load_3_1[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_3 = 0; j_3 < 4; j_3++) {
                    o_1[j_3] = o_1[j_3] + _vec_load_2_1[j_3];
                    o_1[j_3 + 4] = o_1[j_3 + 4] + _vec_load_3_1[j_3];
                }
            }
            if (rl_live_2 != 0) {
                float _fma_35;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(r), "f"(__uint_as_float(words_1[0] << 16)), "f"(o_1[0]));
                o_1[0] = _fma_35;
                float _fma_36;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(r), "f"(__uint_as_float(words_1[0] & 4294901760u)), "f"(o_1[1]));
                o_1[1] = _fma_36;
                float _fma_37;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(r), "f"(__uint_as_float(words_1[1] << 16)), "f"(o_1[2]));
                o_1[2] = _fma_37;
                float _fma_38;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(r), "f"(__uint_as_float(words_1[1] & 4294901760u)), "f"(o_1[3]));
                o_1[3] = _fma_38;
                float _fma_39;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(r), "f"(__uint_as_float(words_1[2] << 16)), "f"(o_1[4]));
                o_1[4] = _fma_39;
                float _fma_40;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(r), "f"(__uint_as_float(words_1[2] & 4294901760u)), "f"(o_1[5]));
                o_1[5] = _fma_40;
                float _fma_41;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(r), "f"(__uint_as_float(words_1[3] << 16)), "f"(o_1[6]));
                o_1[6] = _fma_41;
                float _fma_42;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(r), "f"(__uint_as_float(words_1[3] & 4294901760u)), "f"(o_1[7]));
                o_1[7] = _fma_42;
            }
            {
                float4 _v4 = make_float4(o_1[0 + 0], o_1[0 + 1], o_1[0 + 2], o_1[0 + 3]);
                *reinterpret_cast<float4*>((dx + off_2) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(o_1[4 + 0], o_1[4 + 1], o_1[4 + 2], o_1[4 + 3]);
                *reinterpret_cast<float4*>((dx + (off_2 + 4ull)) + 0) = _v4;
            }
            y = rr_y_1;
            r = rr_r_1;
            col = rr_col_1;
        }
    }
    float z[4];
    #pragma unroll
    for (int j_4 = 0; j_4 < 4; j_4++) {
        z[j_4] = __uint_as_float(2147483648u);
    }
    for (int vi_3 = n_vecs + bid * 256 + tid; vi_3 < num_vecs; vi_3 += stride) {
        unsigned long long zoff = (unsigned long long)vi_3 * 8ull;
        {
            float4 _v4 = make_float4(z[0 + 0], z[0 + 1], z[0 + 2], z[0 + 3]);
            *reinterpret_cast<float4*>((dx + zoff) + 0) = _v4;
        }
        {
            float4 _v4 = make_float4(z[0 + 0], z[0 + 1], z[0 + 2], z[0 + 3]);
            *reinterpret_cast<float4*>((dx + (zoff + 4ull)) + 0) = _v4;
        }
    }
}

} // extern "C"
