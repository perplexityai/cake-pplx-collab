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
kernel_cake_lm_head_det_loss_b9eea8f269340a194207(float* __restrict__ acc, float* __restrict__ g, int* __restrict__ scan, int* __restrict__ idx_lo, __nv_bfloat16* __restrict__ out, int row_vecs, int num_rows, long long* __restrict__ labels, float* __restrict__ rlab, __nv_bfloat16* __restrict__ W, int V, int fold_lo, int fold_hi)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    float scale = g[0];
    float zero = 0.0f;
    int gstride = gridDim.x;
    unsigned long long vecs64 = (unsigned long long)row_vecs * 8ull;
    for (int r = bid; r < num_rows; r += gstride) {
        int n_incl = scan[r];
        int src = ((n_incl > 0) ? n_incl - 1 : 0);
        int _vec_load_0[1];
        {
            _vec_load_0[0] = *reinterpret_cast<const int*>(idx_lo + (unsigned long long)src * 2ull + 0);
        }
        int owner = _vec_load_0[0];
        unsigned long long row_in = (unsigned long long)src * vecs64;
        unsigned long long row_out = (unsigned long long)r * vecs64;
        int in_fold = ((src >= fold_lo) ? 1 : 0);
        if (src >= fold_hi) {
            in_fold = 0;
        }
        if (owner != r) {
            in_fold = 0;
        }
        long long fy = -1ll;
        float fr = 0.0f;
        if (in_fold != 0) {
            fy = labels[src];
            fr = rlab[src];
        }
        int live = (((unsigned long long)fy < (unsigned long long)V) ? 1 : 0);
        if (fr == 0.0f) {
            live = 0;
        }
        #pragma unroll 2
        for (int c = tid; c < row_vecs; c += 256) {
            unsigned long long off = (unsigned long long)c * 8ull;
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(acc + (row_in + off) + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(acc + (row_in + off + 4ull) + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            float o[8];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                o[j] = _vec_load_1[j];
                o[j + 4] = _vec_load_2[j];
            }
            if (live != 0) {
                unsigned int words[4];
                {
                    uint4 _uv4_2 = *reinterpret_cast<const uint4*>(W + ((unsigned long long)fy * (unsigned long long)row_vecs + (unsigned long long)c) * 8ull + 0);
                    words[0 + 0] = _uv4_2.x;
                    words[0 + 1] = _uv4_2.y;
                    words[0 + 2] = _uv4_2.z;
                    words[0 + 3] = _uv4_2.w;
                }
                float _fma_3;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(fr), "f"(__uint_as_float(words[0] << 16)), "f"(o[0]));
                o[0] = _fma_3;
                float _fma_4;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(fr), "f"(__uint_as_float(words[0] & 4294901760u)), "f"(o[1]));
                o[1] = _fma_4;
                float _fma_5;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(fr), "f"(__uint_as_float(words[1] << 16)), "f"(o[2]));
                o[2] = _fma_5;
                float _fma_6;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(fr), "f"(__uint_as_float(words[1] & 4294901760u)), "f"(o[3]));
                o[3] = _fma_6;
                float _fma_7;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(fr), "f"(__uint_as_float(words[2] << 16)), "f"(o[4]));
                o[4] = _fma_7;
                float _fma_8;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(fr), "f"(__uint_as_float(words[2] & 4294901760u)), "f"(o[5]));
                o[5] = _fma_8;
                float _fma_9;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(fr), "f"(__uint_as_float(words[3] << 16)), "f"(o[6]));
                o[6] = _fma_9;
                float _fma_10;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(fr), "f"(__uint_as_float(words[3] & 4294901760u)), "f"(o[7]));
                o[7] = _fma_10;
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                o[j_1] = ((owner == r) ? scale * o[j_1] : zero);
                o[j_1 + 4] = ((owner == r) ? scale * o[j_1 + 4] : zero);
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 0], o[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 2], o[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 4], o[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 6], o[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(out + (row_out + off))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        }
    }
}

} // extern "C"
