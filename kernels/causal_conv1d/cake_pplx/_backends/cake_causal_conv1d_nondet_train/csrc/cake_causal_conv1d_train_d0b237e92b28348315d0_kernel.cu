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

#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_d0b237e92b28348315d0(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ y, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long C64 = (long long)C;
    int ct = bid % n_ch_tiles;
    int rank = bid / n_ch_tiles;
    int g_ct = (((num_bids - ct + n_ch_tiles - 1) / n_ch_tiles) > (1) ? ((num_bids - ct + n_ch_tiles - 1) / n_ch_tiles) : (1));
    int sub = tid / ct_vec;
    int v = tid - sub * ct_vec;
    int c0 = (ct * ct_vec + v) * 2;
    long long c64 = (long long)c0;
    int lane_on = 0;
    if (sub < n_sub && c0 < C) {
        lane_on = 1;
    }
    float wr[8];
    float br[2];
    if (lane_on != 0) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4ll);
                wr[j * 4 + 0] = _v4.x;
                wr[j * 4 + 1] = _v4.y;
                wr[j * 4 + 2] = _v4.z;
                wr[j * 4 + 3] = _v4.w;
            }
        }
    }
    int block = n_sub * 8;
    int n_blocks = (S + block - 1) / block;
    int n_chunks = (num_docs + 32) / 32;
    unsigned int xr[11] = {0};
    float win[8];
    float out[2];
    #pragma unroll 1
    for (int tb = rank; tb < n_blocks; tb += g_ct) {
        int tb0 = tb * block;
        int t0 = tb0 + sub * 8;
        #pragma unroll
        for (int r = 0; r < 11; r++) {
            int row_r = t0 - 3 + r;
            int read = 0;
            if (lane_on != 0 && row_r >= 0 && row_r < S) {
                read = 1;
            }
            int _max_2 = ((row_r) > (0) ? (row_r) : (0));
            int _min_1 = ((_max_2) < (S - 1) ? (_max_2) : (S - 1));
            long long at = (long long)_min_1 * C64 + c64;
            if (read != 0) {
                {
                    xr[r] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<const unsigned int*>(x + at) + 0);
                }
            }
        }
        unsigned int starts = 0;
        int lo_d = 0;
        if (num_docs > 31) {
            int tq = ((tb0 - 3) > (0) ? (tb0 - 3) : (0));
            int hi_d = num_docs - 1;
            #pragma unroll 1
            for (int step_d = 0; step_d < search_steps; step_d++) {
                int mid_d = lo_d + hi_d + 1 >> 1;
                if (tq >= cu[mid_d]) {
                    lo_d = mid_d;
                } else {
                    hi_d = mid_d - 1;
                }
            }
        }
        #pragma unroll 1
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            int entry = S + 11;
            if (lo_d + lane <= num_docs) {
                entry = cu[lo_d + lane];
            }
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, entry, i);
                int pos = _shfl_sync_3 - t0 + 3;
                if (pos >= 1 && pos <= 10) {
                    starts = starts | 1 << (unsigned int)pos;
                }
            }
            int last_d = __shfl_sync(0xFFFFFFFF, entry, 31);
            if (last_d > tb0 + block + 7 || lo_d + 32 > num_docs) {
                break;
            }
            lo_d = lo_d + 32;
        }
        #pragma unroll
        for (int r_1 = 0; r_1 < 11; r_1++) {
            if ((starts & 1 << (unsigned int)r_1) != 0) {
                #pragma unroll
                for (int q = 1; q < 4; q++) {
                    #pragma unroll
                    for (int e = 0; e < 2; e++) {
                        win[(r_1 + 1 + q) % 4 * 2 + e] = 0.0f;
                    }
                }
            }
            #pragma unroll
            for (int kk = 0; kk < 1; kk++) {
                win[(r_1 + 1) % 4 * 2 + 2 * kk] = __uint_as_float(xr[r_1 + kk] << 16);
                win[(r_1 + 1) % 4 * 2 + 2 * kk + 1] = __uint_as_float(xr[r_1 + kk] & 4294901760u);
            }
            if (r_1 >= 3) {
                #pragma unroll
                for (int h = 0; h < 1; h++) {
                    float2 p2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h + 3], wr[8 * h + 4 + 3]), make_float2(win[(r_1 + 1) % 4 * 2 + 2 * h], win[(r_1 + 1) % 4 * 2 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 2], wr[8 * h + 4 + 2]), make_float2(win[((r_1 + 1) % 4 + 3) % 4 * 2 + 2 * h], win[((r_1 + 1) % 4 + 3) % 4 * 2 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 1], wr[8 * h + 4 + 1]), make_float2(win[((r_1 + 1) % 4 + 2) % 4 * 2 + 2 * h], win[((r_1 + 1) % 4 + 2) % 4 * 2 + 2 * h + 1]), mul_f32x2_noftz(make_float2(wr[8 * h], wr[8 * h + 4]), make_float2(win[((r_1 + 1) % 4 + 1) % 4 * 2 + 2 * h], win[((r_1 + 1) % 4 + 1) % 4 * 2 + 2 * h + 1])))));
                    float2 _f32x2_mul_b_4 = make_float2(-1.442695f, -1.442695f);
                    float2 e2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2) : "l"(*(const unsigned long long*)&p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_4));
                    float _exp2_5 = approx_exp2(e2.x);
                    float _exp2_6 = approx_exp2(e2.y);
                    float2 den2 = add_f32x2_noftz(make_float2(_exp2_5, _exp2_6), make_float2(1.0f, 1.0f));
                    float _rcp_7 = approx_rcp(den2.x);
                    float _rcp_8 = approx_rcp(den2.y);
                    float2 _f32x2_mul_b_9 = make_float2(_rcp_7, _rcp_8);
                    float2 o2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2) : "l"(*(const unsigned long long*)&p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                    out[2 * h] = o2.x;
                    out[2 * h + 1] = o2.y;
                }
                if (lane_on != 0 && t0 + r_1 - 3 < S) {
                    unsigned int _packed_l[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                        _packed_l[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(y + ((long long)(t0 + r_1 - 3) * C64 + c64))[0] = reinterpret_cast<unsigned*>(_packed_l)[0];
                }
            }
        }
    }
}

} // extern "C"
