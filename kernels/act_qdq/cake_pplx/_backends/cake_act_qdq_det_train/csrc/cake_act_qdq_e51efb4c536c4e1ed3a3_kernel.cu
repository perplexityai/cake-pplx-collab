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

#if __CUDA_ARCH__ == 1070
#define LAUNCH_MIN_BLOCKS 16
#else
#define LAUNCH_MIN_BLOCKS 32

__device__ __forceinline__ uint32_t cvt_bf16x2_e4m3x2(uint16_t src) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
#else
    uint32_t _f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(src));
    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
    float _f0;
    float _f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(dst) : "f"(_f1), "f"(_f0));
#endif
    return dst;
}
#endif

#include <math_constants.h>

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)
#if __CUDA_ARCH__ == 1070


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
#else
#endif

extern "C" {

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void
kernel_cake_act_qdq_e51efb4c536c4e1ed3a3(const __nv_bfloat16* x, __nv_bfloat16* y, float* __restrict__ scales, int n_vecs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
#if __CUDA_ARCH__ == 1070
    int tiles = n_vecs + 255 >> 8;
#else
    int tiles = n_vecs + 127 >> 7;
#endif
    int last = n_vecs - 1;
    #pragma unroll 1
    for (int c = bid; c < tiles; c += num_bids) {
#if __CUDA_ARCH__ == 1070
        unsigned int w[16];
#else
        float wide[16];
#endif
        #pragma unroll
#if __CUDA_ARCH__ == 1070
        for (int r = 0; r < 4; r++) {
            int i = c * 256 + r * 64 + tid;
#else
        for (int r = 0; r < 2; r++) {
            int i = c * 128 + r * 64 + tid;
#endif
            int ci = ((i < n_vecs) ? i : last);
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (long long)ci * 8ll + 0);
#if __CUDA_ARCH__ == 1070
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&w[r * 4]);
#else
                uint4 _vld_0[1];
#endif
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
#if __CUDA_ARCH__ == 1070
                        : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
#else
                        : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&wide[r * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&wide[r * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
#endif
                }
            }
        }
        #pragma unroll
#if __CUDA_ARCH__ == 1070
        for (int r_1 = 0; r_1 < 4; r_1++) {
            int i_1 = c * 256 + r_1 * 64 + tid;
#else
        for (int r_1 = 0; r_1 < 2; r_1++) {
            int i_1 = c * 128 + r_1 * 64 + tid;
#endif
            float v[8];
            #pragma unroll
#if __CUDA_ARCH__ == 1070
            for (int _pair = 0; _pair < 4; _pair++) {
                (&v[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((w + r_1 * 4)[_pair]) << 16);
                (&v[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((w + r_1 * 4)[_pair]) & 0xffff0000u);
#else
            for (int k = 0; k < 8; k++) {
                v[k] = wide[r_1 * 8 + k];
#endif
            }
            float am = fmaxf(v[0], -v[0]);
            #pragma unroll
#if __CUDA_ARCH__ == 1070
            for (int k = 1; k < 8; k++) {
#else
            for (int k_1 = 1; k_1 < 8; k_1++) {
#endif
                float _max3_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
#if __CUDA_ARCH__ == 1070
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(am), "f"(v[k]), "f"(-v[k]));
#else
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(am), "f"(v[k_1]), "f"(-v[k_1]));
#endif
                am = _max3_1;
            }
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, am, 1);
            am = fmaxf(am, _shfl_xor_sync_2);
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, am, 2);
            am = fmaxf(am, _shfl_xor_sync_3);
            float s = 0.0f;
            float inv = 0.0f;
            int sbits = 0;
            int abits = 0;
            abits = reinterpret_cast<int*>(&am)[0];
            int rounded = (abits >> 23) - 8 + (((abits & 8388607) > 6291456) ? 1 : 0);
            int sbyte = ((rounded > 105) ? rounded : 105);
            sbits = sbyte << 23;
            int ibits = 254 - sbyte << 23;
            s = reinterpret_cast<float*>(&sbits)[0];
            inv = reinterpret_cast<float*>(&ibits)[0];
            unsigned int s_word = (unsigned int)(sbits | sbits >> 16);
            float2 inv2 = make_float2(inv, inv);
            float2 s2 = make_float2(s, s);
            float2 neg_inv2 = make_float2(-inv, -inv);
            float2 u[4];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                float2 x2 = make_float2(v[2 * p], v[2 * p + 1]);
                u[p] = mul_f32x2_noftz(x2, inv2);
            }
#if __CUDA_ARCH__ == 1070
            unsigned int halves[4];
            #pragma unroll
            for (int p_1 = 0; p_1 < 4; p_1++) {
                uint16_t _cvt_satfinite_e4m3x2_f32_5;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_5) : "f"(u[p_1].y), "f"(u[p_1].x));
                uint32_t _cvt_f16x2_e4m3x2_4;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_4) : "h"((uint16_t)(_cvt_satfinite_e4m3x2_f32_5)));
                halves[p_1] = _cvt_f16x2_e4m3x2_4;
            }
            float yq[8];
            #pragma unroll
            for (int p_2 = 0; p_2 < 4; p_2++) {
                float _cvt_f32_f16_6;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(halves[p_2])));
                yq[2 * p_2] = mul_f32_rn_noftz(_cvt_f32_f16_6, s);
                float _cvt_f32_f16_7;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(halves[p_2] >> 16)));
                yq[2 * p_2 + 1] = mul_f32_rn_noftz(_cvt_f32_f16_7, s);
            }
#else
#endif
            unsigned int packed[4];
            #pragma unroll
#if __CUDA_ARCH__ == 1070
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yq[_lp*2 + 0], yq[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_bf2;
#else
            for (int p_1 = 0; p_1 < 4; p_1++) {
                unsigned int q_word = 0;
                uint16_t _cvt_satfinite_e4m3x2_f32_5;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_5) : "f"(u[p_1].y), "f"(u[p_1].x));
                uint32_t _cvt_bf16x2_e4m3x2_4 = cvt_bf16x2_e4m3x2(_cvt_satfinite_e4m3x2_f32_5);
                q_word = _cvt_bf16x2_e4m3x2_4;
                uint32_t y_word;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(y_word) : "r"(q_word), "r"(s_word));
                packed[p_1] = y_word;
#endif
            }
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(y + (long long)i_1 * 8ll), "r"(packed[(0) + 0]), "r"(packed[(0) + 1]), "r"(packed[(0) + 2]), "r"(packed[(0) + 3]), "r"((unsigned int)(i_1 < n_vecs)) : "memory");
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(scales + (long long)(i_1 >> 2))), "f"(s), "r"((unsigned int)(i_1 < n_vecs && tid % 4 == 0)) : "memory");
        }
    }
}

} // extern "C"
