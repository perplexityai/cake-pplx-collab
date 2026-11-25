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

#if __CUDA_ARCH__ == 1070
#define LAUNCH_MIN_BLOCKS 4
#else
#define LAUNCH_MIN_BLOCKS 8
#endif

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2(uint32_t src) {
    uint16_t _fp4_u16 = (uint16_t)(src & 0xFFu);
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
        : "=r"(dst) : "h"(_fp4_u16));
#else
    uint32_t _fp4_f16x2;
    asm("{ .reg .b8 _fp4b, _fp4z0, _fp4z1, _fp4z2;      \n\t"
        "  mov.b32 {_fp4b, _fp4z0, _fp4z1, _fp4z2}, %1;   \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;              }"
        : "=r"(_fp4_f16x2) : "r"((uint32_t)_fp4_u16));
    dst = ((_fp4_f16x2 & 0x7FFF7FFFu) >> 3) | (_fp4_f16x2 & 0x80008000u);
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(dst), "r"(0x77807780u));
#endif
    return dst;
}

#include <math_constants.h>

#if __CUDA_ARCH__ == 1070
#else
__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

#endif

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

#if __CUDA_ARCH__ == 1070
__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
#else
__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
#endif
    float2 r;
#if __CUDA_ARCH__ == 1070
    asm("mul.rz.f32x2 %0, %1, %2;"
#else
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
#endif
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
#if __CUDA_ARCH__ == 1070
          "l"(*(const unsigned long long*)&b));
#else
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
#endif
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_act_qdq_31fe4b142ec4ef695bc0(const __nv_bfloat16* x, __nv_bfloat16* y, int n_vecs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int tiles = n_vecs + 511 >> 9;
    int last = n_vecs - 1;
    #pragma unroll 1
    for (int c = bid; c < tiles; c += num_bids) {
        unsigned int w[8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int i = c * 512 + r * 256 + tid;
            int ci = ((i < n_vecs) ? i : last);
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (long long)ci * 8ll + 0);
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&w[r * 4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        #pragma unroll
        for (int r_1 = 0; r_1 < 2; r_1++) {
            int i_1 = c * 512 + r_1 * 256 + tid;
            float v[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&v[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((w + r_1 * 4)[_pair]) << 16);
                (&v[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((w + r_1 * 4)[_pair]) & 0xffff0000u);
            }
            float am = fmaxf(v[0], -v[0]);
            #pragma unroll
            for (int k = 1; k < 8; k++) {
                float _max3_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(am), "f"(v[k]), "f"(-v[k]));
                am = _max3_1;
            }
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, am, 1);
            am = fmaxf(am, _shfl_xor_sync_2);
            float s = 0.0f;
            float inv = 0.0f;
            int sbits = 0;
            float floor_amax = 0.01171875f;
#if __CUDA_ARCH__ == 1070
            float by_six;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(by_six) : "f"(fmaxf(am, floor_amax)), "f"(0.16666667f));
            uint16_t _e4m3x2_5;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_5) : "f"(0.0f), "f"(by_six));
            uint8_t _cvt_satfinite_e4m3_f32_4 = (uint8_t)_e4m3x2_5;
            uint32_t _f16x2_decode_6;
#else
            float _fdiv_5 = __fdiv_rn(fmaxf(am, floor_amax), 6.0f);
            uint16_t _e4m3x2_6;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_6) : "f"(0.0f), "f"(_fdiv_5));
            uint8_t _cvt_satfinite_e4m3_f32_4 = (uint8_t)_e4m3x2_6;
            uint32_t _f16x2_decode_7;
#endif
            float _fp8_e4m3_decode_3;
#if __CUDA_ARCH__ == 1070
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_6) : "h"(_e4m3x2_5));
            uint16_t _f16_decode_6 = (uint16_t)_f16x2_decode_6;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_3) : "h"(_f16_decode_6));
#else
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_6));
            uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_3) : "h"(_f16_decode_7));
#endif
            s = _fp8_e4m3_decode_3;
#if __CUDA_ARCH__ == 1070
            float _fdiv_7 = __fdiv_rn(1.0f, s);
            inv = _fdiv_7;
#else
            float _fdiv_8 = __fdiv_rn(1.0f, s);
            inv = _fdiv_8;
#endif
            sbits = reinterpret_cast<int*>(&s)[0];
            unsigned int s_word = (unsigned int)(sbits | sbits >> 16);
            float2 inv2 = make_float2(inv, inv);
            float2 s2 = make_float2(s, s);
            float2 neg_inv2 = make_float2(-inv, -inv);
            float2 u[4];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                float2 x2 = make_float2(v[2 * p], v[2 * p + 1]);
#if __CUDA_ARCH__ == 1070
                u[p] = mul_f32x2_rz_noftz(x2, inv2);
#else
                u[p] = mul_f32x2_noftz(x2, inv2);
                float2 neg_x2 = make_float2(-v[2 * p], -v[2 * p + 1]);
                u[p] = fma_f32x2_rn_noftz(fma_f32x2_rn_noftz(u[p], s2, neg_x2), neg_inv2, u[p]);
#endif
            }
            unsigned int packed[4];
            #pragma unroll
            for (int p_1 = 0; p_1 < 4; p_1++) {
                unsigned int q_word = 0;
#if __CUDA_ARCH__ == 1070
                uint32_t _fp4_e2m1_pack_pair_9;
                uint16_t _e2m1_pair_h_10;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(u[p_1].x), "f"(u[p_1].y));
                _fp4_e2m1_pack_pair_9 = (uint32_t)_e2m1_pair_h_10;
                uint32_t _cvt_bf16x2_e2m1x2_8 = cvt_bf16x2_e2m1x2((uint16_t)_fp4_e2m1_pack_pair_9);
                q_word = _cvt_bf16x2_e2m1x2_8;
#else
                uint32_t _fp4_e2m1_pack_pair_10;
                uint16_t _e2m1_pair_h_11;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(u[p_1].x), "f"(u[p_1].y));
                _fp4_e2m1_pack_pair_10 = (uint32_t)_e2m1_pair_h_11;
                uint32_t _cvt_bf16x2_e2m1x2_9 = cvt_bf16x2_e2m1x2((uint16_t)_fp4_e2m1_pack_pair_10);
                q_word = _cvt_bf16x2_e2m1x2_9;
#endif
                uint32_t y_word;
                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(y_word) : "r"(q_word), "r"(s_word));
                packed[p_1] = y_word;
            }
            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(y + (long long)i_1 * 8ll), "r"(packed[(0) + 0]), "r"(packed[(0) + 1]), "r"(packed[(0) + 2]), "r"(packed[(0) + 3]), "r"((unsigned int)(i_1 < n_vecs)) : "memory");
        }
    }
}

} // extern "C"
