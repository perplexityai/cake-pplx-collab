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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_act_qdq_3c9b6d6bcbaf8c45c904(const __nv_bfloat16* x, __nv_bfloat16* y, float* __restrict__ scales, int n_vecs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int chunks = n_vecs + 255 >> 8;
    int last = n_vecs - 1;
    int warm = 1;
#if __CUDA_ARCH__ == 1070
#else
    warm = ((chunks <= num_bids * 4) ? 1 : 0);
#endif
    if (warm != 0) {
        if (tid == 0) {
            #pragma unroll
            for (int k = 1; k < 2; k++) {
                int qc = bid + num_bids * k;
                if (qc < chunks) {
                    int _min_0 = ((n_vecs - qc * 256) < (256) ? (n_vecs - qc * 256) : (256));
                    int qbytes = _min_0 * 16;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)qc * 2048ll), "r"((uint32_t)(qbytes)) : "memory");
                }
            }
        }
    }
    #pragma unroll 1
    for (int c = bid; c < chunks; c += num_bids) {
        if (warm != 0) {
            if (tid == 0) {
                int pc = c + num_bids * 2;
                if (pc < chunks) {
                    int _min_1 = ((n_vecs - pc * 256) < (256) ? (n_vecs - pc * 256) : (256));
                    int pbytes = _min_1 * 16;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pc * 2048ll), "r"((uint32_t)(pbytes)) : "memory");
                }
            }
        }
        int i = c * 256 + tid;
        int ci = ((i < n_vecs) ? i : last);
        long long off = (long long)ci * 8ll;
        float v[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + off + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_2[_blk].x), "=r"(_vld_2[_blk].y), "=r"(_vld_2[_blk].z), "=r"(_vld_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&v[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&v[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        float am = fmaxf(v[0], -v[0]);
        #pragma unroll
        for (int k_1 = 1; k_1 < 8; k_1++) {
            float _max3_3;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_3) : "f"(am), "f"(v[k_1]), "f"(-v[k_1]));
            am = _max3_3;
        }
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, am, 1);
        am = fmaxf(am, _shfl_xor_sync_4);
        float s = 0.0f;
        float inv = 0.0f;
        float floor_amax = 0.01171875f;
        float _fdiv_7 = __fdiv_rn(fmaxf(am, floor_amax), 6.0f);
        uint16_t _e4m3x2_8;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_8) : "f"(0.0f), "f"(_fdiv_7));
        uint8_t _cvt_satfinite_e4m3_f32_6 = (uint8_t)_e4m3x2_8;
        uint32_t _f16x2_decode_9;
        float _fp8_e4m3_decode_5;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_9) : "h"(_e4m3x2_8));
        uint16_t _f16_decode_9 = (uint16_t)_f16x2_decode_9;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_5) : "h"(_f16_decode_9));
        s = _fp8_e4m3_decode_5;
        float _fdiv_10 = __fdiv_rn(1.0f, s);
        inv = _fdiv_10;
        float u[8];
        #pragma unroll
        for (int k_2 = 0; k_2 < 8; k_2++) {
            u[k_2] = mul_f32_rn_noftz(v[k_2], inv);
            float _fma_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(u[k_2]), "f"(s), "f"(-v[k_2]));
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(_fma_12), "f"(-inv), "f"(u[k_2]));
            u[k_2] = _fma_11;
        }
        unsigned int halves[4];
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            uint32_t _fp4_e2m1_pack_pair_14;
            uint16_t _e2m1_pair_h_15;
            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(u[2 * p]), "f"(u[2 * p + 1]));
            _fp4_e2m1_pack_pair_14 = (uint32_t)_e2m1_pair_h_15;
            uint32_t _cvt_f16x2_e2m1x2_13;
            asm("{ .reg .b8 _b;                        \n\t"
        "  mov.b32 {_b, _, _, _}, %1;           \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
        : "=r"(_cvt_f16x2_e2m1x2_13) : "r"((uint32_t)(_fp4_e2m1_pack_pair_14)));
            halves[p] = _cvt_f16x2_e2m1x2_13;
        }
        float yq[8];
        #pragma unroll
        for (int p_1 = 0; p_1 < 4; p_1++) {
            float _cvt_f32_f16_16;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(halves[p_1])));
            yq[2 * p_1] = mul_f32_rn_noftz(_cvt_f32_f16_16, s);
            float _cvt_f32_f16_17;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(halves[p_1] >> 16)));
            yq[2 * p_1 + 1] = mul_f32_rn_noftz(_cvt_f32_f16_17, s);
        }
        unsigned int packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(yq[_lp*2 + 0], yq[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        if (i < n_vecs) {
            reinterpret_cast<int4*>(y + off)[0] = reinterpret_cast<int4*>(packed)[0];
            if (tid % 2 == 0) {
                *(reinterpret_cast<float*>(scales + (long long)(i >> 1)) + (0)) = s;
            }
        }
    }
}

} // extern "C"
