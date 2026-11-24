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

#define LAUNCH_MIN_BLOCKS 8

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
kernel_cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int lane_1 = tid & 31;
    int part = lane_1 & 3;
    int col = lane_1 * 4;
    int n = bid * 4 + (tid >> 5);
    int src = ((n) < (rows - 1) ? (n) : (rows - 1));
    long long off = (long long)src * 128ll;
    int xcol = (lane_1 >> 2) * 16;
    unsigned int xw[8];
    unsigned int zw[2];
    {
        uint32_t _iv_0_0;
        uint32_t _iv_0_1;
        uint32_t _iv_0_2;
        uint32_t _iv_0_3;
        uint32_t _iv_0_4;
        uint32_t _iv_0_5;
        uint32_t _iv_0_6;
        uint32_t _iv_0_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (off + (long long)xcol)) + (0))) : "memory");
        xw[0 + 0] = (uint32_t)_iv_0_0;
        xw[0 + 1] = (uint32_t)_iv_0_1;
        xw[0 + 2] = (uint32_t)_iv_0_2;
        xw[0 + 3] = (uint32_t)_iv_0_3;
        xw[0 + 4] = (uint32_t)_iv_0_4;
        xw[0 + 5] = (uint32_t)_iv_0_5;
        xw[0 + 6] = (uint32_t)_iv_0_6;
        xw[0 + 7] = (uint32_t)_iv_0_7;
    }
    {
        const uint2* _ivptr_1 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(z + (off + (long long)col)) + 0);
        uint2 _ivld_1;
        _ivld_1 = *_ivptr_1;
        zw[0 + 0] = _ivld_1.x;
        zw[0 + 1] = _ivld_1.y;
    }
    unsigned int ww[2];
    {
        const uint2* _ivptr_2 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(w + col) + 0);
        uint2 _ivld_2;
        _ivld_2 = *_ivptr_2;
        ww[0 + 0] = _ivld_2.x;
        ww[0 + 1] = _ivld_2.y;
    }
    float zs[4];
    float wv[4];
    zs[0] = __uint_as_float(zw[0] << 16);
    zs[1] = __uint_as_float(zw[0] & 4294901760u);
    zs[2] = __uint_as_float(zw[1] << 16);
    zs[3] = __uint_as_float(zw[1] & 4294901760u);
    wv[0] = __uint_as_float(ww[0] << 16);
    wv[1] = __uint_as_float(ww[0] & 4294901760u);
    wv[2] = __uint_as_float(ww[1] << 16);
    wv[3] = __uint_as_float(ww[1] & 4294901760u);
    float gs[4];
    float _exp2_4 = approx_exp2(zs[0] * -1.442695f);
    float _rcp_3 = approx_rcp(1.0f + _exp2_4);
    gs[0] = zs[0] * _rcp_3;
    float _exp2_6 = approx_exp2(zs[1] * -1.442695f);
    float _rcp_5 = approx_rcp(1.0f + _exp2_6);
    gs[1] = zs[1] * _rcp_5;
    float _exp2_8 = approx_exp2(zs[2] * -1.442695f);
    float _rcp_7 = approx_rcp(1.0f + _exp2_8);
    gs[2] = zs[2] * _rcp_7;
    float _exp2_10 = approx_exp2(zs[3] * -1.442695f);
    float _rcp_9 = approx_rcp(1.0f + _exp2_10);
    gs[3] = zs[3] * _rcp_9;
    float xs[4];
    float total = 0.0f;
    float xa[16];
    xa[0] = __uint_as_float(xw[0] << 16);
    xa[1] = __uint_as_float(xw[0] & 4294901760u);
    xa[2] = __uint_as_float(xw[1] << 16);
    xa[3] = __uint_as_float(xw[1] & 4294901760u);
    xa[4] = __uint_as_float(xw[2] << 16);
    xa[5] = __uint_as_float(xw[2] & 4294901760u);
    xa[6] = __uint_as_float(xw[3] << 16);
    xa[7] = __uint_as_float(xw[3] & 4294901760u);
    xa[8] = __uint_as_float(xw[4] << 16);
    xa[9] = __uint_as_float(xw[4] & 4294901760u);
    xa[10] = __uint_as_float(xw[5] << 16);
    xa[11] = __uint_as_float(xw[5] & 4294901760u);
    xa[12] = __uint_as_float(xw[6] << 16);
    xa[13] = __uint_as_float(xw[6] & 4294901760u);
    xa[14] = __uint_as_float(xw[7] << 16);
    xa[15] = __uint_as_float(xw[7] & 4294901760u);
    float2 dx2_acc = make_float2(0.0f, 0.0f);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[0], xa[1]), make_float2(xa[0], xa[1]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[2], xa[3]), make_float2(xa[2], xa[3]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[4], xa[5]), make_float2(xa[4], xa[5]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[6], xa[7]), make_float2(xa[6], xa[7]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[8], xa[9]), make_float2(xa[8], xa[9]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[10], xa[11]), make_float2(xa[10], xa[11]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[12], xa[13]), make_float2(xa[12], xa[13]), dx2_acc);
    dx2_acc = fma_f32x2_rn_noftz(make_float2(xa[14], xa[15]), make_float2(xa[14], xa[15]), dx2_acc);
    total = dx2_acc.x + dx2_acc.y;
    xw[0] = (((part & 2) != 0) ? xw[4] : xw[0]);
    xw[1] = (((part & 2) != 0) ? xw[5] : xw[1]);
    xw[2] = (((part & 2) != 0) ? xw[6] : xw[2]);
    xw[3] = (((part & 2) != 0) ? xw[7] : xw[3]);
    xw[0] = (((part & 1) != 0) ? xw[2] : xw[0]);
    xw[1] = (((part & 1) != 0) ? xw[3] : xw[1]);
    xs[0] = __uint_as_float(xw[0] << 16);
    xs[1] = __uint_as_float(xw[0] & 4294901760u);
    xs[2] = __uint_as_float(xw[1] << 16);
    xs[3] = __uint_as_float(xw[1] & 4294901760u);
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
    total = total + _shfl_xor_sync_11;
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
    total = total + _shfl_xor_sync_12;
    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, total, 16);
    total = total + _shfl_xor_sync_13;
    float rstd;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
    float2 rstd2 = make_float2(rstd, rstd);
    float2 _f32x2_mul_a_14 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
    float2 _f32x2_mul_b_15 = make_float2(gs[0], gs[1]);
    float2 y2_0;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
    xs[0] = y2_0.x;
    xs[1] = y2_0.y;
    float2 _f32x2_mul_a_16 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
    float2 _f32x2_mul_b_17 = make_float2(gs[2], gs[3]);
    float2 y2_1;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
    xs[2] = y2_1.x;
    xs[3] = y2_1.y;
    unsigned int packed[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
        packed[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
        packed[_lp + 1] = *(uint32_t*)&_bf2;
    }
    __syncwarp();
    if (n < rows) {
        reinterpret_cast<int2*>(y + ((long long)n * 128ll + (long long)col))[0] = reinterpret_cast<int2*>(packed)[0];
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
