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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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

#define LAUNCH_MIN_BLOCKS 6

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
kernel_cake_rmsnorm_silu_gate_train_92eca43778aed20f8f96(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int group = tid >> 3;
    int col = (tid & 7) * 16;
    float wv[16];
    {
        const void* _v8p_0 = (const void*)(w + col + (0));
        uint32_t _v8_0_0[8];
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
        (&wv[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
        (&wv[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
        (&wv[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
        (&wv[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
        (&wv[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
        (&wv[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
        (&wv[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
        (&wv[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
        (&wv[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
        (&wv[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
        (&wv[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
        (&wv[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
        (&wv[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
        (&wv[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
        (&wv[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
        (&wv[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
    }
    int nblocks = (rows + 15) / 16;
    unsigned int xw[8];
    unsigned int zw[8];
    int src = ((bid * 16 + group) < (rows - 1) ? (bid * 16 + group) : (rows - 1));
    uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
    uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
    long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
    long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
    {
        uint32_t _iv_1_0;
        uint32_t _iv_1_1;
        uint32_t _iv_1_2;
        uint32_t _iv_1_3;
        uint32_t _iv_1_4;
        uint32_t _iv_1_5;
        uint32_t _iv_1_6;
        uint32_t _iv_1_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff) + (0))) : "memory");
        xw[0 + 0] = (uint32_t)_iv_1_0;
        xw[0 + 1] = (uint32_t)_iv_1_1;
        xw[0 + 2] = (uint32_t)_iv_1_2;
        xw[0 + 3] = (uint32_t)_iv_1_3;
        xw[0 + 4] = (uint32_t)_iv_1_4;
        xw[0 + 5] = (uint32_t)_iv_1_5;
        xw[0 + 6] = (uint32_t)_iv_1_6;
        xw[0 + 7] = (uint32_t)_iv_1_7;
    }
    {
        uint32_t _iv_2_0;
        uint32_t _iv_2_1;
        uint32_t _iv_2_2;
        uint32_t _iv_2_3;
        uint32_t _iv_2_4;
        uint32_t _iv_2_5;
        uint32_t _iv_2_6;
        uint32_t _iv_2_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff) + (0))) : "memory");
        zw[0 + 0] = (uint32_t)_iv_2_0;
        zw[0 + 1] = (uint32_t)_iv_2_1;
        zw[0 + 2] = (uint32_t)_iv_2_2;
        zw[0 + 3] = (uint32_t)_iv_2_3;
        zw[0 + 4] = (uint32_t)_iv_2_4;
        zw[0 + 5] = (uint32_t)_iv_2_5;
        zw[0 + 6] = (uint32_t)_iv_2_6;
        zw[0 + 7] = (uint32_t)_iv_2_7;
    }
    int x_mode = 0;
    int z_mode = 0;
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    if (x_head == 128ll) {
        x_mode = 1;
        if (x_tok == heads64 * 128ll) {
            x_mode = 2;
        }
    }
    if (z_head == 128ll) {
        z_mode = 1;
        if (z_tok == heads64 * 128ll) {
            z_mode = 2;
        }
    }
    if (x_mode == 2 && z_mode == 2) {
        if (tid == 0) {
            int pblk = bid + num_bids;
            if (pblk < nblocks) {
                int pfirst = pblk * 16;
                int _min_3 = ((rows - pfirst) < (16) ? (rows - pfirst) : (16));
                int pbytes = _min_3 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst * 128ll), "r"((uint32_t)(pbytes)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst * 128ll), "r"((uint32_t)(pbytes)) : "memory");
            }
            uint32_t _read_special_reg_4;
            asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_4));
            int wblk = bid + 6 * (int)_read_special_reg_4;
            if (wblk < num_bids) {
                if (wblk < nblocks) {
                    int wfirst = wblk * 16;
                    int _min_5 = ((rows - wfirst) < (16) ? (rows - wfirst) : (16));
                    int wbytes = _min_5 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst * 128ll), "r"((uint32_t)(wbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst * 128ll), "r"((uint32_t)(wbytes)) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int blk = bid; blk < nblocks; blk += num_bids) {
            float xs[16];
            float zs[16];
            xs[0] = __uint_as_float(xw[0] << 16);
            xs[1] = __uint_as_float(xw[0] & 4294901760u);
            zs[0] = __uint_as_float(zw[0] << 16);
            zs[1] = __uint_as_float(zw[0] & 4294901760u);
            xs[2] = __uint_as_float(xw[1] << 16);
            xs[3] = __uint_as_float(xw[1] & 4294901760u);
            zs[2] = __uint_as_float(zw[1] << 16);
            zs[3] = __uint_as_float(zw[1] & 4294901760u);
            xs[4] = __uint_as_float(xw[2] << 16);
            xs[5] = __uint_as_float(xw[2] & 4294901760u);
            zs[4] = __uint_as_float(zw[2] << 16);
            zs[5] = __uint_as_float(zw[2] & 4294901760u);
            xs[6] = __uint_as_float(xw[3] << 16);
            xs[7] = __uint_as_float(xw[3] & 4294901760u);
            zs[6] = __uint_as_float(zw[3] << 16);
            zs[7] = __uint_as_float(zw[3] & 4294901760u);
            xs[8] = __uint_as_float(xw[4] << 16);
            xs[9] = __uint_as_float(xw[4] & 4294901760u);
            zs[8] = __uint_as_float(zw[4] << 16);
            zs[9] = __uint_as_float(zw[4] & 4294901760u);
            xs[10] = __uint_as_float(xw[5] << 16);
            xs[11] = __uint_as_float(xw[5] & 4294901760u);
            zs[10] = __uint_as_float(zw[5] << 16);
            zs[11] = __uint_as_float(zw[5] & 4294901760u);
            xs[12] = __uint_as_float(xw[6] << 16);
            xs[13] = __uint_as_float(xw[6] & 4294901760u);
            zs[12] = __uint_as_float(zw[6] << 16);
            zs[13] = __uint_as_float(zw[6] & 4294901760u);
            xs[14] = __uint_as_float(xw[7] << 16);
            xs[15] = __uint_as_float(xw[7] & 4294901760u);
            zs[14] = __uint_as_float(zw[7] << 16);
            zs[15] = __uint_as_float(zw[7] & 4294901760u);
            if (blk + num_bids < nblocks) {
                int src_0 = (((blk + num_bids) * 16 + group) < (rows - 1) ? ((blk + num_bids) * 16 + group) : (rows - 1));
                uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
                uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
                long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
                long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
                {
                    uint32_t _iv_6_0;
                    uint32_t _iv_6_1;
                    uint32_t _iv_6_2;
                    uint32_t _iv_6_3;
                    uint32_t _iv_6_4;
                    uint32_t _iv_6_5;
                    uint32_t _iv_6_6;
                    uint32_t _iv_6_7;
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))) : "memory");
                    xw[0 + 0] = (uint32_t)_iv_6_0;
                    xw[0 + 1] = (uint32_t)_iv_6_1;
                    xw[0 + 2] = (uint32_t)_iv_6_2;
                    xw[0 + 3] = (uint32_t)_iv_6_3;
                    xw[0 + 4] = (uint32_t)_iv_6_4;
                    xw[0 + 5] = (uint32_t)_iv_6_5;
                    xw[0 + 6] = (uint32_t)_iv_6_6;
                    xw[0 + 7] = (uint32_t)_iv_6_7;
                }
                {
                    uint32_t _iv_7_0;
                    uint32_t _iv_7_1;
                    uint32_t _iv_7_2;
                    uint32_t _iv_7_3;
                    uint32_t _iv_7_4;
                    uint32_t _iv_7_5;
                    uint32_t _iv_7_6;
                    uint32_t _iv_7_7;
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))) : "memory");
                    zw[0 + 0] = (uint32_t)_iv_7_0;
                    zw[0 + 1] = (uint32_t)_iv_7_1;
                    zw[0 + 2] = (uint32_t)_iv_7_2;
                    zw[0 + 3] = (uint32_t)_iv_7_3;
                    zw[0 + 4] = (uint32_t)_iv_7_4;
                    zw[0 + 5] = (uint32_t)_iv_7_5;
                    zw[0 + 6] = (uint32_t)_iv_7_6;
                    zw[0 + 7] = (uint32_t)_iv_7_7;
                }
            }
            if (tid == 0) {
                int nblk = blk + 2 * num_bids;
                if (nblk < nblocks) {
                    int nfirst = nblk * 16;
                    int _min_8 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                    int nbytes = _min_8 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                }
            }
            float gs[16];
            float2 scale2 = make_float2(2.3283064e-10f, 2.3283064e-10f);
            float2 z2_0 = make_float2(zs[0], zs[1]);
            float2 _f32x2_mul_b_9 = make_float2(-1.442695f, -1.442695f);
            float2 t2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
            float _exp2_10 = approx_exp2(t2_0.x);
            float _exp2_11 = approx_exp2(t2_0.y);
            float2 d2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_10, _exp2_11), scale2, scale2);
            float _rcp_12 = approx_rcp(d2_0.x);
            float _rcp_13 = approx_rcp(d2_0.y);
            float2 _f32x2_mul_b_14 = mul_f32x2_noftz(make_float2(_rcp_12, _rcp_13), scale2);
            float2 g2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
            gs[0] = g2_0.x;
            gs[1] = g2_0.y;
            float2 z2_1 = make_float2(zs[2], zs[3]);
            float2 _f32x2_mul_b_15 = make_float2(-1.442695f, -1.442695f);
            float2 t2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
            float _exp2_16 = approx_exp2(t2_1.x);
            float _exp2_17 = approx_exp2(t2_1.y);
            float2 d2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_16, _exp2_17), scale2, scale2);
            float _rcp_18 = approx_rcp(d2_1.x);
            float _rcp_19 = approx_rcp(d2_1.y);
            float2 _f32x2_mul_b_20 = mul_f32x2_noftz(make_float2(_rcp_18, _rcp_19), scale2);
            float2 g2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
            gs[2] = g2_1.x;
            gs[3] = g2_1.y;
            float2 z2_2 = make_float2(zs[4], zs[5]);
            float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
            float2 t2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
            float _exp2_22 = approx_exp2(t2_2.x);
            float _exp2_23 = approx_exp2(t2_2.y);
            float2 d2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_22, _exp2_23), scale2, scale2);
            float _rcp_24 = approx_rcp(d2_2.x);
            float _rcp_25 = approx_rcp(d2_2.y);
            float2 _f32x2_mul_b_26 = mul_f32x2_noftz(make_float2(_rcp_24, _rcp_25), scale2);
            float2 g2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
            gs[4] = g2_2.x;
            gs[5] = g2_2.y;
            float2 z2_3 = make_float2(zs[6], zs[7]);
            float2 _f32x2_mul_b_27 = make_float2(-1.442695f, -1.442695f);
            float2 t2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
            float _exp2_28 = approx_exp2(t2_3.x);
            float _exp2_29 = approx_exp2(t2_3.y);
            float2 d2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_28, _exp2_29), scale2, scale2);
            float _rcp_30 = approx_rcp(d2_3.x);
            float _rcp_31 = approx_rcp(d2_3.y);
            float2 _f32x2_mul_b_32 = mul_f32x2_noftz(make_float2(_rcp_30, _rcp_31), scale2);
            float2 g2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
            gs[6] = g2_3.x;
            gs[7] = g2_3.y;
            float2 z2_4 = make_float2(zs[8], zs[9]);
            float2 _f32x2_mul_b_33 = make_float2(-1.442695f, -1.442695f);
            float2 t2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
            float _exp2_34 = approx_exp2(t2_4.x);
            float _exp2_35 = approx_exp2(t2_4.y);
            float2 d2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_34, _exp2_35), scale2, scale2);
            float _rcp_36 = approx_rcp(d2_4.x);
            float _rcp_37 = approx_rcp(d2_4.y);
            float2 _f32x2_mul_b_38 = mul_f32x2_noftz(make_float2(_rcp_36, _rcp_37), scale2);
            float2 g2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
            gs[8] = g2_4.x;
            gs[9] = g2_4.y;
            float2 z2_5 = make_float2(zs[10], zs[11]);
            float2 _f32x2_mul_b_39 = make_float2(-1.442695f, -1.442695f);
            float2 t2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
            float _exp2_40 = approx_exp2(t2_5.x);
            float _exp2_41 = approx_exp2(t2_5.y);
            float2 d2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_40, _exp2_41), scale2, scale2);
            float _rcp_42 = approx_rcp(d2_5.x);
            float _rcp_43 = approx_rcp(d2_5.y);
            float2 _f32x2_mul_b_44 = mul_f32x2_noftz(make_float2(_rcp_42, _rcp_43), scale2);
            float2 g2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
            gs[10] = g2_5.x;
            gs[11] = g2_5.y;
            float2 z2_6 = make_float2(zs[12], zs[13]);
            float2 _f32x2_mul_b_45 = make_float2(-1.442695f, -1.442695f);
            float2 t2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
            float _exp2_46 = approx_exp2(t2_6.x);
            float _exp2_47 = approx_exp2(t2_6.y);
            float2 d2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_46, _exp2_47), scale2, scale2);
            float _rcp_48 = approx_rcp(d2_6.x);
            float _rcp_49 = approx_rcp(d2_6.y);
            float2 _f32x2_mul_b_50 = mul_f32x2_noftz(make_float2(_rcp_48, _rcp_49), scale2);
            float2 g2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
            gs[12] = g2_6.x;
            gs[13] = g2_6.y;
            float2 z2_7 = make_float2(zs[14], zs[15]);
            float2 _f32x2_mul_b_51 = make_float2(-1.442695f, -1.442695f);
            float2 t2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            float _exp2_52 = approx_exp2(t2_7.x);
            float _exp2_53 = approx_exp2(t2_7.y);
            float2 d2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_52, _exp2_53), scale2, scale2);
            float _rcp_54 = approx_rcp(d2_7.x);
            float _rcp_55 = approx_rcp(d2_7.y);
            float2 _f32x2_mul_b_56 = mul_f32x2_noftz(make_float2(_rcp_54, _rcp_55), scale2);
            float2 g2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
            gs[14] = g2_7.x;
            gs[15] = g2_7.y;
            float total = 0.0f;
            float2 dx2_acc = make_float2(0.0f, 0.0f);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[0], xs[1]), make_float2(xs[0], xs[1]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[2], xs[3]), make_float2(xs[2], xs[3]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[4], xs[5]), make_float2(xs[4], xs[5]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[6], xs[7]), make_float2(xs[6], xs[7]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[8], xs[9]), make_float2(xs[8], xs[9]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[10], xs[11]), make_float2(xs[10], xs[11]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[12], xs[13]), make_float2(xs[12], xs[13]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xs[14], xs[15]), make_float2(xs[14], xs[15]), dx2_acc);
            total = dx2_acc.x + dx2_acc.y;
            float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            total = total + _shfl_xor_sync_57;
            float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_58;
            float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_59;
            float rstd;
            asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
            float2 rstd2 = make_float2(rstd, rstd);
            float2 _f32x2_mul_a_60 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
            float2 _f32x2_mul_b_61 = make_float2(gs[0], gs[1]);
            float2 y2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
            xs[0] = y2_0.x;
            xs[1] = y2_0.y;
            float2 _f32x2_mul_a_62 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
            float2 _f32x2_mul_b_63 = make_float2(gs[2], gs[3]);
            float2 y2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
            xs[2] = y2_1.x;
            xs[3] = y2_1.y;
            float2 _f32x2_mul_a_64 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
            float2 _f32x2_mul_b_65 = make_float2(gs[4], gs[5]);
            float2 y2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
            xs[4] = y2_2.x;
            xs[5] = y2_2.y;
            float2 _f32x2_mul_a_66 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
            float2 _f32x2_mul_b_67 = make_float2(gs[6], gs[7]);
            float2 y2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
            xs[6] = y2_3.x;
            xs[7] = y2_3.y;
            float2 _f32x2_mul_a_68 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
            float2 _f32x2_mul_b_69 = make_float2(gs[8], gs[9]);
            float2 y2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
            xs[8] = y2_4.x;
            xs[9] = y2_4.y;
            float2 _f32x2_mul_a_70 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
            float2 _f32x2_mul_b_71 = make_float2(gs[10], gs[11]);
            float2 y2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
            xs[10] = y2_5.x;
            xs[11] = y2_5.y;
            float2 _f32x2_mul_a_72 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
            float2 _f32x2_mul_b_73 = make_float2(gs[12], gs[13]);
            float2 y2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
            xs[12] = y2_6.x;
            xs[13] = y2_6.y;
            float2 _f32x2_mul_a_74 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
            float2 _f32x2_mul_b_75 = make_float2(gs[14], gs[15]);
            float2 y2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
            xs[14] = y2_7.x;
            xs[15] = y2_7.y;
            unsigned int packed[8];
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
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                packed[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                packed[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 8], xs[_lp*2+1 + 8]));
                packed[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 10], xs[_lp*2+1 + 10]));
                packed[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 12], xs[_lp*2+1 + 12]));
                packed[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 14], xs[_lp*2+1 + 14]));
                packed[_lp + 7] = *(uint32_t*)&_bf2;
            }
            if (blk * 16 + group < rows) {
                reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
                reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed + (4))[0];
            }
        }
    } else {
        int oblk = nblocks;
        uint32_t _read_special_reg_76;
        asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_76));
        oblk = bid + 6 * (int)_read_special_reg_76;
        if (bid + num_bids < nblocks || oblk < num_bids && oblk < nblocks) {
            if (tid < 32) {
                int pblk_1 = bid + num_bids;
                if (pblk_1 < nblocks) {
                    int pfirst_1 = pblk_1 * 16;
                    int pn = ((rows - pfirst_1) < (16) ? (rows - pfirst_1) : (16));
                    if (x_mode == 2) {
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst_1 * 128ll), "r"((uint32_t)(pn * 256)) : "memory");
                        }
                    }
                    if (x_mode == 1) {
                        if (lane == 0) {
                            int h = (int)((uint32_t)(heads.divisor));
                            uint32_t q0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)pfirst_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)pfirst_1);
                            uint32_t _h0 = (uint32_t)((unsigned int)pfirst_1) - q0 * (uint32_t)(heads.divisor);
                            uint32_t q1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(pfirst_1 + pn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(pfirst_1 + pn - 1));
                            uint32_t _h1 = (uint32_t)((unsigned int)(pfirst_1 + pn - 1)) - q1 * (uint32_t)(heads.divisor);
                            #pragma unroll 1
                            for (int q = (int)q0; q < (int)q1 + 1; q++) {
                                int lo = ((pfirst_1) > (q * h) ? (pfirst_1) : (q * h));
                                int hi = ((pfirst_1 + pn) < ((q + 1) * h) ? (pfirst_1 + pn) : ((q + 1) * h));
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)q * x_tok + (long long)(lo - q * h) * 128ll)), "r"((uint32_t)((hi - lo) * 256)) : "memory");
                            }
                        }
                    }
                    if (z_mode == 2) {
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst_1 * 128ll), "r"((uint32_t)(pn * 256)) : "memory");
                        }
                    }
                    if (z_mode == 1) {
                        if (lane == 0) {
                            int h_1 = (int)((uint32_t)(heads.divisor));
                            uint32_t q0_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)pfirst_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)pfirst_1);
                            uint32_t _h0_1 = (uint32_t)((unsigned int)pfirst_1) - q0_1 * (uint32_t)(heads.divisor);
                            uint32_t q1_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(pfirst_1 + pn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(pfirst_1 + pn - 1));
                            uint32_t _h1_1 = (uint32_t)((unsigned int)(pfirst_1 + pn - 1)) - q1_1 * (uint32_t)(heads.divisor);
                            #pragma unroll 1
                            for (int q_1 = (int)q0_1; q_1 < (int)q1_1 + 1; q_1++) {
                                int lo_1 = ((pfirst_1) > (q_1 * h_1) ? (pfirst_1) : (q_1 * h_1));
                                int hi_1 = ((pfirst_1 + pn) < ((q_1 + 1) * h_1) ? (pfirst_1 + pn) : ((q_1 + 1) * h_1));
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)q_1 * z_tok + (long long)(lo_1 - q_1 * h_1) * 128ll)), "r"((uint32_t)((hi_1 - lo_1) * 256)) : "memory");
                            }
                        }
                    }
                }
                uint32_t _read_special_reg_77;
                asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_77));
                int wblk_1 = bid + 6 * (int)_read_special_reg_77;
                if (wblk_1 < num_bids) {
                    if (wblk_1 < nblocks) {
                        int wfirst_1 = wblk_1 * 16;
                        int wn = ((rows - wfirst_1) < (16) ? (rows - wfirst_1) : (16));
                        if (x_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst_1 * 128ll), "r"((uint32_t)(wn * 256)) : "memory");
                            }
                        }
                        if (x_mode == 1) {
                            if (lane == 0) {
                                int h_2 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_2 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)wfirst_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)wfirst_1);
                                uint32_t _h0_2 = (uint32_t)((unsigned int)wfirst_1) - q0_2 * (uint32_t)(heads.divisor);
                                uint32_t q1_2 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(wfirst_1 + wn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(wfirst_1 + wn - 1));
                                uint32_t _h1_2 = (uint32_t)((unsigned int)(wfirst_1 + wn - 1)) - q1_2 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_2 = (int)q0_2; q_2 < (int)q1_2 + 1; q_2++) {
                                    int lo_2 = ((wfirst_1) > (q_2 * h_2) ? (wfirst_1) : (q_2 * h_2));
                                    int hi_2 = ((wfirst_1 + wn) < ((q_2 + 1) * h_2) ? (wfirst_1 + wn) : ((q_2 + 1) * h_2));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)q_2 * x_tok + (long long)(lo_2 - q_2 * h_2) * 128ll)), "r"((uint32_t)((hi_2 - lo_2) * 256)) : "memory");
                                }
                            }
                        }
                        if (z_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst_1 * 128ll), "r"((uint32_t)(wn * 256)) : "memory");
                            }
                        }
                        if (z_mode == 1) {
                            if (lane == 0) {
                                int h_3 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_3 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)wfirst_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)wfirst_1);
                                uint32_t _h0_3 = (uint32_t)((unsigned int)wfirst_1) - q0_3 * (uint32_t)(heads.divisor);
                                uint32_t q1_3 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(wfirst_1 + wn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(wfirst_1 + wn - 1));
                                uint32_t _h1_3 = (uint32_t)((unsigned int)(wfirst_1 + wn - 1)) - q1_3 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_3 = (int)q0_3; q_3 < (int)q1_3 + 1; q_3++) {
                                    int lo_3 = ((wfirst_1) > (q_3 * h_3) ? (wfirst_1) : (q_3 * h_3));
                                    int hi_3 = ((wfirst_1 + wn) < ((q_3 + 1) * h_3) ? (wfirst_1 + wn) : ((q_3 + 1) * h_3));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)q_3 * z_tok + (long long)(lo_3 - q_3 * h_3) * 128ll)), "r"((uint32_t)((hi_3 - lo_3) * 256)) : "memory");
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll 1
            for (int blk_1 = bid; blk_1 < nblocks; blk_1 += num_bids) {
                float xs_1[16];
                float zs_1[16];
                xs_1[0] = __uint_as_float(xw[0] << 16);
                xs_1[1] = __uint_as_float(xw[0] & 4294901760u);
                zs_1[0] = __uint_as_float(zw[0] << 16);
                zs_1[1] = __uint_as_float(zw[0] & 4294901760u);
                xs_1[2] = __uint_as_float(xw[1] << 16);
                xs_1[3] = __uint_as_float(xw[1] & 4294901760u);
                zs_1[2] = __uint_as_float(zw[1] << 16);
                zs_1[3] = __uint_as_float(zw[1] & 4294901760u);
                xs_1[4] = __uint_as_float(xw[2] << 16);
                xs_1[5] = __uint_as_float(xw[2] & 4294901760u);
                zs_1[4] = __uint_as_float(zw[2] << 16);
                zs_1[5] = __uint_as_float(zw[2] & 4294901760u);
                xs_1[6] = __uint_as_float(xw[3] << 16);
                xs_1[7] = __uint_as_float(xw[3] & 4294901760u);
                zs_1[6] = __uint_as_float(zw[3] << 16);
                zs_1[7] = __uint_as_float(zw[3] & 4294901760u);
                xs_1[8] = __uint_as_float(xw[4] << 16);
                xs_1[9] = __uint_as_float(xw[4] & 4294901760u);
                zs_1[8] = __uint_as_float(zw[4] << 16);
                zs_1[9] = __uint_as_float(zw[4] & 4294901760u);
                xs_1[10] = __uint_as_float(xw[5] << 16);
                xs_1[11] = __uint_as_float(xw[5] & 4294901760u);
                zs_1[10] = __uint_as_float(zw[5] << 16);
                zs_1[11] = __uint_as_float(zw[5] & 4294901760u);
                xs_1[12] = __uint_as_float(xw[6] << 16);
                xs_1[13] = __uint_as_float(xw[6] & 4294901760u);
                zs_1[12] = __uint_as_float(zw[6] << 16);
                zs_1[13] = __uint_as_float(zw[6] & 4294901760u);
                xs_1[14] = __uint_as_float(xw[7] << 16);
                xs_1[15] = __uint_as_float(xw[7] & 4294901760u);
                zs_1[14] = __uint_as_float(zw[7] << 16);
                zs_1[15] = __uint_as_float(zw[7] & 4294901760u);
                if (blk_1 + num_bids < nblocks) {
                    int src_1 = (((blk_1 + num_bids) * 16 + group) < (rows - 1) ? ((blk_1 + num_bids) * 16 + group) : (rows - 1));
                    uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
                    uint32_t th_1 = (uint32_t)((unsigned int)src_1) - tq_1 * (uint32_t)(heads.divisor);
                    long long xoff_1 = (long long)tq_1 * x_tok + (long long)th_1 * x_head + (long long)col;
                    long long zoff_1 = (long long)tq_1 * z_tok + (long long)th_1 * z_head + (long long)col;
                    {
                        uint32_t _iv_78_0;
                        uint32_t _iv_78_1;
                        uint32_t _iv_78_2;
                        uint32_t _iv_78_3;
                        uint32_t _iv_78_4;
                        uint32_t _iv_78_5;
                        uint32_t _iv_78_6;
                        uint32_t _iv_78_7;
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_78_0), "=r"(_iv_78_1), "=r"(_iv_78_2), "=r"(_iv_78_3), "=r"(_iv_78_4), "=r"(_iv_78_5), "=r"(_iv_78_6), "=r"(_iv_78_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_1) + (0))) : "memory");
                        xw[0 + 0] = (uint32_t)_iv_78_0;
                        xw[0 + 1] = (uint32_t)_iv_78_1;
                        xw[0 + 2] = (uint32_t)_iv_78_2;
                        xw[0 + 3] = (uint32_t)_iv_78_3;
                        xw[0 + 4] = (uint32_t)_iv_78_4;
                        xw[0 + 5] = (uint32_t)_iv_78_5;
                        xw[0 + 6] = (uint32_t)_iv_78_6;
                        xw[0 + 7] = (uint32_t)_iv_78_7;
                    }
                    {
                        uint32_t _iv_79_0;
                        uint32_t _iv_79_1;
                        uint32_t _iv_79_2;
                        uint32_t _iv_79_3;
                        uint32_t _iv_79_4;
                        uint32_t _iv_79_5;
                        uint32_t _iv_79_6;
                        uint32_t _iv_79_7;
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_79_0), "=r"(_iv_79_1), "=r"(_iv_79_2), "=r"(_iv_79_3), "=r"(_iv_79_4), "=r"(_iv_79_5), "=r"(_iv_79_6), "=r"(_iv_79_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_1) + (0))) : "memory");
                        zw[0 + 0] = (uint32_t)_iv_79_0;
                        zw[0 + 1] = (uint32_t)_iv_79_1;
                        zw[0 + 2] = (uint32_t)_iv_79_2;
                        zw[0 + 3] = (uint32_t)_iv_79_3;
                        zw[0 + 4] = (uint32_t)_iv_79_4;
                        zw[0 + 5] = (uint32_t)_iv_79_5;
                        zw[0 + 6] = (uint32_t)_iv_79_6;
                        zw[0 + 7] = (uint32_t)_iv_79_7;
                    }
                }
                if (tid < 32) {
                    int nblk_1 = blk_1 + 2 * num_bids;
                    if (nblk_1 < nblocks) {
                        int nfirst_1 = nblk_1 * 16;
                        int nn = ((rows - nfirst_1) < (16) ? (rows - nfirst_1) : (16));
                        if (x_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nn * 256)) : "memory");
                            }
                        }
                        if (x_mode == 1) {
                            if (lane == 0) {
                                int h_4 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_4 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nfirst_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nfirst_1);
                                uint32_t _h0_4 = (uint32_t)((unsigned int)nfirst_1) - q0_4 * (uint32_t)(heads.divisor);
                                uint32_t q1_4 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(nfirst_1 + nn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(nfirst_1 + nn - 1));
                                uint32_t _h1_4 = (uint32_t)((unsigned int)(nfirst_1 + nn - 1)) - q1_4 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_4 = (int)q0_4; q_4 < (int)q1_4 + 1; q_4++) {
                                    int lo_4 = ((nfirst_1) > (q_4 * h_4) ? (nfirst_1) : (q_4 * h_4));
                                    int hi_4 = ((nfirst_1 + nn) < ((q_4 + 1) * h_4) ? (nfirst_1 + nn) : ((q_4 + 1) * h_4));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)q_4 * x_tok + (long long)(lo_4 - q_4 * h_4) * 128ll)), "r"((uint32_t)((hi_4 - lo_4) * 256)) : "memory");
                                }
                            }
                        }
                        if (z_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nn * 256)) : "memory");
                            }
                        }
                        if (z_mode == 1) {
                            if (lane == 0) {
                                int h_5 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_5 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nfirst_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nfirst_1);
                                uint32_t _h0_5 = (uint32_t)((unsigned int)nfirst_1) - q0_5 * (uint32_t)(heads.divisor);
                                uint32_t q1_5 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(nfirst_1 + nn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(nfirst_1 + nn - 1));
                                uint32_t _h1_5 = (uint32_t)((unsigned int)(nfirst_1 + nn - 1)) - q1_5 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_5 = (int)q0_5; q_5 < (int)q1_5 + 1; q_5++) {
                                    int lo_5 = ((nfirst_1) > (q_5 * h_5) ? (nfirst_1) : (q_5 * h_5));
                                    int hi_5 = ((nfirst_1 + nn) < ((q_5 + 1) * h_5) ? (nfirst_1 + nn) : ((q_5 + 1) * h_5));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)q_5 * z_tok + (long long)(lo_5 - q_5 * h_5) * 128ll)), "r"((uint32_t)((hi_5 - lo_5) * 256)) : "memory");
                                }
                            }
                        }
                    }
                }
                float gs_1[16];
                float2 scale2_1 = make_float2(2.3283064e-10f, 2.3283064e-10f);
                float2 z2_8 = make_float2(zs_1[0], zs_1[1]);
                float2 _f32x2_mul_b_80 = make_float2(-1.442695f, -1.442695f);
                float2 t2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
                float _exp2_81 = approx_exp2(t2_8.x);
                float _exp2_82 = approx_exp2(t2_8.y);
                float2 d2_8 = fma_f32x2_rn_noftz(make_float2(_exp2_81, _exp2_82), scale2_1, scale2_1);
                float _rcp_83 = approx_rcp(d2_8.x);
                float _rcp_84 = approx_rcp(d2_8.y);
                float2 _f32x2_mul_b_85 = mul_f32x2_noftz(make_float2(_rcp_83, _rcp_84), scale2_1);
                float2 g2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
                gs_1[0] = g2_8.x;
                gs_1[1] = g2_8.y;
                float2 z2_9 = make_float2(zs_1[2], zs_1[3]);
                float2 _f32x2_mul_b_86 = make_float2(-1.442695f, -1.442695f);
                float2 t2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                float _exp2_87 = approx_exp2(t2_9.x);
                float _exp2_88 = approx_exp2(t2_9.y);
                float2 d2_9 = fma_f32x2_rn_noftz(make_float2(_exp2_87, _exp2_88), scale2_1, scale2_1);
                float _rcp_89 = approx_rcp(d2_9.x);
                float _rcp_90 = approx_rcp(d2_9.y);
                float2 _f32x2_mul_b_91 = mul_f32x2_noftz(make_float2(_rcp_89, _rcp_90), scale2_1);
                float2 g2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
                gs_1[2] = g2_9.x;
                gs_1[3] = g2_9.y;
                float2 z2_10 = make_float2(zs_1[4], zs_1[5]);
                float2 _f32x2_mul_b_92 = make_float2(-1.442695f, -1.442695f);
                float2 t2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
                float _exp2_93 = approx_exp2(t2_10.x);
                float _exp2_94 = approx_exp2(t2_10.y);
                float2 d2_10 = fma_f32x2_rn_noftz(make_float2(_exp2_93, _exp2_94), scale2_1, scale2_1);
                float _rcp_95 = approx_rcp(d2_10.x);
                float _rcp_96 = approx_rcp(d2_10.y);
                float2 _f32x2_mul_b_97 = mul_f32x2_noftz(make_float2(_rcp_95, _rcp_96), scale2_1);
                float2 g2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_97));
                gs_1[4] = g2_10.x;
                gs_1[5] = g2_10.y;
                float2 z2_11 = make_float2(zs_1[6], zs_1[7]);
                float2 _f32x2_mul_b_98 = make_float2(-1.442695f, -1.442695f);
                float2 t2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
                float _exp2_99 = approx_exp2(t2_11.x);
                float _exp2_100 = approx_exp2(t2_11.y);
                float2 d2_11 = fma_f32x2_rn_noftz(make_float2(_exp2_99, _exp2_100), scale2_1, scale2_1);
                float _rcp_101 = approx_rcp(d2_11.x);
                float _rcp_102 = approx_rcp(d2_11.y);
                float2 _f32x2_mul_b_103 = mul_f32x2_noftz(make_float2(_rcp_101, _rcp_102), scale2_1);
                float2 g2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
                gs_1[6] = g2_11.x;
                gs_1[7] = g2_11.y;
                float2 z2_12 = make_float2(zs_1[8], zs_1[9]);
                float2 _f32x2_mul_b_104 = make_float2(-1.442695f, -1.442695f);
                float2 t2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_104));
                float _exp2_105 = approx_exp2(t2_12.x);
                float _exp2_106 = approx_exp2(t2_12.y);
                float2 d2_12 = fma_f32x2_rn_noftz(make_float2(_exp2_105, _exp2_106), scale2_1, scale2_1);
                float _rcp_107 = approx_rcp(d2_12.x);
                float _rcp_108 = approx_rcp(d2_12.y);
                float2 _f32x2_mul_b_109 = mul_f32x2_noftz(make_float2(_rcp_107, _rcp_108), scale2_1);
                float2 g2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_109));
                gs_1[8] = g2_12.x;
                gs_1[9] = g2_12.y;
                float2 z2_13 = make_float2(zs_1[10], zs_1[11]);
                float2 _f32x2_mul_b_110 = make_float2(-1.442695f, -1.442695f);
                float2 t2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
                float _exp2_111 = approx_exp2(t2_13.x);
                float _exp2_112 = approx_exp2(t2_13.y);
                float2 d2_13 = fma_f32x2_rn_noftz(make_float2(_exp2_111, _exp2_112), scale2_1, scale2_1);
                float _rcp_113 = approx_rcp(d2_13.x);
                float _rcp_114 = approx_rcp(d2_13.y);
                float2 _f32x2_mul_b_115 = mul_f32x2_noftz(make_float2(_rcp_113, _rcp_114), scale2_1);
                float2 g2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_115));
                gs_1[10] = g2_13.x;
                gs_1[11] = g2_13.y;
                float2 z2_14 = make_float2(zs_1[12], zs_1[13]);
                float2 _f32x2_mul_b_116 = make_float2(-1.442695f, -1.442695f);
                float2 t2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
                float _exp2_117 = approx_exp2(t2_14.x);
                float _exp2_118 = approx_exp2(t2_14.y);
                float2 d2_14 = fma_f32x2_rn_noftz(make_float2(_exp2_117, _exp2_118), scale2_1, scale2_1);
                float _rcp_119 = approx_rcp(d2_14.x);
                float _rcp_120 = approx_rcp(d2_14.y);
                float2 _f32x2_mul_b_121 = mul_f32x2_noftz(make_float2(_rcp_119, _rcp_120), scale2_1);
                float2 g2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
                gs_1[12] = g2_14.x;
                gs_1[13] = g2_14.y;
                float2 z2_15 = make_float2(zs_1[14], zs_1[15]);
                float2 _f32x2_mul_b_122 = make_float2(-1.442695f, -1.442695f);
                float2 t2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
                float _exp2_123 = approx_exp2(t2_15.x);
                float _exp2_124 = approx_exp2(t2_15.y);
                float2 d2_15 = fma_f32x2_rn_noftz(make_float2(_exp2_123, _exp2_124), scale2_1, scale2_1);
                float _rcp_125 = approx_rcp(d2_15.x);
                float _rcp_126 = approx_rcp(d2_15.y);
                float2 _f32x2_mul_b_127 = mul_f32x2_noftz(make_float2(_rcp_125, _rcp_126), scale2_1);
                float2 g2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
                gs_1[14] = g2_15.x;
                gs_1[15] = g2_15.y;
                float total_1 = 0.0f;
                float2 dx2_acc_1 = make_float2(0.0f, 0.0f);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[0], xs_1[1]), make_float2(xs_1[0], xs_1[1]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[2], xs_1[3]), make_float2(xs_1[2], xs_1[3]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[4], xs_1[5]), make_float2(xs_1[4], xs_1[5]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[6], xs_1[7]), make_float2(xs_1[6], xs_1[7]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[8], xs_1[9]), make_float2(xs_1[8], xs_1[9]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[10], xs_1[11]), make_float2(xs_1[10], xs_1[11]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[12], xs_1[13]), make_float2(xs_1[12], xs_1[13]), dx2_acc_1);
                dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xs_1[14], xs_1[15]), make_float2(xs_1[14], xs_1[15]), dx2_acc_1);
                total_1 = dx2_acc_1.x + dx2_acc_1.y;
                float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_128;
                float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_129;
                float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_130;
                float rstd_1;
                asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
                float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                float2 _f32x2_mul_a_131 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[0], xs_1[1]), rstd2_1), make_float2(wv[0], wv[1]));
                float2 _f32x2_mul_b_132 = make_float2(gs_1[0], gs_1[1]);
                float2 y2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_131), "l"(*(const unsigned long long*)&_f32x2_mul_b_132));
                xs_1[0] = y2_8.x;
                xs_1[1] = y2_8.y;
                float2 _f32x2_mul_a_133 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[2], xs_1[3]), rstd2_1), make_float2(wv[2], wv[3]));
                float2 _f32x2_mul_b_134 = make_float2(gs_1[2], gs_1[3]);
                float2 y2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&_f32x2_mul_b_134));
                xs_1[2] = y2_9.x;
                xs_1[3] = y2_9.y;
                float2 _f32x2_mul_a_135 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[4], xs_1[5]), rstd2_1), make_float2(wv[4], wv[5]));
                float2 _f32x2_mul_b_136 = make_float2(gs_1[4], gs_1[5]);
                float2 y2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&_f32x2_mul_b_136));
                xs_1[4] = y2_10.x;
                xs_1[5] = y2_10.y;
                float2 _f32x2_mul_a_137 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[6], xs_1[7]), rstd2_1), make_float2(wv[6], wv[7]));
                float2 _f32x2_mul_b_138 = make_float2(gs_1[6], gs_1[7]);
                float2 y2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
                xs_1[6] = y2_11.x;
                xs_1[7] = y2_11.y;
                float2 _f32x2_mul_a_139 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[8], xs_1[9]), rstd2_1), make_float2(wv[8], wv[9]));
                float2 _f32x2_mul_b_140 = make_float2(gs_1[8], gs_1[9]);
                float2 y2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
                xs_1[8] = y2_12.x;
                xs_1[9] = y2_12.y;
                float2 _f32x2_mul_a_141 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[10], xs_1[11]), rstd2_1), make_float2(wv[10], wv[11]));
                float2 _f32x2_mul_b_142 = make_float2(gs_1[10], gs_1[11]);
                float2 y2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&_f32x2_mul_b_142));
                xs_1[10] = y2_13.x;
                xs_1[11] = y2_13.y;
                float2 _f32x2_mul_a_143 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[12], xs_1[13]), rstd2_1), make_float2(wv[12], wv[13]));
                float2 _f32x2_mul_b_144 = make_float2(gs_1[12], gs_1[13]);
                float2 y2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&_f32x2_mul_b_144));
                xs_1[12] = y2_14.x;
                xs_1[13] = y2_14.y;
                float2 _f32x2_mul_a_145 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[14], xs_1[15]), rstd2_1), make_float2(wv[14], wv[15]));
                float2 _f32x2_mul_b_146 = make_float2(gs_1[14], gs_1[15]);
                float2 y2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&_f32x2_mul_b_146));
                xs_1[14] = y2_15.x;
                xs_1[15] = y2_15.y;
                unsigned int packed_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 2], xs_1[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 4], xs_1[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 6], xs_1[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 8], xs_1[_lp*2+1 + 8]));
                    packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 10], xs_1[_lp*2+1 + 10]));
                    packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 12], xs_1[_lp*2+1 + 12]));
                    packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 14], xs_1[_lp*2+1 + 14]));
                    packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                if (blk_1 * 16 + group < rows) {
                    reinterpret_cast<int4*>(y + ((long long)(blk_1 * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    reinterpret_cast<int4*>(y + ((long long)(blk_1 * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_1 + (4))[0];
                }
            }
        } else if (bid < nblocks) {
            float oxs[16];
            float ozs[16];
            oxs[0] = __uint_as_float(xw[0] << 16);
            oxs[1] = __uint_as_float(xw[0] & 4294901760u);
            ozs[0] = __uint_as_float(zw[0] << 16);
            ozs[1] = __uint_as_float(zw[0] & 4294901760u);
            oxs[2] = __uint_as_float(xw[1] << 16);
            oxs[3] = __uint_as_float(xw[1] & 4294901760u);
            ozs[2] = __uint_as_float(zw[1] << 16);
            ozs[3] = __uint_as_float(zw[1] & 4294901760u);
            oxs[4] = __uint_as_float(xw[2] << 16);
            oxs[5] = __uint_as_float(xw[2] & 4294901760u);
            ozs[4] = __uint_as_float(zw[2] << 16);
            ozs[5] = __uint_as_float(zw[2] & 4294901760u);
            oxs[6] = __uint_as_float(xw[3] << 16);
            oxs[7] = __uint_as_float(xw[3] & 4294901760u);
            ozs[6] = __uint_as_float(zw[3] << 16);
            ozs[7] = __uint_as_float(zw[3] & 4294901760u);
            oxs[8] = __uint_as_float(xw[4] << 16);
            oxs[9] = __uint_as_float(xw[4] & 4294901760u);
            ozs[8] = __uint_as_float(zw[4] << 16);
            ozs[9] = __uint_as_float(zw[4] & 4294901760u);
            oxs[10] = __uint_as_float(xw[5] << 16);
            oxs[11] = __uint_as_float(xw[5] & 4294901760u);
            ozs[10] = __uint_as_float(zw[5] << 16);
            ozs[11] = __uint_as_float(zw[5] & 4294901760u);
            oxs[12] = __uint_as_float(xw[6] << 16);
            oxs[13] = __uint_as_float(xw[6] & 4294901760u);
            ozs[12] = __uint_as_float(zw[6] << 16);
            ozs[13] = __uint_as_float(zw[6] & 4294901760u);
            oxs[14] = __uint_as_float(xw[7] << 16);
            oxs[15] = __uint_as_float(xw[7] & 4294901760u);
            ozs[14] = __uint_as_float(zw[7] << 16);
            ozs[15] = __uint_as_float(zw[7] & 4294901760u);
            float gs_2[16];
            float2 scale2_2 = make_float2(2.3283064e-10f, 2.3283064e-10f);
            float2 z2_16 = make_float2(ozs[0], ozs[1]);
            float2 _f32x2_mul_b_147 = make_float2(-1.442695f, -1.442695f);
            float2 t2_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
            float _exp2_148 = approx_exp2(t2_16.x);
            float _exp2_149 = approx_exp2(t2_16.y);
            float2 d2_16 = fma_f32x2_rn_noftz(make_float2(_exp2_148, _exp2_149), scale2_2, scale2_2);
            float _rcp_150 = approx_rcp(d2_16.x);
            float _rcp_151 = approx_rcp(d2_16.y);
            float2 _f32x2_mul_b_152 = mul_f32x2_noftz(make_float2(_rcp_150, _rcp_151), scale2_2);
            float2 g2_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_152));
            gs_2[0] = g2_16.x;
            gs_2[1] = g2_16.y;
            float2 z2_17 = make_float2(ozs[2], ozs[3]);
            float2 _f32x2_mul_b_153 = make_float2(-1.442695f, -1.442695f);
            float2 t2_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
            float _exp2_154 = approx_exp2(t2_17.x);
            float _exp2_155 = approx_exp2(t2_17.y);
            float2 d2_17 = fma_f32x2_rn_noftz(make_float2(_exp2_154, _exp2_155), scale2_2, scale2_2);
            float _rcp_156 = approx_rcp(d2_17.x);
            float _rcp_157 = approx_rcp(d2_17.y);
            float2 _f32x2_mul_b_158 = mul_f32x2_noftz(make_float2(_rcp_156, _rcp_157), scale2_2);
            float2 g2_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_158));
            gs_2[2] = g2_17.x;
            gs_2[3] = g2_17.y;
            float2 z2_18 = make_float2(ozs[4], ozs[5]);
            float2 _f32x2_mul_b_159 = make_float2(-1.442695f, -1.442695f);
            float2 t2_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
            float _exp2_160 = approx_exp2(t2_18.x);
            float _exp2_161 = approx_exp2(t2_18.y);
            float2 d2_18 = fma_f32x2_rn_noftz(make_float2(_exp2_160, _exp2_161), scale2_2, scale2_2);
            float _rcp_162 = approx_rcp(d2_18.x);
            float _rcp_163 = approx_rcp(d2_18.y);
            float2 _f32x2_mul_b_164 = mul_f32x2_noftz(make_float2(_rcp_162, _rcp_163), scale2_2);
            float2 g2_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_164));
            gs_2[4] = g2_18.x;
            gs_2[5] = g2_18.y;
            float2 z2_19 = make_float2(ozs[6], ozs[7]);
            float2 _f32x2_mul_b_165 = make_float2(-1.442695f, -1.442695f);
            float2 t2_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
            float _exp2_166 = approx_exp2(t2_19.x);
            float _exp2_167 = approx_exp2(t2_19.y);
            float2 d2_19 = fma_f32x2_rn_noftz(make_float2(_exp2_166, _exp2_167), scale2_2, scale2_2);
            float _rcp_168 = approx_rcp(d2_19.x);
            float _rcp_169 = approx_rcp(d2_19.y);
            float2 _f32x2_mul_b_170 = mul_f32x2_noftz(make_float2(_rcp_168, _rcp_169), scale2_2);
            float2 g2_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_170));
            gs_2[6] = g2_19.x;
            gs_2[7] = g2_19.y;
            float2 z2_20 = make_float2(ozs[8], ozs[9]);
            float2 _f32x2_mul_b_171 = make_float2(-1.442695f, -1.442695f);
            float2 t2_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
            float _exp2_172 = approx_exp2(t2_20.x);
            float _exp2_173 = approx_exp2(t2_20.y);
            float2 d2_20 = fma_f32x2_rn_noftz(make_float2(_exp2_172, _exp2_173), scale2_2, scale2_2);
            float _rcp_174 = approx_rcp(d2_20.x);
            float _rcp_175 = approx_rcp(d2_20.y);
            float2 _f32x2_mul_b_176 = mul_f32x2_noftz(make_float2(_rcp_174, _rcp_175), scale2_2);
            float2 g2_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_176));
            gs_2[8] = g2_20.x;
            gs_2[9] = g2_20.y;
            float2 z2_21 = make_float2(ozs[10], ozs[11]);
            float2 _f32x2_mul_b_177 = make_float2(-1.442695f, -1.442695f);
            float2 t2_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
            float _exp2_178 = approx_exp2(t2_21.x);
            float _exp2_179 = approx_exp2(t2_21.y);
            float2 d2_21 = fma_f32x2_rn_noftz(make_float2(_exp2_178, _exp2_179), scale2_2, scale2_2);
            float _rcp_180 = approx_rcp(d2_21.x);
            float _rcp_181 = approx_rcp(d2_21.y);
            float2 _f32x2_mul_b_182 = mul_f32x2_noftz(make_float2(_rcp_180, _rcp_181), scale2_2);
            float2 g2_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
            gs_2[10] = g2_21.x;
            gs_2[11] = g2_21.y;
            float2 z2_22 = make_float2(ozs[12], ozs[13]);
            float2 _f32x2_mul_b_183 = make_float2(-1.442695f, -1.442695f);
            float2 t2_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_183));
            float _exp2_184 = approx_exp2(t2_22.x);
            float _exp2_185 = approx_exp2(t2_22.y);
            float2 d2_22 = fma_f32x2_rn_noftz(make_float2(_exp2_184, _exp2_185), scale2_2, scale2_2);
            float _rcp_186 = approx_rcp(d2_22.x);
            float _rcp_187 = approx_rcp(d2_22.y);
            float2 _f32x2_mul_b_188 = mul_f32x2_noftz(make_float2(_rcp_186, _rcp_187), scale2_2);
            float2 g2_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
            gs_2[12] = g2_22.x;
            gs_2[13] = g2_22.y;
            float2 z2_23 = make_float2(ozs[14], ozs[15]);
            float2 _f32x2_mul_b_189 = make_float2(-1.442695f, -1.442695f);
            float2 t2_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
            float _exp2_190 = approx_exp2(t2_23.x);
            float _exp2_191 = approx_exp2(t2_23.y);
            float2 d2_23 = fma_f32x2_rn_noftz(make_float2(_exp2_190, _exp2_191), scale2_2, scale2_2);
            float _rcp_192 = approx_rcp(d2_23.x);
            float _rcp_193 = approx_rcp(d2_23.y);
            float2 _f32x2_mul_b_194 = mul_f32x2_noftz(make_float2(_rcp_192, _rcp_193), scale2_2);
            float2 g2_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_194));
            gs_2[14] = g2_23.x;
            gs_2[15] = g2_23.y;
            float total_2 = 0.0f;
            float2 dx2_acc_2 = make_float2(0.0f, 0.0f);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[0], oxs[1]), make_float2(oxs[0], oxs[1]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[2], oxs[3]), make_float2(oxs[2], oxs[3]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[4], oxs[5]), make_float2(oxs[4], oxs[5]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[6], oxs[7]), make_float2(oxs[6], oxs[7]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[8], oxs[9]), make_float2(oxs[8], oxs[9]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[10], oxs[11]), make_float2(oxs[10], oxs[11]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[12], oxs[13]), make_float2(oxs[12], oxs[13]), dx2_acc_2);
            dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(oxs[14], oxs[15]), make_float2(oxs[14], oxs[15]), dx2_acc_2);
            total_2 = dx2_acc_2.x + dx2_acc_2.y;
            float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, total_2, 1);
            total_2 = total_2 + _shfl_xor_sync_195;
            float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, total_2, 2);
            total_2 = total_2 + _shfl_xor_sync_196;
            float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, total_2, 4);
            total_2 = total_2 + _shfl_xor_sync_197;
            float rstd_2;
            asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd_2) : "f"(total_2 * 0.0078125f + eps));
            float2 rstd2_2 = make_float2(rstd_2, rstd_2);
            float2 _f32x2_mul_a_198 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[0], oxs[1]), rstd2_2), make_float2(wv[0], wv[1]));
            float2 _f32x2_mul_b_199 = make_float2(gs_2[0], gs_2[1]);
            float2 y2_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
            oxs[0] = y2_16.x;
            oxs[1] = y2_16.y;
            float2 _f32x2_mul_a_200 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[2], oxs[3]), rstd2_2), make_float2(wv[2], wv[3]));
            float2 _f32x2_mul_b_201 = make_float2(gs_2[2], gs_2[3]);
            float2 y2_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
            oxs[2] = y2_17.x;
            oxs[3] = y2_17.y;
            float2 _f32x2_mul_a_202 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[4], oxs[5]), rstd2_2), make_float2(wv[4], wv[5]));
            float2 _f32x2_mul_b_203 = make_float2(gs_2[4], gs_2[5]);
            float2 y2_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&_f32x2_mul_b_203));
            oxs[4] = y2_18.x;
            oxs[5] = y2_18.y;
            float2 _f32x2_mul_a_204 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[6], oxs[7]), rstd2_2), make_float2(wv[6], wv[7]));
            float2 _f32x2_mul_b_205 = make_float2(gs_2[6], gs_2[7]);
            float2 y2_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
            oxs[6] = y2_19.x;
            oxs[7] = y2_19.y;
            float2 _f32x2_mul_a_206 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[8], oxs[9]), rstd2_2), make_float2(wv[8], wv[9]));
            float2 _f32x2_mul_b_207 = make_float2(gs_2[8], gs_2[9]);
            float2 y2_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
            oxs[8] = y2_20.x;
            oxs[9] = y2_20.y;
            float2 _f32x2_mul_a_208 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[10], oxs[11]), rstd2_2), make_float2(wv[10], wv[11]));
            float2 _f32x2_mul_b_209 = make_float2(gs_2[10], gs_2[11]);
            float2 y2_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&_f32x2_mul_b_209));
            oxs[10] = y2_21.x;
            oxs[11] = y2_21.y;
            float2 _f32x2_mul_a_210 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[12], oxs[13]), rstd2_2), make_float2(wv[12], wv[13]));
            float2 _f32x2_mul_b_211 = make_float2(gs_2[12], gs_2[13]);
            float2 y2_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&_f32x2_mul_b_211));
            oxs[12] = y2_22.x;
            oxs[13] = y2_22.y;
            float2 _f32x2_mul_a_212 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[14], oxs[15]), rstd2_2), make_float2(wv[14], wv[15]));
            float2 _f32x2_mul_b_213 = make_float2(gs_2[14], gs_2[15]);
            float2 y2_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
            oxs[14] = y2_23.x;
            oxs[15] = y2_23.y;
            unsigned int packed_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 0], oxs[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 2], oxs[_lp*2+1 + 2]));
                packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 4], oxs[_lp*2+1 + 4]));
                packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 6], oxs[_lp*2+1 + 6]));
                packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 8], oxs[_lp*2+1 + 8]));
                packed_2[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 10], oxs[_lp*2+1 + 10]));
                packed_2[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 12], oxs[_lp*2+1 + 12]));
                packed_2[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 14], oxs[_lp*2+1 + 14]));
                packed_2[_lp + 7] = *(uint32_t*)&_bf2;
            }
            if (bid * 16 + group < rows) {
                reinterpret_cast<int4*>(y + ((long long)(bid * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
                reinterpret_cast<int4*>(y + ((long long)(bid * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_2 + (4))[0];
            }
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
