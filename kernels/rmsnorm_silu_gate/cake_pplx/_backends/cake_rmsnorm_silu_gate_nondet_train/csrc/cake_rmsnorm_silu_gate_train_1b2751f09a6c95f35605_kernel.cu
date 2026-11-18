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


extern "C" __device__ float __nv_expf(float);


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_1b2751f09a6c95f35605(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, float* __restrict__ dw_sums, unsigned int* __restrict__ dw_done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    unsigned int wbits[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(w + col + 0);
        wbits[0 + 0] = _uv4_0.x;
        wbits[0 + 1] = _uv4_0.y;
        wbits[0 + 2] = _uv4_0.z;
        wbits[0 + 3] = _uv4_0.w;
    }
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    int packed = 0;
    if (dy_head == 128ll) {
        if (x_head == 128ll) {
            if (z_head == 128ll) {
                if (dy_tok == heads64 * 128ll) {
                    if (x_tok == heads64 * 128ll) {
                        if (z_tok == heads64 * 128ll) {
                            packed = 1;
                        }
                    }
                }
            }
        }
    }
    int s_count = 0;
    if (bid < nblocks) {
        s_count = (nblocks - 1 - bid) / num_bids + 1;
    }
    int s_share = (nblocks - 1) / num_bids + 1;
    int s_ahead = 0;
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
    if (s_share <= 96) {
#else
    if (s_share <= 64) {
#endif
        s_ahead = 1;
    }
    #pragma unroll 1
    for (int i = 0; i < s_count; i++) {
        int u = bid + i * num_bids;
        int row = u * 16 + group;
        int src = ((row) < (rows - 1) ? (row) : (rows - 1));
        uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
        uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
        long long goff = (long long)tq * dy_tok + (long long)th * dy_head + (long long)col;
        long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
        long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
        float gr[8];
        float xr[8];
        float zr[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dy + goff + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&gr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&gr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + xoff + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&xr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(z + zoff + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&zr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&zr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        if (s_ahead != 0) {
            if (i + 1 < s_count) {
                if (packed != 0) {
                    if (tid == 0) {
                        int nfirst = (u + num_bids) * 16;
                        int _min_4 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                        int nbytes = _min_4 * 256;
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                    }
                } else {
                    int nrow = (((u + num_bids) * 16 + group) < (rows - 1) ? ((u + num_bids) * 16 + group) : (rows - 1));
                    uint32_t nq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow);
                    uint32_t nh = (uint32_t)((unsigned int)nrow) - nq * (uint32_t)(heads.divisor);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq * dy_tok + (long long)nh * dy_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq * x_tok + (long long)nh * x_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq * z_tok + (long long)nh * z_head + (long long)col))));
                }
            }
        }
        float total = 0.0f;
#if __CUDA_ARCH__ == 1000
        float2 dx2_acc = make_float2(0.0f, 0.0f);
        dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[0], xr[1]), make_float2(xr[0], xr[1]), dx2_acc);
        dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[2], xr[3]), make_float2(xr[2], xr[3]), dx2_acc);
        dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[4], xr[5]), make_float2(xr[4], xr[5]), dx2_acc);
        dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[6], xr[7]), make_float2(xr[6], xr[7]), dx2_acc);
        total = dx2_acc.x + dx2_acc.y;
#elif __CUDA_ARCH__ == 1030
        float2 dx2_lo = make_float2(0.0f, 0.0f);
        float2 dx2_hi = make_float2(0.0f, 0.0f);
        dx2_lo = fma_f32x2_rn_noftz(make_float2(xr[0], xr[1]), make_float2(xr[0], xr[1]), dx2_lo);
        dx2_hi = fma_f32x2_rn_noftz(make_float2(xr[2], xr[3]), make_float2(xr[2], xr[3]), dx2_hi);
        dx2_lo = fma_f32x2_rn_noftz(make_float2(xr[4], xr[5]), make_float2(xr[4], xr[5]), dx2_lo);
        dx2_hi = fma_f32x2_rn_noftz(make_float2(xr[6], xr[7]), make_float2(xr[6], xr[7]), dx2_hi);
        total = dx2_lo.x + dx2_lo.y + (dx2_hi.x + dx2_hi.y);
#else
        total = __fmaf_rn(xr[0], xr[0], total);
        total = __fmaf_rn(xr[1], xr[1], total);
        total = __fmaf_rn(xr[2], xr[2], total);
        total = __fmaf_rn(xr[3], xr[3], total);
        total = __fmaf_rn(xr[4], xr[4], total);
        total = __fmaf_rn(xr[5], xr[5], total);
        total = __fmaf_rn(xr[6], xr[6], total);
        total = __fmaf_rn(xr[7], xr[7], total);
#endif
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_5;
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_6;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_7;
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
        total = total + _shfl_xor_sync_8;
        float rstd;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
        float dot = 0.0f;
        int fast = 1;
        float wk[2];
        unsigned int packed_dz[4];
        if (fast != 0) {
            float2 rstd2 = make_float2(rstd, rstd);
            float2 z2_0 = make_float2(zr[0], zr[1]);
            float2 _f32x2_mul_b_9 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
            float _fma_10;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(zr[0]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_11;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(zr[1]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_0 = make_float2(_fma_10, _fma_11);
            float _exp2_12 = approx_exp2(ep2_0.x);
            float _exp2_13 = approx_exp2(ep2_0.y);
            float2 dq2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_12, _exp2_13), qr2_0, make_float2(0.25f, 0.25f));
            float _rcp_14 = approx_rcp(dq2_0.x);
            float _rcp_15 = approx_rcp(dq2_0.y);
            float2 _f32x2_mul_a_16 = make_float2(_rcp_14, _rcp_15);
            float2 _f32x2_mul_b_17 = make_float2(0.25f, 0.25f);
            float2 sig2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
            float2 gate2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&sig2_0));
            float2 g2_0 = make_float2(gr[0], gr[1]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[_pair]));
            }
            float2 w2_0 = make_float2(wk[0], wk[1]);
            float2 _f32x2_mul_a_18 = make_float2(xr[0], xr[1]);
            float2 n2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_19 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
            float2 _f32x2_mul_b_20 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
            float2 dz2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
            float2 a2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
            float2 dn2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
            zr[0] = dz2_0.x;
            zr[1] = dz2_0.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 0], zr[_lp*2+1 + 0]));
                packed_dz[_lp] = *(uint32_t*)&_bf2;
            }
            gr[0] = dn2_0.x;
            gr[1] = dn2_0.y;
            float2 z2_1 = make_float2(zr[2], zr[3]);
            float2 _f32x2_mul_b_21 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
            float _fma_22;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(zr[2]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_23;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(zr[3]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_1 = make_float2(_fma_22, _fma_23);
            float _exp2_24 = approx_exp2(ep2_1.x);
            float _exp2_25 = approx_exp2(ep2_1.y);
            float2 dq2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_24, _exp2_25), qr2_1, make_float2(0.25f, 0.25f));
            float _rcp_26 = approx_rcp(dq2_1.x);
            float _rcp_27 = approx_rcp(dq2_1.y);
            float2 _f32x2_mul_a_28 = make_float2(_rcp_26, _rcp_27);
            float2 _f32x2_mul_b_29 = make_float2(0.25f, 0.25f);
            float2 sig2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
            float2 gate2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&sig2_1));
            float2 g2_1 = make_float2(gr[2], gr[3]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[1 + _pair]));
            }
            float2 w2_1 = make_float2(wk[0], wk[1]);
            float2 _f32x2_mul_a_30 = make_float2(xr[2], xr[3]);
            float2 n2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_31 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
            float2 _f32x2_mul_b_32 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
            float2 dz2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
            float2 a2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
            float2 dn2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
            zr[2] = dz2_1.x;
            zr[3] = dz2_1.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 2], zr[_lp*2+1 + 2]));
                packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
            }
            gr[2] = dn2_1.x;
            gr[3] = dn2_1.y;
            float2 z2_2 = make_float2(zr[4], zr[5]);
            float2 _f32x2_mul_b_33 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
            float _fma_34;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(zr[4]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_35;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(zr[5]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_2 = make_float2(_fma_34, _fma_35);
            float _exp2_36 = approx_exp2(ep2_2.x);
            float _exp2_37 = approx_exp2(ep2_2.y);
            float2 dq2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_36, _exp2_37), qr2_2, make_float2(0.25f, 0.25f));
            float _rcp_38 = approx_rcp(dq2_2.x);
            float _rcp_39 = approx_rcp(dq2_2.y);
            float2 _f32x2_mul_a_40 = make_float2(_rcp_38, _rcp_39);
            float2 _f32x2_mul_b_41 = make_float2(0.25f, 0.25f);
            float2 sig2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
            float2 gate2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&sig2_2));
            float2 g2_2 = make_float2(gr[4], gr[5]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[2 + _pair]));
            }
            float2 w2_2 = make_float2(wk[0], wk[1]);
            float2 _f32x2_mul_a_42 = make_float2(xr[4], xr[5]);
            float2 n2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_43 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
            float2 _f32x2_mul_b_44 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
            float2 dz2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
            float2 a2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
            float2 dn2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
            zr[4] = dz2_2.x;
            zr[5] = dz2_2.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 4], zr[_lp*2+1 + 4]));
                packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
            }
            gr[4] = dn2_2.x;
            gr[5] = dn2_2.y;
            float2 z2_3 = make_float2(zr[6], zr[7]);
            float2 _f32x2_mul_b_45 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
            float _fma_46;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(zr[6]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_47;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(zr[7]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_3 = make_float2(_fma_46, _fma_47);
            float _exp2_48 = approx_exp2(ep2_3.x);
            float _exp2_49 = approx_exp2(ep2_3.y);
            float2 dq2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_48, _exp2_49), qr2_3, make_float2(0.25f, 0.25f));
            float _rcp_50 = approx_rcp(dq2_3.x);
            float _rcp_51 = approx_rcp(dq2_3.y);
            float2 _f32x2_mul_a_52 = make_float2(_rcp_50, _rcp_51);
            float2 _f32x2_mul_b_53 = make_float2(0.25f, 0.25f);
            float2 sig2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            float2 gate2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&sig2_3));
            float2 g2_3 = make_float2(gr[6], gr[7]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[3 + _pair]));
            }
            float2 w2_3 = make_float2(wk[0], wk[1]);
            float2 _f32x2_mul_a_54 = make_float2(xr[6], xr[7]);
            float2 n2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_55 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
            float2 _f32x2_mul_b_56 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
            float2 dz2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
            float2 a2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
            float2 dn2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
            zr[6] = dz2_3.x;
            zr[7] = dz2_3.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 6], zr[_lp*2+1 + 6]));
                packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
            }
            gr[6] = dn2_3.x;
            gr[7] = dn2_3.y;
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(gr[0], xr[0], dot);
            dot = __fmaf_rn(gr[1], xr[1], dot);
            dot = __fmaf_rn(gr[2], xr[2], dot);
            dot = __fmaf_rn(gr[3], xr[3], dot);
            dot = __fmaf_rn(gr[4], xr[4], dot);
            dot = __fmaf_rn(gr[5], xr[5], dot);
            dot = __fmaf_rn(gr[6], xr[6], dot);
            dot = __fmaf_rn(gr[7], xr[7], dot);
#endif
        } else {
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[_pair]));
            }
            float _exp_57 = __nv_expf(zr[0] * -1.0f);
            float den_0 = 1.0f + _exp_57;
            float sig_0 = __fdiv_rn(1.0f, den_0);
            float gate_0 = __fdiv_rn(zr[0], den_0);
            float nj_0 = xr[0] * rstd;
            float aj_0 = gr[0] * gate_0;
            float dzj_0 = gr[0] * (nj_0 * wk[0]) * sig_0 * __fmaf_rn(zr[0], 1.0f - sig_0, 1.0f);
            float dnj_0 = aj_0 * wk[0];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_0, xr[0], dot);
#endif
            zr[0] = dzj_0;
            gr[0] = dnj_0;
            float _exp_58 = __nv_expf(zr[1] * -1.0f);
            float den_1 = 1.0f + _exp_58;
            float sig_1 = __fdiv_rn(1.0f, den_1);
            float gate_1 = __fdiv_rn(zr[1], den_1);
            float nj_1 = xr[1] * rstd;
            float aj_1 = gr[1] * gate_1;
            float dzj_1 = gr[1] * (nj_1 * wk[1]) * sig_1 * __fmaf_rn(zr[1], 1.0f - sig_1, 1.0f);
            float dnj_1 = aj_1 * wk[1];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_1, xr[1], dot);
#endif
            zr[1] = dzj_1;
            gr[1] = dnj_1;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 0], zr[_lp*2+1 + 0]));
                packed_dz[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[1 + _pair]));
            }
            float _exp_59 = __nv_expf(zr[2] * -1.0f);
            float den_2 = 1.0f + _exp_59;
            float sig_2 = __fdiv_rn(1.0f, den_2);
            float gate_2 = __fdiv_rn(zr[2], den_2);
            float nj_2 = xr[2] * rstd;
            float aj_2 = gr[2] * gate_2;
            float dzj_2 = gr[2] * (nj_2 * wk[0]) * sig_2 * __fmaf_rn(zr[2], 1.0f - sig_2, 1.0f);
            float dnj_2 = aj_2 * wk[0];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_2, xr[2], dot);
#endif
            zr[2] = dzj_2;
            gr[2] = dnj_2;
            float _exp_60 = __nv_expf(zr[3] * -1.0f);
            float den_3 = 1.0f + _exp_60;
            float sig_3 = __fdiv_rn(1.0f, den_3);
            float gate_3 = __fdiv_rn(zr[3], den_3);
            float nj_3 = xr[3] * rstd;
            float aj_3 = gr[3] * gate_3;
            float dzj_3 = gr[3] * (nj_3 * wk[1]) * sig_3 * __fmaf_rn(zr[3], 1.0f - sig_3, 1.0f);
            float dnj_3 = aj_3 * wk[1];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_3, xr[3], dot);
#endif
            zr[3] = dzj_3;
            gr[3] = dnj_3;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 2], zr[_lp*2+1 + 2]));
                packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[2 + _pair]));
            }
            float _exp_61 = __nv_expf(zr[4] * -1.0f);
            float den_4 = 1.0f + _exp_61;
            float sig_4 = __fdiv_rn(1.0f, den_4);
            float gate_4 = __fdiv_rn(zr[4], den_4);
            float nj_4 = xr[4] * rstd;
            float aj_4 = gr[4] * gate_4;
            float dzj_4 = gr[4] * (nj_4 * wk[0]) * sig_4 * __fmaf_rn(zr[4], 1.0f - sig_4, 1.0f);
            float dnj_4 = aj_4 * wk[0];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_4, xr[4], dot);
#endif
            zr[4] = dzj_4;
            gr[4] = dnj_4;
            float _exp_62 = __nv_expf(zr[5] * -1.0f);
            float den_5 = 1.0f + _exp_62;
            float sig_5 = __fdiv_rn(1.0f, den_5);
            float gate_5 = __fdiv_rn(zr[5], den_5);
            float nj_5 = xr[5] * rstd;
            float aj_5 = gr[5] * gate_5;
            float dzj_5 = gr[5] * (nj_5 * wk[1]) * sig_5 * __fmaf_rn(zr[5], 1.0f - sig_5, 1.0f);
            float dnj_5 = aj_5 * wk[1];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_5, xr[5], dot);
#endif
            zr[5] = dzj_5;
            gr[5] = dnj_5;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 4], zr[_lp*2+1 + 4]));
                packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                    : "r"(wbits[3 + _pair]));
            }
            float _exp_63 = __nv_expf(zr[6] * -1.0f);
            float den_6 = 1.0f + _exp_63;
            float sig_6 = __fdiv_rn(1.0f, den_6);
            float gate_6 = __fdiv_rn(zr[6], den_6);
            float nj_6 = xr[6] * rstd;
            float aj_6 = gr[6] * gate_6;
            float dzj_6 = gr[6] * (nj_6 * wk[0]) * sig_6 * __fmaf_rn(zr[6], 1.0f - sig_6, 1.0f);
            float dnj_6 = aj_6 * wk[0];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_6, xr[6], dot);
#endif
            zr[6] = dzj_6;
            gr[6] = dnj_6;
            float _exp_64 = __nv_expf(zr[7] * -1.0f);
            float den_7 = 1.0f + _exp_64;
            float sig_7 = __fdiv_rn(1.0f, den_7);
            float gate_7 = __fdiv_rn(zr[7], den_7);
            float nj_7 = xr[7] * rstd;
            float aj_7 = gr[7] * gate_7;
            float dzj_7 = gr[7] * (nj_7 * wk[1]) * sig_7 * __fmaf_rn(zr[7], 1.0f - sig_7, 1.0f);
            float dnj_7 = aj_7 * wk[1];
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
#else
            dot = __fmaf_rn(dnj_7, xr[7], dot);
#endif
            zr[7] = dzj_7;
            gr[7] = dnj_7;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 6], zr[_lp*2+1 + 6]));
                packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
            }
        }
#if __CUDA_ARCH__ == 1000
        float2 dx2_acc_0 = make_float2(0.0f, 0.0f);
        dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[0], gr[1]), make_float2(xr[0], xr[1]), dx2_acc_0);
        dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[2], gr[3]), make_float2(xr[2], xr[3]), dx2_acc_0);
        dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[4], gr[5]), make_float2(xr[4], xr[5]), dx2_acc_0);
        dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[6], gr[7]), make_float2(xr[6], xr[7]), dx2_acc_0);
        dot = dx2_acc_0.x + dx2_acc_0.y;
#elif __CUDA_ARCH__ == 1030
        float2 dx2_lo_0 = make_float2(0.0f, 0.0f);
        float2 dx2_hi_0 = make_float2(0.0f, 0.0f);
        dx2_lo_0 = fma_f32x2_rn_noftz(make_float2(gr[0], gr[1]), make_float2(xr[0], xr[1]), dx2_lo_0);
        dx2_hi_0 = fma_f32x2_rn_noftz(make_float2(gr[2], gr[3]), make_float2(xr[2], xr[3]), dx2_hi_0);
        dx2_lo_0 = fma_f32x2_rn_noftz(make_float2(gr[4], gr[5]), make_float2(xr[4], xr[5]), dx2_lo_0);
        dx2_hi_0 = fma_f32x2_rn_noftz(make_float2(gr[6], gr[7]), make_float2(xr[6], xr[7]), dx2_hi_0);
        dot = dx2_lo_0.x + dx2_lo_0.y + (dx2_hi_0.x + dx2_hi_0.y);
#endif
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
        dot = dot + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
        dot = dot + _shfl_xor_sync_66;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
        dot = dot + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
        dot = dot + _shfl_xor_sync_68;
        float coef = rstd * rstd * rstd * (dot * -0.0078125f);
        float2 coef2 = make_float2(coef, coef);
        float2 rstd2b = make_float2(rstd, rstd);
        float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xr[0], xr[1]), coef2, mul_f32x2_noftz(make_float2(gr[0], gr[1]), rstd2b));
        xr[0] = dx2_0.x;
        xr[1] = dx2_0.y;
        float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xr[2], xr[3]), coef2, mul_f32x2_noftz(make_float2(gr[2], gr[3]), rstd2b));
        xr[2] = dx2_1.x;
        xr[3] = dx2_1.y;
        float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xr[4], xr[5]), coef2, mul_f32x2_noftz(make_float2(gr[4], gr[5]), rstd2b));
        xr[4] = dx2_2.x;
        xr[5] = dx2_2.y;
        float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xr[6], xr[7]), coef2, mul_f32x2_noftz(make_float2(gr[6], gr[7]), rstd2b));
        xr[6] = dx2_3.x;
        xr[7] = dx2_3.y;
        unsigned int packed_dx[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 0], xr[_lp*2+1 + 0]));
            packed_dx[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 2], xr[_lp*2+1 + 2]));
            packed_dx[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 4], xr[_lp*2+1 + 4]));
            packed_dx[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 6], xr[_lp*2+1 + 6]));
            packed_dx[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row < rows) {
            long long orow = (long long)row * 128ll + (long long)col;
            reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
            reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
        }
    }
}

} // extern "C"
