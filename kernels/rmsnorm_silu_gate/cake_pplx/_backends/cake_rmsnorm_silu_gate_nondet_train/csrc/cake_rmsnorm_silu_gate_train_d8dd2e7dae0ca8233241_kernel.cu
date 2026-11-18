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
kernel_cake_rmsnorm_silu_gate_train_d8dd2e7dae0ca8233241(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ acc, unsigned int* __restrict__ flags, unsigned int* __restrict__ done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* cell = reinterpret_cast<int*>(smem_raw + 0);
    const int cell_addr = smem + 0;
    int* flag = reinterpret_cast<int*>(smem_raw + 16);
    const int flag_addr = smem + 16;
    int* tab = reinterpret_cast<int*>(smem_raw + 32);
    const int tab_addr = smem + 32;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    int bound = 0;
    if (nblocks <= num_bids) {
        bound = nblocks;
    }
    int rest = nblocks - bound;
    unsigned int wbits[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(w + col + 0);
        wbits[0 + 0] = _uv4_0.x;
        wbits[0 + 1] = _uv4_0.y;
        wbits[0 + 2] = _uv4_0.z;
        wbits[0 + 3] = _uv4_0.w;
    }
    int ticket = 2147483647;
    if (tid == 0) {
        if (rest > 0) {
            uint32_t _atomic_fetch_inc_1;
            asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_fetch_inc_1) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
            ticket = (int)_atomic_fetch_inc_1;
        }
    }
    if (rest > 0) {
        int _min_2 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int sl0 = ((_min_2) > (1) ? (_min_2) : (1));
        if (bid < sl0) {
            int pblk = bid;
            int prow = ((pblk * 16 + group) < (rows - 1) ? (pblk * 16 + group) : (rows - 1));
            uint32_t pq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)prow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)prow);
            uint32_t ph = (uint32_t)((unsigned int)prow) - pq * (uint32_t)(heads.divisor);
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)pq * dy_tok + (long long)ph * dy_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)pq * x_tok + (long long)ph * x_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)pq * z_tok + (long long)ph * z_head + (long long)col))));
        }
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
    unsigned int la_x[4];
    unsigned int la_g[4];
    unsigned int la_z[4];
    if (tid == 0) {
        if (rest > 0) {
            int _min_3 = ((num_bids) < (rest) ? (num_bids) : (rest));
            int sl = ((_min_3) > (1) ? (_min_3) : (1));
            int share = (rest + sl - 1) / sl;
            int tail = (((share + 63) / 64) > (1) ? ((share + 63) / 64) : (1));
            int first_chunk = 0;
            int base = 0;
            int rem = rest;
            int size = (share + 1) / 2;
            int live = 1;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[0] = ((live != 0) ? first_chunk : 2147483647);
            tab[1] = base;
            tab[2] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[4] = ((live != 0) ? first_chunk : 2147483647);
            tab[5] = base;
            tab[6] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[8] = ((live != 0) ? first_chunk : 2147483647);
            tab[9] = base;
            tab[10] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[12] = ((live != 0) ? first_chunk : 2147483647);
            tab[13] = base;
            tab[14] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[16] = ((live != 0) ? first_chunk : 2147483647);
            tab[17] = base;
            tab[18] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[20] = ((live != 0) ? first_chunk : 2147483647);
            tab[21] = base;
            tab[22] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            tab[24] = first_chunk;
            tab[25] = base;
            tab[26] = 1;
            tab[28] = first_chunk + rem;
            tab[29] = 0;
            tab[30] = 0;
        }
    }
    if (bid < bound) {
        if (packed != 0) {
            int tail_i = 0;
            int u = bid + tail_i;
            int row = u * 16 + group;
            int src = ((row) < (rows - 1) ? (row) : (rows - 1));
            long long goff = (long long)src * 128ll + (long long)col;
            long long xoff = goff;
            long long zoff = goff;
            float gr[8];
            float xr[8];
            float zr[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(dy + goff + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&gr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + xoff + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&xr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(z + zoff + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&zr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                    }
                }
            }
            float total = 0.0f;
            float2 dx2_acc = make_float2(0.0f, 0.0f);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[0], xr[1]), make_float2(xr[0], xr[1]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[2], xr[3]), make_float2(xr[2], xr[3]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[4], xr[5]), make_float2(xr[4], xr[5]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[6], xr[7]), make_float2(xr[6], xr[7]), dx2_acc);
            total = dx2_acc.x + dx2_acc.y;
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            total = total + _shfl_xor_sync_7;
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_8;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_9;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
            total = total + _shfl_xor_sync_10;
            float rstd;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
            float dot = 0.0f;
            float wk[2];
            unsigned int packed_dz[4];
            float2 rstd2 = make_float2(rstd, rstd);
            float2 z2_0 = make_float2(zr[0], zr[1]);
            float2 _f32x2_mul_b_11 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
            float _fma_12;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(zr[0]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_13;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(zr[1]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_0 = make_float2(_fma_12, _fma_13);
            float _exp2_14 = approx_exp2(ep2_0.x);
            float _exp2_15 = approx_exp2(ep2_0.y);
            float2 dq2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_14, _exp2_15), qr2_0, make_float2(0.25f, 0.25f));
            float _rcp_16 = approx_rcp(dq2_0.x);
            float _rcp_17 = approx_rcp(dq2_0.y);
            float2 _f32x2_mul_a_18 = make_float2(_rcp_16, _rcp_17);
            float2 _f32x2_mul_b_19 = make_float2(0.25f, 0.25f);
            float2 sig2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
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
            float2 _f32x2_mul_a_20 = make_float2(xr[0], xr[1]);
            float2 n2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_21 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
            float2 _f32x2_mul_b_22 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
            float2 dz2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
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
            float2 _f32x2_mul_b_23 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
            float _fma_24;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(zr[2]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_25;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(zr[3]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_1 = make_float2(_fma_24, _fma_25);
            float _exp2_26 = approx_exp2(ep2_1.x);
            float _exp2_27 = approx_exp2(ep2_1.y);
            float2 dq2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_26, _exp2_27), qr2_1, make_float2(0.25f, 0.25f));
            float _rcp_28 = approx_rcp(dq2_1.x);
            float _rcp_29 = approx_rcp(dq2_1.y);
            float2 _f32x2_mul_a_30 = make_float2(_rcp_28, _rcp_29);
            float2 _f32x2_mul_b_31 = make_float2(0.25f, 0.25f);
            float2 sig2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
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
            float2 _f32x2_mul_a_32 = make_float2(xr[2], xr[3]);
            float2 n2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_33 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
            float2 _f32x2_mul_b_34 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
            float2 dz2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
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
            float2 _f32x2_mul_b_35 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
            float _fma_36;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(zr[4]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_37;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(zr[5]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_2 = make_float2(_fma_36, _fma_37);
            float _exp2_38 = approx_exp2(ep2_2.x);
            float _exp2_39 = approx_exp2(ep2_2.y);
            float2 dq2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_38, _exp2_39), qr2_2, make_float2(0.25f, 0.25f));
            float _rcp_40 = approx_rcp(dq2_2.x);
            float _rcp_41 = approx_rcp(dq2_2.y);
            float2 _f32x2_mul_a_42 = make_float2(_rcp_40, _rcp_41);
            float2 _f32x2_mul_b_43 = make_float2(0.25f, 0.25f);
            float2 sig2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
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
            float2 _f32x2_mul_a_44 = make_float2(xr[4], xr[5]);
            float2 n2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_45 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
            float2 _f32x2_mul_b_46 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
            float2 dz2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
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
            float2 _f32x2_mul_b_47 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
            float _fma_48;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(zr[6]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_49;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(zr[7]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_3 = make_float2(_fma_48, _fma_49);
            float _exp2_50 = approx_exp2(ep2_3.x);
            float _exp2_51 = approx_exp2(ep2_3.y);
            float2 dq2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_50, _exp2_51), qr2_3, make_float2(0.25f, 0.25f));
            float _rcp_52 = approx_rcp(dq2_3.x);
            float _rcp_53 = approx_rcp(dq2_3.y);
            float2 _f32x2_mul_a_54 = make_float2(_rcp_52, _rcp_53);
            float2 _f32x2_mul_b_55 = make_float2(0.25f, 0.25f);
            float2 sig2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
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
            float2 _f32x2_mul_a_56 = make_float2(xr[6], xr[7]);
            float2 n2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&rstd2));
            float2 _f32x2_mul_a_57 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
            float2 _f32x2_mul_b_58 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
            float2 dz2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
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
            float2 dx2_acc_0 = make_float2(0.0f, 0.0f);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[0], gr[1]), make_float2(xr[0], xr[1]), dx2_acc_0);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[2], gr[3]), make_float2(xr[2], xr[3]), dx2_acc_0);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[4], gr[5]), make_float2(xr[4], xr[5]), dx2_acc_0);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[6], gr[7]), make_float2(xr[6], xr[7]), dx2_acc_0);
            dot = dx2_acc_0.x + dx2_acc_0.y;
            float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot = dot + _shfl_xor_sync_59;
            float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot = dot + _shfl_xor_sync_60;
            float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot = dot + _shfl_xor_sync_61;
            float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot = dot + _shfl_xor_sync_62;
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
            long long orow = goff;
            if (row < rows) {
                reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
                reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
            }
        } else {
            int tail_i_1 = 0;
            int u_1 = bid + tail_i_1;
            int row_1 = u_1 * 16 + group;
            int src_1 = ((row_1) < (rows - 1) ? (row_1) : (rows - 1));
            uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
            uint32_t th = (uint32_t)((unsigned int)src_1) - tq * (uint32_t)(heads.divisor);
            long long goff_1 = (long long)tq * dy_tok + (long long)th * dy_head + (long long)col;
            long long xoff_1 = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
            long long zoff_1 = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
            float gr_1[8];
            float xr_1[8];
            float zr_1[8];
            {
                const uint4* _vptr_63 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                uint4 _vld_63[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_63[_blk] = _vptr_63[_blk];
                    uint32_t* _vpairs_63 = reinterpret_cast<uint32_t*>(&_vld_63[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) << 16);
                        (&gr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_64 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                uint4 _vld_64[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_64[_blk] = _vptr_64[_blk];
                    uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                        (&xr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_65 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                uint4 _vld_65[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_65[_blk] = _vptr_65[_blk];
                    uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                        (&zr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
                    }
                }
            }
            float total_1 = 0.0f;
            float2 dx2_acc_1 = make_float2(0.0f, 0.0f);
            dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xr_1[0], xr_1[1]), make_float2(xr_1[0], xr_1[1]), dx2_acc_1);
            dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xr_1[2], xr_1[3]), make_float2(xr_1[2], xr_1[3]), dx2_acc_1);
            dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xr_1[4], xr_1[5]), make_float2(xr_1[4], xr_1[5]), dx2_acc_1);
            dx2_acc_1 = fma_f32x2_rn_noftz(make_float2(xr_1[6], xr_1[7]), make_float2(xr_1[6], xr_1[7]), dx2_acc_1);
            total_1 = dx2_acc_1.x + dx2_acc_1.y;
            float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
            total_1 = total_1 + _shfl_xor_sync_66;
            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
            total_1 = total_1 + _shfl_xor_sync_67;
            float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
            total_1 = total_1 + _shfl_xor_sync_68;
            float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
            total_1 = total_1 + _shfl_xor_sync_69;
            float rstd_1;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
            float dot_1 = 0.0f;
            float wk_1[2];
            unsigned int packed_dz_1[4];
            float2 rstd2_1 = make_float2(rstd_1, rstd_1);
            float2 z2_4 = make_float2(zr_1[0], zr_1[1]);
            float2 _f32x2_mul_b_70 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
            float _fma_71;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(zr_1[0]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_72;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(zr_1[1]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_4 = make_float2(_fma_71, _fma_72);
            float _exp2_73 = approx_exp2(ep2_4.x);
            float _exp2_74 = approx_exp2(ep2_4.y);
            float2 dq2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_73, _exp2_74), qr2_4, make_float2(0.25f, 0.25f));
            float _rcp_75 = approx_rcp(dq2_4.x);
            float _rcp_76 = approx_rcp(dq2_4.y);
            float2 _f32x2_mul_a_77 = make_float2(_rcp_75, _rcp_76);
            float2 _f32x2_mul_b_78 = make_float2(0.25f, 0.25f);
            float2 sig2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
            float2 gate2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&sig2_4));
            float2 g2_4 = make_float2(gr_1[0], gr_1[1]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                    : "r"(wbits[_pair]));
            }
            float2 w2_4 = make_float2(wk_1[0], wk_1[1]);
            float2 _f32x2_mul_a_79 = make_float2(xr_1[0], xr_1[1]);
            float2 n2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_80 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
            float2 _f32x2_mul_b_81 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
            float2 dz2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
            float2 a2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
            float2 dn2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
            zr_1[0] = dz2_4.x;
            zr_1[1] = dz2_4.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 0], zr_1[_lp*2+1 + 0]));
                packed_dz_1[_lp] = *(uint32_t*)&_bf2;
            }
            gr_1[0] = dn2_4.x;
            gr_1[1] = dn2_4.y;
            float2 z2_5 = make_float2(zr_1[2], zr_1[3]);
            float2 _f32x2_mul_b_82 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
            float _fma_83;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(zr_1[2]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_84;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(zr_1[3]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_5 = make_float2(_fma_83, _fma_84);
            float _exp2_85 = approx_exp2(ep2_5.x);
            float _exp2_86 = approx_exp2(ep2_5.y);
            float2 dq2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_85, _exp2_86), qr2_5, make_float2(0.25f, 0.25f));
            float _rcp_87 = approx_rcp(dq2_5.x);
            float _rcp_88 = approx_rcp(dq2_5.y);
            float2 _f32x2_mul_a_89 = make_float2(_rcp_87, _rcp_88);
            float2 _f32x2_mul_b_90 = make_float2(0.25f, 0.25f);
            float2 sig2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
            float2 gate2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&sig2_5));
            float2 g2_5 = make_float2(gr_1[2], gr_1[3]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                    : "r"(wbits[1 + _pair]));
            }
            float2 w2_5 = make_float2(wk_1[0], wk_1[1]);
            float2 _f32x2_mul_a_91 = make_float2(xr_1[2], xr_1[3]);
            float2 n2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_92 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
            float2 _f32x2_mul_b_93 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
            float2 dz2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
            float2 a2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
            float2 dn2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
            zr_1[2] = dz2_5.x;
            zr_1[3] = dz2_5.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 2], zr_1[_lp*2+1 + 2]));
                packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            gr_1[2] = dn2_5.x;
            gr_1[3] = dn2_5.y;
            float2 z2_6 = make_float2(zr_1[4], zr_1[5]);
            float2 _f32x2_mul_b_94 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
            float _fma_95;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(zr_1[4]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_96;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(zr_1[5]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_6 = make_float2(_fma_95, _fma_96);
            float _exp2_97 = approx_exp2(ep2_6.x);
            float _exp2_98 = approx_exp2(ep2_6.y);
            float2 dq2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_97, _exp2_98), qr2_6, make_float2(0.25f, 0.25f));
            float _rcp_99 = approx_rcp(dq2_6.x);
            float _rcp_100 = approx_rcp(dq2_6.y);
            float2 _f32x2_mul_a_101 = make_float2(_rcp_99, _rcp_100);
            float2 _f32x2_mul_b_102 = make_float2(0.25f, 0.25f);
            float2 sig2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
            float2 gate2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&sig2_6));
            float2 g2_6 = make_float2(gr_1[4], gr_1[5]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                    : "r"(wbits[2 + _pair]));
            }
            float2 w2_6 = make_float2(wk_1[0], wk_1[1]);
            float2 _f32x2_mul_a_103 = make_float2(xr_1[4], xr_1[5]);
            float2 n2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_104 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
            float2 _f32x2_mul_b_105 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
            float2 dz2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
            float2 a2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
            float2 dn2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
            zr_1[4] = dz2_6.x;
            zr_1[5] = dz2_6.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 4], zr_1[_lp*2+1 + 4]));
                packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            gr_1[4] = dn2_6.x;
            gr_1[5] = dn2_6.y;
            float2 z2_7 = make_float2(zr_1[6], zr_1[7]);
            float2 _f32x2_mul_b_106 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
            float _fma_107;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(zr_1[6]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_108;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(zr_1[7]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_7 = make_float2(_fma_107, _fma_108);
            float _exp2_109 = approx_exp2(ep2_7.x);
            float _exp2_110 = approx_exp2(ep2_7.y);
            float2 dq2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_109, _exp2_110), qr2_7, make_float2(0.25f, 0.25f));
            float _rcp_111 = approx_rcp(dq2_7.x);
            float _rcp_112 = approx_rcp(dq2_7.y);
            float2 _f32x2_mul_a_113 = make_float2(_rcp_111, _rcp_112);
            float2 _f32x2_mul_b_114 = make_float2(0.25f, 0.25f);
            float2 sig2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
            float2 gate2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&sig2_7));
            float2 g2_7 = make_float2(gr_1[6], gr_1[7]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                    : "r"(wbits[3 + _pair]));
            }
            float2 w2_7 = make_float2(wk_1[0], wk_1[1]);
            float2 _f32x2_mul_a_115 = make_float2(xr_1[6], xr_1[7]);
            float2 n2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_116 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
            float2 _f32x2_mul_b_117 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
            float2 dz2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
            float2 a2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
            float2 dn2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
            zr_1[6] = dz2_7.x;
            zr_1[7] = dz2_7.y;
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 6], zr_1[_lp*2+1 + 6]));
                packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            gr_1[6] = dn2_7.x;
            gr_1[7] = dn2_7.y;
            float2 dx2_acc_1_1 = make_float2(0.0f, 0.0f);
            dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[0], gr_1[1]), make_float2(xr_1[0], xr_1[1]), dx2_acc_1_1);
            dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[2], gr_1[3]), make_float2(xr_1[2], xr_1[3]), dx2_acc_1_1);
            dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[4], gr_1[5]), make_float2(xr_1[4], xr_1[5]), dx2_acc_1_1);
            dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[6], gr_1[7]), make_float2(xr_1[6], xr_1[7]), dx2_acc_1_1);
            dot_1 = dx2_acc_1_1.x + dx2_acc_1_1.y;
            float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
            dot_1 = dot_1 + _shfl_xor_sync_118;
            float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
            dot_1 = dot_1 + _shfl_xor_sync_119;
            float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
            dot_1 = dot_1 + _shfl_xor_sync_120;
            float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
            dot_1 = dot_1 + _shfl_xor_sync_121;
            float coef_1 = rstd_1 * rstd_1 * rstd_1 * (dot_1 * -0.0078125f);
            float2 coef2_1 = make_float2(coef_1, coef_1);
            float2 rstd2b_1 = make_float2(rstd_1, rstd_1);
            float2 dx2_4 = fma_f32x2_rn_noftz(make_float2(xr_1[0], xr_1[1]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[0], gr_1[1]), rstd2b_1));
            xr_1[0] = dx2_4.x;
            xr_1[1] = dx2_4.y;
            float2 dx2_5 = fma_f32x2_rn_noftz(make_float2(xr_1[2], xr_1[3]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[2], gr_1[3]), rstd2b_1));
            xr_1[2] = dx2_5.x;
            xr_1[3] = dx2_5.y;
            float2 dx2_6 = fma_f32x2_rn_noftz(make_float2(xr_1[4], xr_1[5]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[4], gr_1[5]), rstd2b_1));
            xr_1[4] = dx2_6.x;
            xr_1[5] = dx2_6.y;
            float2 dx2_7 = fma_f32x2_rn_noftz(make_float2(xr_1[6], xr_1[7]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[6], gr_1[7]), rstd2b_1));
            xr_1[6] = dx2_7.x;
            xr_1[7] = dx2_7.y;
            unsigned int packed_dx_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 0], xr_1[_lp*2+1 + 0]));
                packed_dx_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 2], xr_1[_lp*2+1 + 2]));
                packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 4], xr_1[_lp*2+1 + 4]));
                packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 6], xr_1[_lp*2+1 + 6]));
                packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            long long orow_1 = (long long)row_1 * 128ll + (long long)col;
            if (row_1 < rows) {
                reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
            }
        }
    }
    int trips = 0;
    int ahead = 0;
    if (rest > 0) {
        int _min_122 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_122) > (1) ? (_min_122) : (1));
        int t_share = (rest + t_sl - 1) / t_sl;
#if __CUDA_ARCH__ == 1070
        if (t_share <= 64) {
#else
        if (t_share <= 96) {
#endif
            ahead = 1;
        }
        int t_tail = (((t_share + 63) / 64) > (1) ? ((t_share + 63) / 64) : (1));
        int t_rem = rest;
        int t_size = (t_share + 1) / 2;
        int t_live = 1;
        trips = 1;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        trips = trips + (t_rem + t_sl - 1) / t_sl;
    }
    if (tid == 0) {
        flag[1] = ticket;
    }
    int _min_123 = ((num_bids) < (rest) ? (num_bids) : (rest));
    int stride = ((_min_123) > (1) ? (_min_123) : (1));
    #pragma unroll 1
    for (int it = 0; it < trips; it++) {
        int half = it % 2 * 2;
        if (tid == 0) {
            int tk = flag[1];
            flag[1] = 2147483647;
            int begin = 0;
            int count = 0;
            int e_first_0 = tab[0];
            int e_base_0 = tab[1];
            int e_size_0 = tab[2];
            begin = ((e_first_0 <= tk) ? e_base_0 + (tk - e_first_0) : begin);
            count = ((e_first_0 <= tk) ? e_size_0 : count);
            int e_first_1 = tab[4];
            int e_base_1 = tab[5];
            int e_size_1 = tab[6];
            begin = ((e_first_1 <= tk) ? e_base_1 + (tk - e_first_1) : begin);
            count = ((e_first_1 <= tk) ? e_size_1 : count);
            int e_first_2 = tab[8];
            int e_base_2 = tab[9];
            int e_size_2 = tab[10];
            begin = ((e_first_2 <= tk) ? e_base_2 + (tk - e_first_2) : begin);
            count = ((e_first_2 <= tk) ? e_size_2 : count);
            int e_first_3 = tab[12];
            int e_base_3 = tab[13];
            int e_size_3 = tab[14];
            begin = ((e_first_3 <= tk) ? e_base_3 + (tk - e_first_3) : begin);
            count = ((e_first_3 <= tk) ? e_size_3 : count);
            int e_first_4 = tab[16];
            int e_base_4 = tab[17];
            int e_size_4 = tab[18];
            begin = ((e_first_4 <= tk) ? e_base_4 + (tk - e_first_4) : begin);
            count = ((e_first_4 <= tk) ? e_size_4 : count);
            int e_first_5 = tab[20];
            int e_base_5 = tab[21];
            int e_size_5 = tab[22];
            begin = ((e_first_5 <= tk) ? e_base_5 + (tk - e_first_5) : begin);
            count = ((e_first_5 <= tk) ? e_size_5 : count);
            int e_first_6 = tab[24];
            int e_base_6 = tab[25];
            int e_size_6 = tab[26];
            begin = ((e_first_6 <= tk) ? e_base_6 + (tk - e_first_6) : begin);
            count = ((e_first_6 <= tk) ? e_size_6 : count);
            int e_first_7 = tab[28];
            int e_base_7 = tab[29];
            int e_size_7 = tab[30];
            begin = ((e_first_7 <= tk) ? e_base_7 + (tk - e_first_7) : begin);
            count = ((e_first_7 <= tk) ? e_size_7 : count);
            cell[half] = begin;
            cell[half + 1] = count;
        }
        __syncthreads();
        int chunk_lo = cell[half];
        int chunk_n = cell[half + 1];
        if (chunk_n > 0) {
            if (packed != 0) {
                if (chunk_n > 1) {
                    long long voff = (long long)((bound + chunk_lo) * 16 + group) * 128ll + (long long)col;
                    {
                        uint4 _uv4_124 = *reinterpret_cast<const uint4*>(dy + voff + 0);
                        la_g[0 + 0] = _uv4_124.x;
                        la_g[0 + 1] = _uv4_124.y;
                        la_g[0 + 2] = _uv4_124.z;
                        la_g[0 + 3] = _uv4_124.w;
                    }
                    {
                        uint4 _uv4_125 = *reinterpret_cast<const uint4*>(x + voff + 0);
                        la_x[0 + 0] = _uv4_125.x;
                        la_x[0 + 1] = _uv4_125.y;
                        la_x[0 + 2] = _uv4_125.z;
                        la_x[0 + 3] = _uv4_125.w;
                    }
                    {
                        uint4 _uv4_126 = *reinterpret_cast<const uint4*>(z + voff + 0);
                        la_z[0 + 0] = _uv4_126.x;
                        la_z[0 + 1] = _uv4_126.y;
                        la_z[0 + 2] = _uv4_126.z;
                        la_z[0 + 3] = _uv4_126.w;
                    }
                }
                #pragma unroll 1
                for (int i = 0; i < chunk_n - 1; i++) {
                    int u_2 = bound + chunk_lo + i * stride;
                    int row_2 = u_2 * 16 + group;
                    long long goff_2 = (long long)row_2 * 128ll + (long long)col;
                    float gr_2[8];
                    float xr_2[8];
                    float zr_2[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) << 16);
                        (&xr_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) << 16);
                        (&gr_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) << 16);
                        (&zr_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) & 0xffff0000u);
                    }
                    if (i + 2 < chunk_n) {
                        long long voff_1 = (long long)(row_2 + stride * 16) * 128ll + (long long)col;
                        {
                            uint4 _uv4_127 = *reinterpret_cast<const uint4*>(dy + voff_1 + 0);
                            la_g[0 + 0] = _uv4_127.x;
                            la_g[0 + 1] = _uv4_127.y;
                            la_g[0 + 2] = _uv4_127.z;
                            la_g[0 + 3] = _uv4_127.w;
                        }
                        {
                            uint4 _uv4_128 = *reinterpret_cast<const uint4*>(x + voff_1 + 0);
                            la_x[0 + 0] = _uv4_128.x;
                            la_x[0 + 1] = _uv4_128.y;
                            la_x[0 + 2] = _uv4_128.z;
                            la_x[0 + 3] = _uv4_128.w;
                        }
                        {
                            uint4 _uv4_129 = *reinterpret_cast<const uint4*>(z + voff_1 + 0);
                            la_z[0 + 0] = _uv4_129.x;
                            la_z[0 + 1] = _uv4_129.y;
                            la_z[0 + 2] = _uv4_129.z;
                            la_z[0 + 3] = _uv4_129.w;
                        }
                    }
                    if (ahead != 0) {
                        if (tid == 0) {
                            int nfirst = (u_2 + stride) * 16;
                            int _min_130 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                            int nbytes = _min_130 * 256;
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                        }
                    }
                    float total_2 = 0.0f;
                    float2 dx2_acc_2 = make_float2(0.0f, 0.0f);
                    dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(xr_2[0], xr_2[1]), make_float2(xr_2[0], xr_2[1]), dx2_acc_2);
                    dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(xr_2[2], xr_2[3]), make_float2(xr_2[2], xr_2[3]), dx2_acc_2);
                    dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(xr_2[4], xr_2[5]), make_float2(xr_2[4], xr_2[5]), dx2_acc_2);
                    dx2_acc_2 = fma_f32x2_rn_noftz(make_float2(xr_2[6], xr_2[7]), make_float2(xr_2[6], xr_2[7]), dx2_acc_2);
                    total_2 = dx2_acc_2.x + dx2_acc_2.y;
                    float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, total_2, 1);
                    total_2 = total_2 + _shfl_xor_sync_131;
                    float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, total_2, 2);
                    total_2 = total_2 + _shfl_xor_sync_132;
                    float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, total_2, 4);
                    total_2 = total_2 + _shfl_xor_sync_133;
                    float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, total_2, 8);
                    total_2 = total_2 + _shfl_xor_sync_134;
                    float rstd_2;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_2) : "f"(total_2 * 0.0078125f + eps));
                    float dot_2 = 0.0f;
                    float wk_2[2];
                    unsigned int packed_dz_2[4];
                    float2 rstd2_2 = make_float2(rstd_2, rstd_2);
                    float2 z2_8 = make_float2(zr_2[0], zr_2[1]);
                    float2 _f32x2_mul_b_135 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
                    float _fma_136;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(zr_2[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_137;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(zr_2[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_8 = make_float2(_fma_136, _fma_137);
                    float _exp2_138 = approx_exp2(ep2_8.x);
                    float _exp2_139 = approx_exp2(ep2_8.y);
                    float2 dq2_8 = fma_f32x2_rn_noftz(make_float2(_exp2_138, _exp2_139), qr2_8, make_float2(0.25f, 0.25f));
                    float _rcp_140 = approx_rcp(dq2_8.x);
                    float _rcp_141 = approx_rcp(dq2_8.y);
                    float2 _f32x2_mul_a_142 = make_float2(_rcp_140, _rcp_141);
                    float2 _f32x2_mul_b_143 = make_float2(0.25f, 0.25f);
                    float2 sig2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
                    float2 gate2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&sig2_8));
                    float2 g2_8 = make_float2(gr_2[0], gr_2[1]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[_pair]));
                    }
                    float2 w2_8 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_144 = make_float2(xr_2[0], xr_2[1]);
                    float2 n2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_145 = mul_f32x2_noftz(mul_f32x2_noftz(g2_8, mul_f32x2_noftz(n2_8, w2_8)), sig2_8);
                    float2 _f32x2_mul_b_146 = fma_f32x2_rn_noftz(z2_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_8), make_float2(1.0f, 1.0f));
                    float2 dz2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&_f32x2_mul_b_146));
                    float2 a2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_8) : "l"(*(const unsigned long long*)&g2_8), "l"(*(const unsigned long long*)&gate2_8));
                    float2 dn2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_8) : "l"(*(const unsigned long long*)&a2_8), "l"(*(const unsigned long long*)&w2_8));
                    zr_2[0] = dz2_8.x;
                    zr_2[1] = dz2_8.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 0], zr_2[_lp*2+1 + 0]));
                        packed_dz_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    gr_2[0] = dn2_8.x;
                    gr_2[1] = dn2_8.y;
                    float2 z2_9 = make_float2(zr_2[2], zr_2[3]);
                    float2 _f32x2_mul_b_147 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
                    float _fma_148;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(zr_2[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_149;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(zr_2[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_9 = make_float2(_fma_148, _fma_149);
                    float _exp2_150 = approx_exp2(ep2_9.x);
                    float _exp2_151 = approx_exp2(ep2_9.y);
                    float2 dq2_9 = fma_f32x2_rn_noftz(make_float2(_exp2_150, _exp2_151), qr2_9, make_float2(0.25f, 0.25f));
                    float _rcp_152 = approx_rcp(dq2_9.x);
                    float _rcp_153 = approx_rcp(dq2_9.y);
                    float2 _f32x2_mul_a_154 = make_float2(_rcp_152, _rcp_153);
                    float2 _f32x2_mul_b_155 = make_float2(0.25f, 0.25f);
                    float2 sig2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
                    float2 gate2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&sig2_9));
                    float2 g2_9 = make_float2(gr_2[2], gr_2[3]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[1 + _pair]));
                    }
                    float2 w2_9 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_156 = make_float2(xr_2[2], xr_2[3]);
                    float2 n2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_157 = mul_f32x2_noftz(mul_f32x2_noftz(g2_9, mul_f32x2_noftz(n2_9, w2_9)), sig2_9);
                    float2 _f32x2_mul_b_158 = fma_f32x2_rn_noftz(z2_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_9), make_float2(1.0f, 1.0f));
                    float2 dz2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&_f32x2_mul_b_158));
                    float2 a2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_9) : "l"(*(const unsigned long long*)&g2_9), "l"(*(const unsigned long long*)&gate2_9));
                    float2 dn2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_9) : "l"(*(const unsigned long long*)&a2_9), "l"(*(const unsigned long long*)&w2_9));
                    zr_2[2] = dz2_9.x;
                    zr_2[3] = dz2_9.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 2], zr_2[_lp*2+1 + 2]));
                        packed_dz_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    gr_2[2] = dn2_9.x;
                    gr_2[3] = dn2_9.y;
                    float2 z2_10 = make_float2(zr_2[4], zr_2[5]);
                    float2 _f32x2_mul_b_159 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
                    float _fma_160;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(zr_2[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_161;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(zr_2[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_10 = make_float2(_fma_160, _fma_161);
                    float _exp2_162 = approx_exp2(ep2_10.x);
                    float _exp2_163 = approx_exp2(ep2_10.y);
                    float2 dq2_10 = fma_f32x2_rn_noftz(make_float2(_exp2_162, _exp2_163), qr2_10, make_float2(0.25f, 0.25f));
                    float _rcp_164 = approx_rcp(dq2_10.x);
                    float _rcp_165 = approx_rcp(dq2_10.y);
                    float2 _f32x2_mul_a_166 = make_float2(_rcp_164, _rcp_165);
                    float2 _f32x2_mul_b_167 = make_float2(0.25f, 0.25f);
                    float2 sig2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
                    float2 gate2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&sig2_10));
                    float2 g2_10 = make_float2(gr_2[4], gr_2[5]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[2 + _pair]));
                    }
                    float2 w2_10 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_168 = make_float2(xr_2[4], xr_2[5]);
                    float2 n2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_169 = mul_f32x2_noftz(mul_f32x2_noftz(g2_10, mul_f32x2_noftz(n2_10, w2_10)), sig2_10);
                    float2 _f32x2_mul_b_170 = fma_f32x2_rn_noftz(z2_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_10), make_float2(1.0f, 1.0f));
                    float2 dz2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&_f32x2_mul_b_170));
                    float2 a2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_10) : "l"(*(const unsigned long long*)&g2_10), "l"(*(const unsigned long long*)&gate2_10));
                    float2 dn2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_10) : "l"(*(const unsigned long long*)&a2_10), "l"(*(const unsigned long long*)&w2_10));
                    zr_2[4] = dz2_10.x;
                    zr_2[5] = dz2_10.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 4], zr_2[_lp*2+1 + 4]));
                        packed_dz_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    gr_2[4] = dn2_10.x;
                    gr_2[5] = dn2_10.y;
                    float2 z2_11 = make_float2(zr_2[6], zr_2[7]);
                    float2 _f32x2_mul_b_171 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
                    float _fma_172;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(zr_2[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_173;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(zr_2[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_11 = make_float2(_fma_172, _fma_173);
                    float _exp2_174 = approx_exp2(ep2_11.x);
                    float _exp2_175 = approx_exp2(ep2_11.y);
                    float2 dq2_11 = fma_f32x2_rn_noftz(make_float2(_exp2_174, _exp2_175), qr2_11, make_float2(0.25f, 0.25f));
                    float _rcp_176 = approx_rcp(dq2_11.x);
                    float _rcp_177 = approx_rcp(dq2_11.y);
                    float2 _f32x2_mul_a_178 = make_float2(_rcp_176, _rcp_177);
                    float2 _f32x2_mul_b_179 = make_float2(0.25f, 0.25f);
                    float2 sig2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&_f32x2_mul_b_179));
                    float2 gate2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&sig2_11));
                    float2 g2_11 = make_float2(gr_2[6], gr_2[7]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[3 + _pair]));
                    }
                    float2 w2_11 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_180 = make_float2(xr_2[6], xr_2[7]);
                    float2 n2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_181 = mul_f32x2_noftz(mul_f32x2_noftz(g2_11, mul_f32x2_noftz(n2_11, w2_11)), sig2_11);
                    float2 _f32x2_mul_b_182 = fma_f32x2_rn_noftz(z2_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_11), make_float2(1.0f, 1.0f));
                    float2 dz2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
                    float2 a2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_11) : "l"(*(const unsigned long long*)&g2_11), "l"(*(const unsigned long long*)&gate2_11));
                    float2 dn2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_11) : "l"(*(const unsigned long long*)&a2_11), "l"(*(const unsigned long long*)&w2_11));
                    zr_2[6] = dz2_11.x;
                    zr_2[7] = dz2_11.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 6], zr_2[_lp*2+1 + 6]));
                        packed_dz_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    gr_2[6] = dn2_11.x;
                    gr_2[7] = dn2_11.y;
                    float2 dx2_acc_2_1 = make_float2(0.0f, 0.0f);
                    dx2_acc_2_1 = fma_f32x2_rn_noftz(make_float2(gr_2[0], gr_2[1]), make_float2(xr_2[0], xr_2[1]), dx2_acc_2_1);
                    dx2_acc_2_1 = fma_f32x2_rn_noftz(make_float2(gr_2[2], gr_2[3]), make_float2(xr_2[2], xr_2[3]), dx2_acc_2_1);
                    dx2_acc_2_1 = fma_f32x2_rn_noftz(make_float2(gr_2[4], gr_2[5]), make_float2(xr_2[4], xr_2[5]), dx2_acc_2_1);
                    dx2_acc_2_1 = fma_f32x2_rn_noftz(make_float2(gr_2[6], gr_2[7]), make_float2(xr_2[6], xr_2[7]), dx2_acc_2_1);
                    dot_2 = dx2_acc_2_1.x + dx2_acc_2_1.y;
                    float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                    dot_2 = dot_2 + _shfl_xor_sync_183;
                    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                    dot_2 = dot_2 + _shfl_xor_sync_184;
                    float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                    dot_2 = dot_2 + _shfl_xor_sync_185;
                    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 8);
                    dot_2 = dot_2 + _shfl_xor_sync_186;
                    float coef_2 = rstd_2 * rstd_2 * rstd_2 * (dot_2 * -0.0078125f);
                    float2 coef2_2 = make_float2(coef_2, coef_2);
                    float2 rstd2b_2 = make_float2(rstd_2, rstd_2);
                    float2 dx2_8 = fma_f32x2_rn_noftz(make_float2(xr_2[0], xr_2[1]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[0], gr_2[1]), rstd2b_2));
                    xr_2[0] = dx2_8.x;
                    xr_2[1] = dx2_8.y;
                    float2 dx2_9 = fma_f32x2_rn_noftz(make_float2(xr_2[2], xr_2[3]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[2], gr_2[3]), rstd2b_2));
                    xr_2[2] = dx2_9.x;
                    xr_2[3] = dx2_9.y;
                    float2 dx2_10 = fma_f32x2_rn_noftz(make_float2(xr_2[4], xr_2[5]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[4], gr_2[5]), rstd2b_2));
                    xr_2[4] = dx2_10.x;
                    xr_2[5] = dx2_10.y;
                    float2 dx2_11 = fma_f32x2_rn_noftz(make_float2(xr_2[6], xr_2[7]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[6], gr_2[7]), rstd2b_2));
                    xr_2[6] = dx2_11.x;
                    xr_2[7] = dx2_11.y;
                    unsigned int packed_dx_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 0], xr_2[_lp*2+1 + 0]));
                        packed_dx_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 2], xr_2[_lp*2+1 + 2]));
                        packed_dx_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 4], xr_2[_lp*2+1 + 4]));
                        packed_dx_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 6], xr_2[_lp*2+1 + 6]));
                        packed_dx_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    long long orow_2 = goff_2;
                    reinterpret_cast<int4*>(dx + orow_2)[0] = reinterpret_cast<int4*>(packed_dx_2)[0];
                    reinterpret_cast<int4*>(dz + orow_2)[0] = reinterpret_cast<int4*>(packed_dz_2)[0];
                }
                int tail_i_2 = chunk_n - 1;
                int u_3 = bound + chunk_lo + tail_i_2 * stride;
                int row_3 = u_3 * 16 + group;
                int src_2 = ((row_3) < (rows - 1) ? (row_3) : (rows - 1));
                long long goff_3 = (long long)src_2 * 128ll + (long long)col;
                long long xoff_2 = goff_3;
                long long zoff_2 = goff_3;
                float gr_3[8];
                float xr_3[8];
                float zr_3[8];
                {
                    const uint4* _vptr_187 = reinterpret_cast<const uint4*>(dy + goff_3 + 0);
                    uint4 _vld_187[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_187[_blk] = _vptr_187[_blk];
                        uint32_t* _vpairs_187 = reinterpret_cast<uint32_t*>(&_vld_187[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_187[_pair]) << 16);
                            (&gr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_187[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_188 = reinterpret_cast<const uint4*>(x + xoff_2 + 0);
                    uint4 _vld_188[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_188[_blk] = _vptr_188[_blk];
                        uint32_t* _vpairs_188 = reinterpret_cast<uint32_t*>(&_vld_188[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_188[_pair]) << 16);
                            (&xr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_188[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_189 = reinterpret_cast<const uint4*>(z + zoff_2 + 0);
                    uint4 _vld_189[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_189[_blk] = _vptr_189[_blk];
                        uint32_t* _vpairs_189 = reinterpret_cast<uint32_t*>(&_vld_189[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_189[_pair]) << 16);
                            (&zr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_189[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        uint32_t _atomic_fetch_inc_190;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_190) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        flag[1] = (int)_atomic_fetch_inc_190;
                    }
                }
                float total_3 = 0.0f;
                float2 dx2_acc_3 = make_float2(0.0f, 0.0f);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[0], xr_3[1]), make_float2(xr_3[0], xr_3[1]), dx2_acc_3);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[2], xr_3[3]), make_float2(xr_3[2], xr_3[3]), dx2_acc_3);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[4], xr_3[5]), make_float2(xr_3[4], xr_3[5]), dx2_acc_3);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[6], xr_3[7]), make_float2(xr_3[6], xr_3[7]), dx2_acc_3);
                total_3 = dx2_acc_3.x + dx2_acc_3.y;
                float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, total_3, 1);
                total_3 = total_3 + _shfl_xor_sync_191;
                float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, total_3, 2);
                total_3 = total_3 + _shfl_xor_sync_192;
                float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, total_3, 4);
                total_3 = total_3 + _shfl_xor_sync_193;
                float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, total_3, 8);
                total_3 = total_3 + _shfl_xor_sync_194;
                float rstd_3;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_3) : "f"(total_3 * 0.0078125f + eps));
                float dot_3 = 0.0f;
                float wk_3[2];
                unsigned int packed_dz_3[4];
                float2 rstd2_3 = make_float2(rstd_3, rstd_3);
                float2 z2_12 = make_float2(zr_3[0], zr_3[1]);
                float2 _f32x2_mul_b_195 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_195));
                float _fma_196;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(zr_3[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_197;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(zr_3[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_12 = make_float2(_fma_196, _fma_197);
                float _exp2_198 = approx_exp2(ep2_12.x);
                float _exp2_199 = approx_exp2(ep2_12.y);
                float2 dq2_12 = fma_f32x2_rn_noftz(make_float2(_exp2_198, _exp2_199), qr2_12, make_float2(0.25f, 0.25f));
                float _rcp_200 = approx_rcp(dq2_12.x);
                float _rcp_201 = approx_rcp(dq2_12.y);
                float2 _f32x2_mul_a_202 = make_float2(_rcp_200, _rcp_201);
                float2 _f32x2_mul_b_203 = make_float2(0.25f, 0.25f);
                float2 sig2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&_f32x2_mul_b_203));
                float2 gate2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&sig2_12));
                float2 g2_12 = make_float2(gr_3[0], gr_3[1]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[_pair]));
                }
                float2 w2_12 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_204 = make_float2(xr_3[0], xr_3[1]);
                float2 n2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_205 = mul_f32x2_noftz(mul_f32x2_noftz(g2_12, mul_f32x2_noftz(n2_12, w2_12)), sig2_12);
                float2 _f32x2_mul_b_206 = fma_f32x2_rn_noftz(z2_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_12), make_float2(1.0f, 1.0f));
                float2 dz2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_205), "l"(*(const unsigned long long*)&_f32x2_mul_b_206));
                float2 a2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_12) : "l"(*(const unsigned long long*)&g2_12), "l"(*(const unsigned long long*)&gate2_12));
                float2 dn2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_12) : "l"(*(const unsigned long long*)&a2_12), "l"(*(const unsigned long long*)&w2_12));
                zr_3[0] = dz2_12.x;
                zr_3[1] = dz2_12.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 0], zr_3[_lp*2+1 + 0]));
                    packed_dz_3[_lp] = *(uint32_t*)&_bf2;
                }
                gr_3[0] = dn2_12.x;
                gr_3[1] = dn2_12.y;
                float2 z2_13 = make_float2(zr_3[2], zr_3[3]);
                float2 _f32x2_mul_b_207 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
                float _fma_208;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(zr_3[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_209;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(zr_3[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_13 = make_float2(_fma_208, _fma_209);
                float _exp2_210 = approx_exp2(ep2_13.x);
                float _exp2_211 = approx_exp2(ep2_13.y);
                float2 dq2_13 = fma_f32x2_rn_noftz(make_float2(_exp2_210, _exp2_211), qr2_13, make_float2(0.25f, 0.25f));
                float _rcp_212 = approx_rcp(dq2_13.x);
                float _rcp_213 = approx_rcp(dq2_13.y);
                float2 _f32x2_mul_a_214 = make_float2(_rcp_212, _rcp_213);
                float2 _f32x2_mul_b_215 = make_float2(0.25f, 0.25f);
                float2 sig2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&_f32x2_mul_b_215));
                float2 gate2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&sig2_13));
                float2 g2_13 = make_float2(gr_3[2], gr_3[3]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[1 + _pair]));
                }
                float2 w2_13 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_216 = make_float2(xr_3[2], xr_3[3]);
                float2 n2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_217 = mul_f32x2_noftz(mul_f32x2_noftz(g2_13, mul_f32x2_noftz(n2_13, w2_13)), sig2_13);
                float2 _f32x2_mul_b_218 = fma_f32x2_rn_noftz(z2_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_13), make_float2(1.0f, 1.0f));
                float2 dz2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_217), "l"(*(const unsigned long long*)&_f32x2_mul_b_218));
                float2 a2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_13) : "l"(*(const unsigned long long*)&g2_13), "l"(*(const unsigned long long*)&gate2_13));
                float2 dn2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_13) : "l"(*(const unsigned long long*)&a2_13), "l"(*(const unsigned long long*)&w2_13));
                zr_3[2] = dz2_13.x;
                zr_3[3] = dz2_13.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 2], zr_3[_lp*2+1 + 2]));
                    packed_dz_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                gr_3[2] = dn2_13.x;
                gr_3[3] = dn2_13.y;
                float2 z2_14 = make_float2(zr_3[4], zr_3[5]);
                float2 _f32x2_mul_b_219 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_219));
                float _fma_220;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(zr_3[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_221;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(zr_3[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_14 = make_float2(_fma_220, _fma_221);
                float _exp2_222 = approx_exp2(ep2_14.x);
                float _exp2_223 = approx_exp2(ep2_14.y);
                float2 dq2_14 = fma_f32x2_rn_noftz(make_float2(_exp2_222, _exp2_223), qr2_14, make_float2(0.25f, 0.25f));
                float _rcp_224 = approx_rcp(dq2_14.x);
                float _rcp_225 = approx_rcp(dq2_14.y);
                float2 _f32x2_mul_a_226 = make_float2(_rcp_224, _rcp_225);
                float2 _f32x2_mul_b_227 = make_float2(0.25f, 0.25f);
                float2 sig2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&_f32x2_mul_b_227));
                float2 gate2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&sig2_14));
                float2 g2_14 = make_float2(gr_3[4], gr_3[5]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[2 + _pair]));
                }
                float2 w2_14 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_228 = make_float2(xr_3[4], xr_3[5]);
                float2 n2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_229 = mul_f32x2_noftz(mul_f32x2_noftz(g2_14, mul_f32x2_noftz(n2_14, w2_14)), sig2_14);
                float2 _f32x2_mul_b_230 = fma_f32x2_rn_noftz(z2_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_14), make_float2(1.0f, 1.0f));
                float2 dz2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
                float2 a2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_14) : "l"(*(const unsigned long long*)&g2_14), "l"(*(const unsigned long long*)&gate2_14));
                float2 dn2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_14) : "l"(*(const unsigned long long*)&a2_14), "l"(*(const unsigned long long*)&w2_14));
                zr_3[4] = dz2_14.x;
                zr_3[5] = dz2_14.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 4], zr_3[_lp*2+1 + 4]));
                    packed_dz_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                gr_3[4] = dn2_14.x;
                gr_3[5] = dn2_14.y;
                float2 z2_15 = make_float2(zr_3[6], zr_3[7]);
                float2 _f32x2_mul_b_231 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_231));
                float _fma_232;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(zr_3[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_233;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(zr_3[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_15 = make_float2(_fma_232, _fma_233);
                float _exp2_234 = approx_exp2(ep2_15.x);
                float _exp2_235 = approx_exp2(ep2_15.y);
                float2 dq2_15 = fma_f32x2_rn_noftz(make_float2(_exp2_234, _exp2_235), qr2_15, make_float2(0.25f, 0.25f));
                float _rcp_236 = approx_rcp(dq2_15.x);
                float _rcp_237 = approx_rcp(dq2_15.y);
                float2 _f32x2_mul_a_238 = make_float2(_rcp_236, _rcp_237);
                float2 _f32x2_mul_b_239 = make_float2(0.25f, 0.25f);
                float2 sig2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&_f32x2_mul_b_239));
                float2 gate2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&sig2_15));
                float2 g2_15 = make_float2(gr_3[6], gr_3[7]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[3 + _pair]));
                }
                float2 w2_15 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_240 = make_float2(xr_3[6], xr_3[7]);
                float2 n2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_240), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_241 = mul_f32x2_noftz(mul_f32x2_noftz(g2_15, mul_f32x2_noftz(n2_15, w2_15)), sig2_15);
                float2 _f32x2_mul_b_242 = fma_f32x2_rn_noftz(z2_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_15), make_float2(1.0f, 1.0f));
                float2 dz2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_241), "l"(*(const unsigned long long*)&_f32x2_mul_b_242));
                float2 a2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_15) : "l"(*(const unsigned long long*)&g2_15), "l"(*(const unsigned long long*)&gate2_15));
                float2 dn2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_15) : "l"(*(const unsigned long long*)&a2_15), "l"(*(const unsigned long long*)&w2_15));
                zr_3[6] = dz2_15.x;
                zr_3[7] = dz2_15.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 6], zr_3[_lp*2+1 + 6]));
                    packed_dz_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                gr_3[6] = dn2_15.x;
                gr_3[7] = dn2_15.y;
                float2 dx2_acc_3_1 = make_float2(0.0f, 0.0f);
                dx2_acc_3_1 = fma_f32x2_rn_noftz(make_float2(gr_3[0], gr_3[1]), make_float2(xr_3[0], xr_3[1]), dx2_acc_3_1);
                dx2_acc_3_1 = fma_f32x2_rn_noftz(make_float2(gr_3[2], gr_3[3]), make_float2(xr_3[2], xr_3[3]), dx2_acc_3_1);
                dx2_acc_3_1 = fma_f32x2_rn_noftz(make_float2(gr_3[4], gr_3[5]), make_float2(xr_3[4], xr_3[5]), dx2_acc_3_1);
                dx2_acc_3_1 = fma_f32x2_rn_noftz(make_float2(gr_3[6], gr_3[7]), make_float2(xr_3[6], xr_3[7]), dx2_acc_3_1);
                dot_3 = dx2_acc_3_1.x + dx2_acc_3_1.y;
                float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_243;
                float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_244;
                float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_245;
                float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 8);
                dot_3 = dot_3 + _shfl_xor_sync_246;
                float coef_3 = rstd_3 * rstd_3 * rstd_3 * (dot_3 * -0.0078125f);
                float2 coef2_3 = make_float2(coef_3, coef_3);
                float2 rstd2b_3 = make_float2(rstd_3, rstd_3);
                float2 dx2_12 = fma_f32x2_rn_noftz(make_float2(xr_3[0], xr_3[1]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[0], gr_3[1]), rstd2b_3));
                xr_3[0] = dx2_12.x;
                xr_3[1] = dx2_12.y;
                float2 dx2_13 = fma_f32x2_rn_noftz(make_float2(xr_3[2], xr_3[3]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[2], gr_3[3]), rstd2b_3));
                xr_3[2] = dx2_13.x;
                xr_3[3] = dx2_13.y;
                float2 dx2_14 = fma_f32x2_rn_noftz(make_float2(xr_3[4], xr_3[5]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[4], gr_3[5]), rstd2b_3));
                xr_3[4] = dx2_14.x;
                xr_3[5] = dx2_14.y;
                float2 dx2_15 = fma_f32x2_rn_noftz(make_float2(xr_3[6], xr_3[7]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[6], gr_3[7]), rstd2b_3));
                xr_3[6] = dx2_15.x;
                xr_3[7] = dx2_15.y;
                unsigned int packed_dx_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 0], xr_3[_lp*2+1 + 0]));
                    packed_dx_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 2], xr_3[_lp*2+1 + 2]));
                    packed_dx_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 4], xr_3[_lp*2+1 + 4]));
                    packed_dx_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 6], xr_3[_lp*2+1 + 6]));
                    packed_dx_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long orow_3 = goff_3;
                if (row_3 < rows) {
                    reinterpret_cast<int4*>(dx + orow_3)[0] = reinterpret_cast<int4*>(packed_dx_3)[0];
                    reinterpret_cast<int4*>(dz + orow_3)[0] = reinterpret_cast<int4*>(packed_dz_3)[0];
                }
            } else {
                if (chunk_n > 1) {
                    uint32_t vq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)((bound + chunk_lo) * 16 + group)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)((bound + chunk_lo) * 16 + group));
                    uint32_t vh = (uint32_t)((unsigned int)((bound + chunk_lo) * 16 + group)) - vq * (uint32_t)(heads.divisor);
                    long long vgoff = (long long)vq * dy_tok + (long long)vh * dy_head + (long long)col;
                    {
                        uint4 _uv4_247 = *reinterpret_cast<const uint4*>(dy + vgoff + 0);
                        la_g[0 + 0] = _uv4_247.x;
                        la_g[0 + 1] = _uv4_247.y;
                        la_g[0 + 2] = _uv4_247.z;
                        la_g[0 + 3] = _uv4_247.w;
                    }
                    long long vxoff = (long long)vq * x_tok + (long long)vh * x_head + (long long)col;
                    {
                        uint4 _uv4_248 = *reinterpret_cast<const uint4*>(x + vxoff + 0);
                        la_x[0 + 0] = _uv4_248.x;
                        la_x[0 + 1] = _uv4_248.y;
                        la_x[0 + 2] = _uv4_248.z;
                        la_x[0 + 3] = _uv4_248.w;
                    }
                    long long vzoff = (long long)vq * z_tok + (long long)vh * z_head + (long long)col;
                    {
                        uint4 _uv4_249 = *reinterpret_cast<const uint4*>(z + vzoff + 0);
                        la_z[0 + 0] = _uv4_249.x;
                        la_z[0 + 1] = _uv4_249.y;
                        la_z[0 + 2] = _uv4_249.z;
                        la_z[0 + 3] = _uv4_249.w;
                    }
                }
                #pragma unroll 1
                for (int i_1 = 0; i_1 < chunk_n - 1; i_1++) {
                    int u_4 = bound + chunk_lo + i_1 * stride;
                    int row_4 = u_4 * 16 + group;
                    float gr_4[8];
                    float xr_4[8];
                    float zr_4[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) << 16);
                        (&xr_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) << 16);
                        (&gr_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) << 16);
                        (&zr_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) & 0xffff0000u);
                    }
                    if (i_1 + 2 < chunk_n) {
                        uint32_t vq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(row_4 + stride * 16)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(row_4 + stride * 16));
                        uint32_t vh_1 = (uint32_t)((unsigned int)(row_4 + stride * 16)) - vq_1 * (uint32_t)(heads.divisor);
                        long long vgoff_1 = (long long)vq_1 * dy_tok + (long long)vh_1 * dy_head + (long long)col;
                        {
                            uint4 _uv4_250 = *reinterpret_cast<const uint4*>(dy + vgoff_1 + 0);
                            la_g[0 + 0] = _uv4_250.x;
                            la_g[0 + 1] = _uv4_250.y;
                            la_g[0 + 2] = _uv4_250.z;
                            la_g[0 + 3] = _uv4_250.w;
                        }
                        long long vxoff_1 = (long long)vq_1 * x_tok + (long long)vh_1 * x_head + (long long)col;
                        {
                            uint4 _uv4_251 = *reinterpret_cast<const uint4*>(x + vxoff_1 + 0);
                            la_x[0 + 0] = _uv4_251.x;
                            la_x[0 + 1] = _uv4_251.y;
                            la_x[0 + 2] = _uv4_251.z;
                            la_x[0 + 3] = _uv4_251.w;
                        }
                        long long vzoff_1 = (long long)vq_1 * z_tok + (long long)vh_1 * z_head + (long long)col;
                        {
                            uint4 _uv4_252 = *reinterpret_cast<const uint4*>(z + vzoff_1 + 0);
                            la_z[0 + 0] = _uv4_252.x;
                            la_z[0 + 1] = _uv4_252.y;
                            la_z[0 + 2] = _uv4_252.z;
                            la_z[0 + 3] = _uv4_252.w;
                        }
                    }
                    if (ahead != 0) {
                        int nrow = (((u_4 + stride) * 16 + group) < (rows - 1) ? ((u_4 + stride) * 16 + group) : (rows - 1));
                        uint32_t nq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow);
                        uint32_t nh = (uint32_t)((unsigned int)nrow) - nq * (uint32_t)(heads.divisor);
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq * dy_tok + (long long)nh * dy_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq * x_tok + (long long)nh * x_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq * z_tok + (long long)nh * z_head + (long long)col))));
                    }
                    float total_4 = 0.0f;
                    float2 dx2_acc_4 = make_float2(0.0f, 0.0f);
                    dx2_acc_4 = fma_f32x2_rn_noftz(make_float2(xr_4[0], xr_4[1]), make_float2(xr_4[0], xr_4[1]), dx2_acc_4);
                    dx2_acc_4 = fma_f32x2_rn_noftz(make_float2(xr_4[2], xr_4[3]), make_float2(xr_4[2], xr_4[3]), dx2_acc_4);
                    dx2_acc_4 = fma_f32x2_rn_noftz(make_float2(xr_4[4], xr_4[5]), make_float2(xr_4[4], xr_4[5]), dx2_acc_4);
                    dx2_acc_4 = fma_f32x2_rn_noftz(make_float2(xr_4[6], xr_4[7]), make_float2(xr_4[6], xr_4[7]), dx2_acc_4);
                    total_4 = dx2_acc_4.x + dx2_acc_4.y;
                    float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, total_4, 1);
                    total_4 = total_4 + _shfl_xor_sync_253;
                    float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, total_4, 2);
                    total_4 = total_4 + _shfl_xor_sync_254;
                    float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, total_4, 4);
                    total_4 = total_4 + _shfl_xor_sync_255;
                    float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, total_4, 8);
                    total_4 = total_4 + _shfl_xor_sync_256;
                    float rstd_4;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_4) : "f"(total_4 * 0.0078125f + eps));
                    float dot_4 = 0.0f;
                    float wk_4[2];
                    unsigned int packed_dz_4[4];
                    float2 rstd2_4 = make_float2(rstd_4, rstd_4);
                    float2 z2_16 = make_float2(zr_4[0], zr_4[1]);
                    float2 _f32x2_mul_b_257 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_257));
                    float _fma_258;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(zr_4[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_259;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(zr_4[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_16 = make_float2(_fma_258, _fma_259);
                    float _exp2_260 = approx_exp2(ep2_16.x);
                    float _exp2_261 = approx_exp2(ep2_16.y);
                    float2 dq2_16 = fma_f32x2_rn_noftz(make_float2(_exp2_260, _exp2_261), qr2_16, make_float2(0.25f, 0.25f));
                    float _rcp_262 = approx_rcp(dq2_16.x);
                    float _rcp_263 = approx_rcp(dq2_16.y);
                    float2 _f32x2_mul_a_264 = make_float2(_rcp_262, _rcp_263);
                    float2 _f32x2_mul_b_265 = make_float2(0.25f, 0.25f);
                    float2 sig2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_264), "l"(*(const unsigned long long*)&_f32x2_mul_b_265));
                    float2 gate2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&sig2_16));
                    float2 g2_16 = make_float2(gr_4[0], gr_4[1]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[_pair]));
                    }
                    float2 w2_16 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_266 = make_float2(xr_4[0], xr_4[1]);
                    float2 n2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_266), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_267 = mul_f32x2_noftz(mul_f32x2_noftz(g2_16, mul_f32x2_noftz(n2_16, w2_16)), sig2_16);
                    float2 _f32x2_mul_b_268 = fma_f32x2_rn_noftz(z2_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_16), make_float2(1.0f, 1.0f));
                    float2 dz2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&_f32x2_mul_b_268));
                    float2 a2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_16) : "l"(*(const unsigned long long*)&g2_16), "l"(*(const unsigned long long*)&gate2_16));
                    float2 dn2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_16) : "l"(*(const unsigned long long*)&a2_16), "l"(*(const unsigned long long*)&w2_16));
                    zr_4[0] = dz2_16.x;
                    zr_4[1] = dz2_16.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 0], zr_4[_lp*2+1 + 0]));
                        packed_dz_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    gr_4[0] = dn2_16.x;
                    gr_4[1] = dn2_16.y;
                    float2 z2_17 = make_float2(zr_4[2], zr_4[3]);
                    float2 _f32x2_mul_b_269 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_269));
                    float _fma_270;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(zr_4[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_271;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(zr_4[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_17 = make_float2(_fma_270, _fma_271);
                    float _exp2_272 = approx_exp2(ep2_17.x);
                    float _exp2_273 = approx_exp2(ep2_17.y);
                    float2 dq2_17 = fma_f32x2_rn_noftz(make_float2(_exp2_272, _exp2_273), qr2_17, make_float2(0.25f, 0.25f));
                    float _rcp_274 = approx_rcp(dq2_17.x);
                    float _rcp_275 = approx_rcp(dq2_17.y);
                    float2 _f32x2_mul_a_276 = make_float2(_rcp_274, _rcp_275);
                    float2 _f32x2_mul_b_277 = make_float2(0.25f, 0.25f);
                    float2 sig2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_276), "l"(*(const unsigned long long*)&_f32x2_mul_b_277));
                    float2 gate2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&sig2_17));
                    float2 g2_17 = make_float2(gr_4[2], gr_4[3]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[1 + _pair]));
                    }
                    float2 w2_17 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_278 = make_float2(xr_4[2], xr_4[3]);
                    float2 n2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_278), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_279 = mul_f32x2_noftz(mul_f32x2_noftz(g2_17, mul_f32x2_noftz(n2_17, w2_17)), sig2_17);
                    float2 _f32x2_mul_b_280 = fma_f32x2_rn_noftz(z2_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_17), make_float2(1.0f, 1.0f));
                    float2 dz2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_279), "l"(*(const unsigned long long*)&_f32x2_mul_b_280));
                    float2 a2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_17) : "l"(*(const unsigned long long*)&g2_17), "l"(*(const unsigned long long*)&gate2_17));
                    float2 dn2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_17) : "l"(*(const unsigned long long*)&a2_17), "l"(*(const unsigned long long*)&w2_17));
                    zr_4[2] = dz2_17.x;
                    zr_4[3] = dz2_17.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 2], zr_4[_lp*2+1 + 2]));
                        packed_dz_4[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    gr_4[2] = dn2_17.x;
                    gr_4[3] = dn2_17.y;
                    float2 z2_18 = make_float2(zr_4[4], zr_4[5]);
                    float2 _f32x2_mul_b_281 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_281));
                    float _fma_282;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(zr_4[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_283;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(zr_4[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_18 = make_float2(_fma_282, _fma_283);
                    float _exp2_284 = approx_exp2(ep2_18.x);
                    float _exp2_285 = approx_exp2(ep2_18.y);
                    float2 dq2_18 = fma_f32x2_rn_noftz(make_float2(_exp2_284, _exp2_285), qr2_18, make_float2(0.25f, 0.25f));
                    float _rcp_286 = approx_rcp(dq2_18.x);
                    float _rcp_287 = approx_rcp(dq2_18.y);
                    float2 _f32x2_mul_a_288 = make_float2(_rcp_286, _rcp_287);
                    float2 _f32x2_mul_b_289 = make_float2(0.25f, 0.25f);
                    float2 sig2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_288), "l"(*(const unsigned long long*)&_f32x2_mul_b_289));
                    float2 gate2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&sig2_18));
                    float2 g2_18 = make_float2(gr_4[4], gr_4[5]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[2 + _pair]));
                    }
                    float2 w2_18 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_290 = make_float2(xr_4[4], xr_4[5]);
                    float2 n2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_290), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_291 = mul_f32x2_noftz(mul_f32x2_noftz(g2_18, mul_f32x2_noftz(n2_18, w2_18)), sig2_18);
                    float2 _f32x2_mul_b_292 = fma_f32x2_rn_noftz(z2_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_18), make_float2(1.0f, 1.0f));
                    float2 dz2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_291), "l"(*(const unsigned long long*)&_f32x2_mul_b_292));
                    float2 a2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_18) : "l"(*(const unsigned long long*)&g2_18), "l"(*(const unsigned long long*)&gate2_18));
                    float2 dn2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_18) : "l"(*(const unsigned long long*)&a2_18), "l"(*(const unsigned long long*)&w2_18));
                    zr_4[4] = dz2_18.x;
                    zr_4[5] = dz2_18.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 4], zr_4[_lp*2+1 + 4]));
                        packed_dz_4[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    gr_4[4] = dn2_18.x;
                    gr_4[5] = dn2_18.y;
                    float2 z2_19 = make_float2(zr_4[6], zr_4[7]);
                    float2 _f32x2_mul_b_293 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_293));
                    float _fma_294;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(zr_4[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_295;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(zr_4[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_19 = make_float2(_fma_294, _fma_295);
                    float _exp2_296 = approx_exp2(ep2_19.x);
                    float _exp2_297 = approx_exp2(ep2_19.y);
                    float2 dq2_19 = fma_f32x2_rn_noftz(make_float2(_exp2_296, _exp2_297), qr2_19, make_float2(0.25f, 0.25f));
                    float _rcp_298 = approx_rcp(dq2_19.x);
                    float _rcp_299 = approx_rcp(dq2_19.y);
                    float2 _f32x2_mul_a_300 = make_float2(_rcp_298, _rcp_299);
                    float2 _f32x2_mul_b_301 = make_float2(0.25f, 0.25f);
                    float2 sig2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_300), "l"(*(const unsigned long long*)&_f32x2_mul_b_301));
                    float2 gate2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&sig2_19));
                    float2 g2_19 = make_float2(gr_4[6], gr_4[7]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[3 + _pair]));
                    }
                    float2 w2_19 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_302 = make_float2(xr_4[6], xr_4[7]);
                    float2 n2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_302), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_303 = mul_f32x2_noftz(mul_f32x2_noftz(g2_19, mul_f32x2_noftz(n2_19, w2_19)), sig2_19);
                    float2 _f32x2_mul_b_304 = fma_f32x2_rn_noftz(z2_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_19), make_float2(1.0f, 1.0f));
                    float2 dz2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_303), "l"(*(const unsigned long long*)&_f32x2_mul_b_304));
                    float2 a2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_19) : "l"(*(const unsigned long long*)&g2_19), "l"(*(const unsigned long long*)&gate2_19));
                    float2 dn2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_19) : "l"(*(const unsigned long long*)&a2_19), "l"(*(const unsigned long long*)&w2_19));
                    zr_4[6] = dz2_19.x;
                    zr_4[7] = dz2_19.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 6], zr_4[_lp*2+1 + 6]));
                        packed_dz_4[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    gr_4[6] = dn2_19.x;
                    gr_4[7] = dn2_19.y;
                    float2 dx2_acc_4_1 = make_float2(0.0f, 0.0f);
                    dx2_acc_4_1 = fma_f32x2_rn_noftz(make_float2(gr_4[0], gr_4[1]), make_float2(xr_4[0], xr_4[1]), dx2_acc_4_1);
                    dx2_acc_4_1 = fma_f32x2_rn_noftz(make_float2(gr_4[2], gr_4[3]), make_float2(xr_4[2], xr_4[3]), dx2_acc_4_1);
                    dx2_acc_4_1 = fma_f32x2_rn_noftz(make_float2(gr_4[4], gr_4[5]), make_float2(xr_4[4], xr_4[5]), dx2_acc_4_1);
                    dx2_acc_4_1 = fma_f32x2_rn_noftz(make_float2(gr_4[6], gr_4[7]), make_float2(xr_4[6], xr_4[7]), dx2_acc_4_1);
                    dot_4 = dx2_acc_4_1.x + dx2_acc_4_1.y;
                    float _shfl_xor_sync_305 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                    dot_4 = dot_4 + _shfl_xor_sync_305;
                    float _shfl_xor_sync_306 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                    dot_4 = dot_4 + _shfl_xor_sync_306;
                    float _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                    dot_4 = dot_4 + _shfl_xor_sync_307;
                    float _shfl_xor_sync_308 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 8);
                    dot_4 = dot_4 + _shfl_xor_sync_308;
                    float coef_4 = rstd_4 * rstd_4 * rstd_4 * (dot_4 * -0.0078125f);
                    float2 coef2_4 = make_float2(coef_4, coef_4);
                    float2 rstd2b_4 = make_float2(rstd_4, rstd_4);
                    float2 dx2_16 = fma_f32x2_rn_noftz(make_float2(xr_4[0], xr_4[1]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[0], gr_4[1]), rstd2b_4));
                    xr_4[0] = dx2_16.x;
                    xr_4[1] = dx2_16.y;
                    float2 dx2_17 = fma_f32x2_rn_noftz(make_float2(xr_4[2], xr_4[3]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[2], gr_4[3]), rstd2b_4));
                    xr_4[2] = dx2_17.x;
                    xr_4[3] = dx2_17.y;
                    float2 dx2_18 = fma_f32x2_rn_noftz(make_float2(xr_4[4], xr_4[5]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[4], gr_4[5]), rstd2b_4));
                    xr_4[4] = dx2_18.x;
                    xr_4[5] = dx2_18.y;
                    float2 dx2_19 = fma_f32x2_rn_noftz(make_float2(xr_4[6], xr_4[7]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[6], gr_4[7]), rstd2b_4));
                    xr_4[6] = dx2_19.x;
                    xr_4[7] = dx2_19.y;
                    unsigned int packed_dx_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 0], xr_4[_lp*2+1 + 0]));
                        packed_dx_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 2], xr_4[_lp*2+1 + 2]));
                        packed_dx_4[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 4], xr_4[_lp*2+1 + 4]));
                        packed_dx_4[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 6], xr_4[_lp*2+1 + 6]));
                        packed_dx_4[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    long long orow_4 = (long long)row_4 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_4)[0] = reinterpret_cast<int4*>(packed_dx_4)[0];
                    reinterpret_cast<int4*>(dz + orow_4)[0] = reinterpret_cast<int4*>(packed_dz_4)[0];
                }
                int tail_i_3 = chunk_n - 1;
                int u_5 = bound + chunk_lo + tail_i_3 * stride;
                int row_5 = u_5 * 16 + group;
                int src_3 = ((row_5) < (rows - 1) ? (row_5) : (rows - 1));
                uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_3), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_3);
                uint32_t th_1 = (uint32_t)((unsigned int)src_3) - tq_1 * (uint32_t)(heads.divisor);
                long long goff_4 = (long long)tq_1 * dy_tok + (long long)th_1 * dy_head + (long long)col;
                long long xoff_3 = (long long)tq_1 * x_tok + (long long)th_1 * x_head + (long long)col;
                long long zoff_3 = (long long)tq_1 * z_tok + (long long)th_1 * z_head + (long long)col;
                float gr_5[8];
                float xr_5[8];
                float zr_5[8];
                {
                    const uint4* _vptr_309 = reinterpret_cast<const uint4*>(dy + goff_4 + 0);
                    uint4 _vld_309[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_309[_blk] = _vptr_309[_blk];
                        uint32_t* _vpairs_309 = reinterpret_cast<uint32_t*>(&_vld_309[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_309[_pair]) << 16);
                            (&gr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_309[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_310 = reinterpret_cast<const uint4*>(x + xoff_3 + 0);
                    uint4 _vld_310[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_310[_blk] = _vptr_310[_blk];
                        uint32_t* _vpairs_310 = reinterpret_cast<uint32_t*>(&_vld_310[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_310[_pair]) << 16);
                            (&xr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_310[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_311 = reinterpret_cast<const uint4*>(z + zoff_3 + 0);
                    uint4 _vld_311[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_311[_blk] = _vptr_311[_blk];
                        uint32_t* _vpairs_311 = reinterpret_cast<uint32_t*>(&_vld_311[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_311[_pair]) << 16);
                            (&zr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_311[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        uint32_t _atomic_fetch_inc_312;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_312) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        flag[1] = (int)_atomic_fetch_inc_312;
                    }
                }
                float total_5 = 0.0f;
                float2 dx2_acc_5 = make_float2(0.0f, 0.0f);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[0], xr_5[1]), make_float2(xr_5[0], xr_5[1]), dx2_acc_5);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[2], xr_5[3]), make_float2(xr_5[2], xr_5[3]), dx2_acc_5);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[4], xr_5[5]), make_float2(xr_5[4], xr_5[5]), dx2_acc_5);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[6], xr_5[7]), make_float2(xr_5[6], xr_5[7]), dx2_acc_5);
                total_5 = dx2_acc_5.x + dx2_acc_5.y;
                float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, total_5, 1);
                total_5 = total_5 + _shfl_xor_sync_313;
                float _shfl_xor_sync_314 = __shfl_xor_sync(0xFFFFFFFF, total_5, 2);
                total_5 = total_5 + _shfl_xor_sync_314;
                float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, total_5, 4);
                total_5 = total_5 + _shfl_xor_sync_315;
                float _shfl_xor_sync_316 = __shfl_xor_sync(0xFFFFFFFF, total_5, 8);
                total_5 = total_5 + _shfl_xor_sync_316;
                float rstd_5;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_5) : "f"(total_5 * 0.0078125f + eps));
                float dot_5 = 0.0f;
                float wk_5[2];
                unsigned int packed_dz_5[4];
                float2 rstd2_5 = make_float2(rstd_5, rstd_5);
                float2 z2_20 = make_float2(zr_5[0], zr_5[1]);
                float2 _f32x2_mul_b_317 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_317));
                float _fma_318;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(zr_5[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_319;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(zr_5[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_20 = make_float2(_fma_318, _fma_319);
                float _exp2_320 = approx_exp2(ep2_20.x);
                float _exp2_321 = approx_exp2(ep2_20.y);
                float2 dq2_20 = fma_f32x2_rn_noftz(make_float2(_exp2_320, _exp2_321), qr2_20, make_float2(0.25f, 0.25f));
                float _rcp_322 = approx_rcp(dq2_20.x);
                float _rcp_323 = approx_rcp(dq2_20.y);
                float2 _f32x2_mul_a_324 = make_float2(_rcp_322, _rcp_323);
                float2 _f32x2_mul_b_325 = make_float2(0.25f, 0.25f);
                float2 sig2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_324), "l"(*(const unsigned long long*)&_f32x2_mul_b_325));
                float2 gate2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&sig2_20));
                float2 g2_20 = make_float2(gr_5[0], gr_5[1]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[_pair]));
                }
                float2 w2_20 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_326 = make_float2(xr_5[0], xr_5[1]);
                float2 n2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_326), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_327 = mul_f32x2_noftz(mul_f32x2_noftz(g2_20, mul_f32x2_noftz(n2_20, w2_20)), sig2_20);
                float2 _f32x2_mul_b_328 = fma_f32x2_rn_noftz(z2_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_20), make_float2(1.0f, 1.0f));
                float2 dz2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_327), "l"(*(const unsigned long long*)&_f32x2_mul_b_328));
                float2 a2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_20) : "l"(*(const unsigned long long*)&g2_20), "l"(*(const unsigned long long*)&gate2_20));
                float2 dn2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_20) : "l"(*(const unsigned long long*)&a2_20), "l"(*(const unsigned long long*)&w2_20));
                zr_5[0] = dz2_20.x;
                zr_5[1] = dz2_20.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 0], zr_5[_lp*2+1 + 0]));
                    packed_dz_5[_lp] = *(uint32_t*)&_bf2;
                }
                gr_5[0] = dn2_20.x;
                gr_5[1] = dn2_20.y;
                float2 z2_21 = make_float2(zr_5[2], zr_5[3]);
                float2 _f32x2_mul_b_329 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_329));
                float _fma_330;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(zr_5[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_331;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(zr_5[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_21 = make_float2(_fma_330, _fma_331);
                float _exp2_332 = approx_exp2(ep2_21.x);
                float _exp2_333 = approx_exp2(ep2_21.y);
                float2 dq2_21 = fma_f32x2_rn_noftz(make_float2(_exp2_332, _exp2_333), qr2_21, make_float2(0.25f, 0.25f));
                float _rcp_334 = approx_rcp(dq2_21.x);
                float _rcp_335 = approx_rcp(dq2_21.y);
                float2 _f32x2_mul_a_336 = make_float2(_rcp_334, _rcp_335);
                float2 _f32x2_mul_b_337 = make_float2(0.25f, 0.25f);
                float2 sig2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_336), "l"(*(const unsigned long long*)&_f32x2_mul_b_337));
                float2 gate2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&sig2_21));
                float2 g2_21 = make_float2(gr_5[2], gr_5[3]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[1 + _pair]));
                }
                float2 w2_21 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_338 = make_float2(xr_5[2], xr_5[3]);
                float2 n2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_338), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_339 = mul_f32x2_noftz(mul_f32x2_noftz(g2_21, mul_f32x2_noftz(n2_21, w2_21)), sig2_21);
                float2 _f32x2_mul_b_340 = fma_f32x2_rn_noftz(z2_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_21), make_float2(1.0f, 1.0f));
                float2 dz2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_339), "l"(*(const unsigned long long*)&_f32x2_mul_b_340));
                float2 a2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_21) : "l"(*(const unsigned long long*)&g2_21), "l"(*(const unsigned long long*)&gate2_21));
                float2 dn2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_21) : "l"(*(const unsigned long long*)&a2_21), "l"(*(const unsigned long long*)&w2_21));
                zr_5[2] = dz2_21.x;
                zr_5[3] = dz2_21.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 2], zr_5[_lp*2+1 + 2]));
                    packed_dz_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                gr_5[2] = dn2_21.x;
                gr_5[3] = dn2_21.y;
                float2 z2_22 = make_float2(zr_5[4], zr_5[5]);
                float2 _f32x2_mul_b_341 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_341));
                float _fma_342;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(zr_5[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_343;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(zr_5[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_22 = make_float2(_fma_342, _fma_343);
                float _exp2_344 = approx_exp2(ep2_22.x);
                float _exp2_345 = approx_exp2(ep2_22.y);
                float2 dq2_22 = fma_f32x2_rn_noftz(make_float2(_exp2_344, _exp2_345), qr2_22, make_float2(0.25f, 0.25f));
                float _rcp_346 = approx_rcp(dq2_22.x);
                float _rcp_347 = approx_rcp(dq2_22.y);
                float2 _f32x2_mul_a_348 = make_float2(_rcp_346, _rcp_347);
                float2 _f32x2_mul_b_349 = make_float2(0.25f, 0.25f);
                float2 sig2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_348), "l"(*(const unsigned long long*)&_f32x2_mul_b_349));
                float2 gate2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&sig2_22));
                float2 g2_22 = make_float2(gr_5[4], gr_5[5]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[2 + _pair]));
                }
                float2 w2_22 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_350 = make_float2(xr_5[4], xr_5[5]);
                float2 n2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_350), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_351 = mul_f32x2_noftz(mul_f32x2_noftz(g2_22, mul_f32x2_noftz(n2_22, w2_22)), sig2_22);
                float2 _f32x2_mul_b_352 = fma_f32x2_rn_noftz(z2_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_22), make_float2(1.0f, 1.0f));
                float2 dz2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_351), "l"(*(const unsigned long long*)&_f32x2_mul_b_352));
                float2 a2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_22) : "l"(*(const unsigned long long*)&g2_22), "l"(*(const unsigned long long*)&gate2_22));
                float2 dn2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_22) : "l"(*(const unsigned long long*)&a2_22), "l"(*(const unsigned long long*)&w2_22));
                zr_5[4] = dz2_22.x;
                zr_5[5] = dz2_22.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 4], zr_5[_lp*2+1 + 4]));
                    packed_dz_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                gr_5[4] = dn2_22.x;
                gr_5[5] = dn2_22.y;
                float2 z2_23 = make_float2(zr_5[6], zr_5[7]);
                float2 _f32x2_mul_b_353 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_353));
                float _fma_354;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(zr_5[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_355;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(zr_5[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_23 = make_float2(_fma_354, _fma_355);
                float _exp2_356 = approx_exp2(ep2_23.x);
                float _exp2_357 = approx_exp2(ep2_23.y);
                float2 dq2_23 = fma_f32x2_rn_noftz(make_float2(_exp2_356, _exp2_357), qr2_23, make_float2(0.25f, 0.25f));
                float _rcp_358 = approx_rcp(dq2_23.x);
                float _rcp_359 = approx_rcp(dq2_23.y);
                float2 _f32x2_mul_a_360 = make_float2(_rcp_358, _rcp_359);
                float2 _f32x2_mul_b_361 = make_float2(0.25f, 0.25f);
                float2 sig2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_360), "l"(*(const unsigned long long*)&_f32x2_mul_b_361));
                float2 gate2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&sig2_23));
                float2 g2_23 = make_float2(gr_5[6], gr_5[7]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[3 + _pair]));
                }
                float2 w2_23 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_362 = make_float2(xr_5[6], xr_5[7]);
                float2 n2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_362), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_363 = mul_f32x2_noftz(mul_f32x2_noftz(g2_23, mul_f32x2_noftz(n2_23, w2_23)), sig2_23);
                float2 _f32x2_mul_b_364 = fma_f32x2_rn_noftz(z2_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_23), make_float2(1.0f, 1.0f));
                float2 dz2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_363), "l"(*(const unsigned long long*)&_f32x2_mul_b_364));
                float2 a2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_23) : "l"(*(const unsigned long long*)&g2_23), "l"(*(const unsigned long long*)&gate2_23));
                float2 dn2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_23) : "l"(*(const unsigned long long*)&a2_23), "l"(*(const unsigned long long*)&w2_23));
                zr_5[6] = dz2_23.x;
                zr_5[7] = dz2_23.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 6], zr_5[_lp*2+1 + 6]));
                    packed_dz_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                gr_5[6] = dn2_23.x;
                gr_5[7] = dn2_23.y;
                float2 dx2_acc_5_1 = make_float2(0.0f, 0.0f);
                dx2_acc_5_1 = fma_f32x2_rn_noftz(make_float2(gr_5[0], gr_5[1]), make_float2(xr_5[0], xr_5[1]), dx2_acc_5_1);
                dx2_acc_5_1 = fma_f32x2_rn_noftz(make_float2(gr_5[2], gr_5[3]), make_float2(xr_5[2], xr_5[3]), dx2_acc_5_1);
                dx2_acc_5_1 = fma_f32x2_rn_noftz(make_float2(gr_5[4], gr_5[5]), make_float2(xr_5[4], xr_5[5]), dx2_acc_5_1);
                dx2_acc_5_1 = fma_f32x2_rn_noftz(make_float2(gr_5[6], gr_5[7]), make_float2(xr_5[6], xr_5[7]), dx2_acc_5_1);
                dot_5 = dx2_acc_5_1.x + dx2_acc_5_1.y;
                float _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_365;
                float _shfl_xor_sync_366 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_366;
                float _shfl_xor_sync_367 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_367;
                float _shfl_xor_sync_368 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 8);
                dot_5 = dot_5 + _shfl_xor_sync_368;
                float coef_5 = rstd_5 * rstd_5 * rstd_5 * (dot_5 * -0.0078125f);
                float2 coef2_5 = make_float2(coef_5, coef_5);
                float2 rstd2b_5 = make_float2(rstd_5, rstd_5);
                float2 dx2_20 = fma_f32x2_rn_noftz(make_float2(xr_5[0], xr_5[1]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[0], gr_5[1]), rstd2b_5));
                xr_5[0] = dx2_20.x;
                xr_5[1] = dx2_20.y;
                float2 dx2_21 = fma_f32x2_rn_noftz(make_float2(xr_5[2], xr_5[3]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[2], gr_5[3]), rstd2b_5));
                xr_5[2] = dx2_21.x;
                xr_5[3] = dx2_21.y;
                float2 dx2_22 = fma_f32x2_rn_noftz(make_float2(xr_5[4], xr_5[5]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[4], gr_5[5]), rstd2b_5));
                xr_5[4] = dx2_22.x;
                xr_5[5] = dx2_22.y;
                float2 dx2_23 = fma_f32x2_rn_noftz(make_float2(xr_5[6], xr_5[7]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[6], gr_5[7]), rstd2b_5));
                xr_5[6] = dx2_23.x;
                xr_5[7] = dx2_23.y;
                unsigned int packed_dx_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 0], xr_5[_lp*2+1 + 0]));
                    packed_dx_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 2], xr_5[_lp*2+1 + 2]));
                    packed_dx_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 4], xr_5[_lp*2+1 + 4]));
                    packed_dx_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 6], xr_5[_lp*2+1 + 6]));
                    packed_dx_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long orow_5 = (long long)row_5 * 128ll + (long long)col;
                if (row_5 < rows) {
                    reinterpret_cast<int4*>(dx + orow_5)[0] = reinterpret_cast<int4*>(packed_dx_5)[0];
                    reinterpret_cast<int4*>(dz + orow_5)[0] = reinterpret_cast<int4*>(packed_dz_5)[0];
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            done[1] = 0;
        }
    }
    __syncthreads();
}

} // extern "C"
