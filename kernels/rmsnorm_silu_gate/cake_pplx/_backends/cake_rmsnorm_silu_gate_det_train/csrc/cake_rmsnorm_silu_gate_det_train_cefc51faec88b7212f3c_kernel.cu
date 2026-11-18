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
kernel_cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ acc, unsigned int* __restrict__ flags, unsigned int* __restrict__ done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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
        #pragma unroll 1
        for (int i = 0; i < 1; i++) {
            int u = bid + i;
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
            int fast = 1;
            float wk[2];
            unsigned int packed_dz[4];
            if (fast != 0) {
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
                float _exp_59 = __nv_expf(zr[0] * -1.0f);
                float den_0 = 1.0f + _exp_59;
                float sig_0 = __fdiv_rn(1.0f, den_0);
                float gate_0 = __fdiv_rn(zr[0], den_0);
                float nj_0 = xr[0] * rstd;
                float aj_0 = gr[0] * gate_0;
                float dzj_0 = gr[0] * (nj_0 * wk[0]) * sig_0 * __fmaf_rn(zr[0], 1.0f - sig_0, 1.0f);
                float dnj_0 = aj_0 * wk[0];
                zr[0] = dzj_0;
                gr[0] = dnj_0;
                float _exp_60 = __nv_expf(zr[1] * -1.0f);
                float den_1 = 1.0f + _exp_60;
                float sig_1 = __fdiv_rn(1.0f, den_1);
                float gate_1 = __fdiv_rn(zr[1], den_1);
                float nj_1 = xr[1] * rstd;
                float aj_1 = gr[1] * gate_1;
                float dzj_1 = gr[1] * (nj_1 * wk[1]) * sig_1 * __fmaf_rn(zr[1], 1.0f - sig_1, 1.0f);
                float dnj_1 = aj_1 * wk[1];
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
                float _exp_61 = __nv_expf(zr[2] * -1.0f);
                float den_2 = 1.0f + _exp_61;
                float sig_2 = __fdiv_rn(1.0f, den_2);
                float gate_2 = __fdiv_rn(zr[2], den_2);
                float nj_2 = xr[2] * rstd;
                float aj_2 = gr[2] * gate_2;
                float dzj_2 = gr[2] * (nj_2 * wk[0]) * sig_2 * __fmaf_rn(zr[2], 1.0f - sig_2, 1.0f);
                float dnj_2 = aj_2 * wk[0];
                zr[2] = dzj_2;
                gr[2] = dnj_2;
                float _exp_62 = __nv_expf(zr[3] * -1.0f);
                float den_3 = 1.0f + _exp_62;
                float sig_3 = __fdiv_rn(1.0f, den_3);
                float gate_3 = __fdiv_rn(zr[3], den_3);
                float nj_3 = xr[3] * rstd;
                float aj_3 = gr[3] * gate_3;
                float dzj_3 = gr[3] * (nj_3 * wk[1]) * sig_3 * __fmaf_rn(zr[3], 1.0f - sig_3, 1.0f);
                float dnj_3 = aj_3 * wk[1];
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
                float _exp_63 = __nv_expf(zr[4] * -1.0f);
                float den_4 = 1.0f + _exp_63;
                float sig_4 = __fdiv_rn(1.0f, den_4);
                float gate_4 = __fdiv_rn(zr[4], den_4);
                float nj_4 = xr[4] * rstd;
                float aj_4 = gr[4] * gate_4;
                float dzj_4 = gr[4] * (nj_4 * wk[0]) * sig_4 * __fmaf_rn(zr[4], 1.0f - sig_4, 1.0f);
                float dnj_4 = aj_4 * wk[0];
                zr[4] = dzj_4;
                gr[4] = dnj_4;
                float _exp_64 = __nv_expf(zr[5] * -1.0f);
                float den_5 = 1.0f + _exp_64;
                float sig_5 = __fdiv_rn(1.0f, den_5);
                float gate_5 = __fdiv_rn(zr[5], den_5);
                float nj_5 = xr[5] * rstd;
                float aj_5 = gr[5] * gate_5;
                float dzj_5 = gr[5] * (nj_5 * wk[1]) * sig_5 * __fmaf_rn(zr[5], 1.0f - sig_5, 1.0f);
                float dnj_5 = aj_5 * wk[1];
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
                float _exp_65 = __nv_expf(zr[6] * -1.0f);
                float den_6 = 1.0f + _exp_65;
                float sig_6 = __fdiv_rn(1.0f, den_6);
                float gate_6 = __fdiv_rn(zr[6], den_6);
                float nj_6 = xr[6] * rstd;
                float aj_6 = gr[6] * gate_6;
                float dzj_6 = gr[6] * (nj_6 * wk[0]) * sig_6 * __fmaf_rn(zr[6], 1.0f - sig_6, 1.0f);
                float dnj_6 = aj_6 * wk[0];
                zr[6] = dzj_6;
                gr[6] = dnj_6;
                float _exp_66 = __nv_expf(zr[7] * -1.0f);
                float den_7 = 1.0f + _exp_66;
                float sig_7 = __fdiv_rn(1.0f, den_7);
                float gate_7 = __fdiv_rn(zr[7], den_7);
                float nj_7 = xr[7] * rstd;
                float aj_7 = gr[7] * gate_7;
                float dzj_7 = gr[7] * (nj_7 * wk[1]) * sig_7 * __fmaf_rn(zr[7], 1.0f - sig_7, 1.0f);
                float dnj_7 = aj_7 * wk[1];
                zr[7] = dzj_7;
                gr[7] = dnj_7;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 6], zr[_lp*2+1 + 6]));
                    packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
                }
            }
            float2 dx2_acc_0 = make_float2(0.0f, 0.0f);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[0], gr[1]), make_float2(xr[0], xr[1]), dx2_acc_0);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[2], gr[3]), make_float2(xr[2], xr[3]), dx2_acc_0);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[4], gr[5]), make_float2(xr[4], xr[5]), dx2_acc_0);
            dx2_acc_0 = fma_f32x2_rn_noftz(make_float2(gr[6], gr[7]), make_float2(xr[6], xr[7]), dx2_acc_0);
            dot = dx2_acc_0.x + dx2_acc_0.y;
            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot = dot + _shfl_xor_sync_67;
            float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot = dot + _shfl_xor_sync_68;
            float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot = dot + _shfl_xor_sync_69;
            float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot = dot + _shfl_xor_sync_70;
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
    int trips = 0;
    int ahead = 0;
    if (rest > 0) {
        int _min_71 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_71) > (1) ? (_min_71) : (1));
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
    int _min_72 = ((num_bids) < (rest) ? (num_bids) : (rest));
    int stride = ((_min_72) > (1) ? (_min_72) : (1));
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
            #pragma unroll 1
            for (int i_1 = 0; i_1 < chunk_n; i_1++) {
                int u_1 = bound + chunk_lo + i_1 * stride;
                int row_1 = u_1 * 16 + group;
                int src_1 = ((row_1) < (rows - 1) ? (row_1) : (rows - 1));
                uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
                uint32_t th_1 = (uint32_t)((unsigned int)src_1) - tq_1 * (uint32_t)(heads.divisor);
                long long goff_1 = (long long)tq_1 * dy_tok + (long long)th_1 * dy_head + (long long)col;
                long long xoff_1 = (long long)tq_1 * x_tok + (long long)th_1 * x_head + (long long)col;
                long long zoff_1 = (long long)tq_1 * z_tok + (long long)th_1 * z_head + (long long)col;
                float gr_1[8];
                float xr_1[8];
                float zr_1[8];
                {
                    const uint4* _vptr_73 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                    uint4 _vld_73[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_73[_blk] = _vptr_73[_blk];
                        uint32_t* _vpairs_73 = reinterpret_cast<uint32_t*>(&_vld_73[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) << 16);
                            (&gr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_74 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                    uint4 _vld_74[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_74[_blk] = _vptr_74[_blk];
                        uint32_t* _vpairs_74 = reinterpret_cast<uint32_t*>(&_vld_74[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) << 16);
                            (&xr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_75 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                    uint4 _vld_75[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_75[_blk] = _vptr_75[_blk];
                        uint32_t* _vpairs_75 = reinterpret_cast<uint32_t*>(&_vld_75[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) << 16);
                            (&zr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (ahead != 0) {
                    if (i_1 + 1 < chunk_n) {
                        if (packed != 0) {
                            if (tid == 0) {
                                int nfirst = (u_1 + stride) * 16;
                                int _min_76 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                                int nbytes = _min_76 * 256;
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                            }
                        } else {
                            int nrow = (((u_1 + stride) * 16 + group) < (rows - 1) ? ((u_1 + stride) * 16 + group) : (rows - 1));
                            uint32_t nq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow);
                            uint32_t nh = (uint32_t)((unsigned int)nrow) - nq * (uint32_t)(heads.divisor);
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq * dy_tok + (long long)nh * dy_head + (long long)col))));
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq * x_tok + (long long)nh * x_head + (long long)col))));
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq * z_tok + (long long)nh * z_head + (long long)col))));
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        if (i_1 + 1 == chunk_n) {
                            uint32_t _atomic_fetch_inc_77;
                            asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                                : "=r"(_atomic_fetch_inc_77) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                            flag[1] = (int)_atomic_fetch_inc_77;
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
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_78;
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_81;
                float rstd_1;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
                float dot_1 = 0.0f;
                int fast_1 = 1;
                float wk_1[2];
                unsigned int packed_dz_1[4];
                if (fast_1 != 0) {
                    float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                    float2 z2_4 = make_float2(zr_1[0], zr_1[1]);
                    float2 _f32x2_mul_b_82 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
                    float _fma_83;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(zr_1[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_84;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(zr_1[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_4 = make_float2(_fma_83, _fma_84);
                    float _exp2_85 = approx_exp2(ep2_4.x);
                    float _exp2_86 = approx_exp2(ep2_4.y);
                    float2 dq2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_85, _exp2_86), qr2_4, make_float2(0.25f, 0.25f));
                    float _rcp_87 = approx_rcp(dq2_4.x);
                    float _rcp_88 = approx_rcp(dq2_4.y);
                    float2 _f32x2_mul_a_89 = make_float2(_rcp_87, _rcp_88);
                    float2 _f32x2_mul_b_90 = make_float2(0.25f, 0.25f);
                    float2 sig2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
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
                    float2 _f32x2_mul_a_91 = make_float2(xr_1[0], xr_1[1]);
                    float2 n2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_92 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                    float2 _f32x2_mul_b_93 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                    float2 dz2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
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
                    float2 _f32x2_mul_b_94 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
                    float _fma_95;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(zr_1[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_96;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(zr_1[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_5 = make_float2(_fma_95, _fma_96);
                    float _exp2_97 = approx_exp2(ep2_5.x);
                    float _exp2_98 = approx_exp2(ep2_5.y);
                    float2 dq2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_97, _exp2_98), qr2_5, make_float2(0.25f, 0.25f));
                    float _rcp_99 = approx_rcp(dq2_5.x);
                    float _rcp_100 = approx_rcp(dq2_5.y);
                    float2 _f32x2_mul_a_101 = make_float2(_rcp_99, _rcp_100);
                    float2 _f32x2_mul_b_102 = make_float2(0.25f, 0.25f);
                    float2 sig2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
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
                    float2 _f32x2_mul_a_103 = make_float2(xr_1[2], xr_1[3]);
                    float2 n2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_104 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                    float2 _f32x2_mul_b_105 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                    float2 dz2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
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
                    float2 _f32x2_mul_b_106 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
                    float _fma_107;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(zr_1[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_108;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(zr_1[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_6 = make_float2(_fma_107, _fma_108);
                    float _exp2_109 = approx_exp2(ep2_6.x);
                    float _exp2_110 = approx_exp2(ep2_6.y);
                    float2 dq2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_109, _exp2_110), qr2_6, make_float2(0.25f, 0.25f));
                    float _rcp_111 = approx_rcp(dq2_6.x);
                    float _rcp_112 = approx_rcp(dq2_6.y);
                    float2 _f32x2_mul_a_113 = make_float2(_rcp_111, _rcp_112);
                    float2 _f32x2_mul_b_114 = make_float2(0.25f, 0.25f);
                    float2 sig2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
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
                    float2 _f32x2_mul_a_115 = make_float2(xr_1[4], xr_1[5]);
                    float2 n2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_116 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                    float2 _f32x2_mul_b_117 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                    float2 dz2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
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
                    float2 _f32x2_mul_b_118 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
                    float _fma_119;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(zr_1[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_120;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(zr_1[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_7 = make_float2(_fma_119, _fma_120);
                    float _exp2_121 = approx_exp2(ep2_7.x);
                    float _exp2_122 = approx_exp2(ep2_7.y);
                    float2 dq2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_121, _exp2_122), qr2_7, make_float2(0.25f, 0.25f));
                    float _rcp_123 = approx_rcp(dq2_7.x);
                    float _rcp_124 = approx_rcp(dq2_7.y);
                    float2 _f32x2_mul_a_125 = make_float2(_rcp_123, _rcp_124);
                    float2 _f32x2_mul_b_126 = make_float2(0.25f, 0.25f);
                    float2 sig2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_126));
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
                    float2 _f32x2_mul_a_127 = make_float2(xr_1[6], xr_1[7]);
                    float2 n2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_128 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                    float2 _f32x2_mul_b_129 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                    float2 dz2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
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
                } else {
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
                    float _exp_130 = __nv_expf(zr_1[0] * -1.0f);
                    float den_8 = 1.0f + _exp_130;
                    float sig_8 = __fdiv_rn(1.0f, den_8);
                    float gate_8 = __fdiv_rn(zr_1[0], den_8);
                    float nj_8 = xr_1[0] * rstd_1;
                    float aj_8 = gr_1[0] * gate_8;
                    float dzj_8 = gr_1[0] * (nj_8 * wk_1[0]) * sig_8 * __fmaf_rn(zr_1[0], 1.0f - sig_8, 1.0f);
                    float dnj_8 = aj_8 * wk_1[0];
                    zr_1[0] = dzj_8;
                    gr_1[0] = dnj_8;
                    float _exp_131 = __nv_expf(zr_1[1] * -1.0f);
                    float den_9 = 1.0f + _exp_131;
                    float sig_9 = __fdiv_rn(1.0f, den_9);
                    float gate_9 = __fdiv_rn(zr_1[1], den_9);
                    float nj_9 = xr_1[1] * rstd_1;
                    float aj_9 = gr_1[1] * gate_9;
                    float dzj_9 = gr_1[1] * (nj_9 * wk_1[1]) * sig_9 * __fmaf_rn(zr_1[1], 1.0f - sig_9, 1.0f);
                    float dnj_9 = aj_9 * wk_1[1];
                    zr_1[1] = dzj_9;
                    gr_1[1] = dnj_9;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 0], zr_1[_lp*2+1 + 0]));
                        packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                    }
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
                    float _exp_132 = __nv_expf(zr_1[2] * -1.0f);
                    float den_10 = 1.0f + _exp_132;
                    float sig_10 = __fdiv_rn(1.0f, den_10);
                    float gate_10 = __fdiv_rn(zr_1[2], den_10);
                    float nj_10 = xr_1[2] * rstd_1;
                    float aj_10 = gr_1[2] * gate_10;
                    float dzj_10 = gr_1[2] * (nj_10 * wk_1[0]) * sig_10 * __fmaf_rn(zr_1[2], 1.0f - sig_10, 1.0f);
                    float dnj_10 = aj_10 * wk_1[0];
                    zr_1[2] = dzj_10;
                    gr_1[2] = dnj_10;
                    float _exp_133 = __nv_expf(zr_1[3] * -1.0f);
                    float den_11 = 1.0f + _exp_133;
                    float sig_11 = __fdiv_rn(1.0f, den_11);
                    float gate_11 = __fdiv_rn(zr_1[3], den_11);
                    float nj_11 = xr_1[3] * rstd_1;
                    float aj_11 = gr_1[3] * gate_11;
                    float dzj_11 = gr_1[3] * (nj_11 * wk_1[1]) * sig_11 * __fmaf_rn(zr_1[3], 1.0f - sig_11, 1.0f);
                    float dnj_11 = aj_11 * wk_1[1];
                    zr_1[3] = dzj_11;
                    gr_1[3] = dnj_11;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 2], zr_1[_lp*2+1 + 2]));
                        packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
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
                    float _exp_134 = __nv_expf(zr_1[4] * -1.0f);
                    float den_12 = 1.0f + _exp_134;
                    float sig_12 = __fdiv_rn(1.0f, den_12);
                    float gate_12 = __fdiv_rn(zr_1[4], den_12);
                    float nj_12 = xr_1[4] * rstd_1;
                    float aj_12 = gr_1[4] * gate_12;
                    float dzj_12 = gr_1[4] * (nj_12 * wk_1[0]) * sig_12 * __fmaf_rn(zr_1[4], 1.0f - sig_12, 1.0f);
                    float dnj_12 = aj_12 * wk_1[0];
                    zr_1[4] = dzj_12;
                    gr_1[4] = dnj_12;
                    float _exp_135 = __nv_expf(zr_1[5] * -1.0f);
                    float den_13 = 1.0f + _exp_135;
                    float sig_13 = __fdiv_rn(1.0f, den_13);
                    float gate_13 = __fdiv_rn(zr_1[5], den_13);
                    float nj_13 = xr_1[5] * rstd_1;
                    float aj_13 = gr_1[5] * gate_13;
                    float dzj_13 = gr_1[5] * (nj_13 * wk_1[1]) * sig_13 * __fmaf_rn(zr_1[5], 1.0f - sig_13, 1.0f);
                    float dnj_13 = aj_13 * wk_1[1];
                    zr_1[5] = dzj_13;
                    gr_1[5] = dnj_13;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 4], zr_1[_lp*2+1 + 4]));
                        packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
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
                    float _exp_136 = __nv_expf(zr_1[6] * -1.0f);
                    float den_14 = 1.0f + _exp_136;
                    float sig_14 = __fdiv_rn(1.0f, den_14);
                    float gate_14 = __fdiv_rn(zr_1[6], den_14);
                    float nj_14 = xr_1[6] * rstd_1;
                    float aj_14 = gr_1[6] * gate_14;
                    float dzj_14 = gr_1[6] * (nj_14 * wk_1[0]) * sig_14 * __fmaf_rn(zr_1[6], 1.0f - sig_14, 1.0f);
                    float dnj_14 = aj_14 * wk_1[0];
                    zr_1[6] = dzj_14;
                    gr_1[6] = dnj_14;
                    float _exp_137 = __nv_expf(zr_1[7] * -1.0f);
                    float den_15 = 1.0f + _exp_137;
                    float sig_15 = __fdiv_rn(1.0f, den_15);
                    float gate_15 = __fdiv_rn(zr_1[7], den_15);
                    float nj_15 = xr_1[7] * rstd_1;
                    float aj_15 = gr_1[7] * gate_15;
                    float dzj_15 = gr_1[7] * (nj_15 * wk_1[1]) * sig_15 * __fmaf_rn(zr_1[7], 1.0f - sig_15, 1.0f);
                    float dnj_15 = aj_15 * wk_1[1];
                    zr_1[7] = dzj_15;
                    gr_1[7] = dnj_15;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 6], zr_1[_lp*2+1 + 6]));
                        packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                }
                float2 dx2_acc_1_1 = make_float2(0.0f, 0.0f);
                dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[0], gr_1[1]), make_float2(xr_1[0], xr_1[1]), dx2_acc_1_1);
                dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[2], gr_1[3]), make_float2(xr_1[2], xr_1[3]), dx2_acc_1_1);
                dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[4], gr_1[5]), make_float2(xr_1[4], xr_1[5]), dx2_acc_1_1);
                dx2_acc_1_1 = fma_f32x2_rn_noftz(make_float2(gr_1[6], gr_1[7]), make_float2(xr_1[6], xr_1[7]), dx2_acc_1_1);
                dot_1 = dx2_acc_1_1.x + dx2_acc_1_1.y;
                float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_138;
                float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_139;
                float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_140;
                float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_141;
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
                if (row_1 < rows) {
                    long long orow_1 = (long long)row_1 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
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
