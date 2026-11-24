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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_fef9d1cc2cc267219992(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ acc, unsigned int* __restrict__ flags, unsigned int* __restrict__ done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(acc));
    __builtin_assume(__isGlobal(acc));
    asm volatile("" : "+l"(flags));
    __builtin_assume(__isGlobal(flags));

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
    unsigned long long* exa = reinterpret_cast<unsigned long long*>(smem_raw + 0);
    const int exa_addr = smem + 0;
    unsigned int* exf = reinterpret_cast<unsigned int*>(smem_raw + 9216);
    const int exf_addr = smem + 9216;
    int* cell = reinterpret_cast<int*>(smem_raw + 9728);
    const int cell_addr = smem + 9728;
    int* flag = reinterpret_cast<int*>(smem_raw + 9744);
    const int flag_addr = smem + 9744;
    int* tab = reinterpret_cast<int*>(smem_raw + 9760);
    const int tab_addr = smem + 9760;
    float* fold = reinterpret_cast<float*>(smem_raw + 9888);
    const int fold_addr = smem + 9888;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    if (tid < 128) {
        exa[tid] = 0ull;
        exa[128 + tid] = 0ull;
        exa[256 + tid] = 0ull;
        exa[384 + tid] = 0ull;
        exa[512 + tid] = 0ull;
        exa[640 + tid] = 0ull;
        exa[768 + tid] = 0ull;
        exa[896 + tid] = 0ull;
        exa[1024 + tid] = 0ull;
        exf[tid] = 0;
    }
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
    float acc_r[8];
    acc_r[0] = 0.0f;
    acc_r[1] = 0.0f;
    acc_r[2] = 0.0f;
    acc_r[3] = 0.0f;
    acc_r[4] = 0.0f;
    acc_r[5] = 0.0f;
    acc_r[6] = 0.0f;
    acc_r[7] = 0.0f;
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
            float2 acc2_0 = fma_f32x2_rn_noftz(a2_0, n2_0, make_float2(acc_r[0], acc_r[1]));
            acc_r[0] = ((row < rows) ? acc2_0.x : acc_r[0]);
            acc_r[1] = ((row < rows) ? acc2_0.y : acc_r[1]);
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
            float2 acc2_1 = fma_f32x2_rn_noftz(a2_1, n2_1, make_float2(acc_r[2], acc_r[3]));
            acc_r[2] = ((row < rows) ? acc2_1.x : acc_r[2]);
            acc_r[3] = ((row < rows) ? acc2_1.y : acc_r[3]);
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
            float2 acc2_2 = fma_f32x2_rn_noftz(a2_2, n2_2, make_float2(acc_r[4], acc_r[5]));
            acc_r[4] = ((row < rows) ? acc2_2.x : acc_r[4]);
            acc_r[5] = ((row < rows) ? acc2_2.y : acc_r[5]);
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
            float2 acc2_3 = fma_f32x2_rn_noftz(a2_3, n2_3, make_float2(acc_r[6], acc_r[7]));
            acc_r[6] = ((row < rows) ? acc2_3.x : acc_r[6]);
            acc_r[7] = ((row < rows) ? acc2_3.y : acc_r[7]);
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
            if ((tail_i & 15) == 15) {
                if (tail_i >= 16) {
                    float _add_63;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    fold[group * 128 + col] = _add_63;
                    acc_r[0] = 0.0f;
                    float _add_64;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    fold[group * 128 + col + 1] = _add_64;
                    acc_r[1] = 0.0f;
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    fold[group * 128 + col + 2] = _add_65;
                    acc_r[2] = 0.0f;
                    float _add_66;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    fold[group * 128 + col + 3] = _add_66;
                    acc_r[3] = 0.0f;
                    float _add_67;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    fold[group * 128 + col + 4] = _add_67;
                    acc_r[4] = 0.0f;
                    float _add_68;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    fold[group * 128 + col + 5] = _add_68;
                    acc_r[5] = 0.0f;
                    float _add_69;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    fold[group * 128 + col + 6] = _add_69;
                    acc_r[6] = 0.0f;
                    float _add_70;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    fold[group * 128 + col + 7] = _add_70;
                    acc_r[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = acc_r[0];
                    acc_r[0] = 0.0f;
                    fold[group * 128 + col + 1] = acc_r[1];
                    acc_r[1] = 0.0f;
                    fold[group * 128 + col + 2] = acc_r[2];
                    acc_r[2] = 0.0f;
                    fold[group * 128 + col + 3] = acc_r[3];
                    acc_r[3] = 0.0f;
                    fold[group * 128 + col + 4] = acc_r[4];
                    acc_r[4] = 0.0f;
                    fold[group * 128 + col + 5] = acc_r[5];
                    acc_r[5] = 0.0f;
                    fold[group * 128 + col + 6] = acc_r[6];
                    acc_r[6] = 0.0f;
                    fold[group * 128 + col + 7] = acc_r[7];
                    acc_r[7] = 0.0f;
                }
            }
            {
                fold[group * 128 + col] = acc_r[0];
                acc_r[0] = 0.0f;
                fold[group * 128 + col + 1] = acc_r[1];
                acc_r[1] = 0.0f;
                fold[group * 128 + col + 2] = acc_r[2];
                acc_r[2] = 0.0f;
                fold[group * 128 + col + 3] = acc_r[3];
                acc_r[3] = 0.0f;
                fold[group * 128 + col + 4] = acc_r[4];
                acc_r[4] = 0.0f;
                fold[group * 128 + col + 5] = acc_r[5];
                acc_r[5] = 0.0f;
                fold[group * 128 + col + 6] = acc_r[6];
                acc_r[6] = 0.0f;
                fold[group * 128 + col + 7] = acc_r[7];
                acc_r[7] = 0.0f;
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
                const uint4* _vptr_71 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                uint4 _vld_71[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_71[_blk] = _vptr_71[_blk];
                    uint32_t* _vpairs_71 = reinterpret_cast<uint32_t*>(&_vld_71[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) << 16);
                        (&gr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_72 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                uint4 _vld_72[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_72[_blk] = _vptr_72[_blk];
                    uint32_t* _vpairs_72 = reinterpret_cast<uint32_t*>(&_vld_72[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) << 16);
                        (&xr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_73 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                uint4 _vld_73[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_73[_blk] = _vptr_73[_blk];
                    uint32_t* _vpairs_73 = reinterpret_cast<uint32_t*>(&_vld_73[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) << 16);
                        (&zr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) & 0xffff0000u);
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
            float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
            total_1 = total_1 + _shfl_xor_sync_74;
            float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
            total_1 = total_1 + _shfl_xor_sync_75;
            float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
            total_1 = total_1 + _shfl_xor_sync_76;
            float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
            total_1 = total_1 + _shfl_xor_sync_77;
            float rstd_1;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
            float dot_1 = 0.0f;
            float wk_1[2];
            unsigned int packed_dz_1[4];
            float2 rstd2_1 = make_float2(rstd_1, rstd_1);
            float2 z2_4 = make_float2(zr_1[0], zr_1[1]);
            float2 _f32x2_mul_b_78 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
            float _fma_79;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(zr_1[0]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_80;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(zr_1[1]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_4 = make_float2(_fma_79, _fma_80);
            float _exp2_81 = approx_exp2(ep2_4.x);
            float _exp2_82 = approx_exp2(ep2_4.y);
            float2 dq2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_81, _exp2_82), qr2_4, make_float2(0.25f, 0.25f));
            float _rcp_83 = approx_rcp(dq2_4.x);
            float _rcp_84 = approx_rcp(dq2_4.y);
            float2 _f32x2_mul_a_85 = make_float2(_rcp_83, _rcp_84);
            float2 _f32x2_mul_b_86 = make_float2(0.25f, 0.25f);
            float2 sig2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
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
            float2 _f32x2_mul_a_87 = make_float2(xr_1[0], xr_1[1]);
            float2 n2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_88 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
            float2 _f32x2_mul_b_89 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
            float2 dz2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
            float2 a2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
            float2 dn2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
            float2 acc2_4 = fma_f32x2_rn_noftz(a2_4, n2_4, make_float2(acc_r[0], acc_r[1]));
            acc_r[0] = ((row_1 < rows) ? acc2_4.x : acc_r[0]);
            acc_r[1] = ((row_1 < rows) ? acc2_4.y : acc_r[1]);
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
            float2 _f32x2_mul_b_90 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
            float _fma_91;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(zr_1[2]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_92;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(zr_1[3]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_5 = make_float2(_fma_91, _fma_92);
            float _exp2_93 = approx_exp2(ep2_5.x);
            float _exp2_94 = approx_exp2(ep2_5.y);
            float2 dq2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_93, _exp2_94), qr2_5, make_float2(0.25f, 0.25f));
            float _rcp_95 = approx_rcp(dq2_5.x);
            float _rcp_96 = approx_rcp(dq2_5.y);
            float2 _f32x2_mul_a_97 = make_float2(_rcp_95, _rcp_96);
            float2 _f32x2_mul_b_98 = make_float2(0.25f, 0.25f);
            float2 sig2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
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
            float2 _f32x2_mul_a_99 = make_float2(xr_1[2], xr_1[3]);
            float2 n2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_100 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
            float2 _f32x2_mul_b_101 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
            float2 dz2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
            float2 a2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
            float2 dn2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
            float2 acc2_5 = fma_f32x2_rn_noftz(a2_5, n2_5, make_float2(acc_r[2], acc_r[3]));
            acc_r[2] = ((row_1 < rows) ? acc2_5.x : acc_r[2]);
            acc_r[3] = ((row_1 < rows) ? acc2_5.y : acc_r[3]);
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
            float2 _f32x2_mul_b_102 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
            float _fma_103;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(zr_1[4]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_104;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(zr_1[5]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_6 = make_float2(_fma_103, _fma_104);
            float _exp2_105 = approx_exp2(ep2_6.x);
            float _exp2_106 = approx_exp2(ep2_6.y);
            float2 dq2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_105, _exp2_106), qr2_6, make_float2(0.25f, 0.25f));
            float _rcp_107 = approx_rcp(dq2_6.x);
            float _rcp_108 = approx_rcp(dq2_6.y);
            float2 _f32x2_mul_a_109 = make_float2(_rcp_107, _rcp_108);
            float2 _f32x2_mul_b_110 = make_float2(0.25f, 0.25f);
            float2 sig2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
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
            float2 _f32x2_mul_a_111 = make_float2(xr_1[4], xr_1[5]);
            float2 n2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_112 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
            float2 _f32x2_mul_b_113 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
            float2 dz2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_113));
            float2 a2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
            float2 dn2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
            float2 acc2_6 = fma_f32x2_rn_noftz(a2_6, n2_6, make_float2(acc_r[4], acc_r[5]));
            acc_r[4] = ((row_1 < rows) ? acc2_6.x : acc_r[4]);
            acc_r[5] = ((row_1 < rows) ? acc2_6.y : acc_r[5]);
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
            float2 _f32x2_mul_b_114 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
            float _fma_115;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(zr_1[6]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_116;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(zr_1[7]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_7 = make_float2(_fma_115, _fma_116);
            float _exp2_117 = approx_exp2(ep2_7.x);
            float _exp2_118 = approx_exp2(ep2_7.y);
            float2 dq2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_117, _exp2_118), qr2_7, make_float2(0.25f, 0.25f));
            float _rcp_119 = approx_rcp(dq2_7.x);
            float _rcp_120 = approx_rcp(dq2_7.y);
            float2 _f32x2_mul_a_121 = make_float2(_rcp_119, _rcp_120);
            float2 _f32x2_mul_b_122 = make_float2(0.25f, 0.25f);
            float2 sig2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
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
            float2 _f32x2_mul_a_123 = make_float2(xr_1[6], xr_1[7]);
            float2 n2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_124 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
            float2 _f32x2_mul_b_125 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
            float2 dz2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_125));
            float2 a2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
            float2 dn2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
            float2 acc2_7 = fma_f32x2_rn_noftz(a2_7, n2_7, make_float2(acc_r[6], acc_r[7]));
            acc_r[6] = ((row_1 < rows) ? acc2_7.x : acc_r[6]);
            acc_r[7] = ((row_1 < rows) ? acc2_7.y : acc_r[7]);
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
            float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
            dot_1 = dot_1 + _shfl_xor_sync_126;
            float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
            dot_1 = dot_1 + _shfl_xor_sync_127;
            float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
            dot_1 = dot_1 + _shfl_xor_sync_128;
            float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
            dot_1 = dot_1 + _shfl_xor_sync_129;
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
            if ((tail_i_1 & 15) == 15) {
                if (tail_i_1 >= 16) {
                    float _add_130;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    fold[group * 128 + col] = _add_130;
                    acc_r[0] = 0.0f;
                    float _add_131;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    fold[group * 128 + col + 1] = _add_131;
                    acc_r[1] = 0.0f;
                    float _add_132;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    fold[group * 128 + col + 2] = _add_132;
                    acc_r[2] = 0.0f;
                    float _add_133;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    fold[group * 128 + col + 3] = _add_133;
                    acc_r[3] = 0.0f;
                    float _add_134;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    fold[group * 128 + col + 4] = _add_134;
                    acc_r[4] = 0.0f;
                    float _add_135;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    fold[group * 128 + col + 5] = _add_135;
                    acc_r[5] = 0.0f;
                    float _add_136;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    fold[group * 128 + col + 6] = _add_136;
                    acc_r[6] = 0.0f;
                    float _add_137;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    fold[group * 128 + col + 7] = _add_137;
                    acc_r[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = acc_r[0];
                    acc_r[0] = 0.0f;
                    fold[group * 128 + col + 1] = acc_r[1];
                    acc_r[1] = 0.0f;
                    fold[group * 128 + col + 2] = acc_r[2];
                    acc_r[2] = 0.0f;
                    fold[group * 128 + col + 3] = acc_r[3];
                    acc_r[3] = 0.0f;
                    fold[group * 128 + col + 4] = acc_r[4];
                    acc_r[4] = 0.0f;
                    fold[group * 128 + col + 5] = acc_r[5];
                    acc_r[5] = 0.0f;
                    fold[group * 128 + col + 6] = acc_r[6];
                    acc_r[6] = 0.0f;
                    fold[group * 128 + col + 7] = acc_r[7];
                    acc_r[7] = 0.0f;
                }
            }
            {
                fold[group * 128 + col] = acc_r[0];
                acc_r[0] = 0.0f;
                fold[group * 128 + col + 1] = acc_r[1];
                acc_r[1] = 0.0f;
                fold[group * 128 + col + 2] = acc_r[2];
                acc_r[2] = 0.0f;
                fold[group * 128 + col + 3] = acc_r[3];
                acc_r[3] = 0.0f;
                fold[group * 128 + col + 4] = acc_r[4];
                acc_r[4] = 0.0f;
                fold[group * 128 + col + 5] = acc_r[5];
                acc_r[5] = 0.0f;
                fold[group * 128 + col + 6] = acc_r[6];
                acc_r[6] = 0.0f;
                fold[group * 128 + col + 7] = acc_r[7];
                acc_r[7] = 0.0f;
            }
        }
        __syncthreads();
        if (tid < 128) {
            float hs = fold[tid];
            #pragma unroll
            for (int g = 1; g < 16; g++) {
                float _add_138;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(hs), "f"(fold[g * 128 + tid]));
                hs = _add_138;
            }
            unsigned int bits = __as_u32(hs);
            unsigned int ex = bits >> 23 & 255;
            unsigned int man = bits & 8388607;
            if (ex == 255) {
                unsigned int fbit = 4;
                if (man == 0) {
                    fbit = 2;
                    if (bits >> 31 == 0) {
                        fbit = 1;
                    }
                }
                exf[tid] = exf[tid] | fbit;
            } else {
                if (ex != 0) {
                    man = man | 8388608;
                }
                if (man != 0) {
                    unsigned int _max_139 = ((ex) > (1) ? (ex) : (1));
                    unsigned int pos = _max_139 - 1;
                    unsigned long long wide = (unsigned long long)man << (unsigned long long)(pos & 31);
                    unsigned long long d0 = wide & 4294967295ull;
                    unsigned long long d1 = wide >> 32ull;
                    if (bits >> 31 != 0) {
                        d0 = -d0;
                        d1 = -d1;
                    }
                    int at = (int)(pos >> 5) * 128 + tid;
                    exa[at] = exa[at] + d0;
                    if (d1 != 0ull) {
                        exa[at + 128] = exa[at + 128] + d1;
                    }
                }
            }
        }
    }
    int trips = 0;
    int ahead = 0;
    if (rest > 0) {
        int _min_140 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_140) > (1) ? (_min_140) : (1));
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
    int _min_141 = ((num_bids) < (rest) ? (num_bids) : (rest));
    int stride = ((_min_141) > (1) ? (_min_141) : (1));
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
                        uint4 _uv4_142 = *reinterpret_cast<const uint4*>(dy + voff + 0);
                        la_g[0 + 0] = _uv4_142.x;
                        la_g[0 + 1] = _uv4_142.y;
                        la_g[0 + 2] = _uv4_142.z;
                        la_g[0 + 3] = _uv4_142.w;
                    }
                    {
                        uint4 _uv4_143 = *reinterpret_cast<const uint4*>(x + voff + 0);
                        la_x[0 + 0] = _uv4_143.x;
                        la_x[0 + 1] = _uv4_143.y;
                        la_x[0 + 2] = _uv4_143.z;
                        la_x[0 + 3] = _uv4_143.w;
                    }
                    {
                        uint4 _uv4_144 = *reinterpret_cast<const uint4*>(z + voff + 0);
                        la_z[0 + 0] = _uv4_144.x;
                        la_z[0 + 1] = _uv4_144.y;
                        la_z[0 + 2] = _uv4_144.z;
                        la_z[0 + 3] = _uv4_144.w;
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
                            uint4 _uv4_145 = *reinterpret_cast<const uint4*>(dy + voff_1 + 0);
                            la_g[0 + 0] = _uv4_145.x;
                            la_g[0 + 1] = _uv4_145.y;
                            la_g[0 + 2] = _uv4_145.z;
                            la_g[0 + 3] = _uv4_145.w;
                        }
                        {
                            uint4 _uv4_146 = *reinterpret_cast<const uint4*>(x + voff_1 + 0);
                            la_x[0 + 0] = _uv4_146.x;
                            la_x[0 + 1] = _uv4_146.y;
                            la_x[0 + 2] = _uv4_146.z;
                            la_x[0 + 3] = _uv4_146.w;
                        }
                        {
                            uint4 _uv4_147 = *reinterpret_cast<const uint4*>(z + voff_1 + 0);
                            la_z[0 + 0] = _uv4_147.x;
                            la_z[0 + 1] = _uv4_147.y;
                            la_z[0 + 2] = _uv4_147.z;
                            la_z[0 + 3] = _uv4_147.w;
                        }
                    }
                    if (ahead != 0) {
                        if (tid == 0) {
                            int nfirst = (u_2 + stride) * 16;
                            int _min_148 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                            int nbytes = _min_148 * 256;
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
                    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, total_2, 1);
                    total_2 = total_2 + _shfl_xor_sync_149;
                    float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, total_2, 2);
                    total_2 = total_2 + _shfl_xor_sync_150;
                    float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, total_2, 4);
                    total_2 = total_2 + _shfl_xor_sync_151;
                    float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, total_2, 8);
                    total_2 = total_2 + _shfl_xor_sync_152;
                    float rstd_2;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_2) : "f"(total_2 * 0.0078125f + eps));
                    float dot_2 = 0.0f;
                    float wk_2[2];
                    unsigned int packed_dz_2[4];
                    float2 rstd2_2 = make_float2(rstd_2, rstd_2);
                    float2 z2_8 = make_float2(zr_2[0], zr_2[1]);
                    float2 _f32x2_mul_b_153 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
                    float _fma_154;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(zr_2[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_155;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(zr_2[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_8 = make_float2(_fma_154, _fma_155);
                    float _exp2_156 = approx_exp2(ep2_8.x);
                    float _exp2_157 = approx_exp2(ep2_8.y);
                    float2 dq2_8 = fma_f32x2_rn_noftz(make_float2(_exp2_156, _exp2_157), qr2_8, make_float2(0.25f, 0.25f));
                    float _rcp_158 = approx_rcp(dq2_8.x);
                    float _rcp_159 = approx_rcp(dq2_8.y);
                    float2 _f32x2_mul_a_160 = make_float2(_rcp_158, _rcp_159);
                    float2 _f32x2_mul_b_161 = make_float2(0.25f, 0.25f);
                    float2 sig2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
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
                    float2 _f32x2_mul_a_162 = make_float2(xr_2[0], xr_2[1]);
                    float2 n2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_163 = mul_f32x2_noftz(mul_f32x2_noftz(g2_8, mul_f32x2_noftz(n2_8, w2_8)), sig2_8);
                    float2 _f32x2_mul_b_164 = fma_f32x2_rn_noftz(z2_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_8), make_float2(1.0f, 1.0f));
                    float2 dz2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&_f32x2_mul_b_164));
                    float2 a2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_8) : "l"(*(const unsigned long long*)&g2_8), "l"(*(const unsigned long long*)&gate2_8));
                    float2 dn2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_8) : "l"(*(const unsigned long long*)&a2_8), "l"(*(const unsigned long long*)&w2_8));
                    float2 acc2_8 = fma_f32x2_rn_noftz(a2_8, n2_8, make_float2(acc_r[0], acc_r[1]));
                    acc_r[0] = acc2_8.x;
                    acc_r[1] = acc2_8.y;
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
                    float2 _f32x2_mul_b_165 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
                    float _fma_166;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(zr_2[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_167;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(zr_2[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_9 = make_float2(_fma_166, _fma_167);
                    float _exp2_168 = approx_exp2(ep2_9.x);
                    float _exp2_169 = approx_exp2(ep2_9.y);
                    float2 dq2_9 = fma_f32x2_rn_noftz(make_float2(_exp2_168, _exp2_169), qr2_9, make_float2(0.25f, 0.25f));
                    float _rcp_170 = approx_rcp(dq2_9.x);
                    float _rcp_171 = approx_rcp(dq2_9.y);
                    float2 _f32x2_mul_a_172 = make_float2(_rcp_170, _rcp_171);
                    float2 _f32x2_mul_b_173 = make_float2(0.25f, 0.25f);
                    float2 sig2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
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
                    float2 _f32x2_mul_a_174 = make_float2(xr_2[2], xr_2[3]);
                    float2 n2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_174), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_175 = mul_f32x2_noftz(mul_f32x2_noftz(g2_9, mul_f32x2_noftz(n2_9, w2_9)), sig2_9);
                    float2 _f32x2_mul_b_176 = fma_f32x2_rn_noftz(z2_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_9), make_float2(1.0f, 1.0f));
                    float2 dz2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&_f32x2_mul_b_176));
                    float2 a2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_9) : "l"(*(const unsigned long long*)&g2_9), "l"(*(const unsigned long long*)&gate2_9));
                    float2 dn2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_9) : "l"(*(const unsigned long long*)&a2_9), "l"(*(const unsigned long long*)&w2_9));
                    float2 acc2_9 = fma_f32x2_rn_noftz(a2_9, n2_9, make_float2(acc_r[2], acc_r[3]));
                    acc_r[2] = acc2_9.x;
                    acc_r[3] = acc2_9.y;
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
                    float2 _f32x2_mul_b_177 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
                    float _fma_178;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(zr_2[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_179;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(zr_2[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_10 = make_float2(_fma_178, _fma_179);
                    float _exp2_180 = approx_exp2(ep2_10.x);
                    float _exp2_181 = approx_exp2(ep2_10.y);
                    float2 dq2_10 = fma_f32x2_rn_noftz(make_float2(_exp2_180, _exp2_181), qr2_10, make_float2(0.25f, 0.25f));
                    float _rcp_182 = approx_rcp(dq2_10.x);
                    float _rcp_183 = approx_rcp(dq2_10.y);
                    float2 _f32x2_mul_a_184 = make_float2(_rcp_182, _rcp_183);
                    float2 _f32x2_mul_b_185 = make_float2(0.25f, 0.25f);
                    float2 sig2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&_f32x2_mul_b_185));
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
                    float2 _f32x2_mul_a_186 = make_float2(xr_2[4], xr_2[5]);
                    float2 n2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_187 = mul_f32x2_noftz(mul_f32x2_noftz(g2_10, mul_f32x2_noftz(n2_10, w2_10)), sig2_10);
                    float2 _f32x2_mul_b_188 = fma_f32x2_rn_noftz(z2_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_10), make_float2(1.0f, 1.0f));
                    float2 dz2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
                    float2 a2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_10) : "l"(*(const unsigned long long*)&g2_10), "l"(*(const unsigned long long*)&gate2_10));
                    float2 dn2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_10) : "l"(*(const unsigned long long*)&a2_10), "l"(*(const unsigned long long*)&w2_10));
                    float2 acc2_10 = fma_f32x2_rn_noftz(a2_10, n2_10, make_float2(acc_r[4], acc_r[5]));
                    acc_r[4] = acc2_10.x;
                    acc_r[5] = acc2_10.y;
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
                    float2 _f32x2_mul_b_189 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
                    float _fma_190;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(zr_2[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_191;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(zr_2[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_11 = make_float2(_fma_190, _fma_191);
                    float _exp2_192 = approx_exp2(ep2_11.x);
                    float _exp2_193 = approx_exp2(ep2_11.y);
                    float2 dq2_11 = fma_f32x2_rn_noftz(make_float2(_exp2_192, _exp2_193), qr2_11, make_float2(0.25f, 0.25f));
                    float _rcp_194 = approx_rcp(dq2_11.x);
                    float _rcp_195 = approx_rcp(dq2_11.y);
                    float2 _f32x2_mul_a_196 = make_float2(_rcp_194, _rcp_195);
                    float2 _f32x2_mul_b_197 = make_float2(0.25f, 0.25f);
                    float2 sig2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
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
                    float2 _f32x2_mul_a_198 = make_float2(xr_2[6], xr_2[7]);
                    float2 n2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_199 = mul_f32x2_noftz(mul_f32x2_noftz(g2_11, mul_f32x2_noftz(n2_11, w2_11)), sig2_11);
                    float2 _f32x2_mul_b_200 = fma_f32x2_rn_noftz(z2_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_11), make_float2(1.0f, 1.0f));
                    float2 dz2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_199), "l"(*(const unsigned long long*)&_f32x2_mul_b_200));
                    float2 a2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_11) : "l"(*(const unsigned long long*)&g2_11), "l"(*(const unsigned long long*)&gate2_11));
                    float2 dn2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_11) : "l"(*(const unsigned long long*)&a2_11), "l"(*(const unsigned long long*)&w2_11));
                    float2 acc2_11 = fma_f32x2_rn_noftz(a2_11, n2_11, make_float2(acc_r[6], acc_r[7]));
                    acc_r[6] = acc2_11.x;
                    acc_r[7] = acc2_11.y;
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
                    float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                    dot_2 = dot_2 + _shfl_xor_sync_201;
                    float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                    dot_2 = dot_2 + _shfl_xor_sync_202;
                    float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                    dot_2 = dot_2 + _shfl_xor_sync_203;
                    float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 8);
                    dot_2 = dot_2 + _shfl_xor_sync_204;
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
                    if ((i & 15) == 15) {
                        if (i >= 16) {
                            float _add_205;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                            fold[group * 128 + col] = _add_205;
                            acc_r[0] = 0.0f;
                            float _add_206;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                            fold[group * 128 + col + 1] = _add_206;
                            acc_r[1] = 0.0f;
                            float _add_207;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                            fold[group * 128 + col + 2] = _add_207;
                            acc_r[2] = 0.0f;
                            float _add_208;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                            fold[group * 128 + col + 3] = _add_208;
                            acc_r[3] = 0.0f;
                            float _add_209;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                            fold[group * 128 + col + 4] = _add_209;
                            acc_r[4] = 0.0f;
                            float _add_210;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_210) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                            fold[group * 128 + col + 5] = _add_210;
                            acc_r[5] = 0.0f;
                            float _add_211;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                            fold[group * 128 + col + 6] = _add_211;
                            acc_r[6] = 0.0f;
                            float _add_212;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                            fold[group * 128 + col + 7] = _add_212;
                            acc_r[7] = 0.0f;
                        } else {
                            fold[group * 128 + col] = acc_r[0];
                            acc_r[0] = 0.0f;
                            fold[group * 128 + col + 1] = acc_r[1];
                            acc_r[1] = 0.0f;
                            fold[group * 128 + col + 2] = acc_r[2];
                            acc_r[2] = 0.0f;
                            fold[group * 128 + col + 3] = acc_r[3];
                            acc_r[3] = 0.0f;
                            fold[group * 128 + col + 4] = acc_r[4];
                            acc_r[4] = 0.0f;
                            fold[group * 128 + col + 5] = acc_r[5];
                            acc_r[5] = 0.0f;
                            fold[group * 128 + col + 6] = acc_r[6];
                            acc_r[6] = 0.0f;
                            fold[group * 128 + col + 7] = acc_r[7];
                            acc_r[7] = 0.0f;
                        }
                    }
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
                    const uint4* _vptr_213 = reinterpret_cast<const uint4*>(dy + goff_3 + 0);
                    uint4 _vld_213[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_213[_blk] = _vptr_213[_blk];
                        uint32_t* _vpairs_213 = reinterpret_cast<uint32_t*>(&_vld_213[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_213[_pair]) << 16);
                            (&gr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_213[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_214 = reinterpret_cast<const uint4*>(x + xoff_2 + 0);
                    uint4 _vld_214[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_214[_blk] = _vptr_214[_blk];
                        uint32_t* _vpairs_214 = reinterpret_cast<uint32_t*>(&_vld_214[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_214[_pair]) << 16);
                            (&xr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_214[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_215 = reinterpret_cast<const uint4*>(z + zoff_2 + 0);
                    uint4 _vld_215[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_215[_blk] = _vptr_215[_blk];
                        uint32_t* _vpairs_215 = reinterpret_cast<uint32_t*>(&_vld_215[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_215[_pair]) << 16);
                            (&zr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_215[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        uint32_t _atomic_fetch_inc_216;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_216) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        flag[1] = (int)_atomic_fetch_inc_216;
                    }
                }
                float total_3 = 0.0f;
                float2 dx2_acc_3 = make_float2(0.0f, 0.0f);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[0], xr_3[1]), make_float2(xr_3[0], xr_3[1]), dx2_acc_3);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[2], xr_3[3]), make_float2(xr_3[2], xr_3[3]), dx2_acc_3);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[4], xr_3[5]), make_float2(xr_3[4], xr_3[5]), dx2_acc_3);
                dx2_acc_3 = fma_f32x2_rn_noftz(make_float2(xr_3[6], xr_3[7]), make_float2(xr_3[6], xr_3[7]), dx2_acc_3);
                total_3 = dx2_acc_3.x + dx2_acc_3.y;
                float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, total_3, 1);
                total_3 = total_3 + _shfl_xor_sync_217;
                float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, total_3, 2);
                total_3 = total_3 + _shfl_xor_sync_218;
                float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, total_3, 4);
                total_3 = total_3 + _shfl_xor_sync_219;
                float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, total_3, 8);
                total_3 = total_3 + _shfl_xor_sync_220;
                float rstd_3;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_3) : "f"(total_3 * 0.0078125f + eps));
                float dot_3 = 0.0f;
                float wk_3[2];
                unsigned int packed_dz_3[4];
                float2 rstd2_3 = make_float2(rstd_3, rstd_3);
                float2 z2_12 = make_float2(zr_3[0], zr_3[1]);
                float2 _f32x2_mul_b_221 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_221));
                float _fma_222;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(zr_3[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_223;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(zr_3[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_12 = make_float2(_fma_222, _fma_223);
                float _exp2_224 = approx_exp2(ep2_12.x);
                float _exp2_225 = approx_exp2(ep2_12.y);
                float2 dq2_12 = fma_f32x2_rn_noftz(make_float2(_exp2_224, _exp2_225), qr2_12, make_float2(0.25f, 0.25f));
                float _rcp_226 = approx_rcp(dq2_12.x);
                float _rcp_227 = approx_rcp(dq2_12.y);
                float2 _f32x2_mul_a_228 = make_float2(_rcp_226, _rcp_227);
                float2 _f32x2_mul_b_229 = make_float2(0.25f, 0.25f);
                float2 sig2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
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
                float2 _f32x2_mul_a_230 = make_float2(xr_3[0], xr_3[1]);
                float2 n2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_230), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_231 = mul_f32x2_noftz(mul_f32x2_noftz(g2_12, mul_f32x2_noftz(n2_12, w2_12)), sig2_12);
                float2 _f32x2_mul_b_232 = fma_f32x2_rn_noftz(z2_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_12), make_float2(1.0f, 1.0f));
                float2 dz2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&_f32x2_mul_b_232));
                float2 a2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_12) : "l"(*(const unsigned long long*)&g2_12), "l"(*(const unsigned long long*)&gate2_12));
                float2 dn2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_12) : "l"(*(const unsigned long long*)&a2_12), "l"(*(const unsigned long long*)&w2_12));
                float2 acc2_12 = fma_f32x2_rn_noftz(a2_12, n2_12, make_float2(acc_r[0], acc_r[1]));
                acc_r[0] = ((row_3 < rows) ? acc2_12.x : acc_r[0]);
                acc_r[1] = ((row_3 < rows) ? acc2_12.y : acc_r[1]);
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
                float2 _f32x2_mul_b_233 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_233));
                float _fma_234;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(zr_3[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_235;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(zr_3[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_13 = make_float2(_fma_234, _fma_235);
                float _exp2_236 = approx_exp2(ep2_13.x);
                float _exp2_237 = approx_exp2(ep2_13.y);
                float2 dq2_13 = fma_f32x2_rn_noftz(make_float2(_exp2_236, _exp2_237), qr2_13, make_float2(0.25f, 0.25f));
                float _rcp_238 = approx_rcp(dq2_13.x);
                float _rcp_239 = approx_rcp(dq2_13.y);
                float2 _f32x2_mul_a_240 = make_float2(_rcp_238, _rcp_239);
                float2 _f32x2_mul_b_241 = make_float2(0.25f, 0.25f);
                float2 sig2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_240), "l"(*(const unsigned long long*)&_f32x2_mul_b_241));
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
                float2 _f32x2_mul_a_242 = make_float2(xr_3[2], xr_3[3]);
                float2 n2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_242), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_243 = mul_f32x2_noftz(mul_f32x2_noftz(g2_13, mul_f32x2_noftz(n2_13, w2_13)), sig2_13);
                float2 _f32x2_mul_b_244 = fma_f32x2_rn_noftz(z2_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_13), make_float2(1.0f, 1.0f));
                float2 dz2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_243), "l"(*(const unsigned long long*)&_f32x2_mul_b_244));
                float2 a2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_13) : "l"(*(const unsigned long long*)&g2_13), "l"(*(const unsigned long long*)&gate2_13));
                float2 dn2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_13) : "l"(*(const unsigned long long*)&a2_13), "l"(*(const unsigned long long*)&w2_13));
                float2 acc2_13 = fma_f32x2_rn_noftz(a2_13, n2_13, make_float2(acc_r[2], acc_r[3]));
                acc_r[2] = ((row_3 < rows) ? acc2_13.x : acc_r[2]);
                acc_r[3] = ((row_3 < rows) ? acc2_13.y : acc_r[3]);
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
                float2 _f32x2_mul_b_245 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_245));
                float _fma_246;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(zr_3[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_247;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(zr_3[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_14 = make_float2(_fma_246, _fma_247);
                float _exp2_248 = approx_exp2(ep2_14.x);
                float _exp2_249 = approx_exp2(ep2_14.y);
                float2 dq2_14 = fma_f32x2_rn_noftz(make_float2(_exp2_248, _exp2_249), qr2_14, make_float2(0.25f, 0.25f));
                float _rcp_250 = approx_rcp(dq2_14.x);
                float _rcp_251 = approx_rcp(dq2_14.y);
                float2 _f32x2_mul_a_252 = make_float2(_rcp_250, _rcp_251);
                float2 _f32x2_mul_b_253 = make_float2(0.25f, 0.25f);
                float2 sig2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_252), "l"(*(const unsigned long long*)&_f32x2_mul_b_253));
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
                float2 _f32x2_mul_a_254 = make_float2(xr_3[4], xr_3[5]);
                float2 n2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_254), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_255 = mul_f32x2_noftz(mul_f32x2_noftz(g2_14, mul_f32x2_noftz(n2_14, w2_14)), sig2_14);
                float2 _f32x2_mul_b_256 = fma_f32x2_rn_noftz(z2_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_14), make_float2(1.0f, 1.0f));
                float2 dz2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_255), "l"(*(const unsigned long long*)&_f32x2_mul_b_256));
                float2 a2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_14) : "l"(*(const unsigned long long*)&g2_14), "l"(*(const unsigned long long*)&gate2_14));
                float2 dn2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_14) : "l"(*(const unsigned long long*)&a2_14), "l"(*(const unsigned long long*)&w2_14));
                float2 acc2_14 = fma_f32x2_rn_noftz(a2_14, n2_14, make_float2(acc_r[4], acc_r[5]));
                acc_r[4] = ((row_3 < rows) ? acc2_14.x : acc_r[4]);
                acc_r[5] = ((row_3 < rows) ? acc2_14.y : acc_r[5]);
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
                float2 _f32x2_mul_b_257 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_257));
                float _fma_258;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(zr_3[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_259;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(zr_3[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_15 = make_float2(_fma_258, _fma_259);
                float _exp2_260 = approx_exp2(ep2_15.x);
                float _exp2_261 = approx_exp2(ep2_15.y);
                float2 dq2_15 = fma_f32x2_rn_noftz(make_float2(_exp2_260, _exp2_261), qr2_15, make_float2(0.25f, 0.25f));
                float _rcp_262 = approx_rcp(dq2_15.x);
                float _rcp_263 = approx_rcp(dq2_15.y);
                float2 _f32x2_mul_a_264 = make_float2(_rcp_262, _rcp_263);
                float2 _f32x2_mul_b_265 = make_float2(0.25f, 0.25f);
                float2 sig2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_264), "l"(*(const unsigned long long*)&_f32x2_mul_b_265));
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
                float2 _f32x2_mul_a_266 = make_float2(xr_3[6], xr_3[7]);
                float2 n2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_266), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_267 = mul_f32x2_noftz(mul_f32x2_noftz(g2_15, mul_f32x2_noftz(n2_15, w2_15)), sig2_15);
                float2 _f32x2_mul_b_268 = fma_f32x2_rn_noftz(z2_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_15), make_float2(1.0f, 1.0f));
                float2 dz2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&_f32x2_mul_b_268));
                float2 a2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_15) : "l"(*(const unsigned long long*)&g2_15), "l"(*(const unsigned long long*)&gate2_15));
                float2 dn2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_15) : "l"(*(const unsigned long long*)&a2_15), "l"(*(const unsigned long long*)&w2_15));
                float2 acc2_15 = fma_f32x2_rn_noftz(a2_15, n2_15, make_float2(acc_r[6], acc_r[7]));
                acc_r[6] = ((row_3 < rows) ? acc2_15.x : acc_r[6]);
                acc_r[7] = ((row_3 < rows) ? acc2_15.y : acc_r[7]);
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
                float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_269;
                float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_270;
                float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_271;
                float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 8);
                dot_3 = dot_3 + _shfl_xor_sync_272;
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
                if ((tail_i_2 & 15) == 15) {
                    if (tail_i_2 >= 16) {
                        float _add_273;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        fold[group * 128 + col] = _add_273;
                        acc_r[0] = 0.0f;
                        float _add_274;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        fold[group * 128 + col + 1] = _add_274;
                        acc_r[1] = 0.0f;
                        float _add_275;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        fold[group * 128 + col + 2] = _add_275;
                        acc_r[2] = 0.0f;
                        float _add_276;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        fold[group * 128 + col + 3] = _add_276;
                        acc_r[3] = 0.0f;
                        float _add_277;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        fold[group * 128 + col + 4] = _add_277;
                        acc_r[4] = 0.0f;
                        float _add_278;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        fold[group * 128 + col + 5] = _add_278;
                        acc_r[5] = 0.0f;
                        float _add_279;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        fold[group * 128 + col + 6] = _add_279;
                        acc_r[6] = 0.0f;
                        float _add_280;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        fold[group * 128 + col + 7] = _add_280;
                        acc_r[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc_r[0];
                        acc_r[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc_r[1];
                        acc_r[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc_r[2];
                        acc_r[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc_r[3];
                        acc_r[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc_r[4];
                        acc_r[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc_r[5];
                        acc_r[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc_r[6];
                        acc_r[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc_r[7];
                        acc_r[7] = 0.0f;
                    }
                }
                if ((chunk_n & 15) != 0) {
                    if (chunk_n - 1 >= 16) {
                        float _add_281;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        fold[group * 128 + col] = _add_281;
                        acc_r[0] = 0.0f;
                        float _add_282;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        fold[group * 128 + col + 1] = _add_282;
                        acc_r[1] = 0.0f;
                        float _add_283;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        fold[group * 128 + col + 2] = _add_283;
                        acc_r[2] = 0.0f;
                        float _add_284;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        fold[group * 128 + col + 3] = _add_284;
                        acc_r[3] = 0.0f;
                        float _add_285;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_285) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        fold[group * 128 + col + 4] = _add_285;
                        acc_r[4] = 0.0f;
                        float _add_286;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        fold[group * 128 + col + 5] = _add_286;
                        acc_r[5] = 0.0f;
                        float _add_287;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        fold[group * 128 + col + 6] = _add_287;
                        acc_r[6] = 0.0f;
                        float _add_288;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        fold[group * 128 + col + 7] = _add_288;
                        acc_r[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc_r[0];
                        acc_r[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc_r[1];
                        acc_r[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc_r[2];
                        acc_r[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc_r[3];
                        acc_r[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc_r[4];
                        acc_r[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc_r[5];
                        acc_r[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc_r[6];
                        acc_r[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc_r[7];
                        acc_r[7] = 0.0f;
                    }
                }
            } else {
                if (chunk_n > 1) {
                    uint32_t vq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)((bound + chunk_lo) * 16 + group)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)((bound + chunk_lo) * 16 + group));
                    uint32_t vh = (uint32_t)((unsigned int)((bound + chunk_lo) * 16 + group)) - vq * (uint32_t)(heads.divisor);
                    long long vgoff = (long long)vq * dy_tok + (long long)vh * dy_head + (long long)col;
                    {
                        uint4 _uv4_289 = *reinterpret_cast<const uint4*>(dy + vgoff + 0);
                        la_g[0 + 0] = _uv4_289.x;
                        la_g[0 + 1] = _uv4_289.y;
                        la_g[0 + 2] = _uv4_289.z;
                        la_g[0 + 3] = _uv4_289.w;
                    }
                    long long vxoff = (long long)vq * x_tok + (long long)vh * x_head + (long long)col;
                    {
                        uint4 _uv4_290 = *reinterpret_cast<const uint4*>(x + vxoff + 0);
                        la_x[0 + 0] = _uv4_290.x;
                        la_x[0 + 1] = _uv4_290.y;
                        la_x[0 + 2] = _uv4_290.z;
                        la_x[0 + 3] = _uv4_290.w;
                    }
                    long long vzoff = (long long)vq * z_tok + (long long)vh * z_head + (long long)col;
                    {
                        uint4 _uv4_291 = *reinterpret_cast<const uint4*>(z + vzoff + 0);
                        la_z[0 + 0] = _uv4_291.x;
                        la_z[0 + 1] = _uv4_291.y;
                        la_z[0 + 2] = _uv4_291.z;
                        la_z[0 + 3] = _uv4_291.w;
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
                            uint4 _uv4_292 = *reinterpret_cast<const uint4*>(dy + vgoff_1 + 0);
                            la_g[0 + 0] = _uv4_292.x;
                            la_g[0 + 1] = _uv4_292.y;
                            la_g[0 + 2] = _uv4_292.z;
                            la_g[0 + 3] = _uv4_292.w;
                        }
                        long long vxoff_1 = (long long)vq_1 * x_tok + (long long)vh_1 * x_head + (long long)col;
                        {
                            uint4 _uv4_293 = *reinterpret_cast<const uint4*>(x + vxoff_1 + 0);
                            la_x[0 + 0] = _uv4_293.x;
                            la_x[0 + 1] = _uv4_293.y;
                            la_x[0 + 2] = _uv4_293.z;
                            la_x[0 + 3] = _uv4_293.w;
                        }
                        long long vzoff_1 = (long long)vq_1 * z_tok + (long long)vh_1 * z_head + (long long)col;
                        {
                            uint4 _uv4_294 = *reinterpret_cast<const uint4*>(z + vzoff_1 + 0);
                            la_z[0 + 0] = _uv4_294.x;
                            la_z[0 + 1] = _uv4_294.y;
                            la_z[0 + 2] = _uv4_294.z;
                            la_z[0 + 3] = _uv4_294.w;
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
                    float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, total_4, 1);
                    total_4 = total_4 + _shfl_xor_sync_295;
                    float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, total_4, 2);
                    total_4 = total_4 + _shfl_xor_sync_296;
                    float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, total_4, 4);
                    total_4 = total_4 + _shfl_xor_sync_297;
                    float _shfl_xor_sync_298 = __shfl_xor_sync(0xFFFFFFFF, total_4, 8);
                    total_4 = total_4 + _shfl_xor_sync_298;
                    float rstd_4;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_4) : "f"(total_4 * 0.0078125f + eps));
                    float dot_4 = 0.0f;
                    float wk_4[2];
                    unsigned int packed_dz_4[4];
                    float2 rstd2_4 = make_float2(rstd_4, rstd_4);
                    float2 z2_16 = make_float2(zr_4[0], zr_4[1]);
                    float2 _f32x2_mul_b_299 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_299));
                    float _fma_300;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(zr_4[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_301;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(zr_4[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_16 = make_float2(_fma_300, _fma_301);
                    float _exp2_302 = approx_exp2(ep2_16.x);
                    float _exp2_303 = approx_exp2(ep2_16.y);
                    float2 dq2_16 = fma_f32x2_rn_noftz(make_float2(_exp2_302, _exp2_303), qr2_16, make_float2(0.25f, 0.25f));
                    float _rcp_304 = approx_rcp(dq2_16.x);
                    float _rcp_305 = approx_rcp(dq2_16.y);
                    float2 _f32x2_mul_a_306 = make_float2(_rcp_304, _rcp_305);
                    float2 _f32x2_mul_b_307 = make_float2(0.25f, 0.25f);
                    float2 sig2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_306), "l"(*(const unsigned long long*)&_f32x2_mul_b_307));
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
                    float2 _f32x2_mul_a_308 = make_float2(xr_4[0], xr_4[1]);
                    float2 n2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_309 = mul_f32x2_noftz(mul_f32x2_noftz(g2_16, mul_f32x2_noftz(n2_16, w2_16)), sig2_16);
                    float2 _f32x2_mul_b_310 = fma_f32x2_rn_noftz(z2_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_16), make_float2(1.0f, 1.0f));
                    float2 dz2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_309), "l"(*(const unsigned long long*)&_f32x2_mul_b_310));
                    float2 a2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_16) : "l"(*(const unsigned long long*)&g2_16), "l"(*(const unsigned long long*)&gate2_16));
                    float2 dn2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_16) : "l"(*(const unsigned long long*)&a2_16), "l"(*(const unsigned long long*)&w2_16));
                    float2 acc2_16 = fma_f32x2_rn_noftz(a2_16, n2_16, make_float2(acc_r[0], acc_r[1]));
                    acc_r[0] = acc2_16.x;
                    acc_r[1] = acc2_16.y;
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
                    float2 _f32x2_mul_b_311 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_311));
                    float _fma_312;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(zr_4[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_313;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(zr_4[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_17 = make_float2(_fma_312, _fma_313);
                    float _exp2_314 = approx_exp2(ep2_17.x);
                    float _exp2_315 = approx_exp2(ep2_17.y);
                    float2 dq2_17 = fma_f32x2_rn_noftz(make_float2(_exp2_314, _exp2_315), qr2_17, make_float2(0.25f, 0.25f));
                    float _rcp_316 = approx_rcp(dq2_17.x);
                    float _rcp_317 = approx_rcp(dq2_17.y);
                    float2 _f32x2_mul_a_318 = make_float2(_rcp_316, _rcp_317);
                    float2 _f32x2_mul_b_319 = make_float2(0.25f, 0.25f);
                    float2 sig2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_318), "l"(*(const unsigned long long*)&_f32x2_mul_b_319));
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
                    float2 _f32x2_mul_a_320 = make_float2(xr_4[2], xr_4[3]);
                    float2 n2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_320), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_321 = mul_f32x2_noftz(mul_f32x2_noftz(g2_17, mul_f32x2_noftz(n2_17, w2_17)), sig2_17);
                    float2 _f32x2_mul_b_322 = fma_f32x2_rn_noftz(z2_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_17), make_float2(1.0f, 1.0f));
                    float2 dz2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_321), "l"(*(const unsigned long long*)&_f32x2_mul_b_322));
                    float2 a2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_17) : "l"(*(const unsigned long long*)&g2_17), "l"(*(const unsigned long long*)&gate2_17));
                    float2 dn2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_17) : "l"(*(const unsigned long long*)&a2_17), "l"(*(const unsigned long long*)&w2_17));
                    float2 acc2_17 = fma_f32x2_rn_noftz(a2_17, n2_17, make_float2(acc_r[2], acc_r[3]));
                    acc_r[2] = acc2_17.x;
                    acc_r[3] = acc2_17.y;
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
                    float2 _f32x2_mul_b_323 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_323));
                    float _fma_324;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(zr_4[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_325;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(zr_4[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_18 = make_float2(_fma_324, _fma_325);
                    float _exp2_326 = approx_exp2(ep2_18.x);
                    float _exp2_327 = approx_exp2(ep2_18.y);
                    float2 dq2_18 = fma_f32x2_rn_noftz(make_float2(_exp2_326, _exp2_327), qr2_18, make_float2(0.25f, 0.25f));
                    float _rcp_328 = approx_rcp(dq2_18.x);
                    float _rcp_329 = approx_rcp(dq2_18.y);
                    float2 _f32x2_mul_a_330 = make_float2(_rcp_328, _rcp_329);
                    float2 _f32x2_mul_b_331 = make_float2(0.25f, 0.25f);
                    float2 sig2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_330), "l"(*(const unsigned long long*)&_f32x2_mul_b_331));
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
                    float2 _f32x2_mul_a_332 = make_float2(xr_4[4], xr_4[5]);
                    float2 n2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_332), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_333 = mul_f32x2_noftz(mul_f32x2_noftz(g2_18, mul_f32x2_noftz(n2_18, w2_18)), sig2_18);
                    float2 _f32x2_mul_b_334 = fma_f32x2_rn_noftz(z2_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_18), make_float2(1.0f, 1.0f));
                    float2 dz2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_333), "l"(*(const unsigned long long*)&_f32x2_mul_b_334));
                    float2 a2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_18) : "l"(*(const unsigned long long*)&g2_18), "l"(*(const unsigned long long*)&gate2_18));
                    float2 dn2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_18) : "l"(*(const unsigned long long*)&a2_18), "l"(*(const unsigned long long*)&w2_18));
                    float2 acc2_18 = fma_f32x2_rn_noftz(a2_18, n2_18, make_float2(acc_r[4], acc_r[5]));
                    acc_r[4] = acc2_18.x;
                    acc_r[5] = acc2_18.y;
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
                    float2 _f32x2_mul_b_335 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_335));
                    float _fma_336;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(zr_4[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_337;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(zr_4[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_19 = make_float2(_fma_336, _fma_337);
                    float _exp2_338 = approx_exp2(ep2_19.x);
                    float _exp2_339 = approx_exp2(ep2_19.y);
                    float2 dq2_19 = fma_f32x2_rn_noftz(make_float2(_exp2_338, _exp2_339), qr2_19, make_float2(0.25f, 0.25f));
                    float _rcp_340 = approx_rcp(dq2_19.x);
                    float _rcp_341 = approx_rcp(dq2_19.y);
                    float2 _f32x2_mul_a_342 = make_float2(_rcp_340, _rcp_341);
                    float2 _f32x2_mul_b_343 = make_float2(0.25f, 0.25f);
                    float2 sig2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_342), "l"(*(const unsigned long long*)&_f32x2_mul_b_343));
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
                    float2 _f32x2_mul_a_344 = make_float2(xr_4[6], xr_4[7]);
                    float2 n2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_344), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_345 = mul_f32x2_noftz(mul_f32x2_noftz(g2_19, mul_f32x2_noftz(n2_19, w2_19)), sig2_19);
                    float2 _f32x2_mul_b_346 = fma_f32x2_rn_noftz(z2_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_19), make_float2(1.0f, 1.0f));
                    float2 dz2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_345), "l"(*(const unsigned long long*)&_f32x2_mul_b_346));
                    float2 a2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_19) : "l"(*(const unsigned long long*)&g2_19), "l"(*(const unsigned long long*)&gate2_19));
                    float2 dn2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_19) : "l"(*(const unsigned long long*)&a2_19), "l"(*(const unsigned long long*)&w2_19));
                    float2 acc2_19 = fma_f32x2_rn_noftz(a2_19, n2_19, make_float2(acc_r[6], acc_r[7]));
                    acc_r[6] = acc2_19.x;
                    acc_r[7] = acc2_19.y;
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
                    float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                    dot_4 = dot_4 + _shfl_xor_sync_347;
                    float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                    dot_4 = dot_4 + _shfl_xor_sync_348;
                    float _shfl_xor_sync_349 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                    dot_4 = dot_4 + _shfl_xor_sync_349;
                    float _shfl_xor_sync_350 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 8);
                    dot_4 = dot_4 + _shfl_xor_sync_350;
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
                    if ((i_1 & 15) == 15) {
                        if (i_1 >= 16) {
                            float _add_351;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_351) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                            fold[group * 128 + col] = _add_351;
                            acc_r[0] = 0.0f;
                            float _add_352;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                            fold[group * 128 + col + 1] = _add_352;
                            acc_r[1] = 0.0f;
                            float _add_353;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_353) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                            fold[group * 128 + col + 2] = _add_353;
                            acc_r[2] = 0.0f;
                            float _add_354;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_354) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                            fold[group * 128 + col + 3] = _add_354;
                            acc_r[3] = 0.0f;
                            float _add_355;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_355) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                            fold[group * 128 + col + 4] = _add_355;
                            acc_r[4] = 0.0f;
                            float _add_356;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_356) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                            fold[group * 128 + col + 5] = _add_356;
                            acc_r[5] = 0.0f;
                            float _add_357;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_357) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                            fold[group * 128 + col + 6] = _add_357;
                            acc_r[6] = 0.0f;
                            float _add_358;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_358) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                            fold[group * 128 + col + 7] = _add_358;
                            acc_r[7] = 0.0f;
                        } else {
                            fold[group * 128 + col] = acc_r[0];
                            acc_r[0] = 0.0f;
                            fold[group * 128 + col + 1] = acc_r[1];
                            acc_r[1] = 0.0f;
                            fold[group * 128 + col + 2] = acc_r[2];
                            acc_r[2] = 0.0f;
                            fold[group * 128 + col + 3] = acc_r[3];
                            acc_r[3] = 0.0f;
                            fold[group * 128 + col + 4] = acc_r[4];
                            acc_r[4] = 0.0f;
                            fold[group * 128 + col + 5] = acc_r[5];
                            acc_r[5] = 0.0f;
                            fold[group * 128 + col + 6] = acc_r[6];
                            acc_r[6] = 0.0f;
                            fold[group * 128 + col + 7] = acc_r[7];
                            acc_r[7] = 0.0f;
                        }
                    }
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
                    const uint4* _vptr_359 = reinterpret_cast<const uint4*>(dy + goff_4 + 0);
                    uint4 _vld_359[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_359[_blk] = _vptr_359[_blk];
                        uint32_t* _vpairs_359 = reinterpret_cast<uint32_t*>(&_vld_359[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_359[_pair]) << 16);
                            (&gr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_359[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_360 = reinterpret_cast<const uint4*>(x + xoff_3 + 0);
                    uint4 _vld_360[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_360[_blk] = _vptr_360[_blk];
                        uint32_t* _vpairs_360 = reinterpret_cast<uint32_t*>(&_vld_360[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_360[_pair]) << 16);
                            (&xr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_360[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_361 = reinterpret_cast<const uint4*>(z + zoff_3 + 0);
                    uint4 _vld_361[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_361[_blk] = _vptr_361[_blk];
                        uint32_t* _vpairs_361 = reinterpret_cast<uint32_t*>(&_vld_361[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_361[_pair]) << 16);
                            (&zr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_361[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        uint32_t _atomic_fetch_inc_362;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_362) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        flag[1] = (int)_atomic_fetch_inc_362;
                    }
                }
                float total_5 = 0.0f;
                float2 dx2_acc_5 = make_float2(0.0f, 0.0f);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[0], xr_5[1]), make_float2(xr_5[0], xr_5[1]), dx2_acc_5);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[2], xr_5[3]), make_float2(xr_5[2], xr_5[3]), dx2_acc_5);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[4], xr_5[5]), make_float2(xr_5[4], xr_5[5]), dx2_acc_5);
                dx2_acc_5 = fma_f32x2_rn_noftz(make_float2(xr_5[6], xr_5[7]), make_float2(xr_5[6], xr_5[7]), dx2_acc_5);
                total_5 = dx2_acc_5.x + dx2_acc_5.y;
                float _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, total_5, 1);
                total_5 = total_5 + _shfl_xor_sync_363;
                float _shfl_xor_sync_364 = __shfl_xor_sync(0xFFFFFFFF, total_5, 2);
                total_5 = total_5 + _shfl_xor_sync_364;
                float _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, total_5, 4);
                total_5 = total_5 + _shfl_xor_sync_365;
                float _shfl_xor_sync_366 = __shfl_xor_sync(0xFFFFFFFF, total_5, 8);
                total_5 = total_5 + _shfl_xor_sync_366;
                float rstd_5;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_5) : "f"(total_5 * 0.0078125f + eps));
                float dot_5 = 0.0f;
                float wk_5[2];
                unsigned int packed_dz_5[4];
                float2 rstd2_5 = make_float2(rstd_5, rstd_5);
                float2 z2_20 = make_float2(zr_5[0], zr_5[1]);
                float2 _f32x2_mul_b_367 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_367));
                float _fma_368;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(zr_5[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_369;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(zr_5[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_20 = make_float2(_fma_368, _fma_369);
                float _exp2_370 = approx_exp2(ep2_20.x);
                float _exp2_371 = approx_exp2(ep2_20.y);
                float2 dq2_20 = fma_f32x2_rn_noftz(make_float2(_exp2_370, _exp2_371), qr2_20, make_float2(0.25f, 0.25f));
                float _rcp_372 = approx_rcp(dq2_20.x);
                float _rcp_373 = approx_rcp(dq2_20.y);
                float2 _f32x2_mul_a_374 = make_float2(_rcp_372, _rcp_373);
                float2 _f32x2_mul_b_375 = make_float2(0.25f, 0.25f);
                float2 sig2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_374), "l"(*(const unsigned long long*)&_f32x2_mul_b_375));
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
                float2 _f32x2_mul_a_376 = make_float2(xr_5[0], xr_5[1]);
                float2 n2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_376), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_377 = mul_f32x2_noftz(mul_f32x2_noftz(g2_20, mul_f32x2_noftz(n2_20, w2_20)), sig2_20);
                float2 _f32x2_mul_b_378 = fma_f32x2_rn_noftz(z2_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_20), make_float2(1.0f, 1.0f));
                float2 dz2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_377), "l"(*(const unsigned long long*)&_f32x2_mul_b_378));
                float2 a2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_20) : "l"(*(const unsigned long long*)&g2_20), "l"(*(const unsigned long long*)&gate2_20));
                float2 dn2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_20) : "l"(*(const unsigned long long*)&a2_20), "l"(*(const unsigned long long*)&w2_20));
                float2 acc2_20 = fma_f32x2_rn_noftz(a2_20, n2_20, make_float2(acc_r[0], acc_r[1]));
                acc_r[0] = ((row_5 < rows) ? acc2_20.x : acc_r[0]);
                acc_r[1] = ((row_5 < rows) ? acc2_20.y : acc_r[1]);
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
                float2 _f32x2_mul_b_379 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_379));
                float _fma_380;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(zr_5[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_381;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(zr_5[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_21 = make_float2(_fma_380, _fma_381);
                float _exp2_382 = approx_exp2(ep2_21.x);
                float _exp2_383 = approx_exp2(ep2_21.y);
                float2 dq2_21 = fma_f32x2_rn_noftz(make_float2(_exp2_382, _exp2_383), qr2_21, make_float2(0.25f, 0.25f));
                float _rcp_384 = approx_rcp(dq2_21.x);
                float _rcp_385 = approx_rcp(dq2_21.y);
                float2 _f32x2_mul_a_386 = make_float2(_rcp_384, _rcp_385);
                float2 _f32x2_mul_b_387 = make_float2(0.25f, 0.25f);
                float2 sig2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_386), "l"(*(const unsigned long long*)&_f32x2_mul_b_387));
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
                float2 _f32x2_mul_a_388 = make_float2(xr_5[2], xr_5[3]);
                float2 n2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_388), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_389 = mul_f32x2_noftz(mul_f32x2_noftz(g2_21, mul_f32x2_noftz(n2_21, w2_21)), sig2_21);
                float2 _f32x2_mul_b_390 = fma_f32x2_rn_noftz(z2_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_21), make_float2(1.0f, 1.0f));
                float2 dz2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_389), "l"(*(const unsigned long long*)&_f32x2_mul_b_390));
                float2 a2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_21) : "l"(*(const unsigned long long*)&g2_21), "l"(*(const unsigned long long*)&gate2_21));
                float2 dn2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_21) : "l"(*(const unsigned long long*)&a2_21), "l"(*(const unsigned long long*)&w2_21));
                float2 acc2_21 = fma_f32x2_rn_noftz(a2_21, n2_21, make_float2(acc_r[2], acc_r[3]));
                acc_r[2] = ((row_5 < rows) ? acc2_21.x : acc_r[2]);
                acc_r[3] = ((row_5 < rows) ? acc2_21.y : acc_r[3]);
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
                float2 _f32x2_mul_b_391 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_391));
                float _fma_392;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(zr_5[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_393;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(zr_5[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_22 = make_float2(_fma_392, _fma_393);
                float _exp2_394 = approx_exp2(ep2_22.x);
                float _exp2_395 = approx_exp2(ep2_22.y);
                float2 dq2_22 = fma_f32x2_rn_noftz(make_float2(_exp2_394, _exp2_395), qr2_22, make_float2(0.25f, 0.25f));
                float _rcp_396 = approx_rcp(dq2_22.x);
                float _rcp_397 = approx_rcp(dq2_22.y);
                float2 _f32x2_mul_a_398 = make_float2(_rcp_396, _rcp_397);
                float2 _f32x2_mul_b_399 = make_float2(0.25f, 0.25f);
                float2 sig2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_398), "l"(*(const unsigned long long*)&_f32x2_mul_b_399));
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
                float2 _f32x2_mul_a_400 = make_float2(xr_5[4], xr_5[5]);
                float2 n2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_400), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_401 = mul_f32x2_noftz(mul_f32x2_noftz(g2_22, mul_f32x2_noftz(n2_22, w2_22)), sig2_22);
                float2 _f32x2_mul_b_402 = fma_f32x2_rn_noftz(z2_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_22), make_float2(1.0f, 1.0f));
                float2 dz2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_401), "l"(*(const unsigned long long*)&_f32x2_mul_b_402));
                float2 a2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_22) : "l"(*(const unsigned long long*)&g2_22), "l"(*(const unsigned long long*)&gate2_22));
                float2 dn2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_22) : "l"(*(const unsigned long long*)&a2_22), "l"(*(const unsigned long long*)&w2_22));
                float2 acc2_22 = fma_f32x2_rn_noftz(a2_22, n2_22, make_float2(acc_r[4], acc_r[5]));
                acc_r[4] = ((row_5 < rows) ? acc2_22.x : acc_r[4]);
                acc_r[5] = ((row_5 < rows) ? acc2_22.y : acc_r[5]);
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
                float2 _f32x2_mul_b_403 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_403));
                float _fma_404;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(zr_5[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_405;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(zr_5[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_23 = make_float2(_fma_404, _fma_405);
                float _exp2_406 = approx_exp2(ep2_23.x);
                float _exp2_407 = approx_exp2(ep2_23.y);
                float2 dq2_23 = fma_f32x2_rn_noftz(make_float2(_exp2_406, _exp2_407), qr2_23, make_float2(0.25f, 0.25f));
                float _rcp_408 = approx_rcp(dq2_23.x);
                float _rcp_409 = approx_rcp(dq2_23.y);
                float2 _f32x2_mul_a_410 = make_float2(_rcp_408, _rcp_409);
                float2 _f32x2_mul_b_411 = make_float2(0.25f, 0.25f);
                float2 sig2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_410), "l"(*(const unsigned long long*)&_f32x2_mul_b_411));
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
                float2 _f32x2_mul_a_412 = make_float2(xr_5[6], xr_5[7]);
                float2 n2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_412), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_413 = mul_f32x2_noftz(mul_f32x2_noftz(g2_23, mul_f32x2_noftz(n2_23, w2_23)), sig2_23);
                float2 _f32x2_mul_b_414 = fma_f32x2_rn_noftz(z2_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_23), make_float2(1.0f, 1.0f));
                float2 dz2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_413), "l"(*(const unsigned long long*)&_f32x2_mul_b_414));
                float2 a2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_23) : "l"(*(const unsigned long long*)&g2_23), "l"(*(const unsigned long long*)&gate2_23));
                float2 dn2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_23) : "l"(*(const unsigned long long*)&a2_23), "l"(*(const unsigned long long*)&w2_23));
                float2 acc2_23 = fma_f32x2_rn_noftz(a2_23, n2_23, make_float2(acc_r[6], acc_r[7]));
                acc_r[6] = ((row_5 < rows) ? acc2_23.x : acc_r[6]);
                acc_r[7] = ((row_5 < rows) ? acc2_23.y : acc_r[7]);
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
                float _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_415;
                float _shfl_xor_sync_416 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_416;
                float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_417;
                float _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 8);
                dot_5 = dot_5 + _shfl_xor_sync_418;
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
                if ((tail_i_3 & 15) == 15) {
                    if (tail_i_3 >= 16) {
                        float _add_419;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_419) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        fold[group * 128 + col] = _add_419;
                        acc_r[0] = 0.0f;
                        float _add_420;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_420) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        fold[group * 128 + col + 1] = _add_420;
                        acc_r[1] = 0.0f;
                        float _add_421;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_421) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        fold[group * 128 + col + 2] = _add_421;
                        acc_r[2] = 0.0f;
                        float _add_422;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_422) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        fold[group * 128 + col + 3] = _add_422;
                        acc_r[3] = 0.0f;
                        float _add_423;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_423) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        fold[group * 128 + col + 4] = _add_423;
                        acc_r[4] = 0.0f;
                        float _add_424;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_424) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        fold[group * 128 + col + 5] = _add_424;
                        acc_r[5] = 0.0f;
                        float _add_425;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_425) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        fold[group * 128 + col + 6] = _add_425;
                        acc_r[6] = 0.0f;
                        float _add_426;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_426) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        fold[group * 128 + col + 7] = _add_426;
                        acc_r[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc_r[0];
                        acc_r[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc_r[1];
                        acc_r[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc_r[2];
                        acc_r[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc_r[3];
                        acc_r[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc_r[4];
                        acc_r[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc_r[5];
                        acc_r[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc_r[6];
                        acc_r[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc_r[7];
                        acc_r[7] = 0.0f;
                    }
                }
                if ((chunk_n & 15) != 0) {
                    if (chunk_n - 1 >= 16) {
                        float _add_427;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_427) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        fold[group * 128 + col] = _add_427;
                        acc_r[0] = 0.0f;
                        float _add_428;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_428) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        fold[group * 128 + col + 1] = _add_428;
                        acc_r[1] = 0.0f;
                        float _add_429;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_429) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        fold[group * 128 + col + 2] = _add_429;
                        acc_r[2] = 0.0f;
                        float _add_430;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_430) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        fold[group * 128 + col + 3] = _add_430;
                        acc_r[3] = 0.0f;
                        float _add_431;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_431) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        fold[group * 128 + col + 4] = _add_431;
                        acc_r[4] = 0.0f;
                        float _add_432;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_432) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        fold[group * 128 + col + 5] = _add_432;
                        acc_r[5] = 0.0f;
                        float _add_433;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_433) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        fold[group * 128 + col + 6] = _add_433;
                        acc_r[6] = 0.0f;
                        float _add_434;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_434) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        fold[group * 128 + col + 7] = _add_434;
                        acc_r[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc_r[0];
                        acc_r[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc_r[1];
                        acc_r[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc_r[2];
                        acc_r[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc_r[3];
                        acc_r[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc_r[4];
                        acc_r[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc_r[5];
                        acc_r[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc_r[6];
                        acc_r[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc_r[7];
                        acc_r[7] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();
        if (chunk_n > 0) {
            if (tid < 128) {
                float hs_1 = fold[tid];
                #pragma unroll
                for (int g_1 = 1; g_1 < 16; g_1++) {
                    float _add_435;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_435) : "f"(hs_1), "f"(fold[g_1 * 128 + tid]));
                    hs_1 = _add_435;
                }
                unsigned int bits_1 = __as_u32(hs_1);
                unsigned int ex_1 = bits_1 >> 23 & 255;
                unsigned int man_1 = bits_1 & 8388607;
                if (ex_1 == 255) {
                    unsigned int fbit_1 = 4;
                    if (man_1 == 0) {
                        fbit_1 = 2;
                        if (bits_1 >> 31 == 0) {
                            fbit_1 = 1;
                        }
                    }
                    exf[tid] = exf[tid] | fbit_1;
                } else {
                    if (ex_1 != 0) {
                        man_1 = man_1 | 8388608;
                    }
                    if (man_1 != 0) {
                        unsigned int _max_436 = ((ex_1) > (1) ? (ex_1) : (1));
                        unsigned int pos_1 = _max_436 - 1;
                        unsigned long long wide_1 = (unsigned long long)man_1 << (unsigned long long)(pos_1 & 31);
                        unsigned long long d0_1 = wide_1 & 4294967295ull;
                        unsigned long long d1_1 = wide_1 >> 32ull;
                        if (bits_1 >> 31 != 0) {
                            d0_1 = -d0_1;
                            d1_1 = -d1_1;
                        }
                        int at_1 = (int)(pos_1 >> 5) * 128 + tid;
                        exa[at_1] = exa[at_1] + d0_1;
                        if (d1_1 != 0ull) {
                            exa[at_1 + 128] = exa[at_1 + 128] + d1_1;
                        }
                    }
                }
            }
        }
    }
    unsigned long long dsum[9];
    __syncthreads();
    if (weight_grad != 0) {
        if (tid < 128) {
            int slot = bid & 1;
            unsigned long long dg_0 = exa[tid];
            if (dg_0 != 0ull) {
                atomicAdd(&acc[slot * 9 * 128 + tid], dg_0);
            }
            unsigned long long dg_1 = exa[128 + tid];
            if (dg_1 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 1) * 128 + tid], dg_1);
            }
            unsigned long long dg_2 = exa[256 + tid];
            if (dg_2 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 2) * 128 + tid], dg_2);
            }
            unsigned long long dg_3 = exa[384 + tid];
            if (dg_3 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 3) * 128 + tid], dg_3);
            }
            unsigned long long dg_4 = exa[512 + tid];
            if (dg_4 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 4) * 128 + tid], dg_4);
            }
            unsigned long long dg_5 = exa[640 + tid];
            if (dg_5 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 5) * 128 + tid], dg_5);
            }
            unsigned long long dg_6 = exa[768 + tid];
            if (dg_6 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 6) * 128 + tid], dg_6);
            }
            unsigned long long dg_7 = exa[896 + tid];
            if (dg_7 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 7) * 128 + tid], dg_7);
            }
            unsigned long long dg_8 = exa[1024 + tid];
            if (dg_8 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 8) * 128 + tid], dg_8);
            }
            unsigned int fv = exf[tid];
            if (fv != 0) {
                atomicOr(&flags[tid], fv);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int is_last = 0;
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            is_last = 1;
            done[1] = 0;
        }
        flag[0] = is_last;
    }
    __syncthreads();
    int flag_word[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag_word[0])) : "r"(flag_addr));
    if (flag_word[0] != 0) {
        if (weight_grad != 0) {
            if (tid < 128) {
                unsigned int flw[1];
                {
                    uint32_t _scalar_bits_437;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_437) : "l"((const void*)(flags + (tid))) : "memory");
                    flw[0] = (unsigned int)_scalar_bits_437;
                }
                unsigned long long dig[18];
                dig[0] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_438;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_438) : "l"((const void*)(acc + (tid))) : "memory");
                        dig[0] = (unsigned long long)_scalar_bits_438;
                    }
                }
                dig[1] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_439;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_439) : "l"((const void*)(acc + (1152 + tid))) : "memory");
                        dig[1] = (unsigned long long)_scalar_bits_439;
                    }
                }
                dig[2] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_440;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_440) : "l"((const void*)(acc + (128 + tid))) : "memory");
                        dig[2] = (unsigned long long)_scalar_bits_440;
                    }
                }
                dig[3] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_441;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_441) : "l"((const void*)(acc + (1280 + tid))) : "memory");
                        dig[3] = (unsigned long long)_scalar_bits_441;
                    }
                }
                dig[4] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_442;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_442) : "l"((const void*)(acc + (256 + tid))) : "memory");
                        dig[4] = (unsigned long long)_scalar_bits_442;
                    }
                }
                dig[5] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_443;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_443) : "l"((const void*)(acc + (1408 + tid))) : "memory");
                        dig[5] = (unsigned long long)_scalar_bits_443;
                    }
                }
                dig[6] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_444;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_444) : "l"((const void*)(acc + (384 + tid))) : "memory");
                        dig[6] = (unsigned long long)_scalar_bits_444;
                    }
                }
                dig[7] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_445;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_445) : "l"((const void*)(acc + (1536 + tid))) : "memory");
                        dig[7] = (unsigned long long)_scalar_bits_445;
                    }
                }
                dig[8] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_446;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_446) : "l"((const void*)(acc + (512 + tid))) : "memory");
                        dig[8] = (unsigned long long)_scalar_bits_446;
                    }
                }
                dig[9] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_447;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_447) : "l"((const void*)(acc + (1664 + tid))) : "memory");
                        dig[9] = (unsigned long long)_scalar_bits_447;
                    }
                }
                dig[10] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_448;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_448) : "l"((const void*)(acc + (640 + tid))) : "memory");
                        dig[10] = (unsigned long long)_scalar_bits_448;
                    }
                }
                dig[11] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_449;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_449) : "l"((const void*)(acc + (1792 + tid))) : "memory");
                        dig[11] = (unsigned long long)_scalar_bits_449;
                    }
                }
                dig[12] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_450;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_450) : "l"((const void*)(acc + (768 + tid))) : "memory");
                        dig[12] = (unsigned long long)_scalar_bits_450;
                    }
                }
                dig[13] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_451;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_451) : "l"((const void*)(acc + (1920 + tid))) : "memory");
                        dig[13] = (unsigned long long)_scalar_bits_451;
                    }
                }
                dig[14] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_452;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_452) : "l"((const void*)(acc + (896 + tid))) : "memory");
                        dig[14] = (unsigned long long)_scalar_bits_452;
                    }
                }
                dig[15] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_453;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_453) : "l"((const void*)(acc + (2048 + tid))) : "memory");
                        dig[15] = (unsigned long long)_scalar_bits_453;
                    }
                }
                dig[16] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_454;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_454) : "l"((const void*)(acc + (1024 + tid))) : "memory");
                        dig[16] = (unsigned long long)_scalar_bits_454;
                    }
                }
                dig[17] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_455;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_455) : "l"((const void*)(acc + (2176 + tid))) : "memory");
                        dig[17] = (unsigned long long)_scalar_bits_455;
                    }
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1152 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (128 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1280 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (256 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1408 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (384 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1536 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (512 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1664 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (640 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1792 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (768 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1920 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (896 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (2048 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1024 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (2176 + tid)) = 0ull;
                }
                unsigned long long digit_0 = dig[0];
                digit_0 = digit_0 + dig[1];
                dsum[0] = digit_0;
                unsigned long long digit_1 = dig[2];
                digit_1 = digit_1 + dig[3];
                dsum[1] = digit_1;
                unsigned long long digit_2 = dig[4];
                digit_2 = digit_2 + dig[5];
                dsum[2] = digit_2;
                unsigned long long digit_3 = dig[6];
                digit_3 = digit_3 + dig[7];
                dsum[3] = digit_3;
                unsigned long long digit_4 = dig[8];
                digit_4 = digit_4 + dig[9];
                dsum[4] = digit_4;
                unsigned long long digit_5 = dig[10];
                digit_5 = digit_5 + dig[11];
                dsum[5] = digit_5;
                unsigned long long digit_6 = dig[12];
                digit_6 = digit_6 + dig[13];
                dsum[6] = digit_6;
                unsigned long long digit_7 = dig[14];
                digit_7 = digit_7 + dig[15];
                dsum[7] = digit_7;
                unsigned long long digit_8 = dig[16];
                digit_8 = digit_8 + dig[17];
                dsum[8] = digit_8;
                *(reinterpret_cast<unsigned int*>(flags) + (tid)) = 0;
                unsigned int flv = flw[0];
                if (accumulate != 0) {
                    float pv = (float)dw[tid];
                    unsigned int pbits = __as_u32(pv);
                    unsigned int pex = pbits >> 23 & 255;
                    unsigned int pman = pbits & 8388607;
                    if (pex == 255) {
                        unsigned int pf = 4;
                        if (pman == 0) {
                            pf = 2;
                            if (pbits >> 31 == 0) {
                                pf = 1;
                            }
                        }
                        flv = flv | pf;
                    } else {
                        if (pex != 0) {
                            pman = pman | 8388608;
                        }
                        if (pman != 0) {
                            unsigned int _max_456 = ((pex) > (1) ? (pex) : (1));
                            unsigned int ppos = _max_456 - 1;
                            unsigned long long pwide = (unsigned long long)pman << (unsigned long long)(ppos & 31);
                            unsigned long long p0 = pwide & 4294967295ull;
                            unsigned long long p1 = pwide >> 32ull;
                            if (pbits >> 31 != 0) {
                                p0 = -p0;
                                p1 = -p1;
                            }
                            int pk = (int)(ppos >> 5);
                            if (pk == 0) {
                                dsum[0] = dsum[0] + p0;
                            }
                            if (pk + 1 == 0) {
                                dsum[0] = dsum[0] + p1;
                            }
                            if (pk == 1) {
                                dsum[1] = dsum[1] + p0;
                            }
                            if (pk + 1 == 1) {
                                dsum[1] = dsum[1] + p1;
                            }
                            if (pk == 2) {
                                dsum[2] = dsum[2] + p0;
                            }
                            if (pk + 1 == 2) {
                                dsum[2] = dsum[2] + p1;
                            }
                            if (pk == 3) {
                                dsum[3] = dsum[3] + p0;
                            }
                            if (pk + 1 == 3) {
                                dsum[3] = dsum[3] + p1;
                            }
                            if (pk == 4) {
                                dsum[4] = dsum[4] + p0;
                            }
                            if (pk + 1 == 4) {
                                dsum[4] = dsum[4] + p1;
                            }
                            if (pk == 5) {
                                dsum[5] = dsum[5] + p0;
                            }
                            if (pk + 1 == 5) {
                                dsum[5] = dsum[5] + p1;
                            }
                            if (pk == 6) {
                                dsum[6] = dsum[6] + p0;
                            }
                            if (pk + 1 == 6) {
                                dsum[6] = dsum[6] + p1;
                            }
                            if (pk == 7) {
                                dsum[7] = dsum[7] + p0;
                            }
                            if (pk + 1 == 7) {
                                dsum[7] = dsum[7] + p1;
                            }
                            if (pk == 8) {
                                dsum[8] = dsum[8] + p0;
                            }
                            if (pk + 1 == 8) {
                                dsum[8] = dsum[8] + p1;
                            }
                        }
                    }
                }
                unsigned int words[10];
                long long carry = 0ll;
                unsigned long long t_0 = dsum[0] + (unsigned long long)carry;
                words[0] = (unsigned int)(t_0 & 4294967295ull);
                carry = (long long)t_0 >> 32ll;
                unsigned long long t_1 = dsum[1] + (unsigned long long)carry;
                words[1] = (unsigned int)(t_1 & 4294967295ull);
                carry = (long long)t_1 >> 32ll;
                unsigned long long t_2 = dsum[2] + (unsigned long long)carry;
                words[2] = (unsigned int)(t_2 & 4294967295ull);
                carry = (long long)t_2 >> 32ll;
                unsigned long long t_3 = dsum[3] + (unsigned long long)carry;
                words[3] = (unsigned int)(t_3 & 4294967295ull);
                carry = (long long)t_3 >> 32ll;
                unsigned long long t_4 = dsum[4] + (unsigned long long)carry;
                words[4] = (unsigned int)(t_4 & 4294967295ull);
                carry = (long long)t_4 >> 32ll;
                unsigned long long t_5 = dsum[5] + (unsigned long long)carry;
                words[5] = (unsigned int)(t_5 & 4294967295ull);
                carry = (long long)t_5 >> 32ll;
                unsigned long long t_6 = dsum[6] + (unsigned long long)carry;
                words[6] = (unsigned int)(t_6 & 4294967295ull);
                carry = (long long)t_6 >> 32ll;
                unsigned long long t_7 = dsum[7] + (unsigned long long)carry;
                words[7] = (unsigned int)(t_7 & 4294967295ull);
                carry = (long long)t_7 >> 32ll;
                long long top = (long long)dsum[8] + carry;
                int neg = 0;
                if (top < 0ll) {
                    neg = 1;
                    unsigned long long c1 = 1ull;
                    unsigned long long tn_0 = (unsigned long long)(words[0] ^ 4294967295u) + c1;
                    words[0] = (unsigned int)(tn_0 & 4294967295ull);
                    c1 = tn_0 >> 32ull;
                    unsigned long long tn_1 = (unsigned long long)(words[1] ^ 4294967295u) + c1;
                    words[1] = (unsigned int)(tn_1 & 4294967295ull);
                    c1 = tn_1 >> 32ull;
                    unsigned long long tn_2 = (unsigned long long)(words[2] ^ 4294967295u) + c1;
                    words[2] = (unsigned int)(tn_2 & 4294967295ull);
                    c1 = tn_2 >> 32ull;
                    unsigned long long tn_3 = (unsigned long long)(words[3] ^ 4294967295u) + c1;
                    words[3] = (unsigned int)(tn_3 & 4294967295ull);
                    c1 = tn_3 >> 32ull;
                    unsigned long long tn_4 = (unsigned long long)(words[4] ^ 4294967295u) + c1;
                    words[4] = (unsigned int)(tn_4 & 4294967295ull);
                    c1 = tn_4 >> 32ull;
                    unsigned long long tn_5 = (unsigned long long)(words[5] ^ 4294967295u) + c1;
                    words[5] = (unsigned int)(tn_5 & 4294967295ull);
                    c1 = tn_5 >> 32ull;
                    unsigned long long tn_6 = (unsigned long long)(words[6] ^ 4294967295u) + c1;
                    words[6] = (unsigned int)(tn_6 & 4294967295ull);
                    c1 = tn_6 >> 32ull;
                    unsigned long long tn_7 = (unsigned long long)(words[7] ^ 4294967295u) + c1;
                    words[7] = (unsigned int)(tn_7 & 4294967295ull);
                    c1 = tn_7 >> 32ull;
                    top = -1ll - top + (long long)c1;
                }
                words[8] = (unsigned int)((unsigned long long)top & 4294967295ull);
                words[9] = (unsigned int)((unsigned long long)top >> 32ull);
                int tw = -1;
                if (words[0] != 0) {
                    tw = 0;
                }
                if (words[1] != 0) {
                    tw = 1;
                }
                if (words[2] != 0) {
                    tw = 2;
                }
                if (words[3] != 0) {
                    tw = 3;
                }
                if (words[4] != 0) {
                    tw = 4;
                }
                if (words[5] != 0) {
                    tw = 5;
                }
                if (words[6] != 0) {
                    tw = 6;
                }
                if (words[7] != 0) {
                    tw = 7;
                }
                if (words[8] != 0) {
                    tw = 8;
                }
                if (words[9] != 0) {
                    tw = 9;
                }
                unsigned int wa = 0;
                unsigned int wb = 0;
                unsigned int wc = 0;
                unsigned int lowor = 0;
                if (tw == 0) {
                    wa = words[0];
                }
                if (tw == 1) {
                    wb = words[0];
                }
                if (tw == 2) {
                    wc = words[0];
                }
                if (tw > 2) {
                    lowor = lowor | words[0];
                }
                if (tw == 1) {
                    wa = words[1];
                }
                if (tw == 2) {
                    wb = words[1];
                }
                if (tw == 3) {
                    wc = words[1];
                }
                if (tw > 3) {
                    lowor = lowor | words[1];
                }
                if (tw == 2) {
                    wa = words[2];
                }
                if (tw == 3) {
                    wb = words[2];
                }
                if (tw == 4) {
                    wc = words[2];
                }
                if (tw > 4) {
                    lowor = lowor | words[2];
                }
                if (tw == 3) {
                    wa = words[3];
                }
                if (tw == 4) {
                    wb = words[3];
                }
                if (tw == 5) {
                    wc = words[3];
                }
                if (tw > 5) {
                    lowor = lowor | words[3];
                }
                if (tw == 4) {
                    wa = words[4];
                }
                if (tw == 5) {
                    wb = words[4];
                }
                if (tw == 6) {
                    wc = words[4];
                }
                if (tw > 6) {
                    lowor = lowor | words[4];
                }
                if (tw == 5) {
                    wa = words[5];
                }
                if (tw == 6) {
                    wb = words[5];
                }
                if (tw == 7) {
                    wc = words[5];
                }
                if (tw > 7) {
                    lowor = lowor | words[5];
                }
                if (tw == 6) {
                    wa = words[6];
                }
                if (tw == 7) {
                    wb = words[6];
                }
                if (tw == 8) {
                    wc = words[6];
                }
                if (tw > 8) {
                    lowor = lowor | words[6];
                }
                if (tw == 7) {
                    wa = words[7];
                }
                if (tw == 8) {
                    wb = words[7];
                }
                if (tw == 9) {
                    wc = words[7];
                }
                if (tw > 9) {
                    lowor = lowor | words[7];
                }
                if (tw == 8) {
                    wa = words[8];
                }
                if (tw == 9) {
                    wb = words[8];
                }
                if (tw == 10) {
                    wc = words[8];
                }
                if (tw > 10) {
                    lowor = lowor | words[8];
                }
                if (tw == 9) {
                    wa = words[9];
                }
                if (tw == 10) {
                    wb = words[9];
                }
                if (tw == 11) {
                    wc = words[9];
                }
                if (tw > 11) {
                    lowor = lowor | words[9];
                }
                unsigned int rbits = 0;
                if (tw >= 0) {
                    int lz = __clz((int)wa);
                    int bp = 32 * tw + 31 - lz;
                    if (bp <= 23) {
                        rbits = wa;
                    } else {
                        unsigned long long wide_c = (unsigned long long)wc << (unsigned long long)lz;
                        unsigned long long hi = ((unsigned long long)wa << 32ull | (unsigned long long)wb) << (unsigned long long)lz;
                        hi = hi | wide_c >> 32ull;
                        unsigned int mant = (unsigned int)(hi >> 40ull);
                        unsigned int sticky = 0;
                        if ((hi & 549755813887ull) != 0ull) {
                            sticky = 1;
                        }
                        if ((wide_c & 4294967295ull) != 0ull) {
                            sticky = 1;
                        }
                        if (lowor != 0) {
                            sticky = 1;
                        }
                        rbits = ((unsigned int)(bp - 23) << 23) + mant;
                        if ((hi >> 39ull & 1ull) != 0ull) {
                            if ((sticky | mant & 1) != 0) {
                                rbits = rbits + 1;
                            }
                        }
                        if (bp >= 277) {
                            rbits = 2139095040;
                        }
                    }
                    if (neg != 0) {
                        rbits = rbits | 2147483648u;
                    }
                }
                if (flv != 0) {
                    unsigned int special = 4286578688u;
                    if ((flv & 1) != 0) {
                        special = 2139095040;
                    }
                    if ((flv & 4) != 0) {
                        special = 2143289344;
                    }
                    if ((flv & 3) == 3) {
                        special = 2143289344;
                    }
                    rbits = special;
                }
                float res = __uint_as_float(rbits);
                dw[tid] = (__nv_bfloat16)res;
            }
        }
    }
}

} // extern "C"
