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
kernel_cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ acc, unsigned int* __restrict__ flags, unsigned int* __restrict__ done, float* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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
    unsigned long long* exa = reinterpret_cast<unsigned long long*>(smem_raw + 0);
    const int exa_addr = smem + 0;
    unsigned int* exf = reinterpret_cast<unsigned int*>(smem_raw + 9216);
    const int exf_addr = smem + 9216;
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
    unsigned int wbits[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(w + col + 0);
        wbits[0 + 0] = _uv4_0.x;
        wbits[0 + 1] = _uv4_0.y;
        wbits[0 + 2] = _uv4_0.z;
        wbits[0 + 3] = _uv4_0.w;
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
    if (bid < nblocks) {
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
            float total = 0.0f;
            float2 dx2_acc = make_float2(0.0f, 0.0f);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[0], xr[1]), make_float2(xr[0], xr[1]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[2], xr[3]), make_float2(xr[2], xr[3]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[4], xr[5]), make_float2(xr[4], xr[5]), dx2_acc);
            dx2_acc = fma_f32x2_rn_noftz(make_float2(xr[6], xr[7]), make_float2(xr[6], xr[7]), dx2_acc);
            total = dx2_acc.x + dx2_acc.y;
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            total = total + _shfl_xor_sync_4;
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_5;
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_6;
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
            total = total + _shfl_xor_sync_7;
            float rstd;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
            float dot = 0.0f;
            int fast = 1;
            float wk[2];
            unsigned int packed_dz[4];
            if (fast != 0) {
                float2 rstd2 = make_float2(rstd, rstd);
                float2 z2_0 = make_float2(zr[0], zr[1]);
                float2 _f32x2_mul_b_8 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                float _fma_9;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(zr[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_10;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(zr[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_0 = make_float2(_fma_9, _fma_10);
                float _exp2_11 = approx_exp2(ep2_0.x);
                float _exp2_12 = approx_exp2(ep2_0.y);
                float2 dq2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_11, _exp2_12), qr2_0, make_float2(0.25f, 0.25f));
                float _rcp_13 = approx_rcp(dq2_0.x);
                float _rcp_14 = approx_rcp(dq2_0.y);
                float2 _f32x2_mul_a_15 = make_float2(_rcp_13, _rcp_14);
                float2 _f32x2_mul_b_16 = make_float2(0.25f, 0.25f);
                float2 sig2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
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
                float2 _f32x2_mul_a_17 = make_float2(xr[0], xr[1]);
                float2 n2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_18 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                float2 _f32x2_mul_b_19 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                float2 dz2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
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
                float2 _f32x2_mul_b_20 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                float _fma_21;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(zr[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_22;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(zr[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_1 = make_float2(_fma_21, _fma_22);
                float _exp2_23 = approx_exp2(ep2_1.x);
                float _exp2_24 = approx_exp2(ep2_1.y);
                float2 dq2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_23, _exp2_24), qr2_1, make_float2(0.25f, 0.25f));
                float _rcp_25 = approx_rcp(dq2_1.x);
                float _rcp_26 = approx_rcp(dq2_1.y);
                float2 _f32x2_mul_a_27 = make_float2(_rcp_25, _rcp_26);
                float2 _f32x2_mul_b_28 = make_float2(0.25f, 0.25f);
                float2 sig2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
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
                float2 _f32x2_mul_a_29 = make_float2(xr[2], xr[3]);
                float2 n2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_30 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                float2 _f32x2_mul_b_31 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                float2 dz2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
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
                float2 _f32x2_mul_b_32 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
                float _fma_33;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(zr[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_34;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(zr[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_2 = make_float2(_fma_33, _fma_34);
                float _exp2_35 = approx_exp2(ep2_2.x);
                float _exp2_36 = approx_exp2(ep2_2.y);
                float2 dq2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_35, _exp2_36), qr2_2, make_float2(0.25f, 0.25f));
                float _rcp_37 = approx_rcp(dq2_2.x);
                float _rcp_38 = approx_rcp(dq2_2.y);
                float2 _f32x2_mul_a_39 = make_float2(_rcp_37, _rcp_38);
                float2 _f32x2_mul_b_40 = make_float2(0.25f, 0.25f);
                float2 sig2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
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
                float2 _f32x2_mul_a_41 = make_float2(xr[4], xr[5]);
                float2 n2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_42 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                float2 _f32x2_mul_b_43 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                float2 dz2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
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
                float2 _f32x2_mul_b_44 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
                float _fma_45;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(zr[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_46;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(zr[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_3 = make_float2(_fma_45, _fma_46);
                float _exp2_47 = approx_exp2(ep2_3.x);
                float _exp2_48 = approx_exp2(ep2_3.y);
                float2 dq2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_47, _exp2_48), qr2_3, make_float2(0.25f, 0.25f));
                float _rcp_49 = approx_rcp(dq2_3.x);
                float _rcp_50 = approx_rcp(dq2_3.y);
                float2 _f32x2_mul_a_51 = make_float2(_rcp_49, _rcp_50);
                float2 _f32x2_mul_b_52 = make_float2(0.25f, 0.25f);
                float2 sig2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
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
                float2 _f32x2_mul_a_53 = make_float2(xr[6], xr[7]);
                float2 n2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_54 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                float2 _f32x2_mul_b_55 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                float2 dz2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
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
                float _exp_56 = __nv_expf(zr[0] * -1.0f);
                float den_0 = 1.0f + _exp_56;
                float sig_0 = __fdiv_rn(1.0f, den_0);
                float gate_0 = __fdiv_rn(zr[0], den_0);
                float nj_0 = xr[0] * rstd;
                float aj_0 = gr[0] * gate_0;
                float dzj_0 = gr[0] * (nj_0 * wk[0]) * sig_0 * __fmaf_rn(zr[0], 1.0f - sig_0, 1.0f);
                float dnj_0 = aj_0 * wk[0];
                acc_r[0] = ((row < rows) ? __fmaf_rn(aj_0, nj_0, acc_r[0]) : acc_r[0]);
                zr[0] = dzj_0;
                gr[0] = dnj_0;
                float _exp_57 = __nv_expf(zr[1] * -1.0f);
                float den_1 = 1.0f + _exp_57;
                float sig_1 = __fdiv_rn(1.0f, den_1);
                float gate_1 = __fdiv_rn(zr[1], den_1);
                float nj_1 = xr[1] * rstd;
                float aj_1 = gr[1] * gate_1;
                float dzj_1 = gr[1] * (nj_1 * wk[1]) * sig_1 * __fmaf_rn(zr[1], 1.0f - sig_1, 1.0f);
                float dnj_1 = aj_1 * wk[1];
                acc_r[1] = ((row < rows) ? __fmaf_rn(aj_1, nj_1, acc_r[1]) : acc_r[1]);
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
                float _exp_58 = __nv_expf(zr[2] * -1.0f);
                float den_2 = 1.0f + _exp_58;
                float sig_2 = __fdiv_rn(1.0f, den_2);
                float gate_2 = __fdiv_rn(zr[2], den_2);
                float nj_2 = xr[2] * rstd;
                float aj_2 = gr[2] * gate_2;
                float dzj_2 = gr[2] * (nj_2 * wk[0]) * sig_2 * __fmaf_rn(zr[2], 1.0f - sig_2, 1.0f);
                float dnj_2 = aj_2 * wk[0];
                acc_r[2] = ((row < rows) ? __fmaf_rn(aj_2, nj_2, acc_r[2]) : acc_r[2]);
                zr[2] = dzj_2;
                gr[2] = dnj_2;
                float _exp_59 = __nv_expf(zr[3] * -1.0f);
                float den_3 = 1.0f + _exp_59;
                float sig_3 = __fdiv_rn(1.0f, den_3);
                float gate_3 = __fdiv_rn(zr[3], den_3);
                float nj_3 = xr[3] * rstd;
                float aj_3 = gr[3] * gate_3;
                float dzj_3 = gr[3] * (nj_3 * wk[1]) * sig_3 * __fmaf_rn(zr[3], 1.0f - sig_3, 1.0f);
                float dnj_3 = aj_3 * wk[1];
                acc_r[3] = ((row < rows) ? __fmaf_rn(aj_3, nj_3, acc_r[3]) : acc_r[3]);
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
                float _exp_60 = __nv_expf(zr[4] * -1.0f);
                float den_4 = 1.0f + _exp_60;
                float sig_4 = __fdiv_rn(1.0f, den_4);
                float gate_4 = __fdiv_rn(zr[4], den_4);
                float nj_4 = xr[4] * rstd;
                float aj_4 = gr[4] * gate_4;
                float dzj_4 = gr[4] * (nj_4 * wk[0]) * sig_4 * __fmaf_rn(zr[4], 1.0f - sig_4, 1.0f);
                float dnj_4 = aj_4 * wk[0];
                acc_r[4] = ((row < rows) ? __fmaf_rn(aj_4, nj_4, acc_r[4]) : acc_r[4]);
                zr[4] = dzj_4;
                gr[4] = dnj_4;
                float _exp_61 = __nv_expf(zr[5] * -1.0f);
                float den_5 = 1.0f + _exp_61;
                float sig_5 = __fdiv_rn(1.0f, den_5);
                float gate_5 = __fdiv_rn(zr[5], den_5);
                float nj_5 = xr[5] * rstd;
                float aj_5 = gr[5] * gate_5;
                float dzj_5 = gr[5] * (nj_5 * wk[1]) * sig_5 * __fmaf_rn(zr[5], 1.0f - sig_5, 1.0f);
                float dnj_5 = aj_5 * wk[1];
                acc_r[5] = ((row < rows) ? __fmaf_rn(aj_5, nj_5, acc_r[5]) : acc_r[5]);
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
                float _exp_62 = __nv_expf(zr[6] * -1.0f);
                float den_6 = 1.0f + _exp_62;
                float sig_6 = __fdiv_rn(1.0f, den_6);
                float gate_6 = __fdiv_rn(zr[6], den_6);
                float nj_6 = xr[6] * rstd;
                float aj_6 = gr[6] * gate_6;
                float dzj_6 = gr[6] * (nj_6 * wk[0]) * sig_6 * __fmaf_rn(zr[6], 1.0f - sig_6, 1.0f);
                float dnj_6 = aj_6 * wk[0];
                acc_r[6] = ((row < rows) ? __fmaf_rn(aj_6, nj_6, acc_r[6]) : acc_r[6]);
                zr[6] = dzj_6;
                gr[6] = dnj_6;
                float _exp_63 = __nv_expf(zr[7] * -1.0f);
                float den_7 = 1.0f + _exp_63;
                float sig_7 = __fdiv_rn(1.0f, den_7);
                float gate_7 = __fdiv_rn(zr[7], den_7);
                float nj_7 = xr[7] * rstd;
                float aj_7 = gr[7] * gate_7;
                float dzj_7 = gr[7] * (nj_7 * wk[1]) * sig_7 * __fmaf_rn(zr[7], 1.0f - sig_7, 1.0f);
                float dnj_7 = aj_7 * wk[1];
                acc_r[7] = ((row < rows) ? __fmaf_rn(aj_7, nj_7, acc_r[7]) : acc_r[7]);
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
            float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot = dot + _shfl_xor_sync_64;
            float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot = dot + _shfl_xor_sync_65;
            float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot = dot + _shfl_xor_sync_66;
            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot = dot + _shfl_xor_sync_67;
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
            if ((i & 15) == 15) {
                if (i >= 16) {
                    float _add_68;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    fold[group * 128 + col] = _add_68;
                    acc_r[0] = 0.0f;
                    float _add_69;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    fold[group * 128 + col + 1] = _add_69;
                    acc_r[1] = 0.0f;
                    float _add_70;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    fold[group * 128 + col + 2] = _add_70;
                    acc_r[2] = 0.0f;
                    float _add_71;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    fold[group * 128 + col + 3] = _add_71;
                    acc_r[3] = 0.0f;
                    float _add_72;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    fold[group * 128 + col + 4] = _add_72;
                    acc_r[4] = 0.0f;
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    fold[group * 128 + col + 5] = _add_73;
                    acc_r[5] = 0.0f;
                    float _add_74;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    fold[group * 128 + col + 6] = _add_74;
                    acc_r[6] = 0.0f;
                    float _add_75;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    fold[group * 128 + col + 7] = _add_75;
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
    unsigned long long dsum[9];
    __syncthreads();
    if (weight_grad != 0) {
        if (tid < 128) {
            float hs = fold[tid];
            #pragma unroll
            for (int g = 1; g < 16; g++) {
                float _add_76;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(hs), "f"(fold[g * 128 + tid]));
                hs = _add_76;
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
                    unsigned int _max_77 = ((ex) > (1) ? (ex) : (1));
                    unsigned int pos = _max_77 - 1;
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
    if (weight_grad != 0) {
        if (tid >> 7 == 0) {
            dsum[0] = exa[tid];
            dsum[1] = exa[128 + tid];
            dsum[2] = exa[256 + tid];
            dsum[3] = exa[384 + tid];
            dsum[4] = exa[512 + tid];
            dsum[5] = exa[640 + tid];
            dsum[6] = exa[768 + tid];
            dsum[7] = exa[896 + tid];
            dsum[8] = exa[1024 + tid];
            unsigned int flv = exf[tid];
            if (accumulate != 0) {
                float pv = dw[tid];
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
                        unsigned int _max_78 = ((pex) > (1) ? (pex) : (1));
                        unsigned int ppos = _max_78 - 1;
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
            dw[tid] = res;
        }
    }
}

} // extern "C"
