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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>


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
kernel_cake_rmsnorm_train_67cb56412bd46304b711(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    float* sums = reinterpret_cast<float*>(smem_raw + 0);
    const int sums_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int rg = warp;
    int ct = tid - rg * 32;
    int col = ct * 8;
    float ww[8];
    {
        unsigned _ldv8_0_0;
        unsigned _ldv8_0_1;
        unsigned _ldv8_0_2;
        unsigned _ldv8_0_3;
        unsigned _ldv8_0_4;
        unsigned _ldv8_0_5;
        unsigned _ldv8_0_6;
        unsigned _ldv8_0_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + ((unsigned long long)col))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_0_0);
        ww[0 + 1] = __uint_as_float(_ldv8_0_1);
        ww[0 + 2] = __uint_as_float(_ldv8_0_2);
        ww[0 + 3] = __uint_as_float(_ldv8_0_3);
        ww[0 + 4] = __uint_as_float(_ldv8_0_4);
        ww[0 + 5] = __uint_as_float(_ldv8_0_5);
        ww[0 + 6] = __uint_as_float(_ldv8_0_6);
        ww[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    unsigned int gw[4];
    unsigned int xw[4];
    unsigned int hw[1];
    float rr[1];
    int lrow_0 = rg;
    if (lrow_0 < T) {
        rr[0] = r[lrow_0];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_0 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&gw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_0 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&xw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
    }
    float acc[8];
    float dq[1];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    dq[0] = 0.0f;
    if (rg < T) {
        float2 rr2 = make_float2(rr[0], rr[0]);
        float dot = 0.0f;
        float g0_0 = __uint_as_float(gw[0] << 16);
        float g1_0 = __uint_as_float(gw[0] & 4294901760u);
        float x0_0 = __uint_as_float(xw[0] << 16);
        float x1_0 = __uint_as_float(xw[0] & 4294901760u);
        dot = __fmaf_rn(g0_0 * ww[0], x0_0, dot);
        dot = __fmaf_rn(g1_0 * ww[1], x1_0, dot);
        float2 _f32x2_mul_a_3 = make_float2(g0_0, g1_0);
        float2 _f32x2_mul_b_4 = make_float2(x0_0, x1_0);
        float2 gx;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_4));
        float2 a = fma_f32x2_rn_noftz(gx, rr2, make_float2(acc[0], acc[1]));
        acc[0] = a.x;
        acc[1] = a.y;
        float g0_1 = __uint_as_float(gw[1] << 16);
        float g1_1 = __uint_as_float(gw[1] & 4294901760u);
        float x0_1 = __uint_as_float(xw[1] << 16);
        float x1_1 = __uint_as_float(xw[1] & 4294901760u);
        dot = __fmaf_rn(g0_1 * ww[2], x0_1, dot);
        dot = __fmaf_rn(g1_1 * ww[3], x1_1, dot);
        float2 _f32x2_mul_a_5 = make_float2(g0_1, g1_1);
        float2 _f32x2_mul_b_6 = make_float2(x0_1, x1_1);
        float2 gx_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
        float2 a_0 = fma_f32x2_rn_noftz(gx_0, rr2, make_float2(acc[2], acc[3]));
        acc[2] = a_0.x;
        acc[3] = a_0.y;
        float g0_2 = __uint_as_float(gw[2] << 16);
        float g1_2 = __uint_as_float(gw[2] & 4294901760u);
        float x0_2 = __uint_as_float(xw[2] << 16);
        float x1_2 = __uint_as_float(xw[2] & 4294901760u);
        dot = __fmaf_rn(g0_2 * ww[4], x0_2, dot);
        dot = __fmaf_rn(g1_2 * ww[5], x1_2, dot);
        float2 _f32x2_mul_a_7 = make_float2(g0_2, g1_2);
        float2 _f32x2_mul_b_8 = make_float2(x0_2, x1_2);
        float2 gx_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float2 a_1 = fma_f32x2_rn_noftz(gx_1, rr2, make_float2(acc[4], acc[5]));
        acc[4] = a_1.x;
        acc[5] = a_1.y;
        float g0_3 = __uint_as_float(gw[3] << 16);
        float g1_3 = __uint_as_float(gw[3] & 4294901760u);
        float x0_3 = __uint_as_float(xw[3] << 16);
        float x1_3 = __uint_as_float(xw[3] & 4294901760u);
        dot = __fmaf_rn(g0_3 * ww[6], x0_3, dot);
        dot = __fmaf_rn(g1_3 * ww[7], x1_3, dot);
        float2 _f32x2_mul_a_9 = make_float2(g0_3, g1_3);
        float2 _f32x2_mul_b_10 = make_float2(x0_3, x1_3);
        float2 gx_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float2 a_2 = fma_f32x2_rn_noftz(gx_2, rr2, make_float2(acc[6], acc[7]));
        acc[6] = a_2.x;
        acc[7] = a_2.y;
        dq[0] = dot;
    }
    float _warp_reduce_11 = dq[0];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
    dq[0] = _warp_reduce_11;
    sums[rg * 8 * 32 + ct] = acc[0];
    sums[(rg * 8 + 1) * 32 + ct] = acc[1];
    sums[(rg * 8 + 2) * 32 + ct] = acc[2];
    sums[(rg * 8 + 3) * 32 + ct] = acc[3];
    sums[(rg * 8 + 4) * 32 + ct] = acc[4];
    sums[(rg * 8 + 5) * 32 + ct] = acc[5];
    sums[(rg * 8 + 6) * 32 + ct] = acc[6];
    sums[(rg * 8 + 7) * 32 + ct] = acc[7];
    __syncthreads();
    float dwo[2];
    int jc_0 = rg * 2;
    float sk_0 = sums[jc_0 * 32 + ct];
    sk_0 = sk_0 + sums[(8 + jc_0) * 32 + ct];
    sk_0 = sk_0 + sums[(16 + jc_0) * 32 + ct];
    sk_0 = sk_0 + sums[(24 + jc_0) * 32 + ct];
    dwo[0] = sk_0;
    int jc_1 = rg * 2 + 1;
    float sk_1 = sums[jc_1 * 32 + ct];
    sk_1 = sk_1 + sums[(8 + jc_1) * 32 + ct];
    sk_1 = sk_1 + sums[(16 + jc_1) * 32 + ct];
    sk_1 = sk_1 + sums[(24 + jc_1) * 32 + ct];
    dwo[1] = sk_1;
    {
        float2 _v2 = make_float2(dwo[0 + 0], dwo[0 + 1]);
        *reinterpret_cast<float2*>(dw + col + rg * 2) = _v2;
    }
    int row_0 = rg;
    if (row_0 < T) {
        float tot_0 = dq[0];
        float rq = rr[0];
        float cq = tot_0 * 0.00390625f;
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        float out[8];
        float g0_4 = __uint_as_float(gw[0] << 16);
        float g1_4 = __uint_as_float(gw[0] & 4294901760u);
        float x0_4 = __uint_as_float(xw[0] << 16);
        float x1_4 = __uint_as_float(xw[0] & 4294901760u);
        float y0_0 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * ww[0]));
        float y1_0 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * ww[1]));
        out[0] = y0_0;
        out[1] = y1_0;
        float g0_5 = __uint_as_float(gw[1] << 16);
        float g1_5 = __uint_as_float(gw[1] & 4294901760u);
        float x0_5 = __uint_as_float(xw[1] << 16);
        float x1_5 = __uint_as_float(xw[1] & 4294901760u);
        float y0_1 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * ww[2]));
        float y1_1 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * ww[3]));
        out[2] = y0_1;
        out[3] = y1_1;
        float g0_6 = __uint_as_float(gw[2] << 16);
        float g1_6 = __uint_as_float(gw[2] & 4294901760u);
        float x0_6 = __uint_as_float(xw[2] << 16);
        float x1_6 = __uint_as_float(xw[2] & 4294901760u);
        float y0_2 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * ww[4]));
        float y1_2 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * ww[5]));
        out[4] = y0_2;
        out[5] = y1_2;
        float g0_7 = __uint_as_float(gw[3] << 16);
        float g1_7 = __uint_as_float(gw[3] & 4294901760u);
        float x0_7 = __uint_as_float(xw[3] << 16);
        float x1_7 = __uint_as_float(xw[3] & 4294901760u);
        float y0_3 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * ww[6]));
        float y1_3 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * ww[7]));
        out[6] = y0_3;
        out[7] = y1_3;
        unsigned int packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_0 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
    }
}

} // extern "C"
