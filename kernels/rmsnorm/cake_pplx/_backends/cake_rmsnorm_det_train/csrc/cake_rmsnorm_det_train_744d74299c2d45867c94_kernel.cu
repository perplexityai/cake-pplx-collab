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

#define LAUNCH_MIN_BLOCKS 2

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_744d74299c2d45867c94(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    int* flags = reinterpret_cast<int*>(smem_raw + 0);
    const int flags_addr = smem + 0;
    float* dots = reinterpret_cast<float*>(smem_raw + 32);
    const int dots_addr = smem + 32;
    float* sums = reinterpret_cast<float*>(smem_raw + 96);
    const int sums_addr = smem + 96;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 96);
    const int dwbuf_addr = smem + 96;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp / 2;
    int tir = tid - rg * 64;
    float ww[16];
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
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + ((unsigned long long)(tir * 8)))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_0_0);
        ww[0 + 1] = __uint_as_float(_ldv8_0_1);
        ww[0 + 2] = __uint_as_float(_ldv8_0_2);
        ww[0 + 3] = __uint_as_float(_ldv8_0_3);
        ww[0 + 4] = __uint_as_float(_ldv8_0_4);
        ww[0 + 5] = __uint_as_float(_ldv8_0_5);
        ww[0 + 6] = __uint_as_float(_ldv8_0_6);
        ww[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    {
        unsigned _ldv8_1_0;
        unsigned _ldv8_1_1;
        unsigned _ldv8_1_2;
        unsigned _ldv8_1_3;
        unsigned _ldv8_1_4;
        unsigned _ldv8_1_5;
        unsigned _ldv8_1_6;
        unsigned _ldv8_1_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((64 + tir) * 8)))) : "memory");
        ww[8 + 0] = __uint_as_float(_ldv8_1_0);
        ww[8 + 1] = __uint_as_float(_ldv8_1_1);
        ww[8 + 2] = __uint_as_float(_ldv8_1_2);
        ww[8 + 3] = __uint_as_float(_ldv8_1_3);
        ww[8 + 4] = __uint_as_float(_ldv8_1_4);
        ww[8 + 5] = __uint_as_float(_ldv8_1_5);
        ww[8 + 6] = __uint_as_float(_ldv8_1_6);
        ww[8 + 7] = __uint_as_float(_ldv8_1_7);
    }
    float acc[16];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    acc[8] = 0.0f;
    acc[9] = 0.0f;
    acc[10] = 0.0f;
    acc[11] = 0.0f;
    acc[12] = 0.0f;
    acc[13] = 0.0f;
    acc[14] = 0.0f;
    acc[15] = 0.0f;
    int c = 0;
    int first_c = 0;
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[1];
    float rr[1];
    float dot[1];
    float wl[16];
    int n_rows = (T + 3) / 4;
    #pragma unroll 1
    for (int k = 0; k < n_rows + 32; k++) {
        int it = (int)bid + k * (int)gridDim.x;
        if (it >= n_rows + 32) {
            break;
        }
        if (it < n_rows) {
            int row = it * 4 + rg;
            int _min_2 = ((row) < (T - 1) ? (row) : (T - 1));
            unsigned long long g_base = (unsigned long long)_min_2 * (unsigned long long)g_stride;
            int _min_3 = ((row) < (T - 1) ? (row) : (T - 1));
            unsigned long long x_base = (unsigned long long)_min_3 * (unsigned long long)x_stride;
            int _min_4 = ((row) < (T - 1) ? (row) : (T - 1));
            rr[0] = r[_min_4];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)(tir * 8));
                uint4* _vdst_5 = reinterpret_cast<uint4*>(&gw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)(tir * 8));
                uint4* _vdst_6 = reinterpret_cast<uint4*>(&xw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((64 + tir) * 8));
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((64 + tir) * 8));
                uint4* _vdst_8 = reinterpret_cast<uint4*>(&xw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                unsigned _ldv8_9_0;
                unsigned _ldv8_9_1;
                unsigned _ldv8_9_2;
                unsigned _ldv8_9_3;
                unsigned _ldv8_9_4;
                unsigned _ldv8_9_5;
                unsigned _ldv8_9_6;
                unsigned _ldv8_9_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_9_0), "=r"(_ldv8_9_1), "=r"(_ldv8_9_2), "=r"(_ldv8_9_3), "=r"(_ldv8_9_4), "=r"(_ldv8_9_5), "=r"(_ldv8_9_6), "=r"(_ldv8_9_7) : "l"((const void*)(w + ((unsigned long long)(tir * 8)))) : "memory");
                wl[0 + 0] = __uint_as_float(_ldv8_9_0);
                wl[0 + 1] = __uint_as_float(_ldv8_9_1);
                wl[0 + 2] = __uint_as_float(_ldv8_9_2);
                wl[0 + 3] = __uint_as_float(_ldv8_9_3);
                wl[0 + 4] = __uint_as_float(_ldv8_9_4);
                wl[0 + 5] = __uint_as_float(_ldv8_9_5);
                wl[0 + 6] = __uint_as_float(_ldv8_9_6);
                wl[0 + 7] = __uint_as_float(_ldv8_9_7);
            }
            {
                unsigned _ldv8_10_0;
                unsigned _ldv8_10_1;
                unsigned _ldv8_10_2;
                unsigned _ldv8_10_3;
                unsigned _ldv8_10_4;
                unsigned _ldv8_10_5;
                unsigned _ldv8_10_6;
                unsigned _ldv8_10_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_10_0), "=r"(_ldv8_10_1), "=r"(_ldv8_10_2), "=r"(_ldv8_10_3), "=r"(_ldv8_10_4), "=r"(_ldv8_10_5), "=r"(_ldv8_10_6), "=r"(_ldv8_10_7) : "l"((const void*)(w + ((unsigned long long)((64 + tir) * 8)))) : "memory");
                wl[8 + 0] = __uint_as_float(_ldv8_10_0);
                wl[8 + 1] = __uint_as_float(_ldv8_10_1);
                wl[8 + 2] = __uint_as_float(_ldv8_10_2);
                wl[8 + 3] = __uint_as_float(_ldv8_10_3);
                wl[8 + 4] = __uint_as_float(_ldv8_10_4);
                wl[8 + 5] = __uint_as_float(_ldv8_10_5);
                wl[8 + 6] = __uint_as_float(_ldv8_10_6);
                wl[8 + 7] = __uint_as_float(_ldv8_10_7);
            }
            float dq = 0.0f;
            dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * wl[0], __uint_as_float(xw[0] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * wl[1], __uint_as_float(xw[0] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * wl[2], __uint_as_float(xw[1] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * wl[3], __uint_as_float(xw[1] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * wl[4], __uint_as_float(xw[2] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * wl[5], __uint_as_float(xw[2] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * wl[6], __uint_as_float(xw[3] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * wl[7], __uint_as_float(xw[3] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * wl[8], __uint_as_float(xw[4] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * wl[9], __uint_as_float(xw[4] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * wl[10], __uint_as_float(xw[5] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * wl[11], __uint_as_float(xw[5] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * wl[12], __uint_as_float(xw[6] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * wl[13], __uint_as_float(xw[6] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * wl[14], __uint_as_float(xw[7] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * wl[15], __uint_as_float(xw[7] & 4294901760u), dq);
            float _warp_reduce_11 = dq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
            dot[0] = _warp_reduce_11;
            if (lane == 0) {
                dots[(k & 1) * 8 + warp] = dot[0];
            }
            __syncthreads();
            float tot = 0.0f;
            tot += dots[(k & 1) * 8 + rg * 2];
            tot += dots[(k & 1) * 8 + rg * 2 + 1];
            dot[0] = tot;
            float rq = rr[0];
            float cq = __fdiv_rn(dot[0], 1024.0f);
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            if (row < T) {
                float out_0[8];
                float g0_0 = __uint_as_float(gw[0] << 16);
                float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                float x0_0 = __uint_as_float(xw[0] << 16);
                float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * wl[0]));
                float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * wl[1]));
                out_0[0] = y0_0;
                out_0[1] = y1_0;
                float g0_1 = __uint_as_float(gw[1] << 16);
                float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                float x0_1 = __uint_as_float(xw[1] << 16);
                float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * wl[2]));
                float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * wl[3]));
                out_0[2] = y0_1;
                out_0[3] = y1_1;
                float g0_2 = __uint_as_float(gw[2] << 16);
                float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                float x0_2 = __uint_as_float(xw[2] << 16);
                float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * wl[4]));
                float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * wl[5]));
                out_0[4] = y0_2;
                out_0[5] = y1_2;
                float g0_3 = __uint_as_float(gw[3] << 16);
                float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                float x0_3 = __uint_as_float(xw[3] << 16);
                float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * wl[6]));
                float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * wl[7]));
                out_0[6] = y0_3;
                out_0[7] = y1_3;
                unsigned int packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 1024ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                float out_1[8];
                float g0_4 = __uint_as_float(gw[4] << 16);
                float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                float x0_4 = __uint_as_float(xw[4] << 16);
                float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * wl[8]));
                float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * wl[9]));
                out_1[0] = y0_4;
                out_1[1] = y1_4;
                float g0_5 = __uint_as_float(gw[5] << 16);
                float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                float x0_5 = __uint_as_float(xw[5] << 16);
                float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * wl[10]));
                float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * wl[11]));
                out_1[2] = y0_5;
                out_1[3] = y1_5;
                float g0_6 = __uint_as_float(gw[6] << 16);
                float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                float x0_6 = __uint_as_float(xw[6] << 16);
                float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * wl[12]));
                float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * wl[13]));
                out_1[4] = y0_6;
                out_1[5] = y1_6;
                float g0_7 = __uint_as_float(gw[7] << 16);
                float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                float x0_7 = __uint_as_float(xw[7] << 16);
                float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * wl[14]));
                float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * wl[15]));
                out_1[6] = y0_7;
                out_1[7] = y1_7;
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 1024ull + (unsigned long long)((64 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        } else {
            int vl = tid % 4;
            int rl = tid / 4;
            unsigned int gw_0[32];
            unsigned int xw_0[32];
            unsigned int hw_0[1];
            float rr_0[8];
            float acc_0[8];
            acc_0[0] = 0.0f;
            acc_0[1] = 0.0f;
            acc_0[2] = 0.0f;
            acc_0[3] = 0.0f;
            acc_0[4] = 0.0f;
            acc_0[5] = 0.0f;
            acc_0[6] = 0.0f;
            acc_0[7] = 0.0f;
            #pragma unroll 1
            for (int t0 = rl; t0 < T; t0 += 512) {
                if (t0 < T) {
                    unsigned long long g_base_1 = (unsigned long long)t0 * (unsigned long long)g_stride;
                    unsigned long long x_base_1 = (unsigned long long)t0 * (unsigned long long)x_stride;
                    rr_0[0] = r[t0];
                    {
                        const uint4* _vptr_12 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_12 = reinterpret_cast<uint4*>(&gw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_13 = reinterpret_cast<uint4*>(&xw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 64 < T) {
                    unsigned long long g_base_2 = (unsigned long long)(t0 + 64) * (unsigned long long)g_stride;
                    unsigned long long x_base_2 = (unsigned long long)(t0 + 64) * (unsigned long long)x_stride;
                    rr_0[1] = r[t0 + 64];
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&gw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_15 = reinterpret_cast<uint4*>(&xw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 128 < T) {
                    unsigned long long g_base_3 = (unsigned long long)(t0 + 128) * (unsigned long long)g_stride;
                    unsigned long long x_base_3 = (unsigned long long)(t0 + 128) * (unsigned long long)x_stride;
                    rr_0[2] = r[t0 + 128];
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw_0[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw_0[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 192 < T) {
                    unsigned long long g_base_4 = (unsigned long long)(t0 + 192) * (unsigned long long)g_stride;
                    unsigned long long x_base_4 = (unsigned long long)(t0 + 192) * (unsigned long long)x_stride;
                    rr_0[3] = r[t0 + 192];
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw_0[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw_0[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 256 < T) {
                    unsigned long long g_base_5 = (unsigned long long)(t0 + 256) * (unsigned long long)g_stride;
                    unsigned long long x_base_5 = (unsigned long long)(t0 + 256) * (unsigned long long)x_stride;
                    rr_0[4] = r[t0 + 256];
                    {
                        const uint4* _vptr_20 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_20 = reinterpret_cast<uint4*>(&gw_0[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_20[_blk].x), "=r"(_vdst_20[_blk].y), "=r"(_vdst_20[_blk].z), "=r"(_vdst_20[_blk].w) : "l"((const void*)(_vptr_20 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_21 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_21 = reinterpret_cast<uint4*>(&xw_0[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_21[_blk].x), "=r"(_vdst_21[_blk].y), "=r"(_vdst_21[_blk].z), "=r"(_vdst_21[_blk].w) : "l"((const void*)(_vptr_21 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 320 < T) {
                    unsigned long long g_base_6 = (unsigned long long)(t0 + 320) * (unsigned long long)g_stride;
                    unsigned long long x_base_6 = (unsigned long long)(t0 + 320) * (unsigned long long)x_stride;
                    rr_0[5] = r[t0 + 320];
                    {
                        const uint4* _vptr_22 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_22 = reinterpret_cast<uint4*>(&gw_0[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_22[_blk].x), "=r"(_vdst_22[_blk].y), "=r"(_vdst_22[_blk].z), "=r"(_vdst_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&xw_0[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_23[_blk].x), "=r"(_vdst_23[_blk].y), "=r"(_vdst_23[_blk].z), "=r"(_vdst_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 384 < T) {
                    unsigned long long g_base_7 = (unsigned long long)(t0 + 384) * (unsigned long long)g_stride;
                    unsigned long long x_base_7 = (unsigned long long)(t0 + 384) * (unsigned long long)x_stride;
                    rr_0[6] = r[t0 + 384];
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(g + g_base_7 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&gw_0[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_24[_blk].x), "=r"(_vdst_24[_blk].y), "=r"(_vdst_24[_blk].z), "=r"(_vdst_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(x + x_base_7 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&xw_0[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 448 < T) {
                    unsigned long long g_base_8 = (unsigned long long)(t0 + 448) * (unsigned long long)g_stride;
                    unsigned long long x_base_8 = (unsigned long long)(t0 + 448) * (unsigned long long)x_stride;
                    rr_0[7] = r[t0 + 448];
                    {
                        const uint4* _vptr_26 = reinterpret_cast<const uint4*>(g + g_base_8 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_26 = reinterpret_cast<uint4*>(&gw_0[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_26[_blk].x), "=r"(_vdst_26[_blk].y), "=r"(_vdst_26[_blk].z), "=r"(_vdst_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_27 = reinterpret_cast<const uint4*>(x + x_base_8 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_27 = reinterpret_cast<uint4*>(&xw_0[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_27[_blk].x), "=r"(_vdst_27[_blk].y), "=r"(_vdst_27[_blk].z), "=r"(_vdst_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 < T) {
                    float2 rr2_0 = make_float2(rr_0[0], rr_0[0]);
                    float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(gw_0[0] << 16), __uint_as_float(gw_0[0] & 4294901760u));
                    float2 _f32x2_mul_b_29 = make_float2(__uint_as_float(xw_0[0] << 16), __uint_as_float(xw_0[0] & 4294901760u));
                    float2 gx;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                    float2 a = fma_f32x2_rn_noftz(gx, rr2_0, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a.x;
                    acc_0[1] = a.y;
                    float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(gw_0[1] << 16), __uint_as_float(gw_0[1] & 4294901760u));
                    float2 _f32x2_mul_b_31 = make_float2(__uint_as_float(xw_0[1] << 16), __uint_as_float(xw_0[1] & 4294901760u));
                    float2 gx_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                    float2 a_0 = fma_f32x2_rn_noftz(gx_0, rr2_0, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_0.x;
                    acc_0[3] = a_0.y;
                    float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(gw_0[2] << 16), __uint_as_float(gw_0[2] & 4294901760u));
                    float2 _f32x2_mul_b_33 = make_float2(__uint_as_float(xw_0[2] << 16), __uint_as_float(xw_0[2] & 4294901760u));
                    float2 gx_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
                    float2 a_1 = fma_f32x2_rn_noftz(gx_1, rr2_0, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_1.x;
                    acc_0[5] = a_1.y;
                    float2 _f32x2_mul_a_34 = make_float2(__uint_as_float(gw_0[3] << 16), __uint_as_float(gw_0[3] & 4294901760u));
                    float2 _f32x2_mul_b_35 = make_float2(__uint_as_float(xw_0[3] << 16), __uint_as_float(xw_0[3] & 4294901760u));
                    float2 gx_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                    float2 a_2 = fma_f32x2_rn_noftz(gx_2, rr2_0, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_2.x;
                    acc_0[7] = a_2.y;
                }
                if (t0 + 64 < T) {
                    float2 rr2_1 = make_float2(rr_0[1], rr_0[1]);
                    float2 _f32x2_mul_a_36 = make_float2(__uint_as_float(gw_0[4] << 16), __uint_as_float(gw_0[4] & 4294901760u));
                    float2 _f32x2_mul_b_37 = make_float2(__uint_as_float(xw_0[4] << 16), __uint_as_float(xw_0[4] & 4294901760u));
                    float2 gx_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                    float2 a_3 = fma_f32x2_rn_noftz(gx_3, rr2_1, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_3.x;
                    acc_0[1] = a_3.y;
                    float2 _f32x2_mul_a_38 = make_float2(__uint_as_float(gw_0[5] << 16), __uint_as_float(gw_0[5] & 4294901760u));
                    float2 _f32x2_mul_b_39 = make_float2(__uint_as_float(xw_0[5] << 16), __uint_as_float(xw_0[5] & 4294901760u));
                    float2 gx_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
                    float2 a_3_1 = fma_f32x2_rn_noftz(gx_3_1, rr2_1, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_3_1.x;
                    acc_0[3] = a_3_1.y;
                    float2 _f32x2_mul_a_40 = make_float2(__uint_as_float(gw_0[6] << 16), __uint_as_float(gw_0[6] & 4294901760u));
                    float2 _f32x2_mul_b_41 = make_float2(__uint_as_float(xw_0[6] << 16), __uint_as_float(xw_0[6] & 4294901760u));
                    float2 gx_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
                    float2 a_4 = fma_f32x2_rn_noftz(gx_4, rr2_1, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_4.x;
                    acc_0[5] = a_4.y;
                    float2 _f32x2_mul_a_42 = make_float2(__uint_as_float(gw_0[7] << 16), __uint_as_float(gw_0[7] & 4294901760u));
                    float2 _f32x2_mul_b_43 = make_float2(__uint_as_float(xw_0[7] << 16), __uint_as_float(xw_0[7] & 4294901760u));
                    float2 gx_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                    float2 a_5 = fma_f32x2_rn_noftz(gx_5, rr2_1, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_5.x;
                    acc_0[7] = a_5.y;
                }
                if (t0 + 128 < T) {
                    float2 rr2_2 = make_float2(rr_0[2], rr_0[2]);
                    float2 _f32x2_mul_a_44 = make_float2(__uint_as_float(gw_0[8] << 16), __uint_as_float(gw_0[8] & 4294901760u));
                    float2 _f32x2_mul_b_45 = make_float2(__uint_as_float(xw_0[8] << 16), __uint_as_float(xw_0[8] & 4294901760u));
                    float2 gx_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
                    float2 a_6 = fma_f32x2_rn_noftz(gx_6, rr2_2, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_6.x;
                    acc_0[1] = a_6.y;
                    float2 _f32x2_mul_a_46 = make_float2(__uint_as_float(gw_0[9] << 16), __uint_as_float(gw_0[9] & 4294901760u));
                    float2 _f32x2_mul_b_47 = make_float2(__uint_as_float(xw_0[9] << 16), __uint_as_float(xw_0[9] & 4294901760u));
                    float2 gx_6_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                    float2 a_6_1 = fma_f32x2_rn_noftz(gx_6_1, rr2_2, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_6_1.x;
                    acc_0[3] = a_6_1.y;
                    float2 _f32x2_mul_a_48 = make_float2(__uint_as_float(gw_0[10] << 16), __uint_as_float(gw_0[10] & 4294901760u));
                    float2 _f32x2_mul_b_49 = make_float2(__uint_as_float(xw_0[10] << 16), __uint_as_float(xw_0[10] & 4294901760u));
                    float2 gx_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                    float2 a_7 = fma_f32x2_rn_noftz(gx_7, rr2_2, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_7.x;
                    acc_0[5] = a_7.y;
                    float2 _f32x2_mul_a_50 = make_float2(__uint_as_float(gw_0[11] << 16), __uint_as_float(gw_0[11] & 4294901760u));
                    float2 _f32x2_mul_b_51 = make_float2(__uint_as_float(xw_0[11] << 16), __uint_as_float(xw_0[11] & 4294901760u));
                    float2 gx_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
                    float2 a_8 = fma_f32x2_rn_noftz(gx_8, rr2_2, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_8.x;
                    acc_0[7] = a_8.y;
                }
                if (t0 + 192 < T) {
                    float2 rr2_3 = make_float2(rr_0[3], rr_0[3]);
                    float2 _f32x2_mul_a_52 = make_float2(__uint_as_float(gw_0[12] << 16), __uint_as_float(gw_0[12] & 4294901760u));
                    float2 _f32x2_mul_b_53 = make_float2(__uint_as_float(xw_0[12] << 16), __uint_as_float(xw_0[12] & 4294901760u));
                    float2 gx_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
                    float2 a_9 = fma_f32x2_rn_noftz(gx_9, rr2_3, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_9.x;
                    acc_0[1] = a_9.y;
                    float2 _f32x2_mul_a_54 = make_float2(__uint_as_float(gw_0[13] << 16), __uint_as_float(gw_0[13] & 4294901760u));
                    float2 _f32x2_mul_b_55 = make_float2(__uint_as_float(xw_0[13] << 16), __uint_as_float(xw_0[13] & 4294901760u));
                    float2 gx_9_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
                    float2 a_9_1 = fma_f32x2_rn_noftz(gx_9_1, rr2_3, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_9_1.x;
                    acc_0[3] = a_9_1.y;
                    float2 _f32x2_mul_a_56 = make_float2(__uint_as_float(gw_0[14] << 16), __uint_as_float(gw_0[14] & 4294901760u));
                    float2 _f32x2_mul_b_57 = make_float2(__uint_as_float(xw_0[14] << 16), __uint_as_float(xw_0[14] & 4294901760u));
                    float2 gx_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
                    float2 a_10 = fma_f32x2_rn_noftz(gx_10, rr2_3, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_10.x;
                    acc_0[5] = a_10.y;
                    float2 _f32x2_mul_a_58 = make_float2(__uint_as_float(gw_0[15] << 16), __uint_as_float(gw_0[15] & 4294901760u));
                    float2 _f32x2_mul_b_59 = make_float2(__uint_as_float(xw_0[15] << 16), __uint_as_float(xw_0[15] & 4294901760u));
                    float2 gx_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
                    float2 a_11 = fma_f32x2_rn_noftz(gx_11, rr2_3, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_11.x;
                    acc_0[7] = a_11.y;
                }
                if (t0 + 256 < T) {
                    float2 rr2_4 = make_float2(rr_0[4], rr_0[4]);
                    float2 _f32x2_mul_a_60 = make_float2(__uint_as_float(gw_0[16] << 16), __uint_as_float(gw_0[16] & 4294901760u));
                    float2 _f32x2_mul_b_61 = make_float2(__uint_as_float(xw_0[16] << 16), __uint_as_float(xw_0[16] & 4294901760u));
                    float2 gx_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
                    float2 a_12 = fma_f32x2_rn_noftz(gx_12, rr2_4, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_12.x;
                    acc_0[1] = a_12.y;
                    float2 _f32x2_mul_a_62 = make_float2(__uint_as_float(gw_0[17] << 16), __uint_as_float(gw_0[17] & 4294901760u));
                    float2 _f32x2_mul_b_63 = make_float2(__uint_as_float(xw_0[17] << 16), __uint_as_float(xw_0[17] & 4294901760u));
                    float2 gx_12_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
                    float2 a_12_1 = fma_f32x2_rn_noftz(gx_12_1, rr2_4, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_12_1.x;
                    acc_0[3] = a_12_1.y;
                    float2 _f32x2_mul_a_64 = make_float2(__uint_as_float(gw_0[18] << 16), __uint_as_float(gw_0[18] & 4294901760u));
                    float2 _f32x2_mul_b_65 = make_float2(__uint_as_float(xw_0[18] << 16), __uint_as_float(xw_0[18] & 4294901760u));
                    float2 gx_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                    float2 a_13 = fma_f32x2_rn_noftz(gx_13, rr2_4, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_13.x;
                    acc_0[5] = a_13.y;
                    float2 _f32x2_mul_a_66 = make_float2(__uint_as_float(gw_0[19] << 16), __uint_as_float(gw_0[19] & 4294901760u));
                    float2 _f32x2_mul_b_67 = make_float2(__uint_as_float(xw_0[19] << 16), __uint_as_float(xw_0[19] & 4294901760u));
                    float2 gx_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
                    float2 a_14 = fma_f32x2_rn_noftz(gx_14, rr2_4, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_14.x;
                    acc_0[7] = a_14.y;
                }
                if (t0 + 320 < T) {
                    float2 rr2_5 = make_float2(rr_0[5], rr_0[5]);
                    float2 _f32x2_mul_a_68 = make_float2(__uint_as_float(gw_0[20] << 16), __uint_as_float(gw_0[20] & 4294901760u));
                    float2 _f32x2_mul_b_69 = make_float2(__uint_as_float(xw_0[20] << 16), __uint_as_float(xw_0[20] & 4294901760u));
                    float2 gx_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                    float2 a_15 = fma_f32x2_rn_noftz(gx_15, rr2_5, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_15.x;
                    acc_0[1] = a_15.y;
                    float2 _f32x2_mul_a_70 = make_float2(__uint_as_float(gw_0[21] << 16), __uint_as_float(gw_0[21] & 4294901760u));
                    float2 _f32x2_mul_b_71 = make_float2(__uint_as_float(xw_0[21] << 16), __uint_as_float(xw_0[21] & 4294901760u));
                    float2 gx_15_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                    float2 a_15_1 = fma_f32x2_rn_noftz(gx_15_1, rr2_5, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_15_1.x;
                    acc_0[3] = a_15_1.y;
                    float2 _f32x2_mul_a_72 = make_float2(__uint_as_float(gw_0[22] << 16), __uint_as_float(gw_0[22] & 4294901760u));
                    float2 _f32x2_mul_b_73 = make_float2(__uint_as_float(xw_0[22] << 16), __uint_as_float(xw_0[22] & 4294901760u));
                    float2 gx_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                    float2 a_16 = fma_f32x2_rn_noftz(gx_16, rr2_5, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_16.x;
                    acc_0[5] = a_16.y;
                    float2 _f32x2_mul_a_74 = make_float2(__uint_as_float(gw_0[23] << 16), __uint_as_float(gw_0[23] & 4294901760u));
                    float2 _f32x2_mul_b_75 = make_float2(__uint_as_float(xw_0[23] << 16), __uint_as_float(xw_0[23] & 4294901760u));
                    float2 gx_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                    float2 a_17 = fma_f32x2_rn_noftz(gx_17, rr2_5, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_17.x;
                    acc_0[7] = a_17.y;
                }
                if (t0 + 384 < T) {
                    float2 rr2_6 = make_float2(rr_0[6], rr_0[6]);
                    float2 _f32x2_mul_a_76 = make_float2(__uint_as_float(gw_0[24] << 16), __uint_as_float(gw_0[24] & 4294901760u));
                    float2 _f32x2_mul_b_77 = make_float2(__uint_as_float(xw_0[24] << 16), __uint_as_float(xw_0[24] & 4294901760u));
                    float2 gx_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                    float2 a_18 = fma_f32x2_rn_noftz(gx_18, rr2_6, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_18.x;
                    acc_0[1] = a_18.y;
                    float2 _f32x2_mul_a_78 = make_float2(__uint_as_float(gw_0[25] << 16), __uint_as_float(gw_0[25] & 4294901760u));
                    float2 _f32x2_mul_b_79 = make_float2(__uint_as_float(xw_0[25] << 16), __uint_as_float(xw_0[25] & 4294901760u));
                    float2 gx_18_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
                    float2 a_18_1 = fma_f32x2_rn_noftz(gx_18_1, rr2_6, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_18_1.x;
                    acc_0[3] = a_18_1.y;
                    float2 _f32x2_mul_a_80 = make_float2(__uint_as_float(gw_0[26] << 16), __uint_as_float(gw_0[26] & 4294901760u));
                    float2 _f32x2_mul_b_81 = make_float2(__uint_as_float(xw_0[26] << 16), __uint_as_float(xw_0[26] & 4294901760u));
                    float2 gx_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
                    float2 a_19 = fma_f32x2_rn_noftz(gx_19, rr2_6, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_19.x;
                    acc_0[5] = a_19.y;
                    float2 _f32x2_mul_a_82 = make_float2(__uint_as_float(gw_0[27] << 16), __uint_as_float(gw_0[27] & 4294901760u));
                    float2 _f32x2_mul_b_83 = make_float2(__uint_as_float(xw_0[27] << 16), __uint_as_float(xw_0[27] & 4294901760u));
                    float2 gx_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                    float2 a_20 = fma_f32x2_rn_noftz(gx_20, rr2_6, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_20.x;
                    acc_0[7] = a_20.y;
                }
                if (t0 + 448 < T) {
                    float2 rr2_7 = make_float2(rr_0[7], rr_0[7]);
                    float2 _f32x2_mul_a_84 = make_float2(__uint_as_float(gw_0[28] << 16), __uint_as_float(gw_0[28] & 4294901760u));
                    float2 _f32x2_mul_b_85 = make_float2(__uint_as_float(xw_0[28] << 16), __uint_as_float(xw_0[28] & 4294901760u));
                    float2 gx_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
                    float2 a_21 = fma_f32x2_rn_noftz(gx_21, rr2_7, make_float2(acc_0[0], acc_0[1]));
                    acc_0[0] = a_21.x;
                    acc_0[1] = a_21.y;
                    float2 _f32x2_mul_a_86 = make_float2(__uint_as_float(gw_0[29] << 16), __uint_as_float(gw_0[29] & 4294901760u));
                    float2 _f32x2_mul_b_87 = make_float2(__uint_as_float(xw_0[29] << 16), __uint_as_float(xw_0[29] & 4294901760u));
                    float2 gx_21_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
                    float2 a_21_1 = fma_f32x2_rn_noftz(gx_21_1, rr2_7, make_float2(acc_0[2], acc_0[3]));
                    acc_0[2] = a_21_1.x;
                    acc_0[3] = a_21_1.y;
                    float2 _f32x2_mul_a_88 = make_float2(__uint_as_float(gw_0[30] << 16), __uint_as_float(gw_0[30] & 4294901760u));
                    float2 _f32x2_mul_b_89 = make_float2(__uint_as_float(xw_0[30] << 16), __uint_as_float(xw_0[30] & 4294901760u));
                    float2 gx_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
                    float2 a_22 = fma_f32x2_rn_noftz(gx_22, rr2_7, make_float2(acc_0[4], acc_0[5]));
                    acc_0[4] = a_22.x;
                    acc_0[5] = a_22.y;
                    float2 _f32x2_mul_a_90 = make_float2(__uint_as_float(gw_0[31] << 16), __uint_as_float(gw_0[31] & 4294901760u));
                    float2 _f32x2_mul_b_91 = make_float2(__uint_as_float(xw_0[31] << 16), __uint_as_float(xw_0[31] & 4294901760u));
                    float2 gx_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
                    float2 a_23 = fma_f32x2_rn_noftz(gx_23, rr2_7, make_float2(acc_0[6], acc_0[7]));
                    acc_0[6] = a_23.x;
                    acc_0[7] = a_23.y;
                }
            }
            if (rl < T) {
                sums[rl * 32 + vl * 8] = acc_0[0];
                sums[rl * 32 + vl * 8 + 1] = acc_0[1];
                sums[rl * 32 + vl * 8 + 2] = acc_0[2];
                sums[rl * 32 + vl * 8 + 3] = acc_0[3];
                sums[rl * 32 + vl * 8 + 4] = acc_0[4];
                sums[rl * 32 + vl * 8 + 5] = acc_0[5];
                sums[rl * 32 + vl * 8 + 6] = acc_0[6];
                sums[rl * 32 + vl * 8 + 7] = acc_0[7];
            }
            __syncthreads();
            int cl = tid % 32;
            int s8 = tid / 32 * 8;
            if (s8 < T) {
                float lv8[8];
                lv8[0] = 0.0f;
                if (s8 < T) {
                    lv8[0] = sums[s8 * 32 + cl];
                }
                lv8[1] = 0.0f;
                if (s8 + 1 < T) {
                    lv8[1] = sums[(s8 + 1) * 32 + cl];
                }
                lv8[2] = 0.0f;
                if (s8 + 2 < T) {
                    lv8[2] = sums[(s8 + 2) * 32 + cl];
                }
                lv8[3] = 0.0f;
                if (s8 + 3 < T) {
                    lv8[3] = sums[(s8 + 3) * 32 + cl];
                }
                lv8[4] = 0.0f;
                if (s8 + 4 < T) {
                    lv8[4] = sums[(s8 + 4) * 32 + cl];
                }
                lv8[5] = 0.0f;
                if (s8 + 5 < T) {
                    lv8[5] = sums[(s8 + 5) * 32 + cl];
                }
                lv8[6] = 0.0f;
                if (s8 + 6 < T) {
                    lv8[6] = sums[(s8 + 6) * 32 + cl];
                }
                lv8[7] = 0.0f;
                if (s8 + 7 < T) {
                    lv8[7] = sums[(s8 + 7) * 32 + cl];
                }
                if (s8 + 1 < T) {
                    lv8[0] = lv8[0] + lv8[1];
                }
                if (s8 + 3 < T) {
                    lv8[2] = lv8[2] + lv8[3];
                }
                if (s8 + 5 < T) {
                    lv8[4] = lv8[4] + lv8[5];
                }
                if (s8 + 7 < T) {
                    lv8[6] = lv8[6] + lv8[7];
                }
                if (s8 + 2 < T) {
                    lv8[0] = lv8[0] + lv8[2];
                }
                if (s8 + 6 < T) {
                    lv8[4] = lv8[4] + lv8[6];
                }
                if (s8 + 4 < T) {
                    lv8[0] = lv8[0] + lv8[4];
                }
                sums[s8 * 32 + cl] = lv8[0];
            }
            __syncthreads();
            if (tid < 32) {
                float gv[8];
                gv[0] = 0.0f;
                if (0 < T) {
                    gv[0] = sums[cl];
                }
                gv[1] = 0.0f;
                if (8 < T) {
                    gv[1] = sums[256 + cl];
                }
                gv[2] = 0.0f;
                if (16 < T) {
                    gv[2] = sums[512 + cl];
                }
                gv[3] = 0.0f;
                if (24 < T) {
                    gv[3] = sums[768 + cl];
                }
                gv[4] = 0.0f;
                if (32 < T) {
                    gv[4] = sums[1024 + cl];
                }
                gv[5] = 0.0f;
                if (40 < T) {
                    gv[5] = sums[1280 + cl];
                }
                gv[6] = 0.0f;
                if (48 < T) {
                    gv[6] = sums[1536 + cl];
                }
                gv[7] = 0.0f;
                if (56 < T) {
                    gv[7] = sums[1792 + cl];
                }
                if (8 < T) {
                    gv[0] = gv[0] + gv[1];
                }
                if (24 < T) {
                    gv[2] = gv[2] + gv[3];
                }
                if (40 < T) {
                    gv[4] = gv[4] + gv[5];
                }
                if (56 < T) {
                    gv[6] = gv[6] + gv[7];
                }
                if (16 < T) {
                    gv[0] = gv[0] + gv[2];
                }
                if (48 < T) {
                    gv[4] = gv[4] + gv[6];
                }
                if (32 < T) {
                    gv[0] = gv[0] + gv[4];
                }
                *(reinterpret_cast<float*>(dw) + ((it - n_rows) * 32 + cl)) = gv[0];
            }
        }
        __syncthreads();
    }
}

} // extern "C"
