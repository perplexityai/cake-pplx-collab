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

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_7c0dd63f0b011049603e(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial));
    __builtin_assume(__isGlobal(partial));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 0);
    const int flags_addr = smem + 0;
    float* dots = reinterpret_cast<float*>(smem_raw + 32);
    const int dots_addr = smem + 32;
    float* sums = reinterpret_cast<float*>(smem_raw + 80);
    const int sums_addr = smem + 80;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 80);
    const int dwbuf_addr = smem + 80;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp / 3;
    int tir = tid - rg * 96;
    float ww[16];
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
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)(tir * 8)))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_1_0);
        ww[0 + 1] = __uint_as_float(_ldv8_1_1);
        ww[0 + 2] = __uint_as_float(_ldv8_1_2);
        ww[0 + 3] = __uint_as_float(_ldv8_1_3);
        ww[0 + 4] = __uint_as_float(_ldv8_1_4);
        ww[0 + 5] = __uint_as_float(_ldv8_1_5);
        ww[0 + 6] = __uint_as_float(_ldv8_1_6);
        ww[0 + 7] = __uint_as_float(_ldv8_1_7);
    }
    {
        unsigned _ldv8_2_0;
        unsigned _ldv8_2_1;
        unsigned _ldv8_2_2;
        unsigned _ldv8_2_3;
        unsigned _ldv8_2_4;
        unsigned _ldv8_2_5;
        unsigned _ldv8_2_6;
        unsigned _ldv8_2_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + ((unsigned long long)((96 + tir) * 8)))) : "memory");
        ww[8 + 0] = __uint_as_float(_ldv8_2_0);
        ww[8 + 1] = __uint_as_float(_ldv8_2_1);
        ww[8 + 2] = __uint_as_float(_ldv8_2_2);
        ww[8 + 3] = __uint_as_float(_ldv8_2_3);
        ww[8 + 4] = __uint_as_float(_ldv8_2_4);
        ww[8 + 5] = __uint_as_float(_ldv8_2_5);
        ww[8 + 6] = __uint_as_float(_ldv8_2_6);
        ww[8 + 7] = __uint_as_float(_ldv8_2_7);
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
    sums[rg * 1536 + tir] = 0.0f;
    sums[rg * 1536 + 96 + tir] = 0.0f;
    sums[rg * 1536 + 192 + tir] = 0.0f;
    sums[rg * 1536 + 288 + tir] = 0.0f;
    sums[rg * 1536 + 384 + tir] = 0.0f;
    sums[rg * 1536 + 480 + tir] = 0.0f;
    sums[rg * 1536 + 576 + tir] = 0.0f;
    sums[rg * 1536 + 672 + tir] = 0.0f;
    sums[rg * 1536 + 768 + tir] = 0.0f;
    sums[rg * 1536 + 864 + tir] = 0.0f;
    sums[rg * 1536 + 960 + tir] = 0.0f;
    sums[rg * 1536 + 1056 + tir] = 0.0f;
    sums[rg * 1536 + 1152 + tir] = 0.0f;
    sums[rg * 1536 + 1248 + tir] = 0.0f;
    sums[rg * 1536 + 1344 + tir] = 0.0f;
    sums[rg * 1536 + 1440 + tir] = 0.0f;
    unsigned int gw[16];
    unsigned int xw[16];
    unsigned int hw[1];
    float rr[2];
    if (tid == 0) {
        flags[4] = (int)nxt;
    }
    __syncthreads();
    c = flags[4];
    first_c = c;
    int row_begin = 0;
    int row_end = 0;
    int n_iters = 0;
    int last_row = 0;
    int window = 0;
    #pragma unroll 1
    for (int claim_it = 0; claim_it < n_chunks; claim_it++) {
        if (c >= n_chunks) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_3;
        }
        int stages = (T + 1) / 2;
        int sl = rows_per_chunk;
        if (sl > stages) {
            sl = stages;
        }
        if (sl < 1) {
            sl = 1;
        }
        int share = (stages + sl - 1) / sl;
        int tail = (share + 11) / 12;
        if (tail < 1) {
            tail = 1;
        }
        int rem = stages;
        int size = (share + 1) / 2;
        int first = 0;
        int base = 0;
        int span = 0;
        int halving = 1;
        int begin_it = 0;
        #pragma unroll 1
        for (int k = 0; k < 5; k++) {
            if (halving != 0) {
                if (size > tail) {
                    if (rem >= sl * size) {
                        if (c < first + sl) {
                            span = size;
                            begin_it = base + (c - first) * size;
                            halving = 0;
                        } else {
                            first = first + sl;
                            base = base + sl * size;
                            rem = rem - sl * size;
                            size = (size + 1) / 2;
                        }
                    } else {
                        halving = 0;
                    }
                } else {
                    halving = 0;
                }
            }
        }
        if (span == 0) {
            int quot = rem / sl;
            int extra = rem - quot * sl;
            if (c < first + extra) {
                span = quot + 1;
                begin_it = base + (c - first) * (quot + 1);
            } else if (c < first + sl) {
                span = quot;
                begin_it = base + extra * (quot + 1) + (c - first - extra) * quot;
            } else {
                begin_it = stages;
            }
        }
        int _min_4 = ((begin_it * 2) < (T) ? (begin_it * 2) : (T));
        row_begin = _min_4;
        int _min_5 = ((begin_it * 2 + span * 2) < (T) ? (begin_it * 2 + span * 2) : (T));
        row_end = _min_5;
        n_iters = (row_end - row_begin + 1) / 2;
        last_row = row_end - 1;
        if (n_iters > 0) {
            int _min_6 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long g_base = (unsigned long long)_min_6 * (unsigned long long)g_stride;
            int _min_7 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long x_base = (unsigned long long)_min_7 * (unsigned long long)x_stride;
            int _min_8 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            rr[0] = r[_min_8];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)(tir * 8));
                uint4* _vdst_9 = reinterpret_cast<uint4*>(&gw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)(tir * 8));
                uint4* _vdst_10 = reinterpret_cast<uint4*>(&xw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((96 + tir) * 8));
                uint4* _vdst_11 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((96 + tir) * 8));
                uint4* _vdst_12 = reinterpret_cast<uint4*>(&xw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int it0 = 0; it0 < n_iters; it0 += 2) {
            if (n_iters > it0) {
                if (n_iters > it0 + 1) {
                    int _min_13 = ((row_begin + (it0 + 2 - 1) * 2 + rg) < (last_row) ? (row_begin + (it0 + 2 - 1) * 2 + rg) : (last_row));
                    unsigned long long g_base_1 = (unsigned long long)_min_13 * (unsigned long long)g_stride;
                    int _min_14 = ((row_begin + (it0 + 2 - 1) * 2 + rg) < (last_row) ? (row_begin + (it0 + 2 - 1) * 2 + rg) : (last_row));
                    unsigned long long x_base_1 = (unsigned long long)_min_14 * (unsigned long long)x_stride;
                    int _min_15 = ((row_begin + (it0 + 2 - 1) * 2 + rg) < (last_row) ? (row_begin + (it0 + 2 - 1) * 2 + rg) : (last_row));
                    rr[1] = r[_min_15];
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(tir * 8));
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(tir * 8));
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)((96 + tir) * 8));
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)((96 + tir) * 8));
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot[1];
                float dq = 0.0f;
                dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * ww[0], __uint_as_float(xw[0] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * ww[1], __uint_as_float(xw[0] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * ww[2], __uint_as_float(xw[1] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * ww[3], __uint_as_float(xw[1] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * ww[4], __uint_as_float(xw[2] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * ww[5], __uint_as_float(xw[2] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * ww[6], __uint_as_float(xw[3] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * ww[7], __uint_as_float(xw[3] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * ww[8], __uint_as_float(xw[4] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * ww[9], __uint_as_float(xw[4] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * ww[10], __uint_as_float(xw[5] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * ww[11], __uint_as_float(xw[5] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * ww[12], __uint_as_float(xw[6] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * ww[13], __uint_as_float(xw[6] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * ww[14], __uint_as_float(xw[7] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * ww[15], __uint_as_float(xw[7] & 4294901760u), dq);
                float _warp_reduce_20 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_20 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_20, offset);
                dot[0] = _warp_reduce_20;
                if (lane == 0) {
                    dots[(it0 & 1) * 6 + warp] = dot[0];
                }
                __syncthreads();
                float tot = 0.0f;
                tot += dots[(it0 & 1) * 6 + rg * 3];
                tot += dots[(it0 & 1) * 6 + rg * 3 + 1];
                tot += dots[(it0 & 1) * 6 + rg * 3 + 2];
                dot[0] = tot;
                float rq = rr[0];
                float cq = __fdiv_rn(dot[0], 1536.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row_begin + it0 * 2 + rg < row_end) {
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_0 * x0_0, rq, acc[0]);
                    acc[1] = __fmaf_rn(g1_0 * x1_0, rq, acc[1]);
                    float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * ww[0]));
                    float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * ww[1]));
                    out_0[0] = y0_0;
                    out_0[1] = y1_0;
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_1 * x0_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(g1_1 * x1_1, rq, acc[3]);
                    float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * ww[2]));
                    float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * ww[3]));
                    out_0[2] = y0_1;
                    out_0[3] = y1_1;
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_2 * x0_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(g1_2 * x1_2, rq, acc[5]);
                    float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * ww[4]));
                    float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * ww[5]));
                    out_0[4] = y0_2;
                    out_0[5] = y1_2;
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_3 * x0_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(g1_3 * x1_3, rq, acc[7]);
                    float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * ww[6]));
                    float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * ww[7]));
                    out_0[6] = y0_3;
                    out_0[7] = y1_3;
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + it0 * 2 + rg) * 1536ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    acc[8] = __fmaf_rn(g0_4 * x0_4, rq, acc[8]);
                    acc[9] = __fmaf_rn(g1_4 * x1_4, rq, acc[9]);
                    float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * ww[8]));
                    float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * ww[9]));
                    out_1[0] = y0_4;
                    out_1[1] = y1_4;
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    acc[10] = __fmaf_rn(g0_5 * x0_5, rq, acc[10]);
                    acc[11] = __fmaf_rn(g1_5 * x1_5, rq, acc[11]);
                    float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * ww[10]));
                    float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * ww[11]));
                    out_1[2] = y0_5;
                    out_1[3] = y1_5;
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    acc[12] = __fmaf_rn(g0_6 * x0_6, rq, acc[12]);
                    acc[13] = __fmaf_rn(g1_6 * x1_6, rq, acc[13]);
                    float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * ww[12]));
                    float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * ww[13]));
                    out_1[4] = y0_6;
                    out_1[5] = y1_6;
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    acc[14] = __fmaf_rn(g0_7 * x0_7, rq, acc[14]);
                    acc[15] = __fmaf_rn(g1_7 * x1_7, rq, acc[15]);
                    float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * ww[14]));
                    float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * ww[15]));
                    out_1[6] = y0_7;
                    out_1[7] = y1_7;
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + it0 * 2 + rg) * 1536ull + (unsigned long long)((96 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
            }
            if (n_iters > it0 + 1) {
                if (n_iters > it0 + 1 + 1) {
                    int _min_21 = ((row_begin + (it0 + 1 + 2 - 1) * 2 + rg) < (last_row) ? (row_begin + (it0 + 1 + 2 - 1) * 2 + rg) : (last_row));
                    unsigned long long g_base_2 = (unsigned long long)_min_21 * (unsigned long long)g_stride;
                    int _min_22 = ((row_begin + (it0 + 1 + 2 - 1) * 2 + rg) < (last_row) ? (row_begin + (it0 + 1 + 2 - 1) * 2 + rg) : (last_row));
                    unsigned long long x_base_2 = (unsigned long long)_min_22 * (unsigned long long)x_stride;
                    int _min_23 = ((row_begin + (it0 + 1 + 2 - 1) * 2 + rg) < (last_row) ? (row_begin + (it0 + 1 + 2 - 1) * 2 + rg) : (last_row));
                    rr[0] = r[_min_23];
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(tir * 8));
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&gw[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_24[_blk].x), "=r"(_vdst_24[_blk].y), "=r"(_vdst_24[_blk].z), "=r"(_vdst_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(tir * 8));
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&xw[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_26 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)((96 + tir) * 8));
                        uint4* _vdst_26 = reinterpret_cast<uint4*>(&gw[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_26[_blk].x), "=r"(_vdst_26[_blk].y), "=r"(_vdst_26[_blk].z), "=r"(_vdst_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_27 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)((96 + tir) * 8));
                        uint4* _vdst_27 = reinterpret_cast<uint4*>(&xw[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_27[_blk].x), "=r"(_vdst_27[_blk].y), "=r"(_vdst_27[_blk].z), "=r"(_vdst_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot_1[1];
                float dq_1 = 0.0f;
                dq_1 = __fmaf_rn(__uint_as_float(gw[8] << 16) * ww[0], __uint_as_float(xw[8] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[8] & 4294901760u) * ww[1], __uint_as_float(xw[8] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[9] << 16) * ww[2], __uint_as_float(xw[9] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[9] & 4294901760u) * ww[3], __uint_as_float(xw[9] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[10] << 16) * ww[4], __uint_as_float(xw[10] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[10] & 4294901760u) * ww[5], __uint_as_float(xw[10] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[11] << 16) * ww[6], __uint_as_float(xw[11] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[11] & 4294901760u) * ww[7], __uint_as_float(xw[11] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[12] << 16) * ww[8], __uint_as_float(xw[12] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[12] & 4294901760u) * ww[9], __uint_as_float(xw[12] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[13] << 16) * ww[10], __uint_as_float(xw[13] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[13] & 4294901760u) * ww[11], __uint_as_float(xw[13] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[14] << 16) * ww[12], __uint_as_float(xw[14] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[14] & 4294901760u) * ww[13], __uint_as_float(xw[14] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[15] << 16) * ww[14], __uint_as_float(xw[15] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[15] & 4294901760u) * ww[15], __uint_as_float(xw[15] & 4294901760u), dq_1);
                float _warp_reduce_28 = dq_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_28 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_28, offset);
                dot_1[0] = _warp_reduce_28;
                if (lane == 0) {
                    dots[(it0 + 1 & 1) * 6 + warp] = dot_1[0];
                }
                __syncthreads();
                float tot_1 = 0.0f;
                tot_1 += dots[(it0 + 1 & 1) * 6 + rg * 3];
                tot_1 += dots[(it0 + 1 & 1) * 6 + rg * 3 + 1];
                tot_1 += dots[(it0 + 1 & 1) * 6 + rg * 3 + 2];
                dot_1[0] = tot_1;
                float rq_1 = rr[1];
                float cq_1 = __fdiv_rn(dot_1[0], 1536.0f);
                float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
                float nkx_1 = -kx_1;
                if (row_begin + (it0 + 1) * 2 + rg < row_end) {
                    float out_2[8];
                    float g0_8 = __uint_as_float(gw[8] << 16);
                    float g1_8 = __uint_as_float(gw[8] & 4294901760u);
                    float x0_8 = __uint_as_float(xw[8] << 16);
                    float x1_8 = __uint_as_float(xw[8] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_8 * x0_8, rq_1, acc[0]);
                    acc[1] = __fmaf_rn(g1_8 * x1_8, rq_1, acc[1]);
                    float y0_8 = __fmaf_rn(nkx_1, x0_8, rq_1 * (g0_8 * ww[0]));
                    float y1_8 = __fmaf_rn(nkx_1, x1_8, rq_1 * (g1_8 * ww[1]));
                    out_2[0] = y0_8;
                    out_2[1] = y1_8;
                    float g0_9 = __uint_as_float(gw[9] << 16);
                    float g1_9 = __uint_as_float(gw[9] & 4294901760u);
                    float x0_9 = __uint_as_float(xw[9] << 16);
                    float x1_9 = __uint_as_float(xw[9] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_9 * x0_9, rq_1, acc[2]);
                    acc[3] = __fmaf_rn(g1_9 * x1_9, rq_1, acc[3]);
                    float y0_9 = __fmaf_rn(nkx_1, x0_9, rq_1 * (g0_9 * ww[2]));
                    float y1_9 = __fmaf_rn(nkx_1, x1_9, rq_1 * (g1_9 * ww[3]));
                    out_2[2] = y0_9;
                    out_2[3] = y1_9;
                    float g0_10 = __uint_as_float(gw[10] << 16);
                    float g1_10 = __uint_as_float(gw[10] & 4294901760u);
                    float x0_10 = __uint_as_float(xw[10] << 16);
                    float x1_10 = __uint_as_float(xw[10] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_10 * x0_10, rq_1, acc[4]);
                    acc[5] = __fmaf_rn(g1_10 * x1_10, rq_1, acc[5]);
                    float y0_10 = __fmaf_rn(nkx_1, x0_10, rq_1 * (g0_10 * ww[4]));
                    float y1_10 = __fmaf_rn(nkx_1, x1_10, rq_1 * (g1_10 * ww[5]));
                    out_2[4] = y0_10;
                    out_2[5] = y1_10;
                    float g0_11 = __uint_as_float(gw[11] << 16);
                    float g1_11 = __uint_as_float(gw[11] & 4294901760u);
                    float x0_11 = __uint_as_float(xw[11] << 16);
                    float x1_11 = __uint_as_float(xw[11] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_11 * x0_11, rq_1, acc[6]);
                    acc[7] = __fmaf_rn(g1_11 * x1_11, rq_1, acc[7]);
                    float y0_11 = __fmaf_rn(nkx_1, x0_11, rq_1 * (g0_11 * ww[6]));
                    float y1_11 = __fmaf_rn(nkx_1, x1_11, rq_1 * (g1_11 * ww[7]));
                    out_2[6] = y0_11;
                    out_2[7] = y1_11;
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 1) * 2 + rg) * 1536ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
                    float out_3[8];
                    float g0_12 = __uint_as_float(gw[12] << 16);
                    float g1_12 = __uint_as_float(gw[12] & 4294901760u);
                    float x0_12 = __uint_as_float(xw[12] << 16);
                    float x1_12 = __uint_as_float(xw[12] & 4294901760u);
                    acc[8] = __fmaf_rn(g0_12 * x0_12, rq_1, acc[8]);
                    acc[9] = __fmaf_rn(g1_12 * x1_12, rq_1, acc[9]);
                    float y0_12 = __fmaf_rn(nkx_1, x0_12, rq_1 * (g0_12 * ww[8]));
                    float y1_12 = __fmaf_rn(nkx_1, x1_12, rq_1 * (g1_12 * ww[9]));
                    out_3[0] = y0_12;
                    out_3[1] = y1_12;
                    float g0_13 = __uint_as_float(gw[13] << 16);
                    float g1_13 = __uint_as_float(gw[13] & 4294901760u);
                    float x0_13 = __uint_as_float(xw[13] << 16);
                    float x1_13 = __uint_as_float(xw[13] & 4294901760u);
                    acc[10] = __fmaf_rn(g0_13 * x0_13, rq_1, acc[10]);
                    acc[11] = __fmaf_rn(g1_13 * x1_13, rq_1, acc[11]);
                    float y0_13 = __fmaf_rn(nkx_1, x0_13, rq_1 * (g0_13 * ww[10]));
                    float y1_13 = __fmaf_rn(nkx_1, x1_13, rq_1 * (g1_13 * ww[11]));
                    out_3[2] = y0_13;
                    out_3[3] = y1_13;
                    float g0_14 = __uint_as_float(gw[14] << 16);
                    float g1_14 = __uint_as_float(gw[14] & 4294901760u);
                    float x0_14 = __uint_as_float(xw[14] << 16);
                    float x1_14 = __uint_as_float(xw[14] & 4294901760u);
                    acc[12] = __fmaf_rn(g0_14 * x0_14, rq_1, acc[12]);
                    acc[13] = __fmaf_rn(g1_14 * x1_14, rq_1, acc[13]);
                    float y0_14 = __fmaf_rn(nkx_1, x0_14, rq_1 * (g0_14 * ww[12]));
                    float y1_14 = __fmaf_rn(nkx_1, x1_14, rq_1 * (g1_14 * ww[13]));
                    out_3[4] = y0_14;
                    out_3[5] = y1_14;
                    float g0_15 = __uint_as_float(gw[15] << 16);
                    float g1_15 = __uint_as_float(gw[15] & 4294901760u);
                    float x0_15 = __uint_as_float(xw[15] << 16);
                    float x1_15 = __uint_as_float(xw[15] & 4294901760u);
                    acc[14] = __fmaf_rn(g0_15 * x0_15, rq_1, acc[14]);
                    acc[15] = __fmaf_rn(g1_15 * x1_15, rq_1, acc[15]);
                    float y0_15 = __fmaf_rn(nkx_1, x0_15, rq_1 * (g0_15 * ww[14]));
                    float y1_15 = __fmaf_rn(nkx_1, x1_15, rq_1 * (g1_15 * ww[15]));
                    out_3[6] = y0_15;
                    out_3[7] = y1_15;
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 1) * 2 + rg) * 1536ull + (unsigned long long)((96 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
                }
            }
            window = window + 1;
            if (window == 8) {
                sums[rg * 1536 + tir] = sums[rg * 1536 + tir] + acc[0];
                acc[0] = 0.0f;
                sums[rg * 1536 + 96 + tir] = sums[rg * 1536 + 96 + tir] + acc[1];
                acc[1] = 0.0f;
                sums[rg * 1536 + 192 + tir] = sums[rg * 1536 + 192 + tir] + acc[2];
                acc[2] = 0.0f;
                sums[rg * 1536 + 288 + tir] = sums[rg * 1536 + 288 + tir] + acc[3];
                acc[3] = 0.0f;
                sums[rg * 1536 + 384 + tir] = sums[rg * 1536 + 384 + tir] + acc[4];
                acc[4] = 0.0f;
                sums[rg * 1536 + 480 + tir] = sums[rg * 1536 + 480 + tir] + acc[5];
                acc[5] = 0.0f;
                sums[rg * 1536 + 576 + tir] = sums[rg * 1536 + 576 + tir] + acc[6];
                acc[6] = 0.0f;
                sums[rg * 1536 + 672 + tir] = sums[rg * 1536 + 672 + tir] + acc[7];
                acc[7] = 0.0f;
                sums[rg * 1536 + 768 + tir] = sums[rg * 1536 + 768 + tir] + acc[8];
                acc[8] = 0.0f;
                sums[rg * 1536 + 864 + tir] = sums[rg * 1536 + 864 + tir] + acc[9];
                acc[9] = 0.0f;
                sums[rg * 1536 + 960 + tir] = sums[rg * 1536 + 960 + tir] + acc[10];
                acc[10] = 0.0f;
                sums[rg * 1536 + 1056 + tir] = sums[rg * 1536 + 1056 + tir] + acc[11];
                acc[11] = 0.0f;
                sums[rg * 1536 + 1152 + tir] = sums[rg * 1536 + 1152 + tir] + acc[12];
                acc[12] = 0.0f;
                sums[rg * 1536 + 1248 + tir] = sums[rg * 1536 + 1248 + tir] + acc[13];
                acc[13] = 0.0f;
                sums[rg * 1536 + 1344 + tir] = sums[rg * 1536 + 1344 + tir] + acc[14];
                acc[14] = 0.0f;
                sums[rg * 1536 + 1440 + tir] = sums[rg * 1536 + 1440 + tir] + acc[15];
                acc[15] = 0.0f;
                window = 0;
            }
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
    sums[rg * 1536 + tir] = sums[rg * 1536 + tir] + acc[0];
    acc[0] = 0.0f;
    sums[rg * 1536 + 96 + tir] = sums[rg * 1536 + 96 + tir] + acc[1];
    acc[1] = 0.0f;
    sums[rg * 1536 + 192 + tir] = sums[rg * 1536 + 192 + tir] + acc[2];
    acc[2] = 0.0f;
    sums[rg * 1536 + 288 + tir] = sums[rg * 1536 + 288 + tir] + acc[3];
    acc[3] = 0.0f;
    sums[rg * 1536 + 384 + tir] = sums[rg * 1536 + 384 + tir] + acc[4];
    acc[4] = 0.0f;
    sums[rg * 1536 + 480 + tir] = sums[rg * 1536 + 480 + tir] + acc[5];
    acc[5] = 0.0f;
    sums[rg * 1536 + 576 + tir] = sums[rg * 1536 + 576 + tir] + acc[6];
    acc[6] = 0.0f;
    sums[rg * 1536 + 672 + tir] = sums[rg * 1536 + 672 + tir] + acc[7];
    acc[7] = 0.0f;
    sums[rg * 1536 + 768 + tir] = sums[rg * 1536 + 768 + tir] + acc[8];
    acc[8] = 0.0f;
    sums[rg * 1536 + 864 + tir] = sums[rg * 1536 + 864 + tir] + acc[9];
    acc[9] = 0.0f;
    sums[rg * 1536 + 960 + tir] = sums[rg * 1536 + 960 + tir] + acc[10];
    acc[10] = 0.0f;
    sums[rg * 1536 + 1056 + tir] = sums[rg * 1536 + 1056 + tir] + acc[11];
    acc[11] = 0.0f;
    sums[rg * 1536 + 1152 + tir] = sums[rg * 1536 + 1152 + tir] + acc[12];
    acc[12] = 0.0f;
    sums[rg * 1536 + 1248 + tir] = sums[rg * 1536 + 1248 + tir] + acc[13];
    acc[13] = 0.0f;
    sums[rg * 1536 + 1344 + tir] = sums[rg * 1536 + 1344 + tir] + acc[14];
    acc[14] = 0.0f;
    sums[rg * 1536 + 1440 + tir] = sums[rg * 1536 + 1440 + tir] + acc[15];
    acc[15] = 0.0f;
    int _min_29 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
    int nrep = ((_min_29) < (21) ? (_min_29) : (21));
    __syncthreads();
    int at_0 = tid;
    float rv_0[2];
    rv_0[0] = sums[at_0];
    rv_0[1] = sums[1536 + at_0];
    rv_0[0] = rv_0[0] + rv_0[1];
    sums[at_0] = rv_0[0];
    int at_1 = 192 + tid;
    float rv_1[2];
    rv_1[0] = sums[at_1];
    rv_1[1] = sums[1536 + at_1];
    rv_1[0] = rv_1[0] + rv_1[1];
    sums[at_1] = rv_1[0];
    int at_2 = 384 + tid;
    float rv_2[2];
    rv_2[0] = sums[at_2];
    rv_2[1] = sums[1536 + at_2];
    rv_2[0] = rv_2[0] + rv_2[1];
    sums[at_2] = rv_2[0];
    int at_3 = 576 + tid;
    float rv_3[2];
    rv_3[0] = sums[at_3];
    rv_3[1] = sums[1536 + at_3];
    rv_3[0] = rv_3[0] + rv_3[1];
    sums[at_3] = rv_3[0];
    int at_4 = 768 + tid;
    float rv_4[2];
    rv_4[0] = sums[at_4];
    rv_4[1] = sums[1536 + at_4];
    rv_4[0] = rv_4[0] + rv_4[1];
    sums[at_4] = rv_4[0];
    int at_5 = 960 + tid;
    float rv_5[2];
    rv_5[0] = sums[at_5];
    rv_5[1] = sums[1536 + at_5];
    rv_5[0] = rv_5[0] + rv_5[1];
    sums[at_5] = rv_5[0];
    int at_6 = 1152 + tid;
    float rv_6[2];
    rv_6[0] = sums[at_6];
    rv_6[1] = sums[1536 + at_6];
    rv_6[0] = rv_6[0] + rv_6[1];
    sums[at_6] = rv_6[0];
    int at_7 = 1344 + tid;
    float rv_7[2];
    rv_7[0] = sums[at_7];
    rv_7[1] = sums[1536 + at_7];
    rv_7[0] = rv_7[0] + rv_7[1];
    sums[at_7] = rv_7[0];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_30 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1536));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_30), "r"(sums_addr + 0), "r"((uint32_t)(1536))
                    : "memory");
            }
            {
                void* _cpred_dst_31 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1536 + 384));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_31), "r"(sums_addr + 1536), "r"((uint32_t)(1536))
                    : "memory");
            }
            {
                void* _cpred_dst_32 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1536 + 768));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_32), "r"(sums_addr + 3072), "r"((uint32_t)(1536))
                    : "memory");
            }
            {
                void* _cpred_dst_33 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1536 + 1152));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_33), "r"(sums_addr + 4608), "r"((uint32_t)(1536))
                    : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
            asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        unsigned int old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(old) : "l"(&counters[1]), "r"(static_cast<uint32_t>(1)) : "memory");
        int last = 0;
        if (old + 1 == (unsigned int)num_bids) {
            *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
            last = 1;
        }
        flags[2] = last;
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
    __syncthreads();
    if (flags[2] != 0) {
        float buf[42];
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_34;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_34) : "l"((const void*)(partial + (tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_34);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_35;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_35) : "l"((const void*)(partial + (1536 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_35);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_36;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_36) : "l"((const void*)(partial + (3072 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_36);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_37;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_37) : "l"((const void*)(partial + (4608 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_37);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_38;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_38) : "l"((const void*)(partial + (6144 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_38);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_39;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_39) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_39);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_40;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_40) : "l"((const void*)(partial + (9216 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_40);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_41;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_41) : "l"((const void*)(partial + (10752 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_41);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_42;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_42) : "l"((const void*)(partial + (12288 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_42);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_43;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_43) : "l"((const void*)(partial + (13824 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_43);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_44;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_44) : "l"((const void*)(partial + (15360 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_44);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_45;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_45) : "l"((const void*)(partial + (16896 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_45);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_46;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_46) : "l"((const void*)(partial + (18432 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_46);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_47;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_47) : "l"((const void*)(partial + (19968 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_47);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_48;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_48) : "l"((const void*)(partial + (21504 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_48);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_49;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_49) : "l"((const void*)(partial + (23040 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_49);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_50;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_50) : "l"((const void*)(partial + (24576 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_50);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_51;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_51) : "l"((const void*)(partial + (26112 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_51);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_52;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_52) : "l"((const void*)(partial + (27648 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_52);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_53;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_53) : "l"((const void*)(partial + (29184 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_53);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_54;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_54) : "l"((const void*)(partial + (30720 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_54);
            }
        }
        buf[21] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_55;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_55) : "l"((const void*)(partial + (192 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_55);
            }
        }
        buf[22] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_56;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_56) : "l"((const void*)(partial + (1728 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_56);
            }
        }
        buf[23] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_57;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (3264 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_57);
            }
        }
        buf[24] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_58;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (4800 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_58);
            }
        }
        buf[25] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_59;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (6336 + tid))) : "memory");
                buf[25] = __uint_as_float(_scalar_bits_59);
            }
        }
        buf[26] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_60;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (7872 + tid))) : "memory");
                buf[26] = __uint_as_float(_scalar_bits_60);
            }
        }
        buf[27] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_61;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (9408 + tid))) : "memory");
                buf[27] = __uint_as_float(_scalar_bits_61);
            }
        }
        buf[28] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_62;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (10944 + tid))) : "memory");
                buf[28] = __uint_as_float(_scalar_bits_62);
            }
        }
        buf[29] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_63;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (12480 + tid))) : "memory");
                buf[29] = __uint_as_float(_scalar_bits_63);
            }
        }
        buf[30] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_64;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (14016 + tid))) : "memory");
                buf[30] = __uint_as_float(_scalar_bits_64);
            }
        }
        buf[31] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_65;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (15552 + tid))) : "memory");
                buf[31] = __uint_as_float(_scalar_bits_65);
            }
        }
        buf[32] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_66;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (17088 + tid))) : "memory");
                buf[32] = __uint_as_float(_scalar_bits_66);
            }
        }
        buf[33] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_67;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (18624 + tid))) : "memory");
                buf[33] = __uint_as_float(_scalar_bits_67);
            }
        }
        buf[34] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_68;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (20160 + tid))) : "memory");
                buf[34] = __uint_as_float(_scalar_bits_68);
            }
        }
        buf[35] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_69;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (21696 + tid))) : "memory");
                buf[35] = __uint_as_float(_scalar_bits_69);
            }
        }
        buf[36] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_70;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (23232 + tid))) : "memory");
                buf[36] = __uint_as_float(_scalar_bits_70);
            }
        }
        buf[37] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_71;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (24768 + tid))) : "memory");
                buf[37] = __uint_as_float(_scalar_bits_71);
            }
        }
        buf[38] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_72;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (26304 + tid))) : "memory");
                buf[38] = __uint_as_float(_scalar_bits_72);
            }
        }
        buf[39] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_73;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (27840 + tid))) : "memory");
                buf[39] = __uint_as_float(_scalar_bits_73);
            }
        }
        buf[40] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_74;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (29376 + tid))) : "memory");
                buf[40] = __uint_as_float(_scalar_bits_74);
            }
        }
        buf[41] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_75;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (30912 + tid))) : "memory");
                buf[41] = __uint_as_float(_scalar_bits_75);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[0] = buf[0] + buf[16];
        int at_w_0 = tid;
        dwbuf[(at_w_0 / 96 / 8 * 96 + at_w_0 % 96) * 8 + at_w_0 / 96 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_0)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_0)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_0)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_0)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_0)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_0)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_0)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_0)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_0)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_0)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_0)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_0)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_0)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_0)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_0)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_0)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_0)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_0)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_0)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_0)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_76;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (384 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_76);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_77;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_77);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_78;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_78) : "l"((const void*)(partial + (3456 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_78);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_79;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_79) : "l"((const void*)(partial + (4992 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_79);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_80;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_80) : "l"((const void*)(partial + (6528 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_80);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_81;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_81) : "l"((const void*)(partial + (8064 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_81);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_82;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_82) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_82);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_83;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_83) : "l"((const void*)(partial + (11136 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_83);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_84;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_84) : "l"((const void*)(partial + (12672 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_84);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_85;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_85) : "l"((const void*)(partial + (14208 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_85);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_86;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_86) : "l"((const void*)(partial + (15744 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_86);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_87;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_87) : "l"((const void*)(partial + (17280 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_87);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_88;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_88) : "l"((const void*)(partial + (18816 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_88);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_89;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_89) : "l"((const void*)(partial + (20352 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_89);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_90;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_90) : "l"((const void*)(partial + (21888 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_90);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_91;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_91) : "l"((const void*)(partial + (23424 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_91);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_92;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_92) : "l"((const void*)(partial + (24960 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_92);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_93;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_93) : "l"((const void*)(partial + (26496 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_93);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_94;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_94) : "l"((const void*)(partial + (28032 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_94);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_95;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_95) : "l"((const void*)(partial + (29568 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_95);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_96;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_96) : "l"((const void*)(partial + (31104 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_96);
            }
        }
        buf[21] = buf[21] + buf[22];
        buf[23] = buf[23] + buf[24];
        buf[25] = buf[25] + buf[26];
        buf[27] = buf[27] + buf[28];
        buf[29] = buf[29] + buf[30];
        buf[31] = buf[31] + buf[32];
        buf[33] = buf[33] + buf[34];
        buf[35] = buf[35] + buf[36];
        buf[37] = buf[37] + buf[38];
        buf[39] = buf[39] + buf[40];
        buf[21] = buf[21] + buf[23];
        buf[25] = buf[25] + buf[27];
        buf[29] = buf[29] + buf[31];
        buf[33] = buf[33] + buf[35];
        buf[37] = buf[37] + buf[39];
        buf[21] = buf[21] + buf[25];
        buf[29] = buf[29] + buf[33];
        buf[37] = buf[37] + buf[41];
        buf[21] = buf[21] + buf[29];
        buf[21] = buf[21] + buf[37];
        int at_w_1 = 192 + tid;
        dwbuf[(at_w_1 / 96 / 8 * 96 + at_w_1 % 96) * 8 + at_w_1 / 96 % 8] = buf[21];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_1)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_1)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_1)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_1)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_1)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_1)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_1)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_1)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_1)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_1)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_1)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_1)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_1)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_1)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_1)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_1)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_1)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_1)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_1)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_1)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_1)) = 0.0f;
        }
        buf[21] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_97;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_97) : "l"((const void*)(partial + (576 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_97);
            }
        }
        buf[22] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_98;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_98) : "l"((const void*)(partial + (2112 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_98);
            }
        }
        buf[23] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_99;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_99) : "l"((const void*)(partial + (3648 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_99);
            }
        }
        buf[24] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_100;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_100) : "l"((const void*)(partial + (5184 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_100);
            }
        }
        buf[25] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_101;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_101) : "l"((const void*)(partial + (6720 + tid))) : "memory");
                buf[25] = __uint_as_float(_scalar_bits_101);
            }
        }
        buf[26] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_102;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_102) : "l"((const void*)(partial + (8256 + tid))) : "memory");
                buf[26] = __uint_as_float(_scalar_bits_102);
            }
        }
        buf[27] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_103;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_103) : "l"((const void*)(partial + (9792 + tid))) : "memory");
                buf[27] = __uint_as_float(_scalar_bits_103);
            }
        }
        buf[28] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_104;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_104) : "l"((const void*)(partial + (11328 + tid))) : "memory");
                buf[28] = __uint_as_float(_scalar_bits_104);
            }
        }
        buf[29] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_105;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_105) : "l"((const void*)(partial + (12864 + tid))) : "memory");
                buf[29] = __uint_as_float(_scalar_bits_105);
            }
        }
        buf[30] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_106;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_106) : "l"((const void*)(partial + (14400 + tid))) : "memory");
                buf[30] = __uint_as_float(_scalar_bits_106);
            }
        }
        buf[31] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_107;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_107) : "l"((const void*)(partial + (15936 + tid))) : "memory");
                buf[31] = __uint_as_float(_scalar_bits_107);
            }
        }
        buf[32] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_108;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_108) : "l"((const void*)(partial + (17472 + tid))) : "memory");
                buf[32] = __uint_as_float(_scalar_bits_108);
            }
        }
        buf[33] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_109;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_109) : "l"((const void*)(partial + (19008 + tid))) : "memory");
                buf[33] = __uint_as_float(_scalar_bits_109);
            }
        }
        buf[34] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_110;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_110) : "l"((const void*)(partial + (20544 + tid))) : "memory");
                buf[34] = __uint_as_float(_scalar_bits_110);
            }
        }
        buf[35] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_111;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_111) : "l"((const void*)(partial + (22080 + tid))) : "memory");
                buf[35] = __uint_as_float(_scalar_bits_111);
            }
        }
        buf[36] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_112;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_112) : "l"((const void*)(partial + (23616 + tid))) : "memory");
                buf[36] = __uint_as_float(_scalar_bits_112);
            }
        }
        buf[37] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_113;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_113) : "l"((const void*)(partial + (25152 + tid))) : "memory");
                buf[37] = __uint_as_float(_scalar_bits_113);
            }
        }
        buf[38] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_114;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_114) : "l"((const void*)(partial + (26688 + tid))) : "memory");
                buf[38] = __uint_as_float(_scalar_bits_114);
            }
        }
        buf[39] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_115;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_115) : "l"((const void*)(partial + (28224 + tid))) : "memory");
                buf[39] = __uint_as_float(_scalar_bits_115);
            }
        }
        buf[40] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_116;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_116) : "l"((const void*)(partial + (29760 + tid))) : "memory");
                buf[40] = __uint_as_float(_scalar_bits_116);
            }
        }
        buf[41] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_117;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_117) : "l"((const void*)(partial + (31296 + tid))) : "memory");
                buf[41] = __uint_as_float(_scalar_bits_117);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[0] = buf[0] + buf[16];
        int at_w_2 = 384 + tid;
        dwbuf[(at_w_2 / 96 / 8 * 96 + at_w_2 % 96) * 8 + at_w_2 / 96 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_2)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_2)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_2)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_2)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_2)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_2)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_2)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_2)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_2)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_2)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_2)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_2)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_2)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_2)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_2)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_2)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_2)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_2)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_2)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_2)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_2)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_118;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_118) : "l"((const void*)(partial + (768 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_118);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_119;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_119) : "l"((const void*)(partial + (2304 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_119);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_120;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_120) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_120);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_121;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_121) : "l"((const void*)(partial + (5376 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_121);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_122;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_122) : "l"((const void*)(partial + (6912 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_122);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_123;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_123) : "l"((const void*)(partial + (8448 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_123);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_124;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_124) : "l"((const void*)(partial + (9984 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_124);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_125;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_125) : "l"((const void*)(partial + (11520 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_125);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_126;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_126) : "l"((const void*)(partial + (13056 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_126);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_127;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_127) : "l"((const void*)(partial + (14592 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_127);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_128;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_128) : "l"((const void*)(partial + (16128 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_128);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_129;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_129) : "l"((const void*)(partial + (17664 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_129);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_130;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_130) : "l"((const void*)(partial + (19200 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_130);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_131;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_131) : "l"((const void*)(partial + (20736 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_131);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_132;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_132) : "l"((const void*)(partial + (22272 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_132);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_133;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_133) : "l"((const void*)(partial + (23808 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_133);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_134;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_134) : "l"((const void*)(partial + (25344 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_134);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_135;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_135) : "l"((const void*)(partial + (26880 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_135);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_136;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_136) : "l"((const void*)(partial + (28416 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_136);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_137;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_137) : "l"((const void*)(partial + (29952 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_137);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_138;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_138) : "l"((const void*)(partial + (31488 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_138);
            }
        }
        buf[21] = buf[21] + buf[22];
        buf[23] = buf[23] + buf[24];
        buf[25] = buf[25] + buf[26];
        buf[27] = buf[27] + buf[28];
        buf[29] = buf[29] + buf[30];
        buf[31] = buf[31] + buf[32];
        buf[33] = buf[33] + buf[34];
        buf[35] = buf[35] + buf[36];
        buf[37] = buf[37] + buf[38];
        buf[39] = buf[39] + buf[40];
        buf[21] = buf[21] + buf[23];
        buf[25] = buf[25] + buf[27];
        buf[29] = buf[29] + buf[31];
        buf[33] = buf[33] + buf[35];
        buf[37] = buf[37] + buf[39];
        buf[21] = buf[21] + buf[25];
        buf[29] = buf[29] + buf[33];
        buf[37] = buf[37] + buf[41];
        buf[21] = buf[21] + buf[29];
        buf[21] = buf[21] + buf[37];
        int at_w_3 = 576 + tid;
        dwbuf[(at_w_3 / 96 / 8 * 96 + at_w_3 % 96) * 8 + at_w_3 / 96 % 8] = buf[21];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_3)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_3)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_3)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_3)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_3)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_3)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_3)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_3)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_3)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_3)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_3)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_3)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_3)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_3)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_3)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_3)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_3)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_3)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_3)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_3)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_3)) = 0.0f;
        }
        buf[21] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_139;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_139) : "l"((const void*)(partial + (960 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_139);
            }
        }
        buf[22] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_140;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_140) : "l"((const void*)(partial + (2496 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_140);
            }
        }
        buf[23] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_141;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_141) : "l"((const void*)(partial + (4032 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_141);
            }
        }
        buf[24] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_142;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_142) : "l"((const void*)(partial + (5568 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_142);
            }
        }
        buf[25] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_143;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_143) : "l"((const void*)(partial + (7104 + tid))) : "memory");
                buf[25] = __uint_as_float(_scalar_bits_143);
            }
        }
        buf[26] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_144;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_144) : "l"((const void*)(partial + (8640 + tid))) : "memory");
                buf[26] = __uint_as_float(_scalar_bits_144);
            }
        }
        buf[27] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_145;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_145) : "l"((const void*)(partial + (10176 + tid))) : "memory");
                buf[27] = __uint_as_float(_scalar_bits_145);
            }
        }
        buf[28] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_146;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_146) : "l"((const void*)(partial + (11712 + tid))) : "memory");
                buf[28] = __uint_as_float(_scalar_bits_146);
            }
        }
        buf[29] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_147;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_147) : "l"((const void*)(partial + (13248 + tid))) : "memory");
                buf[29] = __uint_as_float(_scalar_bits_147);
            }
        }
        buf[30] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_148;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_148) : "l"((const void*)(partial + (14784 + tid))) : "memory");
                buf[30] = __uint_as_float(_scalar_bits_148);
            }
        }
        buf[31] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_149;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_149) : "l"((const void*)(partial + (16320 + tid))) : "memory");
                buf[31] = __uint_as_float(_scalar_bits_149);
            }
        }
        buf[32] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_150;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_150) : "l"((const void*)(partial + (17856 + tid))) : "memory");
                buf[32] = __uint_as_float(_scalar_bits_150);
            }
        }
        buf[33] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_151;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_151) : "l"((const void*)(partial + (19392 + tid))) : "memory");
                buf[33] = __uint_as_float(_scalar_bits_151);
            }
        }
        buf[34] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_152;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_152) : "l"((const void*)(partial + (20928 + tid))) : "memory");
                buf[34] = __uint_as_float(_scalar_bits_152);
            }
        }
        buf[35] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_153;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_153) : "l"((const void*)(partial + (22464 + tid))) : "memory");
                buf[35] = __uint_as_float(_scalar_bits_153);
            }
        }
        buf[36] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_154;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_154) : "l"((const void*)(partial + (24000 + tid))) : "memory");
                buf[36] = __uint_as_float(_scalar_bits_154);
            }
        }
        buf[37] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_155;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_155) : "l"((const void*)(partial + (25536 + tid))) : "memory");
                buf[37] = __uint_as_float(_scalar_bits_155);
            }
        }
        buf[38] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_156;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_156) : "l"((const void*)(partial + (27072 + tid))) : "memory");
                buf[38] = __uint_as_float(_scalar_bits_156);
            }
        }
        buf[39] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_157;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_157) : "l"((const void*)(partial + (28608 + tid))) : "memory");
                buf[39] = __uint_as_float(_scalar_bits_157);
            }
        }
        buf[40] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_158;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_158) : "l"((const void*)(partial + (30144 + tid))) : "memory");
                buf[40] = __uint_as_float(_scalar_bits_158);
            }
        }
        buf[41] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_159;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_159) : "l"((const void*)(partial + (31680 + tid))) : "memory");
                buf[41] = __uint_as_float(_scalar_bits_159);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[0] = buf[0] + buf[16];
        int at_w_4 = 768 + tid;
        dwbuf[(at_w_4 / 96 / 8 * 96 + at_w_4 % 96) * 8 + at_w_4 / 96 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_4)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_4)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_4)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_4)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_4)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_4)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_4)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_4)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_4)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_4)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_4)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_4)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_4)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_4)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_4)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_4)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_4)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_4)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_4)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_4)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_4)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_160;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_160) : "l"((const void*)(partial + (1152 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_160);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_161;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_161) : "l"((const void*)(partial + (2688 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_161);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_162;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_162) : "l"((const void*)(partial + (4224 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_162);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_163;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_163) : "l"((const void*)(partial + (5760 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_163);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_164;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_164) : "l"((const void*)(partial + (7296 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_164);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_165;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_165) : "l"((const void*)(partial + (8832 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_165);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_166;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_166) : "l"((const void*)(partial + (10368 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_166);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_167;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_167) : "l"((const void*)(partial + (11904 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_167);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_168;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_168) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_168);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_169;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_169) : "l"((const void*)(partial + (14976 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_169);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_170;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_170) : "l"((const void*)(partial + (16512 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_170);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_171;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_171) : "l"((const void*)(partial + (18048 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_171);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_172;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_172) : "l"((const void*)(partial + (19584 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_172);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_173;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_173) : "l"((const void*)(partial + (21120 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_173);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_174;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_174) : "l"((const void*)(partial + (22656 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_174);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_175;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_175) : "l"((const void*)(partial + (24192 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_175);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_176;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_176) : "l"((const void*)(partial + (25728 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_176);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_177;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_177) : "l"((const void*)(partial + (27264 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_177);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_178;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_178) : "l"((const void*)(partial + (28800 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_178);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_179;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_179) : "l"((const void*)(partial + (30336 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_179);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_180;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_180) : "l"((const void*)(partial + (31872 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_180);
            }
        }
        buf[21] = buf[21] + buf[22];
        buf[23] = buf[23] + buf[24];
        buf[25] = buf[25] + buf[26];
        buf[27] = buf[27] + buf[28];
        buf[29] = buf[29] + buf[30];
        buf[31] = buf[31] + buf[32];
        buf[33] = buf[33] + buf[34];
        buf[35] = buf[35] + buf[36];
        buf[37] = buf[37] + buf[38];
        buf[39] = buf[39] + buf[40];
        buf[21] = buf[21] + buf[23];
        buf[25] = buf[25] + buf[27];
        buf[29] = buf[29] + buf[31];
        buf[33] = buf[33] + buf[35];
        buf[37] = buf[37] + buf[39];
        buf[21] = buf[21] + buf[25];
        buf[29] = buf[29] + buf[33];
        buf[37] = buf[37] + buf[41];
        buf[21] = buf[21] + buf[29];
        buf[21] = buf[21] + buf[37];
        int at_w_5 = 960 + tid;
        dwbuf[(at_w_5 / 96 / 8 * 96 + at_w_5 % 96) * 8 + at_w_5 / 96 % 8] = buf[21];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_5)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_5)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_5)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_5)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_5)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_5)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_5)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_5)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_5)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_5)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_5)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_5)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_5)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_5)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_5)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_5)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_5)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_5)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_5)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_5)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_5)) = 0.0f;
        }
        buf[21] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_181;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_181) : "l"((const void*)(partial + (1344 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_181);
            }
        }
        buf[22] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_182;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_182) : "l"((const void*)(partial + (2880 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_182);
            }
        }
        buf[23] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_183;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_183) : "l"((const void*)(partial + (4416 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_183);
            }
        }
        buf[24] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_184;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_184) : "l"((const void*)(partial + (5952 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_184);
            }
        }
        buf[25] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_185;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_185) : "l"((const void*)(partial + (7488 + tid))) : "memory");
                buf[25] = __uint_as_float(_scalar_bits_185);
            }
        }
        buf[26] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_186;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_186) : "l"((const void*)(partial + (9024 + tid))) : "memory");
                buf[26] = __uint_as_float(_scalar_bits_186);
            }
        }
        buf[27] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_187;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_187) : "l"((const void*)(partial + (10560 + tid))) : "memory");
                buf[27] = __uint_as_float(_scalar_bits_187);
            }
        }
        buf[28] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_188;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_188) : "l"((const void*)(partial + (12096 + tid))) : "memory");
                buf[28] = __uint_as_float(_scalar_bits_188);
            }
        }
        buf[29] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_189;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_189) : "l"((const void*)(partial + (13632 + tid))) : "memory");
                buf[29] = __uint_as_float(_scalar_bits_189);
            }
        }
        buf[30] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_190;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_190) : "l"((const void*)(partial + (15168 + tid))) : "memory");
                buf[30] = __uint_as_float(_scalar_bits_190);
            }
        }
        buf[31] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_191;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_191) : "l"((const void*)(partial + (16704 + tid))) : "memory");
                buf[31] = __uint_as_float(_scalar_bits_191);
            }
        }
        buf[32] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_192;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_192) : "l"((const void*)(partial + (18240 + tid))) : "memory");
                buf[32] = __uint_as_float(_scalar_bits_192);
            }
        }
        buf[33] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_193;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_193) : "l"((const void*)(partial + (19776 + tid))) : "memory");
                buf[33] = __uint_as_float(_scalar_bits_193);
            }
        }
        buf[34] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_194;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_194) : "l"((const void*)(partial + (21312 + tid))) : "memory");
                buf[34] = __uint_as_float(_scalar_bits_194);
            }
        }
        buf[35] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_195;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_195) : "l"((const void*)(partial + (22848 + tid))) : "memory");
                buf[35] = __uint_as_float(_scalar_bits_195);
            }
        }
        buf[36] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_196;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_196) : "l"((const void*)(partial + (24384 + tid))) : "memory");
                buf[36] = __uint_as_float(_scalar_bits_196);
            }
        }
        buf[37] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_197;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_197) : "l"((const void*)(partial + (25920 + tid))) : "memory");
                buf[37] = __uint_as_float(_scalar_bits_197);
            }
        }
        buf[38] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_198;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_198) : "l"((const void*)(partial + (27456 + tid))) : "memory");
                buf[38] = __uint_as_float(_scalar_bits_198);
            }
        }
        buf[39] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_199;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_199) : "l"((const void*)(partial + (28992 + tid))) : "memory");
                buf[39] = __uint_as_float(_scalar_bits_199);
            }
        }
        buf[40] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_200;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_200) : "l"((const void*)(partial + (30528 + tid))) : "memory");
                buf[40] = __uint_as_float(_scalar_bits_200);
            }
        }
        buf[41] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_201;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_201) : "l"((const void*)(partial + (32064 + tid))) : "memory");
                buf[41] = __uint_as_float(_scalar_bits_201);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[0] = buf[0] + buf[16];
        int at_w_6 = 1152 + tid;
        dwbuf[(at_w_6 / 96 / 8 * 96 + at_w_6 % 96) * 8 + at_w_6 / 96 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_6)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_6)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_6)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_6)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_6)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_6)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_6)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_6)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_6)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_6)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_6)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_6)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_6)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_6)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_6)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_6)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_6)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_6)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_6)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_6)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_6)) = 0.0f;
        }
        buf[21] = buf[21] + buf[22];
        buf[23] = buf[23] + buf[24];
        buf[25] = buf[25] + buf[26];
        buf[27] = buf[27] + buf[28];
        buf[29] = buf[29] + buf[30];
        buf[31] = buf[31] + buf[32];
        buf[33] = buf[33] + buf[34];
        buf[35] = buf[35] + buf[36];
        buf[37] = buf[37] + buf[38];
        buf[39] = buf[39] + buf[40];
        buf[21] = buf[21] + buf[23];
        buf[25] = buf[25] + buf[27];
        buf[29] = buf[29] + buf[31];
        buf[33] = buf[33] + buf[35];
        buf[37] = buf[37] + buf[39];
        buf[21] = buf[21] + buf[25];
        buf[29] = buf[29] + buf[33];
        buf[37] = buf[37] + buf[41];
        buf[21] = buf[21] + buf[29];
        buf[21] = buf[21] + buf[37];
        int at_w_7 = 1344 + tid;
        dwbuf[(at_w_7 / 96 / 8 * 96 + at_w_7 % 96) * 8 + at_w_7 / 96 % 8] = buf[21];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_7)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_7)) = 0.0f;
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_7)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_7)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_7)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_7)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (9216 + at_w_7)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10752 + at_w_7)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12288 + at_w_7)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (13824 + at_w_7)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_7)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16896 + at_w_7)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (18432 + at_w_7)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19968 + at_w_7)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21504 + at_w_7)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_7)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24576 + at_w_7)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26112 + at_w_7)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (27648 + at_w_7)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29184 + at_w_7)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_7)) = 0.0f;
        }
        __syncthreads();
        int col_0 = tid * 4;
        if (col_0 < 1536) {
            float o4_0[4];
            o4_0[0] = dwbuf[col_0];
            o4_0[1] = dwbuf[col_0 + 1];
            o4_0[2] = dwbuf[col_0 + 2];
            o4_0[3] = dwbuf[col_0 + 3];
            {
                float4 _v4 = make_float4(o4_0[0 + 0], o4_0[0 + 1], o4_0[0 + 2], o4_0[0 + 3]);
                *reinterpret_cast<float4*>(dw + col_0) = _v4;
            }
        }
        int col_1 = (192 + tid) * 4;
        if (col_1 < 1536) {
            float o4_1[4];
            o4_1[0] = dwbuf[col_1];
            o4_1[1] = dwbuf[col_1 + 1];
            o4_1[2] = dwbuf[col_1 + 2];
            o4_1[3] = dwbuf[col_1 + 3];
            {
                float4 _v4 = make_float4(o4_1[0 + 0], o4_1[0 + 1], o4_1[0 + 2], o4_1[0 + 3]);
                *reinterpret_cast<float4*>(dw + col_1) = _v4;
            }
        }
    }
}

} // extern "C"
