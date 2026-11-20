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

#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_8b0b86c827709e73ee26(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial));
    __builtin_assume(__isGlobal(partial));

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
    float* sums = reinterpret_cast<float*>(smem_raw + 64);
    const int sums_addr = smem + 64;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 64);
    const int dwbuf_addr = smem + 64;

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
    int rg = tid / 16;
    int tir = tid - rg * 16;
    unsigned int ww[4];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    float acc[8];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    int c = 0;
    int first_c = 0;
    sums[rg * 128 + tir] = 0.0f;
    sums[rg * 128 + 16 + tir] = 0.0f;
    sums[rg * 128 + 32 + tir] = 0.0f;
    sums[rg * 128 + 48 + tir] = 0.0f;
    sums[rg * 128 + 64 + tir] = 0.0f;
    sums[rg * 128 + 80 + tir] = 0.0f;
    sums[rg * 128 + 96 + tir] = 0.0f;
    sums[rg * 128 + 112 + tir] = 0.0f;
    unsigned int gw[16];
    unsigned int xw[16];
    unsigned int hw[1];
    float rr[4];
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
            unsigned int _atomic_fetch_add_2 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_2;
        }
        int stages = (T + 7) / 8;
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
        int _min_3 = ((begin_it * 8) < (T) ? (begin_it * 8) : (T));
        row_begin = _min_3;
        int _min_4 = ((begin_it * 8 + span * 8) < (T) ? (begin_it * 8 + span * 8) : (T));
        row_end = _min_4;
        n_iters = (row_end - row_begin + 7) / 8;
        last_row = row_end - 1;
        if (n_iters > 0) {
            int _min_5 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long g_base = (unsigned long long)_min_5 * (unsigned long long)g_stride;
            int _min_6 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long x_base = (unsigned long long)_min_6 * (unsigned long long)x_stride;
            int _min_7 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            rr[0] = r[_min_7];
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)(tir * 8));
                uint4* _vdst_8 = reinterpret_cast<uint4*>(&gw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)(tir * 8));
                uint4* _vdst_9 = reinterpret_cast<uint4*>(&xw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        if (n_iters > 1) {
            int _min_10 = ((row_begin + 8 + rg) < (last_row) ? (row_begin + 8 + rg) : (last_row));
            unsigned long long g_base_1 = (unsigned long long)_min_10 * (unsigned long long)g_stride;
            int _min_11 = ((row_begin + 8 + rg) < (last_row) ? (row_begin + 8 + rg) : (last_row));
            unsigned long long x_base_1 = (unsigned long long)_min_11 * (unsigned long long)x_stride;
            int _min_12 = ((row_begin + 8 + rg) < (last_row) ? (row_begin + 8 + rg) : (last_row));
            rr[1] = r[_min_12];
            {
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(tir * 8));
                uint4* _vdst_13 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(tir * 8));
                uint4* _vdst_14 = reinterpret_cast<uint4*>(&xw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        if (n_iters > 2) {
            int _min_15 = ((row_begin + 16 + rg) < (last_row) ? (row_begin + 16 + rg) : (last_row));
            unsigned long long g_base_2 = (unsigned long long)_min_15 * (unsigned long long)g_stride;
            int _min_16 = ((row_begin + 16 + rg) < (last_row) ? (row_begin + 16 + rg) : (last_row));
            unsigned long long x_base_2 = (unsigned long long)_min_16 * (unsigned long long)x_stride;
            int _min_17 = ((row_begin + 16 + rg) < (last_row) ? (row_begin + 16 + rg) : (last_row));
            rr[2] = r[_min_17];
            {
                const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(tir * 8));
                uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(tir * 8));
                uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int it0 = 0; it0 < n_iters; it0 += 4) {
            if (n_iters > it0) {
                if (n_iters > it0 + 3) {
                    int _min_20 = ((row_begin + (it0 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long g_base_3 = (unsigned long long)_min_20 * (unsigned long long)g_stride;
                    int _min_21 = ((row_begin + (it0 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long x_base_3 = (unsigned long long)_min_21 * (unsigned long long)x_stride;
                    int _min_22 = ((row_begin + (it0 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 4 - 1) * 8 + rg) : (last_row));
                    rr[3] = r[_min_22];
                    {
                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)(tir * 8));
                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&gw[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_23[_blk].x), "=r"(_vdst_23[_blk].y), "=r"(_vdst_23[_blk].z), "=r"(_vdst_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)(tir * 8));
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&xw[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_24[_blk].x), "=r"(_vdst_24[_blk].y), "=r"(_vdst_24[_blk].z), "=r"(_vdst_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot[1];
                float dq = 0.0f;
                dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[0] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[0] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[1] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[1] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[2] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[2] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[3] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[3] & 4294901760u), dq);
                float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, dq, 8);
                dq = dq + _shfl_xor_sync_25;
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, dq, 4);
                dq = dq + _shfl_xor_sync_26;
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, dq, 2);
                dq = dq + _shfl_xor_sync_27;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, dq, 1);
                dq = dq + _shfl_xor_sync_28;
                dot[0] = dq;
                float rq = rr[0];
                float cq = __fdiv_rn(dot[0], 128.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row_begin + it0 * 8 + rg < row_end) {
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_0 * x0_0, rq, acc[0]);
                    acc[1] = __fmaf_rn(g1_0 * x1_0, rq, acc[1]);
                    float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
                    float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
                    out_0[0] = y0_0;
                    out_0[1] = y1_0;
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_1 * x0_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(g1_1 * x1_1, rq, acc[3]);
                    float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
                    float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
                    out_0[2] = y0_1;
                    out_0[3] = y1_1;
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_2 * x0_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(g1_2 * x1_2, rq, acc[5]);
                    float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
                    float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
                    out_0[4] = y0_2;
                    out_0[5] = y1_2;
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_3 * x0_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(g1_3 * x1_3, rq, acc[7]);
                    float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * __uint_as_float(ww[3] << 16)));
                    float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * __uint_as_float(ww[3] & 4294901760u)));
                    out_0[6] = y0_3;
                    out_0[7] = y1_3;
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + it0 * 8 + rg) * 128ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                }
            }
            if (n_iters > it0 + 1) {
                if (n_iters > it0 + 1 + 3) {
                    int _min_29 = ((row_begin + (it0 + 1 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 1 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long g_base_4 = (unsigned long long)_min_29 * (unsigned long long)g_stride;
                    int _min_30 = ((row_begin + (it0 + 1 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 1 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long x_base_4 = (unsigned long long)_min_30 * (unsigned long long)x_stride;
                    int _min_31 = ((row_begin + (it0 + 1 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 1 + 4 - 1) * 8 + rg) : (last_row));
                    rr[0] = r[_min_31];
                    {
                        const uint4* _vptr_32 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)(tir * 8));
                        uint4* _vdst_32 = reinterpret_cast<uint4*>(&gw[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_32[_blk].x), "=r"(_vdst_32[_blk].y), "=r"(_vdst_32[_blk].z), "=r"(_vdst_32[_blk].w) : "l"((const void*)(_vptr_32 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_33 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)(tir * 8));
                        uint4* _vdst_33 = reinterpret_cast<uint4*>(&xw[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_33[_blk].x), "=r"(_vdst_33[_blk].y), "=r"(_vdst_33[_blk].z), "=r"(_vdst_33[_blk].w) : "l"((const void*)(_vptr_33 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot_1[1];
                float dq_1 = 0.0f;
                dq_1 = __fmaf_rn(__uint_as_float(gw[4] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[4] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[4] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[5] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[5] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[5] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[6] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[6] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[6] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[7] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[7] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[7] & 4294901760u), dq_1);
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dq_1, 8);
                dq_1 = dq_1 + _shfl_xor_sync_34;
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dq_1, 4);
                dq_1 = dq_1 + _shfl_xor_sync_35;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dq_1, 2);
                dq_1 = dq_1 + _shfl_xor_sync_36;
                float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dq_1, 1);
                dq_1 = dq_1 + _shfl_xor_sync_37;
                dot_1[0] = dq_1;
                float rq_1 = rr[1];
                float cq_1 = __fdiv_rn(dot_1[0], 128.0f);
                float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
                float nkx_1 = -kx_1;
                if (row_begin + (it0 + 1) * 8 + rg < row_end) {
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_4 * x0_4, rq_1, acc[0]);
                    acc[1] = __fmaf_rn(g1_4 * x1_4, rq_1, acc[1]);
                    float y0_4 = __fmaf_rn(nkx_1, x0_4, rq_1 * (g0_4 * __uint_as_float(ww[0] << 16)));
                    float y1_4 = __fmaf_rn(nkx_1, x1_4, rq_1 * (g1_4 * __uint_as_float(ww[0] & 4294901760u)));
                    out_1[0] = y0_4;
                    out_1[1] = y1_4;
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_5 * x0_5, rq_1, acc[2]);
                    acc[3] = __fmaf_rn(g1_5 * x1_5, rq_1, acc[3]);
                    float y0_5 = __fmaf_rn(nkx_1, x0_5, rq_1 * (g0_5 * __uint_as_float(ww[1] << 16)));
                    float y1_5 = __fmaf_rn(nkx_1, x1_5, rq_1 * (g1_5 * __uint_as_float(ww[1] & 4294901760u)));
                    out_1[2] = y0_5;
                    out_1[3] = y1_5;
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_6 * x0_6, rq_1, acc[4]);
                    acc[5] = __fmaf_rn(g1_6 * x1_6, rq_1, acc[5]);
                    float y0_6 = __fmaf_rn(nkx_1, x0_6, rq_1 * (g0_6 * __uint_as_float(ww[2] << 16)));
                    float y1_6 = __fmaf_rn(nkx_1, x1_6, rq_1 * (g1_6 * __uint_as_float(ww[2] & 4294901760u)));
                    out_1[4] = y0_6;
                    out_1[5] = y1_6;
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_7 * x0_7, rq_1, acc[6]);
                    acc[7] = __fmaf_rn(g1_7 * x1_7, rq_1, acc[7]);
                    float y0_7 = __fmaf_rn(nkx_1, x0_7, rq_1 * (g0_7 * __uint_as_float(ww[3] << 16)));
                    float y1_7 = __fmaf_rn(nkx_1, x1_7, rq_1 * (g1_7 * __uint_as_float(ww[3] & 4294901760u)));
                    out_1[6] = y0_7;
                    out_1[7] = y1_7;
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 1) * 8 + rg) * 128ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
            }
            if (n_iters > it0 + 2) {
                if (n_iters > it0 + 2 + 3) {
                    int _min_38 = ((row_begin + (it0 + 2 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 2 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long g_base_5 = (unsigned long long)_min_38 * (unsigned long long)g_stride;
                    int _min_39 = ((row_begin + (it0 + 2 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 2 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long x_base_5 = (unsigned long long)_min_39 * (unsigned long long)x_stride;
                    int _min_40 = ((row_begin + (it0 + 2 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 2 + 4 - 1) * 8 + rg) : (last_row));
                    rr[1] = r[_min_40];
                    {
                        const uint4* _vptr_41 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)(tir * 8));
                        uint4* _vdst_41 = reinterpret_cast<uint4*>(&gw[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_41[_blk].x), "=r"(_vdst_41[_blk].y), "=r"(_vdst_41[_blk].z), "=r"(_vdst_41[_blk].w) : "l"((const void*)(_vptr_41 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_42 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)(tir * 8));
                        uint4* _vdst_42 = reinterpret_cast<uint4*>(&xw[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_42[_blk].x), "=r"(_vdst_42[_blk].y), "=r"(_vdst_42[_blk].z), "=r"(_vdst_42[_blk].w) : "l"((const void*)(_vptr_42 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot_2[1];
                float dq_2 = 0.0f;
                dq_2 = __fmaf_rn(__uint_as_float(gw[8] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[8] << 16), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[8] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[8] & 4294901760u), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[9] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[9] << 16), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[9] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[9] & 4294901760u), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[10] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[10] << 16), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[10] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[10] & 4294901760u), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[11] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[11] << 16), dq_2);
                dq_2 = __fmaf_rn(__uint_as_float(gw[11] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[11] & 4294901760u), dq_2);
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dq_2, 8);
                dq_2 = dq_2 + _shfl_xor_sync_43;
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dq_2, 4);
                dq_2 = dq_2 + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dq_2, 2);
                dq_2 = dq_2 + _shfl_xor_sync_45;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dq_2, 1);
                dq_2 = dq_2 + _shfl_xor_sync_46;
                dot_2[0] = dq_2;
                float rq_2 = rr[2];
                float cq_2 = __fdiv_rn(dot_2[0], 128.0f);
                float kx_2 = rq_2 * rq_2 * rq_2 * cq_2;
                float nkx_2 = -kx_2;
                if (row_begin + (it0 + 2) * 8 + rg < row_end) {
                    float out_2[8];
                    float g0_8 = __uint_as_float(gw[8] << 16);
                    float g1_8 = __uint_as_float(gw[8] & 4294901760u);
                    float x0_8 = __uint_as_float(xw[8] << 16);
                    float x1_8 = __uint_as_float(xw[8] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_8 * x0_8, rq_2, acc[0]);
                    acc[1] = __fmaf_rn(g1_8 * x1_8, rq_2, acc[1]);
                    float y0_8 = __fmaf_rn(nkx_2, x0_8, rq_2 * (g0_8 * __uint_as_float(ww[0] << 16)));
                    float y1_8 = __fmaf_rn(nkx_2, x1_8, rq_2 * (g1_8 * __uint_as_float(ww[0] & 4294901760u)));
                    out_2[0] = y0_8;
                    out_2[1] = y1_8;
                    float g0_9 = __uint_as_float(gw[9] << 16);
                    float g1_9 = __uint_as_float(gw[9] & 4294901760u);
                    float x0_9 = __uint_as_float(xw[9] << 16);
                    float x1_9 = __uint_as_float(xw[9] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_9 * x0_9, rq_2, acc[2]);
                    acc[3] = __fmaf_rn(g1_9 * x1_9, rq_2, acc[3]);
                    float y0_9 = __fmaf_rn(nkx_2, x0_9, rq_2 * (g0_9 * __uint_as_float(ww[1] << 16)));
                    float y1_9 = __fmaf_rn(nkx_2, x1_9, rq_2 * (g1_9 * __uint_as_float(ww[1] & 4294901760u)));
                    out_2[2] = y0_9;
                    out_2[3] = y1_9;
                    float g0_10 = __uint_as_float(gw[10] << 16);
                    float g1_10 = __uint_as_float(gw[10] & 4294901760u);
                    float x0_10 = __uint_as_float(xw[10] << 16);
                    float x1_10 = __uint_as_float(xw[10] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_10 * x0_10, rq_2, acc[4]);
                    acc[5] = __fmaf_rn(g1_10 * x1_10, rq_2, acc[5]);
                    float y0_10 = __fmaf_rn(nkx_2, x0_10, rq_2 * (g0_10 * __uint_as_float(ww[2] << 16)));
                    float y1_10 = __fmaf_rn(nkx_2, x1_10, rq_2 * (g1_10 * __uint_as_float(ww[2] & 4294901760u)));
                    out_2[4] = y0_10;
                    out_2[5] = y1_10;
                    float g0_11 = __uint_as_float(gw[11] << 16);
                    float g1_11 = __uint_as_float(gw[11] & 4294901760u);
                    float x0_11 = __uint_as_float(xw[11] << 16);
                    float x1_11 = __uint_as_float(xw[11] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_11 * x0_11, rq_2, acc[6]);
                    acc[7] = __fmaf_rn(g1_11 * x1_11, rq_2, acc[7]);
                    float y0_11 = __fmaf_rn(nkx_2, x0_11, rq_2 * (g0_11 * __uint_as_float(ww[3] << 16)));
                    float y1_11 = __fmaf_rn(nkx_2, x1_11, rq_2 * (g1_11 * __uint_as_float(ww[3] & 4294901760u)));
                    out_2[6] = y0_11;
                    out_2[7] = y1_11;
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 2) * 8 + rg) * 128ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
                }
            }
            if (n_iters > it0 + 3) {
                if (n_iters > it0 + 3 + 3) {
                    int _min_47 = ((row_begin + (it0 + 3 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 3 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long g_base_6 = (unsigned long long)_min_47 * (unsigned long long)g_stride;
                    int _min_48 = ((row_begin + (it0 + 3 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 3 + 4 - 1) * 8 + rg) : (last_row));
                    unsigned long long x_base_6 = (unsigned long long)_min_48 * (unsigned long long)x_stride;
                    int _min_49 = ((row_begin + (it0 + 3 + 4 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 3 + 4 - 1) * 8 + rg) : (last_row));
                    rr[2] = r[_min_49];
                    {
                        const uint4* _vptr_50 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)(tir * 8));
                        uint4* _vdst_50 = reinterpret_cast<uint4*>(&gw[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_50[_blk].x), "=r"(_vdst_50[_blk].y), "=r"(_vdst_50[_blk].z), "=r"(_vdst_50[_blk].w) : "l"((const void*)(_vptr_50 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_51 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)(tir * 8));
                        uint4* _vdst_51 = reinterpret_cast<uint4*>(&xw[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_51[_blk].x), "=r"(_vdst_51[_blk].y), "=r"(_vdst_51[_blk].z), "=r"(_vdst_51[_blk].w) : "l"((const void*)(_vptr_51 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot_3[1];
                float dq_3 = 0.0f;
                dq_3 = __fmaf_rn(__uint_as_float(gw[12] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[12] << 16), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[12] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[12] & 4294901760u), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[13] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[13] << 16), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[13] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[13] & 4294901760u), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[14] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[14] << 16), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[14] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[14] & 4294901760u), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[15] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[15] << 16), dq_3);
                dq_3 = __fmaf_rn(__uint_as_float(gw[15] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[15] & 4294901760u), dq_3);
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dq_3, 8);
                dq_3 = dq_3 + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dq_3, 4);
                dq_3 = dq_3 + _shfl_xor_sync_53;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dq_3, 2);
                dq_3 = dq_3 + _shfl_xor_sync_54;
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dq_3, 1);
                dq_3 = dq_3 + _shfl_xor_sync_55;
                dot_3[0] = dq_3;
                float rq_3 = rr[3];
                float cq_3 = __fdiv_rn(dot_3[0], 128.0f);
                float kx_3 = rq_3 * rq_3 * rq_3 * cq_3;
                float nkx_3 = -kx_3;
                if (row_begin + (it0 + 3) * 8 + rg < row_end) {
                    float out_3[8];
                    float g0_12 = __uint_as_float(gw[12] << 16);
                    float g1_12 = __uint_as_float(gw[12] & 4294901760u);
                    float x0_12 = __uint_as_float(xw[12] << 16);
                    float x1_12 = __uint_as_float(xw[12] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_12 * x0_12, rq_3, acc[0]);
                    acc[1] = __fmaf_rn(g1_12 * x1_12, rq_3, acc[1]);
                    float y0_12 = __fmaf_rn(nkx_3, x0_12, rq_3 * (g0_12 * __uint_as_float(ww[0] << 16)));
                    float y1_12 = __fmaf_rn(nkx_3, x1_12, rq_3 * (g1_12 * __uint_as_float(ww[0] & 4294901760u)));
                    out_3[0] = y0_12;
                    out_3[1] = y1_12;
                    float g0_13 = __uint_as_float(gw[13] << 16);
                    float g1_13 = __uint_as_float(gw[13] & 4294901760u);
                    float x0_13 = __uint_as_float(xw[13] << 16);
                    float x1_13 = __uint_as_float(xw[13] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_13 * x0_13, rq_3, acc[2]);
                    acc[3] = __fmaf_rn(g1_13 * x1_13, rq_3, acc[3]);
                    float y0_13 = __fmaf_rn(nkx_3, x0_13, rq_3 * (g0_13 * __uint_as_float(ww[1] << 16)));
                    float y1_13 = __fmaf_rn(nkx_3, x1_13, rq_3 * (g1_13 * __uint_as_float(ww[1] & 4294901760u)));
                    out_3[2] = y0_13;
                    out_3[3] = y1_13;
                    float g0_14 = __uint_as_float(gw[14] << 16);
                    float g1_14 = __uint_as_float(gw[14] & 4294901760u);
                    float x0_14 = __uint_as_float(xw[14] << 16);
                    float x1_14 = __uint_as_float(xw[14] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_14 * x0_14, rq_3, acc[4]);
                    acc[5] = __fmaf_rn(g1_14 * x1_14, rq_3, acc[5]);
                    float y0_14 = __fmaf_rn(nkx_3, x0_14, rq_3 * (g0_14 * __uint_as_float(ww[2] << 16)));
                    float y1_14 = __fmaf_rn(nkx_3, x1_14, rq_3 * (g1_14 * __uint_as_float(ww[2] & 4294901760u)));
                    out_3[4] = y0_14;
                    out_3[5] = y1_14;
                    float g0_15 = __uint_as_float(gw[15] << 16);
                    float g1_15 = __uint_as_float(gw[15] & 4294901760u);
                    float x0_15 = __uint_as_float(xw[15] << 16);
                    float x1_15 = __uint_as_float(xw[15] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_15 * x0_15, rq_3, acc[6]);
                    acc[7] = __fmaf_rn(g1_15 * x1_15, rq_3, acc[7]);
                    float y0_15 = __fmaf_rn(nkx_3, x0_15, rq_3 * (g0_15 * __uint_as_float(ww[3] << 16)));
                    float y1_15 = __fmaf_rn(nkx_3, x1_15, rq_3 * (g1_15 * __uint_as_float(ww[3] & 4294901760u)));
                    out_3[6] = y0_15;
                    out_3[7] = y1_15;
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 3) * 8 + rg) * 128ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
                }
            }
            window = window + 1;
            if (window == 4) {
                sums[rg * 128 + tir] = sums[rg * 128 + tir] + acc[0];
                acc[0] = 0.0f;
                sums[rg * 128 + 16 + tir] = sums[rg * 128 + 16 + tir] + acc[1];
                acc[1] = 0.0f;
                sums[rg * 128 + 32 + tir] = sums[rg * 128 + 32 + tir] + acc[2];
                acc[2] = 0.0f;
                sums[rg * 128 + 48 + tir] = sums[rg * 128 + 48 + tir] + acc[3];
                acc[3] = 0.0f;
                sums[rg * 128 + 64 + tir] = sums[rg * 128 + 64 + tir] + acc[4];
                acc[4] = 0.0f;
                sums[rg * 128 + 80 + tir] = sums[rg * 128 + 80 + tir] + acc[5];
                acc[5] = 0.0f;
                sums[rg * 128 + 96 + tir] = sums[rg * 128 + 96 + tir] + acc[6];
                acc[6] = 0.0f;
                sums[rg * 128 + 112 + tir] = sums[rg * 128 + 112 + tir] + acc[7];
                acc[7] = 0.0f;
                window = 0;
            }
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
    sums[rg * 128 + tir] = sums[rg * 128 + tir] + acc[0];
    acc[0] = 0.0f;
    sums[rg * 128 + 16 + tir] = sums[rg * 128 + 16 + tir] + acc[1];
    acc[1] = 0.0f;
    sums[rg * 128 + 32 + tir] = sums[rg * 128 + 32 + tir] + acc[2];
    acc[2] = 0.0f;
    sums[rg * 128 + 48 + tir] = sums[rg * 128 + 48 + tir] + acc[3];
    acc[3] = 0.0f;
    sums[rg * 128 + 64 + tir] = sums[rg * 128 + 64 + tir] + acc[4];
    acc[4] = 0.0f;
    sums[rg * 128 + 80 + tir] = sums[rg * 128 + 80 + tir] + acc[5];
    acc[5] = 0.0f;
    sums[rg * 128 + 96 + tir] = sums[rg * 128 + 96 + tir] + acc[6];
    acc[6] = 0.0f;
    sums[rg * 128 + 112 + tir] = sums[rg * 128 + 112 + tir] + acc[7];
    acc[7] = 0.0f;
    int _min_56 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
#if __CUDA_ARCH__ == 1070
    int nrep = ((_min_56) < (64) ? (_min_56) : (64));
#else
    int nrep = ((_min_56) < (2) ? (_min_56) : (2));
#endif
    __syncthreads();
    int at_0 = tid;
    float rv_0[8];
    rv_0[0] = sums[at_0];
    rv_0[1] = sums[128 + at_0];
    rv_0[2] = sums[256 + at_0];
    rv_0[3] = sums[384 + at_0];
    rv_0[4] = sums[512 + at_0];
    rv_0[5] = sums[640 + at_0];
    rv_0[6] = sums[768 + at_0];
    rv_0[7] = sums[896 + at_0];
    rv_0[0] = rv_0[0] + rv_0[1];
    rv_0[2] = rv_0[2] + rv_0[3];
    rv_0[4] = rv_0[4] + rv_0[5];
    rv_0[6] = rv_0[6] + rv_0[7];
    rv_0[0] = rv_0[0] + rv_0[2];
    rv_0[4] = rv_0[4] + rv_0[6];
    rv_0[0] = rv_0[0] + rv_0[4];
    sums[at_0] = rv_0[0];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_57 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 128));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_57), "r"(sums_addr + 0), "r"((uint32_t)(128))
                    : "memory");
            }
            {
                void* _cpred_dst_58 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 128 + 32));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_58), "r"(sums_addr + 128), "r"((uint32_t)(128))
                    : "memory");
            }
            {
                void* _cpred_dst_59 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 128 + 64));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_59), "r"(sums_addr + 256), "r"((uint32_t)(128))
                    : "memory");
            }
            {
                void* _cpred_dst_60 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 128 + 96));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_60), "r"(sums_addr + 384), "r"((uint32_t)(128))
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
#if __CUDA_ARCH__ == 1070
        float buf[64];
#else
        float buf[4];
#endif
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_61;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_61);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_62;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (128 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_62);
            }
        }
#if __CUDA_ARCH__ == 1070
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_63;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (256 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_63);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_64;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (384 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_64);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_65;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (512 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_65);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_66;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (640 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_66);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_67;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (768 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_67);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_68;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (896 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_68);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_69;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (1024 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_69);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_70;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (1152 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_70);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_71;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (1280 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_71);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_72;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (1408 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_72);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_73;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (1536 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_73);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_74;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (1664 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_74);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_75;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (1792 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_75);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_76;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_76);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_77;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (2048 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_77);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_78;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_78) : "l"((const void*)(partial + (2176 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_78);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_79;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_79) : "l"((const void*)(partial + (2304 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_79);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_80;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_80) : "l"((const void*)(partial + (2432 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_80);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_81;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_81) : "l"((const void*)(partial + (2560 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_81);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_82;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_82) : "l"((const void*)(partial + (2688 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_82);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_83;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_83) : "l"((const void*)(partial + (2816 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_83);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_84;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_84) : "l"((const void*)(partial + (2944 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_84);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_85;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_85) : "l"((const void*)(partial + (3072 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_85);
            }
        }
        buf[25] = 0.0f;
        if (25 < nrep) {
            {
                uint32_t _scalar_bits_86;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_86) : "l"((const void*)(partial + (3200 + tid))) : "memory");
                buf[25] = __uint_as_float(_scalar_bits_86);
            }
        }
        buf[26] = 0.0f;
        if (26 < nrep) {
            {
                uint32_t _scalar_bits_87;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_87) : "l"((const void*)(partial + (3328 + tid))) : "memory");
                buf[26] = __uint_as_float(_scalar_bits_87);
            }
        }
        buf[27] = 0.0f;
        if (27 < nrep) {
            {
                uint32_t _scalar_bits_88;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_88) : "l"((const void*)(partial + (3456 + tid))) : "memory");
                buf[27] = __uint_as_float(_scalar_bits_88);
            }
        }
        buf[28] = 0.0f;
        if (28 < nrep) {
            {
                uint32_t _scalar_bits_89;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_89) : "l"((const void*)(partial + (3584 + tid))) : "memory");
                buf[28] = __uint_as_float(_scalar_bits_89);
            }
        }
        buf[29] = 0.0f;
        if (29 < nrep) {
            {
                uint32_t _scalar_bits_90;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_90) : "l"((const void*)(partial + (3712 + tid))) : "memory");
                buf[29] = __uint_as_float(_scalar_bits_90);
            }
        }
        buf[30] = 0.0f;
        if (30 < nrep) {
            {
                uint32_t _scalar_bits_91;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_91) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                buf[30] = __uint_as_float(_scalar_bits_91);
            }
        }
        buf[31] = 0.0f;
        if (31 < nrep) {
            {
                uint32_t _scalar_bits_92;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_92) : "l"((const void*)(partial + (3968 + tid))) : "memory");
                buf[31] = __uint_as_float(_scalar_bits_92);
            }
        }
        buf[32] = 0.0f;
        if (32 < nrep) {
            {
                uint32_t _scalar_bits_93;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_93) : "l"((const void*)(partial + (4096 + tid))) : "memory");
                buf[32] = __uint_as_float(_scalar_bits_93);
            }
        }
        buf[33] = 0.0f;
        if (33 < nrep) {
            {
                uint32_t _scalar_bits_94;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_94) : "l"((const void*)(partial + (4224 + tid))) : "memory");
                buf[33] = __uint_as_float(_scalar_bits_94);
            }
        }
        buf[34] = 0.0f;
        if (34 < nrep) {
            {
                uint32_t _scalar_bits_95;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_95) : "l"((const void*)(partial + (4352 + tid))) : "memory");
                buf[34] = __uint_as_float(_scalar_bits_95);
            }
        }
        buf[35] = 0.0f;
        if (35 < nrep) {
            {
                uint32_t _scalar_bits_96;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_96) : "l"((const void*)(partial + (4480 + tid))) : "memory");
                buf[35] = __uint_as_float(_scalar_bits_96);
            }
        }
        buf[36] = 0.0f;
        if (36 < nrep) {
            {
                uint32_t _scalar_bits_97;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_97) : "l"((const void*)(partial + (4608 + tid))) : "memory");
                buf[36] = __uint_as_float(_scalar_bits_97);
            }
        }
        buf[37] = 0.0f;
        if (37 < nrep) {
            {
                uint32_t _scalar_bits_98;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_98) : "l"((const void*)(partial + (4736 + tid))) : "memory");
                buf[37] = __uint_as_float(_scalar_bits_98);
            }
        }
        buf[38] = 0.0f;
        if (38 < nrep) {
            {
                uint32_t _scalar_bits_99;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_99) : "l"((const void*)(partial + (4864 + tid))) : "memory");
                buf[38] = __uint_as_float(_scalar_bits_99);
            }
        }
        buf[39] = 0.0f;
        if (39 < nrep) {
            {
                uint32_t _scalar_bits_100;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_100) : "l"((const void*)(partial + (4992 + tid))) : "memory");
                buf[39] = __uint_as_float(_scalar_bits_100);
            }
        }
        buf[40] = 0.0f;
        if (40 < nrep) {
            {
                uint32_t _scalar_bits_101;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_101) : "l"((const void*)(partial + (5120 + tid))) : "memory");
                buf[40] = __uint_as_float(_scalar_bits_101);
            }
        }
        buf[41] = 0.0f;
        if (41 < nrep) {
            {
                uint32_t _scalar_bits_102;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_102) : "l"((const void*)(partial + (5248 + tid))) : "memory");
                buf[41] = __uint_as_float(_scalar_bits_102);
            }
        }
        buf[42] = 0.0f;
        if (42 < nrep) {
            {
                uint32_t _scalar_bits_103;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_103) : "l"((const void*)(partial + (5376 + tid))) : "memory");
                buf[42] = __uint_as_float(_scalar_bits_103);
            }
        }
        buf[43] = 0.0f;
        if (43 < nrep) {
            {
                uint32_t _scalar_bits_104;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_104) : "l"((const void*)(partial + (5504 + tid))) : "memory");
                buf[43] = __uint_as_float(_scalar_bits_104);
            }
        }
        buf[44] = 0.0f;
        if (44 < nrep) {
            {
                uint32_t _scalar_bits_105;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_105) : "l"((const void*)(partial + (5632 + tid))) : "memory");
                buf[44] = __uint_as_float(_scalar_bits_105);
            }
        }
        buf[45] = 0.0f;
        if (45 < nrep) {
            {
                uint32_t _scalar_bits_106;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_106) : "l"((const void*)(partial + (5760 + tid))) : "memory");
                buf[45] = __uint_as_float(_scalar_bits_106);
            }
        }
        buf[46] = 0.0f;
        if (46 < nrep) {
            {
                uint32_t _scalar_bits_107;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_107) : "l"((const void*)(partial + (5888 + tid))) : "memory");
                buf[46] = __uint_as_float(_scalar_bits_107);
            }
        }
        buf[47] = 0.0f;
        if (47 < nrep) {
            {
                uint32_t _scalar_bits_108;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_108) : "l"((const void*)(partial + (6016 + tid))) : "memory");
                buf[47] = __uint_as_float(_scalar_bits_108);
            }
        }
        buf[48] = 0.0f;
        if (48 < nrep) {
            {
                uint32_t _scalar_bits_109;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_109) : "l"((const void*)(partial + (6144 + tid))) : "memory");
                buf[48] = __uint_as_float(_scalar_bits_109);
            }
        }
        buf[49] = 0.0f;
        if (49 < nrep) {
            {
                uint32_t _scalar_bits_110;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_110) : "l"((const void*)(partial + (6272 + tid))) : "memory");
                buf[49] = __uint_as_float(_scalar_bits_110);
            }
        }
        buf[50] = 0.0f;
        if (50 < nrep) {
            {
                uint32_t _scalar_bits_111;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_111) : "l"((const void*)(partial + (6400 + tid))) : "memory");
                buf[50] = __uint_as_float(_scalar_bits_111);
            }
        }
        buf[51] = 0.0f;
        if (51 < nrep) {
            {
                uint32_t _scalar_bits_112;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_112) : "l"((const void*)(partial + (6528 + tid))) : "memory");
                buf[51] = __uint_as_float(_scalar_bits_112);
            }
        }
        buf[52] = 0.0f;
        if (52 < nrep) {
            {
                uint32_t _scalar_bits_113;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_113) : "l"((const void*)(partial + (6656 + tid))) : "memory");
                buf[52] = __uint_as_float(_scalar_bits_113);
            }
        }
        buf[53] = 0.0f;
        if (53 < nrep) {
            {
                uint32_t _scalar_bits_114;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_114) : "l"((const void*)(partial + (6784 + tid))) : "memory");
                buf[53] = __uint_as_float(_scalar_bits_114);
            }
        }
        buf[54] = 0.0f;
        if (54 < nrep) {
            {
                uint32_t _scalar_bits_115;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_115) : "l"((const void*)(partial + (6912 + tid))) : "memory");
                buf[54] = __uint_as_float(_scalar_bits_115);
            }
        }
        buf[55] = 0.0f;
        if (55 < nrep) {
            {
                uint32_t _scalar_bits_116;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_116) : "l"((const void*)(partial + (7040 + tid))) : "memory");
                buf[55] = __uint_as_float(_scalar_bits_116);
            }
        }
        buf[56] = 0.0f;
        if (56 < nrep) {
            {
                uint32_t _scalar_bits_117;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_117) : "l"((const void*)(partial + (7168 + tid))) : "memory");
                buf[56] = __uint_as_float(_scalar_bits_117);
            }
        }
        buf[57] = 0.0f;
        if (57 < nrep) {
            {
                uint32_t _scalar_bits_118;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_118) : "l"((const void*)(partial + (7296 + tid))) : "memory");
                buf[57] = __uint_as_float(_scalar_bits_118);
            }
        }
        buf[58] = 0.0f;
        if (58 < nrep) {
            {
                uint32_t _scalar_bits_119;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_119) : "l"((const void*)(partial + (7424 + tid))) : "memory");
                buf[58] = __uint_as_float(_scalar_bits_119);
            }
        }
        buf[59] = 0.0f;
        if (59 < nrep) {
            {
                uint32_t _scalar_bits_120;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_120) : "l"((const void*)(partial + (7552 + tid))) : "memory");
                buf[59] = __uint_as_float(_scalar_bits_120);
            }
        }
        buf[60] = 0.0f;
        if (60 < nrep) {
            {
                uint32_t _scalar_bits_121;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_121) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                buf[60] = __uint_as_float(_scalar_bits_121);
            }
        }
        buf[61] = 0.0f;
        if (61 < nrep) {
            {
                uint32_t _scalar_bits_122;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_122) : "l"((const void*)(partial + (7808 + tid))) : "memory");
                buf[61] = __uint_as_float(_scalar_bits_122);
            }
        }
        buf[62] = 0.0f;
        if (62 < nrep) {
            {
                uint32_t _scalar_bits_123;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_123) : "l"((const void*)(partial + (7936 + tid))) : "memory");
                buf[62] = __uint_as_float(_scalar_bits_123);
            }
        }
        buf[63] = 0.0f;
        if (63 < nrep) {
            {
                uint32_t _scalar_bits_124;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_124) : "l"((const void*)(partial + (8064 + tid))) : "memory");
                buf[63] = __uint_as_float(_scalar_bits_124);
            }
        }
#endif
        buf[0] = buf[0] + buf[1];
#if __CUDA_ARCH__ == 1070
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[24] = buf[24] + buf[25];
        buf[26] = buf[26] + buf[27];
        buf[28] = buf[28] + buf[29];
        buf[30] = buf[30] + buf[31];
        buf[32] = buf[32] + buf[33];
        buf[34] = buf[34] + buf[35];
        buf[36] = buf[36] + buf[37];
        buf[38] = buf[38] + buf[39];
        buf[40] = buf[40] + buf[41];
        buf[42] = buf[42] + buf[43];
        buf[44] = buf[44] + buf[45];
        buf[46] = buf[46] + buf[47];
        buf[48] = buf[48] + buf[49];
        buf[50] = buf[50] + buf[51];
        buf[52] = buf[52] + buf[53];
        buf[54] = buf[54] + buf[55];
        buf[56] = buf[56] + buf[57];
        buf[58] = buf[58] + buf[59];
        buf[60] = buf[60] + buf[61];
        buf[62] = buf[62] + buf[63];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[24] = buf[24] + buf[26];
        buf[28] = buf[28] + buf[30];
        buf[32] = buf[32] + buf[34];
        buf[36] = buf[36] + buf[38];
        buf[40] = buf[40] + buf[42];
        buf[44] = buf[44] + buf[46];
        buf[48] = buf[48] + buf[50];
        buf[52] = buf[52] + buf[54];
        buf[56] = buf[56] + buf[58];
        buf[60] = buf[60] + buf[62];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[24] = buf[24] + buf[28];
        buf[32] = buf[32] + buf[36];
        buf[40] = buf[40] + buf[44];
        buf[48] = buf[48] + buf[52];
        buf[56] = buf[56] + buf[60];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[32] = buf[32] + buf[40];
        buf[48] = buf[48] + buf[56];
        buf[0] = buf[0] + buf[16];
        buf[32] = buf[32] + buf[48];
        buf[0] = buf[0] + buf[32];
#endif
        int at_w_0 = tid;
        dwbuf[(at_w_0 / 16 / 8 * 16 + at_w_0 % 16) * 8 + at_w_0 / 16 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (128 + at_w_0)) = 0.0f;
#if __CUDA_ARCH__ == 1070
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (256 + at_w_0)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (384 + at_w_0)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (512 + at_w_0)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (640 + at_w_0)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (768 + at_w_0)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (896 + at_w_0)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_0)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1152 + at_w_0)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_0)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1408 + at_w_0)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1536 + at_w_0)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1664 + at_w_0)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1792 + at_w_0)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1920 + at_w_0)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2048 + at_w_0)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2176 + at_w_0)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2304 + at_w_0)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2432 + at_w_0)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_0)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2688 + at_w_0)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2816 + at_w_0)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2944 + at_w_0)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3072 + at_w_0)) = 0.0f;
        }
        if (25 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3200 + at_w_0)) = 0.0f;
        }
        if (26 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3328 + at_w_0)) = 0.0f;
        }
        if (27 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3456 + at_w_0)) = 0.0f;
        }
        if (28 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3584 + at_w_0)) = 0.0f;
        }
        if (29 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3712 + at_w_0)) = 0.0f;
        }
        if (30 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_0)) = 0.0f;
        }
        if (31 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3968 + at_w_0)) = 0.0f;
        }
        if (32 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4096 + at_w_0)) = 0.0f;
        }
        if (33 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4224 + at_w_0)) = 0.0f;
        }
        if (34 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4352 + at_w_0)) = 0.0f;
        }
        if (35 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4480 + at_w_0)) = 0.0f;
        }
        if (36 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4608 + at_w_0)) = 0.0f;
        }
        if (37 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4736 + at_w_0)) = 0.0f;
        }
        if (38 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4864 + at_w_0)) = 0.0f;
        }
        if (39 < nrep) {
            *(reinterpret_cast<float*>(partial) + (4992 + at_w_0)) = 0.0f;
        }
        if (40 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_0)) = 0.0f;
        }
        if (41 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5248 + at_w_0)) = 0.0f;
        }
        if (42 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5376 + at_w_0)) = 0.0f;
        }
        if (43 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5504 + at_w_0)) = 0.0f;
        }
        if (44 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5632 + at_w_0)) = 0.0f;
        }
        if (45 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5760 + at_w_0)) = 0.0f;
        }
        if (46 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5888 + at_w_0)) = 0.0f;
        }
        if (47 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6016 + at_w_0)) = 0.0f;
        }
        if (48 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6144 + at_w_0)) = 0.0f;
        }
        if (49 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6272 + at_w_0)) = 0.0f;
        }
        if (50 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_0)) = 0.0f;
        }
        if (51 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6528 + at_w_0)) = 0.0f;
        }
        if (52 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6656 + at_w_0)) = 0.0f;
        }
        if (53 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6784 + at_w_0)) = 0.0f;
        }
        if (54 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6912 + at_w_0)) = 0.0f;
        }
        if (55 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7040 + at_w_0)) = 0.0f;
        }
        if (56 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7168 + at_w_0)) = 0.0f;
        }
        if (57 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7296 + at_w_0)) = 0.0f;
        }
        if (58 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7424 + at_w_0)) = 0.0f;
        }
        if (59 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7552 + at_w_0)) = 0.0f;
        }
        if (60 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_0)) = 0.0f;
        }
        if (61 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7808 + at_w_0)) = 0.0f;
        }
        if (62 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7936 + at_w_0)) = 0.0f;
        }
        if (63 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8064 + at_w_0)) = 0.0f;
#endif
        }
        __syncthreads();
        int col_0 = tid * 4;
        if (col_0 < 128) {
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
    }
}

} // extern "C"
