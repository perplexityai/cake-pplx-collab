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

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



extern "C" {

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_e7cb6972099b7e10c035(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    float* sums = reinterpret_cast<float*>(smem_raw + 1168);
    const int sums_addr = smem + 1168;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1168);
    const int dwbuf_addr = smem + 1168;
    int* meta = reinterpret_cast<int*>(smem_raw + 1136);
    const int meta_addr = smem + 1136;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 21760);
    const int ring_addr = smem + 21760;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=320
            mbarrier_init(smem + 32, 320);
            mbarrier_init(smem + 40, 320);
            mbarrier_init(smem + 48, 320);
            mbarrier_init(smem + 56, 320);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 9) {
        { // compute_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int nxt = 0;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            long long g_stride = g_stride_p0;
            long long x_stride = x_stride_p0;
            long long gh_stride = g_h_stride_p0;
            int rg = warp / 10;
            int tir = tid - rg * 320;
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
                    : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((320 + tir) * 8)))) : "memory");
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
            int rows_mine = 0;
            int slot = 0;
            sums[rg * 5120 + tir] = 0.0f;
            sums[rg * 5120 + 320 + tir] = 0.0f;
            sums[rg * 5120 + 640 + tir] = 0.0f;
            sums[rg * 5120 + 960 + tir] = 0.0f;
            sums[rg * 5120 + 1280 + tir] = 0.0f;
            sums[rg * 5120 + 1600 + tir] = 0.0f;
            sums[rg * 5120 + 1920 + tir] = 0.0f;
            sums[rg * 5120 + 2240 + tir] = 0.0f;
            sums[rg * 5120 + 2560 + tir] = 0.0f;
            sums[rg * 5120 + 2880 + tir] = 0.0f;
            sums[rg * 5120 + 3200 + tir] = 0.0f;
            sums[rg * 5120 + 3520 + tir] = 0.0f;
            sums[rg * 5120 + 3840 + tir] = 0.0f;
            sums[rg * 5120 + 4160 + tir] = 0.0f;
            sums[rg * 5120 + 4480 + tir] = 0.0f;
            sums[rg * 5120 + 4800 + tir] = 0.0f;
            int ring_at = (int)ring_addr;
            unsigned int gw[8];
            unsigned int xw[8];
            unsigned int hw[1];
            float rr[1];
            float dot[1];
            #pragma unroll 1
            for (int k = 0; k < T + n_chunks + 1; k++) {
                int _stage_full = (int)((unsigned int)k % 4);
                int _parity_full = (int)((unsigned int)k / 4 % 2);
                mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                int row0 = meta[2 * (k % 4)];
                int nval = meta[2 * (k % 4) + 1];
                if (nval < 1) {
                    c = row0;
                    first_c = -1 - nval;
                    break;
                }
                rows_mine = rows_mine + nval;
                if (k == 0) {
                    if (tid == 0) {
                        unsigned int _atomic_fetch_add_2 = atomicAdd(&counters[3], 1);
                        slot = (int)_atomic_fetch_add_2;
                    }
                }
                int rbase = ring_at + k % 4 * 20480 + rg * 10240;
#if defined(_MSC_VER)
                uint64_t _smem_raw_3[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_3[0]), "=l"(_smem_raw_3[1]) : "r"(rbase + tir * 8 * 2));
#else
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(rbase + tir * 8 * 2));
#endif
                gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_4[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_4[0]), "=l"(_smem_raw_4[1]) : "r"(rbase + 10240 + tir * 8 * 2));
#else
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(rbase + 10240 + tir * 8 * 2));
#endif
                xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_5[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_5[0]), "=l"(_smem_raw_5[1]) : "r"(rbase + (320 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_5;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(rbase + (320 + tir) * 8 * 2));
#endif
                gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_6[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_6[0]), "=l"(_smem_raw_6[1]) : "r"(rbase + 10240 + (320 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(rbase + 10240 + (320 + tir) * 8 * 2));
#endif
                xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                mbarrier_arrive(empty_addr + ((int)((unsigned int)k % 4)) * 8);
                rr[0] = 0.0f;
                if (nval > rg) {
                    rr[0] = r[row0 + rg];
                }
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
                float _warp_reduce_7 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
                dot[0] = _warp_reduce_7;
                if (lane == 0) {
                    dots[(k & 1) * 10 + warp] = dot[0];
                }
                asm volatile("barrier.sync 1, 320;" ::: "memory");
                float tot = 0.0f;
                tot += dots[(k & 1) * 10 + rg * 10];
                tot += dots[(k & 1) * 10 + rg * 10 + 1];
                tot += dots[(k & 1) * 10 + rg * 10 + 2];
                tot += dots[(k & 1) * 10 + rg * 10 + 3];
                tot += dots[(k & 1) * 10 + rg * 10 + 4];
                tot += dots[(k & 1) * 10 + rg * 10 + 5];
                tot += dots[(k & 1) * 10 + rg * 10 + 6];
                tot += dots[(k & 1) * 10 + rg * 10 + 7];
                tot += dots[(k & 1) * 10 + rg * 10 + 8];
                tot += dots[(k & 1) * 10 + rg * 10 + 9];
                dot[0] = tot;
                float rq = rr[0];
                float cq = __fdiv_rn(dot[0], 5120.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row0 + rg < row0 + nval) {
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
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 5120ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
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
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 5120ull + (unsigned long long)((320 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
                if ((k + 1 & 7) == 0) {
                    sums[rg * 5120 + tir] = sums[rg * 5120 + tir] + acc[0];
                    acc[0] = 0.0f;
                    sums[rg * 5120 + 320 + tir] = sums[rg * 5120 + 320 + tir] + acc[1];
                    acc[1] = 0.0f;
                    sums[rg * 5120 + 640 + tir] = sums[rg * 5120 + 640 + tir] + acc[2];
                    acc[2] = 0.0f;
                    sums[rg * 5120 + 960 + tir] = sums[rg * 5120 + 960 + tir] + acc[3];
                    acc[3] = 0.0f;
                    sums[rg * 5120 + 1280 + tir] = sums[rg * 5120 + 1280 + tir] + acc[4];
                    acc[4] = 0.0f;
                    sums[rg * 5120 + 1600 + tir] = sums[rg * 5120 + 1600 + tir] + acc[5];
                    acc[5] = 0.0f;
                    sums[rg * 5120 + 1920 + tir] = sums[rg * 5120 + 1920 + tir] + acc[6];
                    acc[6] = 0.0f;
                    sums[rg * 5120 + 2240 + tir] = sums[rg * 5120 + 2240 + tir] + acc[7];
                    acc[7] = 0.0f;
                    sums[rg * 5120 + 2560 + tir] = sums[rg * 5120 + 2560 + tir] + acc[8];
                    acc[8] = 0.0f;
                    sums[rg * 5120 + 2880 + tir] = sums[rg * 5120 + 2880 + tir] + acc[9];
                    acc[9] = 0.0f;
                    sums[rg * 5120 + 3200 + tir] = sums[rg * 5120 + 3200 + tir] + acc[10];
                    acc[10] = 0.0f;
                    sums[rg * 5120 + 3520 + tir] = sums[rg * 5120 + 3520 + tir] + acc[11];
                    acc[11] = 0.0f;
                    sums[rg * 5120 + 3840 + tir] = sums[rg * 5120 + 3840 + tir] + acc[12];
                    acc[12] = 0.0f;
                    sums[rg * 5120 + 4160 + tir] = sums[rg * 5120 + 4160 + tir] + acc[13];
                    acc[13] = 0.0f;
                    sums[rg * 5120 + 4480 + tir] = sums[rg * 5120 + 4480 + tir] + acc[14];
                    acc[14] = 0.0f;
                    sums[rg * 5120 + 4800 + tir] = sums[rg * 5120 + 4800 + tir] + acc[15];
                    acc[15] = 0.0f;
                }
            }
            sums[rg * 5120 + tir] = sums[rg * 5120 + tir] + acc[0];
            acc[0] = 0.0f;
            sums[rg * 5120 + 320 + tir] = sums[rg * 5120 + 320 + tir] + acc[1];
            acc[1] = 0.0f;
            sums[rg * 5120 + 640 + tir] = sums[rg * 5120 + 640 + tir] + acc[2];
            acc[2] = 0.0f;
            sums[rg * 5120 + 960 + tir] = sums[rg * 5120 + 960 + tir] + acc[3];
            acc[3] = 0.0f;
            sums[rg * 5120 + 1280 + tir] = sums[rg * 5120 + 1280 + tir] + acc[4];
            acc[4] = 0.0f;
            sums[rg * 5120 + 1600 + tir] = sums[rg * 5120 + 1600 + tir] + acc[5];
            acc[5] = 0.0f;
            sums[rg * 5120 + 1920 + tir] = sums[rg * 5120 + 1920 + tir] + acc[6];
            acc[6] = 0.0f;
            sums[rg * 5120 + 2240 + tir] = sums[rg * 5120 + 2240 + tir] + acc[7];
            acc[7] = 0.0f;
            sums[rg * 5120 + 2560 + tir] = sums[rg * 5120 + 2560 + tir] + acc[8];
            acc[8] = 0.0f;
            sums[rg * 5120 + 2880 + tir] = sums[rg * 5120 + 2880 + tir] + acc[9];
            acc[9] = 0.0f;
            sums[rg * 5120 + 3200 + tir] = sums[rg * 5120 + 3200 + tir] + acc[10];
            acc[10] = 0.0f;
            sums[rg * 5120 + 3520 + tir] = sums[rg * 5120 + 3520 + tir] + acc[11];
            acc[11] = 0.0f;
            sums[rg * 5120 + 3840 + tir] = sums[rg * 5120 + 3840 + tir] + acc[12];
            acc[12] = 0.0f;
            sums[rg * 5120 + 4160 + tir] = sums[rg * 5120 + 4160 + tir] + acc[13];
            acc[13] = 0.0f;
            sums[rg * 5120 + 4480 + tir] = sums[rg * 5120 + 4480 + tir] + acc[14];
            acc[14] = 0.0f;
            sums[rg * 5120 + 4800 + tir] = sums[rg * 5120 + 4800 + tir] + acc[15];
            acc[15] = 0.0f;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            if (tid == 0) {
                if (first_c < n_chunks) {
                    {
                        void* _cpbulk_dst_8 = reinterpret_cast<void*>(partial + (long long)(slot * 5120));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_8), "r"(sums_addr), "r"((uint32_t)(20480))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(counters) + (1))), "r"(static_cast<unsigned int>((unsigned int)rows_mine)) : "memory");
                }
                int hs = -1;
                if (c - n_chunks < 128) {
                    hs = c - n_chunks;
                }
                flags[3] = hs;
                if (c == n_chunks + (int)gridDim.x - 1) {
                    *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
                }
            }
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            int hs_c = flags[3];
            if (hs_c >= 0) {
                if (tid == 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counters) + (1))) : "memory");
                    } while (_acquire_observed < static_cast<unsigned int>((unsigned int)T));
                    }
                }
                asm volatile("barrier.sync 1, 320;" ::: "memory");
                asm volatile("fence.acquire.gpu;" ::: "memory");
                unsigned int nw[1];
                {
                    uint32_t _scalar_bits_9;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_9) : "l"((const void*)(counters + (3))) : "memory");
                    nw[0] = (unsigned int)_scalar_bits_9;
                }
                int grid = (int)num_bids;
                int helpers = ((grid) < (128) ? (grid) : (128));
                int cl = tid % 10;
                int rl = tid / 10;
                #pragma unroll 1
                for (int kk = 0; kk < 128; kk++) {
                    int sl = hs_c + kk * helpers;
                    if (sl >= 128) {
                        break;
                    }
                    int at0 = sl * 40 + 4 * cl;
                    float p[4];
                    p[0] = 0.0f;
                    p[1] = 0.0f;
                    p[2] = 0.0f;
                    p[3] = 0.0f;
                    float buf[32];
                    #pragma unroll 1
                    for (int s0 = rl; s0 < grid; s0 += 256) {
                        if (s0 < grid) {
                            {
                                unsigned _v4_10_0;
                                unsigned _v4_10_1;
                                unsigned _v4_10_2;
                                unsigned _v4_10_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3) : "l"((const void*)(partial + (s0 * 5120 + at0))) : "memory");
                                buf[0 + 0] = __uint_as_float(_v4_10_0);
                                buf[0 + 1] = __uint_as_float(_v4_10_1);
                                buf[0 + 2] = __uint_as_float(_v4_10_2);
                                buf[0 + 3] = __uint_as_float(_v4_10_3);
                            }
                        }
                        if (s0 + 32 < grid) {
                            {
                                unsigned _v4_11_0;
                                unsigned _v4_11_1;
                                unsigned _v4_11_2;
                                unsigned _v4_11_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(partial + ((s0 + 32) * 5120 + at0))) : "memory");
                                buf[4 + 0] = __uint_as_float(_v4_11_0);
                                buf[4 + 1] = __uint_as_float(_v4_11_1);
                                buf[4 + 2] = __uint_as_float(_v4_11_2);
                                buf[4 + 3] = __uint_as_float(_v4_11_3);
                            }
                        }
                        if (s0 + 64 < grid) {
                            {
                                unsigned _v4_12_0;
                                unsigned _v4_12_1;
                                unsigned _v4_12_2;
                                unsigned _v4_12_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(partial + ((s0 + 64) * 5120 + at0))) : "memory");
                                buf[8 + 0] = __uint_as_float(_v4_12_0);
                                buf[8 + 1] = __uint_as_float(_v4_12_1);
                                buf[8 + 2] = __uint_as_float(_v4_12_2);
                                buf[8 + 3] = __uint_as_float(_v4_12_3);
                            }
                        }
                        if (s0 + 96 < grid) {
                            {
                                unsigned _v4_13_0;
                                unsigned _v4_13_1;
                                unsigned _v4_13_2;
                                unsigned _v4_13_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3) : "l"((const void*)(partial + ((s0 + 96) * 5120 + at0))) : "memory");
                                buf[12 + 0] = __uint_as_float(_v4_13_0);
                                buf[12 + 1] = __uint_as_float(_v4_13_1);
                                buf[12 + 2] = __uint_as_float(_v4_13_2);
                                buf[12 + 3] = __uint_as_float(_v4_13_3);
                            }
                        }
                        if (s0 + 128 < grid) {
                            {
                                unsigned _v4_14_0;
                                unsigned _v4_14_1;
                                unsigned _v4_14_2;
                                unsigned _v4_14_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_14_0), "=r"(_v4_14_1), "=r"(_v4_14_2), "=r"(_v4_14_3) : "l"((const void*)(partial + ((s0 + 128) * 5120 + at0))) : "memory");
                                buf[16 + 0] = __uint_as_float(_v4_14_0);
                                buf[16 + 1] = __uint_as_float(_v4_14_1);
                                buf[16 + 2] = __uint_as_float(_v4_14_2);
                                buf[16 + 3] = __uint_as_float(_v4_14_3);
                            }
                        }
                        if (s0 + 160 < grid) {
                            {
                                unsigned _v4_15_0;
                                unsigned _v4_15_1;
                                unsigned _v4_15_2;
                                unsigned _v4_15_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_15_0), "=r"(_v4_15_1), "=r"(_v4_15_2), "=r"(_v4_15_3) : "l"((const void*)(partial + ((s0 + 160) * 5120 + at0))) : "memory");
                                buf[20 + 0] = __uint_as_float(_v4_15_0);
                                buf[20 + 1] = __uint_as_float(_v4_15_1);
                                buf[20 + 2] = __uint_as_float(_v4_15_2);
                                buf[20 + 3] = __uint_as_float(_v4_15_3);
                            }
                        }
                        if (s0 + 192 < grid) {
                            {
                                unsigned _v4_16_0;
                                unsigned _v4_16_1;
                                unsigned _v4_16_2;
                                unsigned _v4_16_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_16_0), "=r"(_v4_16_1), "=r"(_v4_16_2), "=r"(_v4_16_3) : "l"((const void*)(partial + ((s0 + 192) * 5120 + at0))) : "memory");
                                buf[24 + 0] = __uint_as_float(_v4_16_0);
                                buf[24 + 1] = __uint_as_float(_v4_16_1);
                                buf[24 + 2] = __uint_as_float(_v4_16_2);
                                buf[24 + 3] = __uint_as_float(_v4_16_3);
                            }
                        }
                        if (s0 + 224 < grid) {
                            {
                                unsigned _v4_17_0;
                                unsigned _v4_17_1;
                                unsigned _v4_17_2;
                                unsigned _v4_17_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_17_0), "=r"(_v4_17_1), "=r"(_v4_17_2), "=r"(_v4_17_3) : "l"((const void*)(partial + ((s0 + 224) * 5120 + at0))) : "memory");
                                buf[28 + 0] = __uint_as_float(_v4_17_0);
                                buf[28 + 1] = __uint_as_float(_v4_17_1);
                                buf[28 + 2] = __uint_as_float(_v4_17_2);
                                buf[28 + 3] = __uint_as_float(_v4_17_3);
                            }
                        }
                        if (s0 < (int)nw[0]) {
                            p[0] = p[0] + buf[0];
                            p[1] = p[1] + buf[1];
                            p[2] = p[2] + buf[2];
                            p[3] = p[3] + buf[3];
                        }
                        if (s0 + 32 < (int)nw[0]) {
                            p[0] = p[0] + buf[4];
                            p[1] = p[1] + buf[5];
                            p[2] = p[2] + buf[6];
                            p[3] = p[3] + buf[7];
                        }
                        if (s0 + 64 < (int)nw[0]) {
                            p[0] = p[0] + buf[8];
                            p[1] = p[1] + buf[9];
                            p[2] = p[2] + buf[10];
                            p[3] = p[3] + buf[11];
                        }
                        if (s0 + 96 < (int)nw[0]) {
                            p[0] = p[0] + buf[12];
                            p[1] = p[1] + buf[13];
                            p[2] = p[2] + buf[14];
                            p[3] = p[3] + buf[15];
                        }
                        if (s0 + 128 < (int)nw[0]) {
                            p[0] = p[0] + buf[16];
                            p[1] = p[1] + buf[17];
                            p[2] = p[2] + buf[18];
                            p[3] = p[3] + buf[19];
                        }
                        if (s0 + 160 < (int)nw[0]) {
                            p[0] = p[0] + buf[20];
                            p[1] = p[1] + buf[21];
                            p[2] = p[2] + buf[22];
                            p[3] = p[3] + buf[23];
                        }
                        if (s0 + 192 < (int)nw[0]) {
                            p[0] = p[0] + buf[24];
                            p[1] = p[1] + buf[25];
                            p[2] = p[2] + buf[26];
                            p[3] = p[3] + buf[27];
                        }
                        if (s0 + 224 < (int)nw[0]) {
                            p[0] = p[0] + buf[28];
                            p[1] = p[1] + buf[29];
                            p[2] = p[2] + buf[30];
                            p[3] = p[3] + buf[31];
                        }
                    }
                    sums[4 * tid] = p[0];
                    sums[4 * tid + 1] = p[1];
                    sums[4 * tid + 2] = p[2];
                    sums[4 * tid + 3] = p[3];
                    asm volatile("barrier.sync 1, 320;" ::: "memory");
                    if (tid < 10) {
                        p[0] = p[0] + sums[4 * (10 + cl)];
                        p[1] = p[1] + sums[4 * (10 + cl) + 1];
                        p[2] = p[2] + sums[4 * (10 + cl) + 2];
                        p[3] = p[3] + sums[4 * (10 + cl) + 3];
                        p[0] = p[0] + sums[4 * (20 + cl)];
                        p[1] = p[1] + sums[4 * (20 + cl) + 1];
                        p[2] = p[2] + sums[4 * (20 + cl) + 2];
                        p[3] = p[3] + sums[4 * (20 + cl) + 3];
                        p[0] = p[0] + sums[4 * (30 + cl)];
                        p[1] = p[1] + sums[4 * (30 + cl) + 1];
                        p[2] = p[2] + sums[4 * (30 + cl) + 2];
                        p[3] = p[3] + sums[4 * (30 + cl) + 3];
                        p[0] = p[0] + sums[4 * (40 + cl)];
                        p[1] = p[1] + sums[4 * (40 + cl) + 1];
                        p[2] = p[2] + sums[4 * (40 + cl) + 2];
                        p[3] = p[3] + sums[4 * (40 + cl) + 3];
                        p[0] = p[0] + sums[4 * (50 + cl)];
                        p[1] = p[1] + sums[4 * (50 + cl) + 1];
                        p[2] = p[2] + sums[4 * (50 + cl) + 2];
                        p[3] = p[3] + sums[4 * (50 + cl) + 3];
                        p[0] = p[0] + sums[4 * (60 + cl)];
                        p[1] = p[1] + sums[4 * (60 + cl) + 1];
                        p[2] = p[2] + sums[4 * (60 + cl) + 2];
                        p[3] = p[3] + sums[4 * (60 + cl) + 3];
                        p[0] = p[0] + sums[4 * (70 + cl)];
                        p[1] = p[1] + sums[4 * (70 + cl) + 1];
                        p[2] = p[2] + sums[4 * (70 + cl) + 2];
                        p[3] = p[3] + sums[4 * (70 + cl) + 3];
                        p[0] = p[0] + sums[4 * (80 + cl)];
                        p[1] = p[1] + sums[4 * (80 + cl) + 1];
                        p[2] = p[2] + sums[4 * (80 + cl) + 2];
                        p[3] = p[3] + sums[4 * (80 + cl) + 3];
                        p[0] = p[0] + sums[4 * (90 + cl)];
                        p[1] = p[1] + sums[4 * (90 + cl) + 1];
                        p[2] = p[2] + sums[4 * (90 + cl) + 2];
                        p[3] = p[3] + sums[4 * (90 + cl) + 3];
                        p[0] = p[0] + sums[4 * (100 + cl)];
                        p[1] = p[1] + sums[4 * (100 + cl) + 1];
                        p[2] = p[2] + sums[4 * (100 + cl) + 2];
                        p[3] = p[3] + sums[4 * (100 + cl) + 3];
                        p[0] = p[0] + sums[4 * (110 + cl)];
                        p[1] = p[1] + sums[4 * (110 + cl) + 1];
                        p[2] = p[2] + sums[4 * (110 + cl) + 2];
                        p[3] = p[3] + sums[4 * (110 + cl) + 3];
                        p[0] = p[0] + sums[4 * (120 + cl)];
                        p[1] = p[1] + sums[4 * (120 + cl) + 1];
                        p[2] = p[2] + sums[4 * (120 + cl) + 2];
                        p[3] = p[3] + sums[4 * (120 + cl) + 3];
                        p[0] = p[0] + sums[4 * (130 + cl)];
                        p[1] = p[1] + sums[4 * (130 + cl) + 1];
                        p[2] = p[2] + sums[4 * (130 + cl) + 2];
                        p[3] = p[3] + sums[4 * (130 + cl) + 3];
                        p[0] = p[0] + sums[4 * (140 + cl)];
                        p[1] = p[1] + sums[4 * (140 + cl) + 1];
                        p[2] = p[2] + sums[4 * (140 + cl) + 2];
                        p[3] = p[3] + sums[4 * (140 + cl) + 3];
                        p[0] = p[0] + sums[4 * (150 + cl)];
                        p[1] = p[1] + sums[4 * (150 + cl) + 1];
                        p[2] = p[2] + sums[4 * (150 + cl) + 2];
                        p[3] = p[3] + sums[4 * (150 + cl) + 3];
                        p[0] = p[0] + sums[4 * (160 + cl)];
                        p[1] = p[1] + sums[4 * (160 + cl) + 1];
                        p[2] = p[2] + sums[4 * (160 + cl) + 2];
                        p[3] = p[3] + sums[4 * (160 + cl) + 3];
                        p[0] = p[0] + sums[4 * (170 + cl)];
                        p[1] = p[1] + sums[4 * (170 + cl) + 1];
                        p[2] = p[2] + sums[4 * (170 + cl) + 2];
                        p[3] = p[3] + sums[4 * (170 + cl) + 3];
                        p[0] = p[0] + sums[4 * (180 + cl)];
                        p[1] = p[1] + sums[4 * (180 + cl) + 1];
                        p[2] = p[2] + sums[4 * (180 + cl) + 2];
                        p[3] = p[3] + sums[4 * (180 + cl) + 3];
                        p[0] = p[0] + sums[4 * (190 + cl)];
                        p[1] = p[1] + sums[4 * (190 + cl) + 1];
                        p[2] = p[2] + sums[4 * (190 + cl) + 2];
                        p[3] = p[3] + sums[4 * (190 + cl) + 3];
                        p[0] = p[0] + sums[4 * (200 + cl)];
                        p[1] = p[1] + sums[4 * (200 + cl) + 1];
                        p[2] = p[2] + sums[4 * (200 + cl) + 2];
                        p[3] = p[3] + sums[4 * (200 + cl) + 3];
                        p[0] = p[0] + sums[4 * (210 + cl)];
                        p[1] = p[1] + sums[4 * (210 + cl) + 1];
                        p[2] = p[2] + sums[4 * (210 + cl) + 2];
                        p[3] = p[3] + sums[4 * (210 + cl) + 3];
                        p[0] = p[0] + sums[4 * (220 + cl)];
                        p[1] = p[1] + sums[4 * (220 + cl) + 1];
                        p[2] = p[2] + sums[4 * (220 + cl) + 2];
                        p[3] = p[3] + sums[4 * (220 + cl) + 3];
                        p[0] = p[0] + sums[4 * (230 + cl)];
                        p[1] = p[1] + sums[4 * (230 + cl) + 1];
                        p[2] = p[2] + sums[4 * (230 + cl) + 2];
                        p[3] = p[3] + sums[4 * (230 + cl) + 3];
                        p[0] = p[0] + sums[4 * (240 + cl)];
                        p[1] = p[1] + sums[4 * (240 + cl) + 1];
                        p[2] = p[2] + sums[4 * (240 + cl) + 2];
                        p[3] = p[3] + sums[4 * (240 + cl) + 3];
                        p[0] = p[0] + sums[4 * (250 + cl)];
                        p[1] = p[1] + sums[4 * (250 + cl) + 1];
                        p[2] = p[2] + sums[4 * (250 + cl) + 2];
                        p[3] = p[3] + sums[4 * (250 + cl) + 3];
                        p[0] = p[0] + sums[4 * (260 + cl)];
                        p[1] = p[1] + sums[4 * (260 + cl) + 1];
                        p[2] = p[2] + sums[4 * (260 + cl) + 2];
                        p[3] = p[3] + sums[4 * (260 + cl) + 3];
                        p[0] = p[0] + sums[4 * (270 + cl)];
                        p[1] = p[1] + sums[4 * (270 + cl) + 1];
                        p[2] = p[2] + sums[4 * (270 + cl) + 2];
                        p[3] = p[3] + sums[4 * (270 + cl) + 3];
                        p[0] = p[0] + sums[4 * (280 + cl)];
                        p[1] = p[1] + sums[4 * (280 + cl) + 1];
                        p[2] = p[2] + sums[4 * (280 + cl) + 2];
                        p[3] = p[3] + sums[4 * (280 + cl) + 3];
                        p[0] = p[0] + sums[4 * (290 + cl)];
                        p[1] = p[1] + sums[4 * (290 + cl) + 1];
                        p[2] = p[2] + sums[4 * (290 + cl) + 2];
                        p[3] = p[3] + sums[4 * (290 + cl) + 3];
                        p[0] = p[0] + sums[4 * (300 + cl)];
                        p[1] = p[1] + sums[4 * (300 + cl) + 1];
                        p[2] = p[2] + sums[4 * (300 + cl) + 2];
                        p[3] = p[3] + sums[4 * (300 + cl) + 3];
                        p[0] = p[0] + sums[4 * (310 + cl)];
                        p[1] = p[1] + sums[4 * (310 + cl) + 1];
                        p[2] = p[2] + sums[4 * (310 + cl) + 2];
                        p[3] = p[3] + sums[4 * (310 + cl) + 3];
                        *(reinterpret_cast<float*>(dw) + ((at0 / 320 / 8 * 320 + at0 % 320) * 8 + at0 / 320 % 8)) = p[0];
                        *(reinterpret_cast<float*>(dw) + (((at0 + 1) / 320 / 8 * 320 + (at0 + 1) % 320) * 8 + (at0 + 1) / 320 % 8)) = p[1];
                        *(reinterpret_cast<float*>(dw) + (((at0 + 2) / 320 / 8 * 320 + (at0 + 2) % 320) * 8 + (at0 + 2) / 320 % 8)) = p[2];
                        *(reinterpret_cast<float*>(dw) + (((at0 + 3) / 320 / 8 * 320 + (at0 + 3) % 320) * 8 + (at0 + 3) / 320 % 8)) = p[3];
                    }
                    asm volatile("barrier.sync 1, 320;" ::: "memory");
                }
                if (tid == 0) {
                    int add2 = 1;
                    if (hs_c == 0) {
                        add2 = 129 - helpers;
                    }
                    unsigned int old2;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old2) : "l"(&counters[2]), "r"(static_cast<uint32_t>(add2)) : "memory");
                    if (old2 + (unsigned int)add2 == 128) {
                        *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
                        *(reinterpret_cast<unsigned int*>(counters) + (2)) = 0;
                        *(reinterpret_cast<unsigned int*>(counters) + (3)) = 0;
                    }
                }
            }
        }
    }
    // ---- Role: prod ----
    if (warp == 10) {
        { // prod_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            if (elect_sync()) {
                unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
                int p_c = (int)_atomic_fetch_add_0;
                int p_first = p_c;
                int p_row = 0;
                int p_end = 0;
                unsigned int p_nxt = 0;
                if (p_c < n_chunks) {
                    int stages = T;
                    int sl_1 = rows_per_chunk;
                    if (sl_1 > stages) {
                        sl_1 = stages;
                    }
                    if (sl_1 < 1) {
                        sl_1 = 1;
                    }
                    int share = (stages + sl_1 - 1) / sl_1;
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
                    for (int k_1 = 0; k_1 < 5; k_1++) {
                        if (halving != 0) {
                            if (size > tail) {
                                if (rem >= sl_1 * size) {
                                    if (p_c < first + sl_1) {
                                        span = size;
                                        begin_it = base + (p_c - first) * size;
                                        halving = 0;
                                    } else {
                                        first = first + sl_1;
                                        base = base + sl_1 * size;
                                        rem = rem - sl_1 * size;
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
                        int quot = rem / sl_1;
                        int extra = rem - quot * sl_1;
                        if (p_c < first + extra) {
                            span = quot + 1;
                            begin_it = base + (p_c - first) * (quot + 1);
                        } else if (p_c < first + sl_1) {
                            span = quot;
                            begin_it = base + extra * (quot + 1) + (p_c - first - extra) * quot;
                        } else {
                            begin_it = stages;
                        }
                    }
                    int _min_1 = ((begin_it) < (T) ? (begin_it) : (T));
                    p_row = _min_1;
                    int _min_2 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
                    p_end = _min_2;
                    unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
                    p_nxt = _atomic_fetch_add_3;
                }
                int p_done = 0;
                int p_ring = (int)ring_addr;
                #pragma unroll 1
                for (int pk = 0; pk < T + n_chunks + 1; pk++) {
                    int _stage_empty = (int)((unsigned int)pk % 4);
                    int _parity_empty = (int)((unsigned int)pk / 4 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int slot_1 = pk % 4;
                    if (p_row >= p_end) {
                        if (p_c < n_chunks) {
                            p_c = (int)p_nxt;
                            if (p_c < n_chunks) {
                                int stages_1 = T;
                                int sl_2 = rows_per_chunk;
                                if (sl_2 > stages_1) {
                                    sl_2 = stages_1;
                                }
                                if (sl_2 < 1) {
                                    sl_2 = 1;
                                }
                                int share_1 = (stages_1 + sl_2 - 1) / sl_2;
                                int tail_1 = (share_1 + 11) / 12;
                                if (tail_1 < 1) {
                                    tail_1 = 1;
                                }
                                int rem_1 = stages_1;
                                int size_1 = (share_1 + 1) / 2;
                                int first_1 = 0;
                                int base_1 = 0;
                                int span_1 = 0;
                                int halving_1 = 1;
                                int begin_it_1 = 0;
                                #pragma unroll 1
                                for (int k_2 = 0; k_2 < 5; k_2++) {
                                    if (halving_1 != 0) {
                                        if (size_1 > tail_1) {
                                            if (rem_1 >= sl_2 * size_1) {
                                                if (p_c < first_1 + sl_2) {
                                                    span_1 = size_1;
                                                    begin_it_1 = base_1 + (p_c - first_1) * size_1;
                                                    halving_1 = 0;
                                                } else {
                                                    first_1 = first_1 + sl_2;
                                                    base_1 = base_1 + sl_2 * size_1;
                                                    rem_1 = rem_1 - sl_2 * size_1;
                                                    size_1 = (size_1 + 1) / 2;
                                                }
                                            } else {
                                                halving_1 = 0;
                                            }
                                        } else {
                                            halving_1 = 0;
                                        }
                                    }
                                }
                                if (span_1 == 0) {
                                    int quot_1 = rem_1 / sl_2;
                                    int extra_1 = rem_1 - quot_1 * sl_2;
                                    if (p_c < first_1 + extra_1) {
                                        span_1 = quot_1 + 1;
                                        begin_it_1 = base_1 + (p_c - first_1) * (quot_1 + 1);
                                    } else if (p_c < first_1 + sl_2) {
                                        span_1 = quot_1;
                                        begin_it_1 = base_1 + extra_1 * (quot_1 + 1) + (p_c - first_1 - extra_1) * quot_1;
                                    } else {
                                        begin_it_1 = stages_1;
                                    }
                                }
                                int _min_4 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                                p_row = _min_4;
                                int _min_5 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                                p_end = _min_5;
                                unsigned int _atomic_fetch_add_6 = atomicAdd(&counters[0], 1);
                                p_nxt = _atomic_fetch_add_6;
                            }
                        }
                    }
                    int nval_1 = 0;
                    if (p_row < p_end) {
                        int _min_7 = ((p_end - p_row) < (1) ? (p_end - p_row) : (1));
                        nval_1 = _min_7;
                    }
                    if (nval_1 > 0) {
                        meta[2 * slot_1] = p_row;
                        meta[2 * slot_1 + 1] = nval_1;
                        mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 4)) * 8, 20480);
                        int sbase = p_ring + slot_1 * 20480;
                        int _min_8 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
                        long long row_i_0 = (long long)(p_row + _min_8);
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride_p0)), "r"((unsigned)(10240)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride_p0)), "r"((unsigned)(10240)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                    } else {
                        meta[2 * slot_1] = p_c;
                        meta[2 * slot_1 + 1] = -1 - p_first;
                        mbarrier_arrive(full_addr + ((int)((unsigned int)pk % 4)) * 8);
                        p_done = 1;
                    }
                    p_row = p_row + nval_1;
                    if (p_done != 0) {
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
