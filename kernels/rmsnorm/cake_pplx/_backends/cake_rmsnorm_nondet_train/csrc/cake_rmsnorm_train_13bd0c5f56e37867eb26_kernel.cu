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
kernel_cake_rmsnorm_train_13bd0c5f56e37867eb26(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
                int rbase = ring_at + k % 4 * 20480 + rg * 10240;
#if defined(_MSC_VER)
                uint64_t _smem_raw_2[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_2[0]), "=l"(_smem_raw_2[1]) : "r"(rbase + tir * 8 * 2));
#else
                __int128_t _smem_raw_2;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(rbase + tir * 8 * 2));
#endif
                gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_3[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_3[0]), "=l"(_smem_raw_3[1]) : "r"(rbase + 10240 + tir * 8 * 2));
#else
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(rbase + 10240 + tir * 8 * 2));
#endif
                xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_4[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_4[0]), "=l"(_smem_raw_4[1]) : "r"(rbase + (320 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(rbase + (320 + tir) * 8 * 2));
#endif
                gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_5[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_5[0]), "=l"(_smem_raw_5[1]) : "r"(rbase + 10240 + (320 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_5;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(rbase + 10240 + (320 + tir) * 8 * 2));
#endif
                xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
                float _warp_reduce_6 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_6 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset);
                dot[0] = _warp_reduce_6;
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
            int _min_7 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
#if __CUDA_ARCH__ == 1070
            int nrep = ((_min_7) < (6) ? (_min_7) : (6));
#else
            int nrep = ((_min_7) < (4) ? (_min_7) : (4));
#endif
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            if (tid == 0) {
                if (first_c < n_chunks) {
                    {
                        void* _cpred_dst_8 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_8), "r"(sums_addr + 0), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_9 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 640));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_9), "r"(sums_addr + 2560), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_10 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 1280));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_10), "r"(sums_addr + 5120), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_11 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 1920));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_11), "r"(sums_addr + 7680), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_12 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 2560));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_12), "r"(sums_addr + 10240), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_13 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 3200));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_13), "r"(sums_addr + 12800), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_14 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 3840));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_14), "r"(sums_addr + 15360), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_15 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 5120 + 4480));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_15), "r"(sums_addr + 17920), "r"((uint32_t)(2560))
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
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            if (flags[2] != 0) {
#if __CUDA_ARCH__ == 1070
                float buf[48];
#else
                float buf[32];
#endif
                buf[0] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_16;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_16) : "l"((const void*)(partial + (tid))) : "memory");
                        buf[0] = __uint_as_float(_scalar_bits_16);
                    }
                }
                buf[1] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_17;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_17) : "l"((const void*)(partial + (5120 + tid))) : "memory");
                        buf[1] = __uint_as_float(_scalar_bits_17);
                    }
                }
                buf[2] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_18;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_18) : "l"((const void*)(partial + (10240 + tid))) : "memory");
                        buf[2] = __uint_as_float(_scalar_bits_18);
                    }
                }
                buf[3] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_19;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_19) : "l"((const void*)(partial + (15360 + tid))) : "memory");
                        buf[3] = __uint_as_float(_scalar_bits_19);
                    }
                }
                buf[4] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_20;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_20) : "l"((const void*)(partial + (20480 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_20) : "l"((const void*)(partial + (320 + tid))) : "memory");
#endif
                        buf[4] = __uint_as_float(_scalar_bits_20);
                    }
                }
                buf[5] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_21;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_21) : "l"((const void*)(partial + (25600 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_21) : "l"((const void*)(partial + (5440 + tid))) : "memory");
#endif
                        buf[5] = __uint_as_float(_scalar_bits_21);
                    }
                }
                buf[6] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (0 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_22;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_22) : "l"((const void*)(partial + (320 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_22) : "l"((const void*)(partial + (10560 + tid))) : "memory");
#endif
                        buf[6] = __uint_as_float(_scalar_bits_22);
                    }
                }
                buf[7] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (1 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_23;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_23) : "l"((const void*)(partial + (5440 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_23) : "l"((const void*)(partial + (15680 + tid))) : "memory");
#endif
                        buf[7] = __uint_as_float(_scalar_bits_23);
                    }
                }
                buf[8] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (2 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_24;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_24) : "l"((const void*)(partial + (10560 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_24) : "l"((const void*)(partial + (640 + tid))) : "memory");
#endif
                        buf[8] = __uint_as_float(_scalar_bits_24);
                    }
                }
                buf[9] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (3 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_25;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_25) : "l"((const void*)(partial + (15680 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_25) : "l"((const void*)(partial + (5760 + tid))) : "memory");
#endif
                        buf[9] = __uint_as_float(_scalar_bits_25);
                    }
                }
                buf[10] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_26;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_26) : "l"((const void*)(partial + (20800 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_26) : "l"((const void*)(partial + (10880 + tid))) : "memory");
#endif
                        buf[10] = __uint_as_float(_scalar_bits_26);
                    }
                }
                buf[11] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_27;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_27) : "l"((const void*)(partial + (25920 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_27) : "l"((const void*)(partial + (16000 + tid))) : "memory");
#endif
                        buf[11] = __uint_as_float(_scalar_bits_27);
                    }
                }
                buf[12] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_28;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_28) : "l"((const void*)(partial + (640 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_28) : "l"((const void*)(partial + (960 + tid))) : "memory");
#endif
                        buf[12] = __uint_as_float(_scalar_bits_28);
                    }
                }
                buf[13] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_29;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_29) : "l"((const void*)(partial + (5760 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_29) : "l"((const void*)(partial + (6080 + tid))) : "memory");
#endif
                        buf[13] = __uint_as_float(_scalar_bits_29);
                    }
                }
                buf[14] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_30;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_30) : "l"((const void*)(partial + (10880 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_30) : "l"((const void*)(partial + (11200 + tid))) : "memory");
#endif
                        buf[14] = __uint_as_float(_scalar_bits_30);
                    }
                }
                buf[15] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_31;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_31) : "l"((const void*)(partial + (16000 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_31) : "l"((const void*)(partial + (16320 + tid))) : "memory");
#endif
                        buf[15] = __uint_as_float(_scalar_bits_31);
                    }
                }
                buf[16] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_32;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_32) : "l"((const void*)(partial + (21120 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_32) : "l"((const void*)(partial + (1280 + tid))) : "memory");
#endif
                        buf[16] = __uint_as_float(_scalar_bits_32);
                    }
                }
                buf[17] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_33;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_33) : "l"((const void*)(partial + (26240 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_33) : "l"((const void*)(partial + (6400 + tid))) : "memory");
#endif
                        buf[17] = __uint_as_float(_scalar_bits_33);
                    }
                }
                buf[18] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (0 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_34;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_34) : "l"((const void*)(partial + (960 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_34) : "l"((const void*)(partial + (11520 + tid))) : "memory");
#endif
                        buf[18] = __uint_as_float(_scalar_bits_34);
                    }
                }
                buf[19] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (1 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_35;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_35) : "l"((const void*)(partial + (6080 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_35) : "l"((const void*)(partial + (16640 + tid))) : "memory");
#endif
                        buf[19] = __uint_as_float(_scalar_bits_35);
                    }
                }
                buf[20] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (2 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_36;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_36) : "l"((const void*)(partial + (11200 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_36) : "l"((const void*)(partial + (1600 + tid))) : "memory");
#endif
                        buf[20] = __uint_as_float(_scalar_bits_36);
                    }
                }
                buf[21] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (3 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_37;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_37) : "l"((const void*)(partial + (16320 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_37) : "l"((const void*)(partial + (6720 + tid))) : "memory");
#endif
                        buf[21] = __uint_as_float(_scalar_bits_37);
                    }
                }
                buf[22] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_38;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_38) : "l"((const void*)(partial + (21440 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_38) : "l"((const void*)(partial + (11840 + tid))) : "memory");
#endif
                        buf[22] = __uint_as_float(_scalar_bits_38);
                    }
                }
                buf[23] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_39;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_39) : "l"((const void*)(partial + (26560 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_39) : "l"((const void*)(partial + (16960 + tid))) : "memory");
#endif
                        buf[23] = __uint_as_float(_scalar_bits_39);
                    }
                }
                buf[24] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_40;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_40) : "l"((const void*)(partial + (1280 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_40) : "l"((const void*)(partial + (1920 + tid))) : "memory");
#endif
                        buf[24] = __uint_as_float(_scalar_bits_40);
                    }
                }
                buf[25] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_41;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_41) : "l"((const void*)(partial + (6400 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_41) : "l"((const void*)(partial + (7040 + tid))) : "memory");
#endif
                        buf[25] = __uint_as_float(_scalar_bits_41);
                    }
                }
                buf[26] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_42;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_42) : "l"((const void*)(partial + (11520 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_42) : "l"((const void*)(partial + (12160 + tid))) : "memory");
#endif
                        buf[26] = __uint_as_float(_scalar_bits_42);
                    }
                }
                buf[27] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_43;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_43) : "l"((const void*)(partial + (16640 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_43) : "l"((const void*)(partial + (17280 + tid))) : "memory");
#endif
                        buf[27] = __uint_as_float(_scalar_bits_43);
                    }
                }
                buf[28] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_44;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_44) : "l"((const void*)(partial + (21760 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_44) : "l"((const void*)(partial + (2240 + tid))) : "memory");
#endif
                        buf[28] = __uint_as_float(_scalar_bits_44);
                    }
                }
                buf[29] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_45;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_45) : "l"((const void*)(partial + (26880 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_45) : "l"((const void*)(partial + (7360 + tid))) : "memory");
#endif
                        buf[29] = __uint_as_float(_scalar_bits_45);
                    }
                }
                buf[30] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (0 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_46;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_46) : "l"((const void*)(partial + (1600 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_46) : "l"((const void*)(partial + (12480 + tid))) : "memory");
#endif
                        buf[30] = __uint_as_float(_scalar_bits_46);
                    }
                }
                buf[31] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (1 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
                        uint32_t _scalar_bits_47;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_47) : "l"((const void*)(partial + (6720 + tid))) : "memory");
#else
                            : "=r"(_scalar_bits_47) : "l"((const void*)(partial + (17600 + tid))) : "memory");
#endif
                        buf[31] = __uint_as_float(_scalar_bits_47);
#if __CUDA_ARCH__ == 1070
                    }
                }
                buf[32] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_48;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_48) : "l"((const void*)(partial + (11840 + tid))) : "memory");
                        buf[32] = __uint_as_float(_scalar_bits_48);
                    }
                }
                buf[33] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_49;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_49) : "l"((const void*)(partial + (16960 + tid))) : "memory");
                        buf[33] = __uint_as_float(_scalar_bits_49);
                    }
                }
                buf[34] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_50;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_50) : "l"((const void*)(partial + (22080 + tid))) : "memory");
                        buf[34] = __uint_as_float(_scalar_bits_50);
                    }
                }
                buf[35] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_51;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_51) : "l"((const void*)(partial + (27200 + tid))) : "memory");
                        buf[35] = __uint_as_float(_scalar_bits_51);
                    }
                }
                buf[36] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_52;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_52) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                        buf[36] = __uint_as_float(_scalar_bits_52);
                    }
                }
                buf[37] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_53;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_53) : "l"((const void*)(partial + (7040 + tid))) : "memory");
                        buf[37] = __uint_as_float(_scalar_bits_53);
                    }
                }
                buf[38] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_54;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_54) : "l"((const void*)(partial + (12160 + tid))) : "memory");
                        buf[38] = __uint_as_float(_scalar_bits_54);
                    }
                }
                buf[39] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_55;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_55) : "l"((const void*)(partial + (17280 + tid))) : "memory");
                        buf[39] = __uint_as_float(_scalar_bits_55);
                    }
                }
                buf[40] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_56;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_56) : "l"((const void*)(partial + (22400 + tid))) : "memory");
                        buf[40] = __uint_as_float(_scalar_bits_56);
                    }
                }
                buf[41] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_57;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (27520 + tid))) : "memory");
                        buf[41] = __uint_as_float(_scalar_bits_57);
                    }
                }
                buf[42] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_58;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (2240 + tid))) : "memory");
                        buf[42] = __uint_as_float(_scalar_bits_58);
                    }
                }
                buf[43] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_59;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (7360 + tid))) : "memory");
                        buf[43] = __uint_as_float(_scalar_bits_59);
                    }
                }
                buf[44] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_60;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (12480 + tid))) : "memory");
                        buf[44] = __uint_as_float(_scalar_bits_60);
                    }
                }
                buf[45] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_61;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (17600 + tid))) : "memory");
                        buf[45] = __uint_as_float(_scalar_bits_61);
                    }
                }
                buf[46] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_62;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (22720 + tid))) : "memory");
                        buf[46] = __uint_as_float(_scalar_bits_62);
                    }
                }
                buf[47] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_63;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (27840 + tid))) : "memory");
                        buf[47] = __uint_as_float(_scalar_bits_63);
#endif
                    }
                }
                buf[0] = buf[0] + buf[1];
                buf[2] = buf[2] + buf[3];
#if __CUDA_ARCH__ == 1070
                buf[4] = buf[4] + buf[5];
#endif
                buf[0] = buf[0] + buf[2];
#if __CUDA_ARCH__ == 1070
                buf[0] = buf[0] + buf[4];
#endif
                int at_w_0 = tid;
                dwbuf[(at_w_0 / 320 / 8 * 320 + at_w_0 % 320) * 8 + at_w_0 / 320 % 8] = buf[0];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_0)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_0)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_0)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_0)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_0)) = 0.0f;
                }
#else
                buf[4] = buf[4] + buf[5];
#endif
                buf[6] = buf[6] + buf[7];
#if __CUDA_ARCH__ == 1070
                buf[8] = buf[8] + buf[9];
                buf[10] = buf[10] + buf[11];
                buf[6] = buf[6] + buf[8];
                buf[6] = buf[6] + buf[10];
#else
                buf[4] = buf[4] + buf[6];
#endif
                int at_w_1 = 320 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_1 / 320 / 8 * 320 + at_w_1 % 320) * 8 + at_w_1 / 320 % 8] = buf[6];
#else
                dwbuf[(at_w_1 / 320 / 8 * 320 + at_w_1 % 320) * 8 + at_w_1 / 320 % 8] = buf[4];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_1)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_1)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_1)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_1)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_1)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_1)) = 0.0f;
                }
                buf[12] = buf[12] + buf[13];
                buf[14] = buf[14] + buf[15];
                buf[16] = buf[16] + buf[17];
                buf[12] = buf[12] + buf[14];
                buf[12] = buf[12] + buf[16];
#else
                buf[8] = buf[8] + buf[9];
                buf[10] = buf[10] + buf[11];
                buf[8] = buf[8] + buf[10];
#endif
                int at_w_2 = 640 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_2 / 320 / 8 * 320 + at_w_2 % 320) * 8 + at_w_2 / 320 % 8] = buf[12];
#else
                dwbuf[(at_w_2 / 320 / 8 * 320 + at_w_2 % 320) * 8 + at_w_2 / 320 % 8] = buf[8];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_2)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_2)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_2)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_2)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_2)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_2)) = 0.0f;
                }
                buf[18] = buf[18] + buf[19];
                buf[20] = buf[20] + buf[21];
                buf[22] = buf[22] + buf[23];
                buf[18] = buf[18] + buf[20];
                buf[18] = buf[18] + buf[22];
#else
                buf[12] = buf[12] + buf[13];
                buf[14] = buf[14] + buf[15];
                buf[12] = buf[12] + buf[14];
#endif
                int at_w_3 = 960 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_3 / 320 / 8 * 320 + at_w_3 % 320) * 8 + at_w_3 / 320 % 8] = buf[18];
#else
                dwbuf[(at_w_3 / 320 / 8 * 320 + at_w_3 % 320) * 8 + at_w_3 / 320 % 8] = buf[12];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_3)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_3)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_3)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_3)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_3)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_3)) = 0.0f;
                }
#endif
                buf[0] = 0.0f;
                if (0 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_64;
#else
                        uint32_t _scalar_bits_48;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (2560 + tid))) : "memory");
                        buf[0] = __uint_as_float(_scalar_bits_64);
#else
                            : "=r"(_scalar_bits_48) : "l"((const void*)(partial + (2560 + tid))) : "memory");
                        buf[0] = __uint_as_float(_scalar_bits_48);
#endif
                    }
                }
                buf[1] = 0.0f;
                if (1 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_65;
#else
                        uint32_t _scalar_bits_49;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                        buf[1] = __uint_as_float(_scalar_bits_65);
#else
                            : "=r"(_scalar_bits_49) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                        buf[1] = __uint_as_float(_scalar_bits_49);
#endif
                    }
                }
                buf[2] = 0.0f;
                if (2 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_66;
#else
                        uint32_t _scalar_bits_50;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (12800 + tid))) : "memory");
                        buf[2] = __uint_as_float(_scalar_bits_66);
#else
                            : "=r"(_scalar_bits_50) : "l"((const void*)(partial + (12800 + tid))) : "memory");
                        buf[2] = __uint_as_float(_scalar_bits_50);
#endif
                    }
                }
                buf[3] = 0.0f;
                if (3 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_67;
#else
                        uint32_t _scalar_bits_51;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (17920 + tid))) : "memory");
                        buf[3] = __uint_as_float(_scalar_bits_67);
#else
                            : "=r"(_scalar_bits_51) : "l"((const void*)(partial + (17920 + tid))) : "memory");
                        buf[3] = __uint_as_float(_scalar_bits_51);
#endif
                    }
                }
                buf[4] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_68;
#else
                        uint32_t _scalar_bits_52;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (23040 + tid))) : "memory");
                        buf[4] = __uint_as_float(_scalar_bits_68);
#else
                            : "=r"(_scalar_bits_52) : "l"((const void*)(partial + (2880 + tid))) : "memory");
                        buf[4] = __uint_as_float(_scalar_bits_52);
#endif
                    }
                }
                buf[5] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_69;
#else
                        uint32_t _scalar_bits_53;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (28160 + tid))) : "memory");
                        buf[5] = __uint_as_float(_scalar_bits_69);
#else
                            : "=r"(_scalar_bits_53) : "l"((const void*)(partial + (8000 + tid))) : "memory");
                        buf[5] = __uint_as_float(_scalar_bits_53);
#endif
                    }
                }
                buf[6] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (0 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_70;
#else
                        uint32_t _scalar_bits_54;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (2880 + tid))) : "memory");
                        buf[6] = __uint_as_float(_scalar_bits_70);
#else
                            : "=r"(_scalar_bits_54) : "l"((const void*)(partial + (13120 + tid))) : "memory");
                        buf[6] = __uint_as_float(_scalar_bits_54);
#endif
                    }
                }
                buf[7] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (1 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_71;
#else
                        uint32_t _scalar_bits_55;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (8000 + tid))) : "memory");
                        buf[7] = __uint_as_float(_scalar_bits_71);
#else
                            : "=r"(_scalar_bits_55) : "l"((const void*)(partial + (18240 + tid))) : "memory");
                        buf[7] = __uint_as_float(_scalar_bits_55);
#endif
                    }
                }
                buf[8] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (2 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_72;
#else
                        uint32_t _scalar_bits_56;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (13120 + tid))) : "memory");
                        buf[8] = __uint_as_float(_scalar_bits_72);
#else
                            : "=r"(_scalar_bits_56) : "l"((const void*)(partial + (3200 + tid))) : "memory");
                        buf[8] = __uint_as_float(_scalar_bits_56);
#endif
                    }
                }
                buf[9] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (3 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_73;
#else
                        uint32_t _scalar_bits_57;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (18240 + tid))) : "memory");
                        buf[9] = __uint_as_float(_scalar_bits_73);
#else
                            : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (8320 + tid))) : "memory");
                        buf[9] = __uint_as_float(_scalar_bits_57);
#endif
                    }
                }
                buf[10] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_74;
#else
                        uint32_t _scalar_bits_58;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (23360 + tid))) : "memory");
                        buf[10] = __uint_as_float(_scalar_bits_74);
#else
                            : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                        buf[10] = __uint_as_float(_scalar_bits_58);
#endif
                    }
                }
                buf[11] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (3 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_75;
#else
                        uint32_t _scalar_bits_59;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (28480 + tid))) : "memory");
                        buf[11] = __uint_as_float(_scalar_bits_75);
#else
                            : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (18560 + tid))) : "memory");
                        buf[11] = __uint_as_float(_scalar_bits_59);
#endif
                    }
                }
                buf[12] = 0.0f;
                if (0 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_76;
#else
                        uint32_t _scalar_bits_60;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (3200 + tid))) : "memory");
                        buf[12] = __uint_as_float(_scalar_bits_76);
#else
                            : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (3520 + tid))) : "memory");
                        buf[12] = __uint_as_float(_scalar_bits_60);
#endif
                    }
                }
                buf[13] = 0.0f;
                if (1 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_77;
#else
                        uint32_t _scalar_bits_61;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (8320 + tid))) : "memory");
                        buf[13] = __uint_as_float(_scalar_bits_77);
#else
                            : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (8640 + tid))) : "memory");
                        buf[13] = __uint_as_float(_scalar_bits_61);
#endif
                    }
                }
                buf[14] = 0.0f;
                if (2 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_78;
#else
                        uint32_t _scalar_bits_62;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_78) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                        buf[14] = __uint_as_float(_scalar_bits_78);
#else
                            : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (13760 + tid))) : "memory");
                        buf[14] = __uint_as_float(_scalar_bits_62);
#endif
                    }
                }
                buf[15] = 0.0f;
                if (3 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_79;
#else
                        uint32_t _scalar_bits_63;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_79) : "l"((const void*)(partial + (18560 + tid))) : "memory");
                        buf[15] = __uint_as_float(_scalar_bits_79);
#else
                            : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (18880 + tid))) : "memory");
                        buf[15] = __uint_as_float(_scalar_bits_63);
#endif
                    }
                }
#if __CUDA_ARCH__ == 1070
                buf[16] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_80;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_80) : "l"((const void*)(partial + (23680 + tid))) : "memory");
                        buf[16] = __uint_as_float(_scalar_bits_80);
                    }
                }
                buf[17] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_81;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_81) : "l"((const void*)(partial + (28800 + tid))) : "memory");
                        buf[17] = __uint_as_float(_scalar_bits_81);
                    }
                }
                buf[18] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_82;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_82) : "l"((const void*)(partial + (3520 + tid))) : "memory");
                        buf[18] = __uint_as_float(_scalar_bits_82);
                    }
                }
                buf[19] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_83;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_83) : "l"((const void*)(partial + (8640 + tid))) : "memory");
                        buf[19] = __uint_as_float(_scalar_bits_83);
                    }
                }
                buf[20] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_84;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_84) : "l"((const void*)(partial + (13760 + tid))) : "memory");
                        buf[20] = __uint_as_float(_scalar_bits_84);
                    }
                }
                buf[21] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_85;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_85) : "l"((const void*)(partial + (18880 + tid))) : "memory");
                        buf[21] = __uint_as_float(_scalar_bits_85);
                    }
                }
                buf[22] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_86;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_86) : "l"((const void*)(partial + (24000 + tid))) : "memory");
                        buf[22] = __uint_as_float(_scalar_bits_86);
                    }
                }
                buf[23] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_87;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_87) : "l"((const void*)(partial + (29120 + tid))) : "memory");
                        buf[23] = __uint_as_float(_scalar_bits_87);
                    }
                }
                buf[24] = buf[24] + buf[25];
                buf[26] = buf[26] + buf[27];
                buf[28] = buf[28] + buf[29];
                buf[24] = buf[24] + buf[26];
                buf[24] = buf[24] + buf[28];
#else
                buf[16] = buf[16] + buf[17];
                buf[18] = buf[18] + buf[19];
                buf[16] = buf[16] + buf[18];
#endif
                int at_w_4 = 1280 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_4 / 320 / 8 * 320 + at_w_4 % 320) * 8 + at_w_4 / 320 % 8] = buf[24];
#else
                dwbuf[(at_w_4 / 320 / 8 * 320 + at_w_4 % 320) * 8 + at_w_4 / 320 % 8] = buf[16];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_4)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_4)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_4)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_4)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_4)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_4)) = 0.0f;
                }
                buf[30] = buf[30] + buf[31];
                buf[32] = buf[32] + buf[33];
                buf[34] = buf[34] + buf[35];
                buf[30] = buf[30] + buf[32];
                buf[30] = buf[30] + buf[34];
#else
                buf[20] = buf[20] + buf[21];
                buf[22] = buf[22] + buf[23];
                buf[20] = buf[20] + buf[22];
#endif
                int at_w_5 = 1600 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_5 / 320 / 8 * 320 + at_w_5 % 320) * 8 + at_w_5 / 320 % 8] = buf[30];
#else
                dwbuf[(at_w_5 / 320 / 8 * 320 + at_w_5 % 320) * 8 + at_w_5 / 320 % 8] = buf[20];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_5)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_5)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_5)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_5)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_5)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_5)) = 0.0f;
                }
                buf[36] = buf[36] + buf[37];
                buf[38] = buf[38] + buf[39];
                buf[40] = buf[40] + buf[41];
                buf[36] = buf[36] + buf[38];
                buf[36] = buf[36] + buf[40];
#else
                buf[24] = buf[24] + buf[25];
                buf[26] = buf[26] + buf[27];
                buf[24] = buf[24] + buf[26];
#endif
                int at_w_6 = 1920 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_6 / 320 / 8 * 320 + at_w_6 % 320) * 8 + at_w_6 / 320 % 8] = buf[36];
#else
                dwbuf[(at_w_6 / 320 / 8 * 320 + at_w_6 % 320) * 8 + at_w_6 / 320 % 8] = buf[24];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_6)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_6)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_6)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_6)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_6)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_6)) = 0.0f;
                }
                buf[42] = buf[42] + buf[43];
                buf[44] = buf[44] + buf[45];
                buf[46] = buf[46] + buf[47];
                buf[42] = buf[42] + buf[44];
                buf[42] = buf[42] + buf[46];
#else
                buf[28] = buf[28] + buf[29];
                buf[30] = buf[30] + buf[31];
                buf[28] = buf[28] + buf[30];
#endif
                int at_w_7 = 2240 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_7 / 320 / 8 * 320 + at_w_7 % 320) * 8 + at_w_7 / 320 % 8] = buf[42];
#else
                dwbuf[(at_w_7 / 320 / 8 * 320 + at_w_7 % 320) * 8 + at_w_7 / 320 % 8] = buf[28];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_7)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_7)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_7)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_7)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_7)) = 0.0f;
#else
                buf[16] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_64;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                        buf[16] = __uint_as_float(_scalar_bits_64);
                    }
#endif
                }
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_7)) = 0.0f;
#else
                buf[17] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_65;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (8960 + tid))) : "memory");
                        buf[17] = __uint_as_float(_scalar_bits_65);
                    }
                }
                buf[18] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_66;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (14080 + tid))) : "memory");
                        buf[18] = __uint_as_float(_scalar_bits_66);
                    }
                }
                buf[19] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_67;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (19200 + tid))) : "memory");
                        buf[19] = __uint_as_float(_scalar_bits_67);
                    }
                }
                buf[20] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_68;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (4160 + tid))) : "memory");
                        buf[20] = __uint_as_float(_scalar_bits_68);
                    }
                }
                buf[21] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_69;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (9280 + tid))) : "memory");
                        buf[21] = __uint_as_float(_scalar_bits_69);
                    }
                }
                buf[22] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_70;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (14400 + tid))) : "memory");
                        buf[22] = __uint_as_float(_scalar_bits_70);
                    }
                }
                buf[23] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_71;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (19520 + tid))) : "memory");
                        buf[23] = __uint_as_float(_scalar_bits_71);
                    }
#endif
                }
                buf[24] = 0.0f;
                if (0 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_88;
#else
                        uint32_t _scalar_bits_72;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_88) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                        buf[24] = __uint_as_float(_scalar_bits_88);
#else
                            : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (4480 + tid))) : "memory");
                        buf[24] = __uint_as_float(_scalar_bits_72);
#endif
                    }
                }
                buf[25] = 0.0f;
                if (1 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_89;
#else
                        uint32_t _scalar_bits_73;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_89) : "l"((const void*)(partial + (8960 + tid))) : "memory");
                        buf[25] = __uint_as_float(_scalar_bits_89);
#else
                            : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                        buf[25] = __uint_as_float(_scalar_bits_73);
#endif
                    }
                }
                buf[26] = 0.0f;
                if (2 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_90;
#else
                        uint32_t _scalar_bits_74;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_90) : "l"((const void*)(partial + (14080 + tid))) : "memory");
                        buf[26] = __uint_as_float(_scalar_bits_90);
#else
                            : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (14720 + tid))) : "memory");
                        buf[26] = __uint_as_float(_scalar_bits_74);
#endif
                    }
                }
                buf[27] = 0.0f;
                if (3 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_91;
#else
                        uint32_t _scalar_bits_75;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_91) : "l"((const void*)(partial + (19200 + tid))) : "memory");
                        buf[27] = __uint_as_float(_scalar_bits_91);
#else
                            : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (19840 + tid))) : "memory");
                        buf[27] = __uint_as_float(_scalar_bits_75);
#endif
                    }
                }
                buf[28] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
#else
                if (0 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_92;
#else
                        uint32_t _scalar_bits_76;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_92) : "l"((const void*)(partial + (24320 + tid))) : "memory");
                        buf[28] = __uint_as_float(_scalar_bits_92);
#else
                            : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (4800 + tid))) : "memory");
                        buf[28] = __uint_as_float(_scalar_bits_76);
#endif
                    }
                }
                buf[29] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (5 < nrep) {
#else
                if (1 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_93;
#else
                        uint32_t _scalar_bits_77;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_93) : "l"((const void*)(partial + (29440 + tid))) : "memory");
                        buf[29] = __uint_as_float(_scalar_bits_93);
#else
                            : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (9920 + tid))) : "memory");
                        buf[29] = __uint_as_float(_scalar_bits_77);
#endif
                    }
                }
                buf[30] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (0 < nrep) {
#else
                if (2 < nrep) {
#endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_94;
#else
                        uint32_t _scalar_bits_78;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_94) : "l"((const void*)(partial + (4160 + tid))) : "memory");
                        buf[30] = __uint_as_float(_scalar_bits_94);
#else
                            : "=r"(_scalar_bits_78) : "l"((const void*)(partial + (15040 + tid))) : "memory");
                        buf[30] = __uint_as_float(_scalar_bits_78);
#endif
                    }
                }
                buf[31] = 0.0f;
#if __CUDA_ARCH__ == 1070
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_95;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_95) : "l"((const void*)(partial + (9280 + tid))) : "memory");
                        buf[31] = __uint_as_float(_scalar_bits_95);
                    }
                }
                buf[32] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_96;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_96) : "l"((const void*)(partial + (14400 + tid))) : "memory");
                        buf[32] = __uint_as_float(_scalar_bits_96);
                    }
                }
                buf[33] = 0.0f;
#endif
                if (3 < nrep) {
                    {
#if __CUDA_ARCH__ == 1070
                        uint32_t _scalar_bits_97;
#else
                        uint32_t _scalar_bits_79;
#endif
                        asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                            : "=r"(_scalar_bits_97) : "l"((const void*)(partial + (19520 + tid))) : "memory");
                        buf[33] = __uint_as_float(_scalar_bits_97);
                    }
                }
                buf[34] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_98;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_98) : "l"((const void*)(partial + (24640 + tid))) : "memory");
                        buf[34] = __uint_as_float(_scalar_bits_98);
                    }
                }
                buf[35] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_99;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_99) : "l"((const void*)(partial + (29760 + tid))) : "memory");
                        buf[35] = __uint_as_float(_scalar_bits_99);
                    }
                }
                buf[36] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_100;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_100) : "l"((const void*)(partial + (4480 + tid))) : "memory");
                        buf[36] = __uint_as_float(_scalar_bits_100);
                    }
                }
                buf[37] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_101;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_101) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                        buf[37] = __uint_as_float(_scalar_bits_101);
                    }
                }
                buf[38] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_102;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_102) : "l"((const void*)(partial + (14720 + tid))) : "memory");
                        buf[38] = __uint_as_float(_scalar_bits_102);
                    }
                }
                buf[39] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_103;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_103) : "l"((const void*)(partial + (19840 + tid))) : "memory");
                        buf[39] = __uint_as_float(_scalar_bits_103);
                    }
                }
                buf[40] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_104;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_104) : "l"((const void*)(partial + (24960 + tid))) : "memory");
                        buf[40] = __uint_as_float(_scalar_bits_104);
                    }
                }
                buf[41] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_105;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_105) : "l"((const void*)(partial + (30080 + tid))) : "memory");
                        buf[41] = __uint_as_float(_scalar_bits_105);
                    }
                }
                buf[42] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_106;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_106) : "l"((const void*)(partial + (4800 + tid))) : "memory");
                        buf[42] = __uint_as_float(_scalar_bits_106);
                    }
                }
                buf[43] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_107;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_107) : "l"((const void*)(partial + (9920 + tid))) : "memory");
                        buf[43] = __uint_as_float(_scalar_bits_107);
                    }
                }
                buf[44] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_108;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_108) : "l"((const void*)(partial + (15040 + tid))) : "memory");
                        buf[44] = __uint_as_float(_scalar_bits_108);
                    }
                }
                buf[45] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_109;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_109) : "l"((const void*)(partial + (20160 + tid))) : "memory");
                        buf[45] = __uint_as_float(_scalar_bits_109);
                    }
                }
                buf[46] = 0.0f;
                if (4 < nrep) {
                    {
                        uint32_t _scalar_bits_110;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_110) : "l"((const void*)(partial + (25280 + tid))) : "memory");
                        buf[46] = __uint_as_float(_scalar_bits_110);
                    }
                }
                buf[47] = 0.0f;
                if (5 < nrep) {
                    {
                        uint32_t _scalar_bits_111;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_111) : "l"((const void*)(partial + (30400 + tid))) : "memory");
                        buf[47] = __uint_as_float(_scalar_bits_111);
#else
                            : "=r"(_scalar_bits_79) : "l"((const void*)(partial + (20160 + tid))) : "memory");
                        buf[31] = __uint_as_float(_scalar_bits_79);
#endif
                    }
                }
                buf[0] = buf[0] + buf[1];
                buf[2] = buf[2] + buf[3];
#if __CUDA_ARCH__ == 1070
                buf[4] = buf[4] + buf[5];
#endif
                buf[0] = buf[0] + buf[2];
#if __CUDA_ARCH__ == 1070
                buf[0] = buf[0] + buf[4];
#endif
                int at_w_8 = 2560 + tid;
                dwbuf[(at_w_8 / 320 / 8 * 320 + at_w_8 % 320) * 8 + at_w_8 / 320 % 8] = buf[0];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_8)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_8)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_8)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_8)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_8)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_8)) = 0.0f;
                }
#else
                buf[4] = buf[4] + buf[5];
#endif
                buf[6] = buf[6] + buf[7];
#if __CUDA_ARCH__ == 1070
                buf[8] = buf[8] + buf[9];
                buf[10] = buf[10] + buf[11];
                buf[6] = buf[6] + buf[8];
                buf[6] = buf[6] + buf[10];
#else
                buf[4] = buf[4] + buf[6];
#endif
                int at_w_9 = 2880 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_9 / 320 / 8 * 320 + at_w_9 % 320) * 8 + at_w_9 / 320 % 8] = buf[6];
#else
                dwbuf[(at_w_9 / 320 / 8 * 320 + at_w_9 % 320) * 8 + at_w_9 / 320 % 8] = buf[4];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_9)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_9)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_9)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_9)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_9)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_9)) = 0.0f;
                }
                buf[12] = buf[12] + buf[13];
                buf[14] = buf[14] + buf[15];
                buf[16] = buf[16] + buf[17];
                buf[12] = buf[12] + buf[14];
                buf[12] = buf[12] + buf[16];
#else
                buf[8] = buf[8] + buf[9];
                buf[10] = buf[10] + buf[11];
                buf[8] = buf[8] + buf[10];
#endif
                int at_w_10 = 3200 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_10 / 320 / 8 * 320 + at_w_10 % 320) * 8 + at_w_10 / 320 % 8] = buf[12];
#else
                dwbuf[(at_w_10 / 320 / 8 * 320 + at_w_10 % 320) * 8 + at_w_10 / 320 % 8] = buf[8];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_10)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_10)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_10)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_10)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_10)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_10)) = 0.0f;
                }
                buf[18] = buf[18] + buf[19];
                buf[20] = buf[20] + buf[21];
                buf[22] = buf[22] + buf[23];
                buf[18] = buf[18] + buf[20];
                buf[18] = buf[18] + buf[22];
#else
                buf[12] = buf[12] + buf[13];
                buf[14] = buf[14] + buf[15];
                buf[12] = buf[12] + buf[14];
#endif
                int at_w_11 = 3520 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_11 / 320 / 8 * 320 + at_w_11 % 320) * 8 + at_w_11 / 320 % 8] = buf[18];
#else
                dwbuf[(at_w_11 / 320 / 8 * 320 + at_w_11 % 320) * 8 + at_w_11 / 320 % 8] = buf[12];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_11)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_11)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_11)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_11)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_11)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_11)) = 0.0f;
                }
                buf[24] = buf[24] + buf[25];
                buf[26] = buf[26] + buf[27];
                buf[28] = buf[28] + buf[29];
                buf[24] = buf[24] + buf[26];
                buf[24] = buf[24] + buf[28];
#else
                buf[16] = buf[16] + buf[17];
                buf[18] = buf[18] + buf[19];
                buf[16] = buf[16] + buf[18];
#endif
                int at_w_12 = 3840 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_12 / 320 / 8 * 320 + at_w_12 % 320) * 8 + at_w_12 / 320 % 8] = buf[24];
#else
                dwbuf[(at_w_12 / 320 / 8 * 320 + at_w_12 % 320) * 8 + at_w_12 / 320 % 8] = buf[16];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_12)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_12)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_12)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_12)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_12)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_12)) = 0.0f;
                }
                buf[30] = buf[30] + buf[31];
                buf[32] = buf[32] + buf[33];
                buf[34] = buf[34] + buf[35];
                buf[30] = buf[30] + buf[32];
                buf[30] = buf[30] + buf[34];
#else
                buf[20] = buf[20] + buf[21];
                buf[22] = buf[22] + buf[23];
                buf[20] = buf[20] + buf[22];
#endif
                int at_w_13 = 4160 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_13 / 320 / 8 * 320 + at_w_13 % 320) * 8 + at_w_13 / 320 % 8] = buf[30];
#else
                dwbuf[(at_w_13 / 320 / 8 * 320 + at_w_13 % 320) * 8 + at_w_13 / 320 % 8] = buf[20];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_13)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_13)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_13)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_13)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_13)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_13)) = 0.0f;
                }
                buf[36] = buf[36] + buf[37];
                buf[38] = buf[38] + buf[39];
                buf[40] = buf[40] + buf[41];
                buf[36] = buf[36] + buf[38];
                buf[36] = buf[36] + buf[40];
#else
                buf[24] = buf[24] + buf[25];
                buf[26] = buf[26] + buf[27];
                buf[24] = buf[24] + buf[26];
#endif
                int at_w_14 = 4480 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_14 / 320 / 8 * 320 + at_w_14 % 320) * 8 + at_w_14 / 320 % 8] = buf[36];
#else
                dwbuf[(at_w_14 / 320 / 8 * 320 + at_w_14 % 320) * 8 + at_w_14 / 320 % 8] = buf[24];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_14)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_14)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_14)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_14)) = 0.0f;
                }
#if __CUDA_ARCH__ == 1070
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_14)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_14)) = 0.0f;
                }
                buf[42] = buf[42] + buf[43];
                buf[44] = buf[44] + buf[45];
                buf[46] = buf[46] + buf[47];
                buf[42] = buf[42] + buf[44];
                buf[42] = buf[42] + buf[46];
#else
                buf[28] = buf[28] + buf[29];
                buf[30] = buf[30] + buf[31];
                buf[28] = buf[28] + buf[30];
#endif
                int at_w_15 = 4800 + tid;
#if __CUDA_ARCH__ == 1070
                dwbuf[(at_w_15 / 320 / 8 * 320 + at_w_15 % 320) * 8 + at_w_15 / 320 % 8] = buf[42];
#else
                dwbuf[(at_w_15 / 320 / 8 * 320 + at_w_15 % 320) * 8 + at_w_15 / 320 % 8] = buf[28];
#endif
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_15)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (5120 + at_w_15)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (10240 + at_w_15)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (15360 + at_w_15)) = 0.0f;
#if __CUDA_ARCH__ == 1070
                }
                if (4 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (20480 + at_w_15)) = 0.0f;
                }
                if (5 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (25600 + at_w_15)) = 0.0f;
#endif
                }
                asm volatile("barrier.sync 1, 320;" ::: "memory");
                int col_0 = tid * 4;
                if (col_0 < 5120) {
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
                int col_1 = (320 + tid) * 4;
                if (col_1 < 5120) {
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
                int col_2 = (640 + tid) * 4;
                if (col_2 < 5120) {
                    float o4_2[4];
                    o4_2[0] = dwbuf[col_2];
                    o4_2[1] = dwbuf[col_2 + 1];
                    o4_2[2] = dwbuf[col_2 + 2];
                    o4_2[3] = dwbuf[col_2 + 3];
                    {
                        float4 _v4 = make_float4(o4_2[0 + 0], o4_2[0 + 1], o4_2[0 + 2], o4_2[0 + 3]);
                        *reinterpret_cast<float4*>(dw + col_2) = _v4;
                    }
                }
                int col_3 = (960 + tid) * 4;
                if (col_3 < 5120) {
                    float o4_3[4];
                    o4_3[0] = dwbuf[col_3];
                    o4_3[1] = dwbuf[col_3 + 1];
                    o4_3[2] = dwbuf[col_3 + 2];
                    o4_3[3] = dwbuf[col_3 + 3];
                    {
                        float4 _v4 = make_float4(o4_3[0 + 0], o4_3[0 + 1], o4_3[0 + 2], o4_3[0 + 3]);
                        *reinterpret_cast<float4*>(dw + col_3) = _v4;
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
                    for (int k_1 = 0; k_1 < 5; k_1++) {
                        if (halving != 0) {
                            if (size > tail) {
                                if (rem >= sl * size) {
                                    if (p_c < first + sl) {
                                        span = size;
                                        begin_it = base + (p_c - first) * size;
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
                        if (p_c < first + extra) {
                            span = quot + 1;
                            begin_it = base + (p_c - first) * (quot + 1);
                        } else if (p_c < first + sl) {
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
                    int slot = pk % 4;
                    if (p_row >= p_end) {
                        if (p_c < n_chunks) {
                            p_c = (int)p_nxt;
                            if (p_c < n_chunks) {
                                int stages_1 = T;
                                int sl_1 = rows_per_chunk;
                                if (sl_1 > stages_1) {
                                    sl_1 = stages_1;
                                }
                                if (sl_1 < 1) {
                                    sl_1 = 1;
                                }
                                int share_1 = (stages_1 + sl_1 - 1) / sl_1;
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
                                            if (rem_1 >= sl_1 * size_1) {
                                                if (p_c < first_1 + sl_1) {
                                                    span_1 = size_1;
                                                    begin_it_1 = base_1 + (p_c - first_1) * size_1;
                                                    halving_1 = 0;
                                                } else {
                                                    first_1 = first_1 + sl_1;
                                                    base_1 = base_1 + sl_1 * size_1;
                                                    rem_1 = rem_1 - sl_1 * size_1;
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
                                    int quot_1 = rem_1 / sl_1;
                                    int extra_1 = rem_1 - quot_1 * sl_1;
                                    if (p_c < first_1 + extra_1) {
                                        span_1 = quot_1 + 1;
                                        begin_it_1 = base_1 + (p_c - first_1) * (quot_1 + 1);
                                    } else if (p_c < first_1 + sl_1) {
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
                        int _min_7_1 = ((p_end - p_row) < (1) ? (p_end - p_row) : (1));
                        nval_1 = _min_7_1;
                    }
                    if (nval_1 > 0) {
                        meta[2 * slot] = p_row;
                        meta[2 * slot + 1] = nval_1;
                        mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 4)) * 8, 20480);
                        int sbase = p_ring + slot * 20480;
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
                        meta[2 * slot] = p_c;
                        meta[2 * slot + 1] = -1 - p_first;
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
