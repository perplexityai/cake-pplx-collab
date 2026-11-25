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

#define LAUNCH_MIN_BLOCKS 2

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



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_64423f2da36401d9de39(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 120)
    #define rows_mb_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* dots = reinterpret_cast<float*>(smem_raw + 1024);
    const int dots_addr = smem + 1024;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 1152);
    const int ring_addr = smem + 1152;
    float* folds = reinterpret_cast<float*>(smem_raw + 62592);
    const int fold_addr = smem + 62592;

    // Mbarrier init (40 barriers at smem_raw[0..320))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 15 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // empty: 15 barriers, init_count=64
            mbarrier_init(smem + 120, 64);
            mbarrier_init(smem + 128, 64);
            mbarrier_init(smem + 136, 64);
            mbarrier_init(smem + 144, 64);
            mbarrier_init(smem + 152, 64);
            mbarrier_init(smem + 160, 64);
            mbarrier_init(smem + 168, 64);
            mbarrier_init(smem + 176, 64);
            mbarrier_init(smem + 184, 64);
            mbarrier_init(smem + 192, 64);
            mbarrier_init(smem + 200, 64);
            mbarrier_init(smem + 208, 64);
            mbarrier_init(smem + 216, 64);
            mbarrier_init(smem + 224, 64);
            mbarrier_init(smem + 232, 64);
            // rows_mb: 10 barriers, init_count=64
            mbarrier_init(smem + 240, 64);
            mbarrier_init(smem + 248, 64);
            mbarrier_init(smem + 256, 64);
            mbarrier_init(smem + 264, 64);
            mbarrier_init(smem + 272, 64);
            mbarrier_init(smem + 280, 64);
            mbarrier_init(smem + 288, 64);
            mbarrier_init(smem + 296, 64);
            mbarrier_init(smem + 304, 64);
            mbarrier_init(smem + 312, 64);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 9) {
        { // compute_main
            int gi = (int)warp / 2;
            int wg = (int)warp - gi * 2;
            int t = (int)tid - gi * 64;
            int ring_at = (int)ring_addr;
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
                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + ((unsigned long long)((0 + t) * 8)))) : "memory");
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
                    : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((64 + t) * 8)))) : "memory");
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
            float part[8];
            unsigned int gw[8];
            unsigned int xw[8];
            int tiles = (T + 4) / 5;
            int k = 0;
            int slot = (int)bid;
            while (slot < n_chunks) {
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
                int tile = slot;
                while (tile < tiles) {
                    int row = tile * 5 + gi;
                    int _min_2 = ((row) < (T - 1) ? (row) : (T - 1));
                    float rq = r[_min_2];
                    int _stage_full = gi + 5 * (int)((unsigned int)k % 3);
                    int _parity_full = (int)((unsigned int)k / 3 % 2);
                    mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                    int sbase = ring_at + (gi + 5 * (k % 3)) * 4096;
#if defined(_MSC_VER)
                    uint64_t _smem_raw_3[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_3[0]), "=l"(_smem_raw_3[1]) : "r"(sbase + (0 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(sbase + (0 + t) * 8 * 2));
#endif
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
#if defined(_MSC_VER)
                    uint64_t _smem_raw_4[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_4[0]), "=l"(_smem_raw_4[1]) : "r"(sbase + 2048 + (0 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(sbase + 2048 + (0 + t) * 8 * 2));
#endif
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
#if defined(_MSC_VER)
                    uint64_t _smem_raw_5[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_5[0]), "=l"(_smem_raw_5[1]) : "r"(sbase + (64 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(sbase + (64 + t) * 8 * 2));
#endif
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
#if defined(_MSC_VER)
                    uint64_t _smem_raw_6[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_6[0]), "=l"(_smem_raw_6[1]) : "r"(sbase + 2048 + (64 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_6;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(sbase + 2048 + (64 + t) * 8 * 2));
#endif
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                    float dq = 0.0f;
                    float ga_0 = __uint_as_float(gw[0] << 16);
                    float gb_0 = __uint_as_float(gw[0] & 4294901760u);
                    float xa_0 = __uint_as_float(xw[0] << 16);
                    float xb_0 = __uint_as_float(xw[0] & 4294901760u);
                    dq = __fmaf_rn(ga_0 * ww[0], xa_0, dq);
                    dq = __fmaf_rn(gb_0 * ww[1], xb_0, dq);
                    float ga_1 = __uint_as_float(gw[1] << 16);
                    float gb_1 = __uint_as_float(gw[1] & 4294901760u);
                    float xa_1 = __uint_as_float(xw[1] << 16);
                    float xb_1 = __uint_as_float(xw[1] & 4294901760u);
                    dq = __fmaf_rn(ga_1 * ww[2], xa_1, dq);
                    dq = __fmaf_rn(gb_1 * ww[3], xb_1, dq);
                    float ga_2 = __uint_as_float(gw[2] << 16);
                    float gb_2 = __uint_as_float(gw[2] & 4294901760u);
                    float xa_2 = __uint_as_float(xw[2] << 16);
                    float xb_2 = __uint_as_float(xw[2] & 4294901760u);
                    dq = __fmaf_rn(ga_2 * ww[4], xa_2, dq);
                    dq = __fmaf_rn(gb_2 * ww[5], xb_2, dq);
                    float ga_3 = __uint_as_float(gw[3] << 16);
                    float gb_3 = __uint_as_float(gw[3] & 4294901760u);
                    float xa_3 = __uint_as_float(xw[3] << 16);
                    float xb_3 = __uint_as_float(xw[3] & 4294901760u);
                    dq = __fmaf_rn(ga_3 * ww[6], xa_3, dq);
                    dq = __fmaf_rn(gb_3 * ww[7], xb_3, dq);
                    float ga_4 = __uint_as_float(gw[4] << 16);
                    float gb_4 = __uint_as_float(gw[4] & 4294901760u);
                    float xa_4 = __uint_as_float(xw[4] << 16);
                    float xb_4 = __uint_as_float(xw[4] & 4294901760u);
                    dq = __fmaf_rn(ga_4 * ww[8], xa_4, dq);
                    dq = __fmaf_rn(gb_4 * ww[9], xb_4, dq);
                    float ga_5 = __uint_as_float(gw[5] << 16);
                    float gb_5 = __uint_as_float(gw[5] & 4294901760u);
                    float xa_5 = __uint_as_float(xw[5] << 16);
                    float xb_5 = __uint_as_float(xw[5] & 4294901760u);
                    dq = __fmaf_rn(ga_5 * ww[10], xa_5, dq);
                    dq = __fmaf_rn(gb_5 * ww[11], xb_5, dq);
                    float ga_6 = __uint_as_float(gw[6] << 16);
                    float gb_6 = __uint_as_float(gw[6] & 4294901760u);
                    float xa_6 = __uint_as_float(xw[6] << 16);
                    float xb_6 = __uint_as_float(xw[6] & 4294901760u);
                    dq = __fmaf_rn(ga_6 * ww[12], xa_6, dq);
                    dq = __fmaf_rn(gb_6 * ww[13], xb_6, dq);
                    float ga_7 = __uint_as_float(gw[7] << 16);
                    float gb_7 = __uint_as_float(gw[7] & 4294901760u);
                    float xa_7 = __uint_as_float(xw[7] << 16);
                    float xb_7 = __uint_as_float(xw[7] & 4294901760u);
                    dq = __fmaf_rn(ga_7 * ww[14], xa_7, dq);
                    dq = __fmaf_rn(gb_7 * ww[15], xb_7, dq);
                    float dsum = dq;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsum += __shfl_xor_sync(0xFFFFFFFF, dsum, offset);
                    int par = gi * 4 + (k & 1) * 2;
                    if (lane == 0) {
                        dots[par + wg] = dsum;
                    }
                    mbarrier_arrive(rows_mb_addr + (gi + 5 * (int)((unsigned int)k % 2)) * 8);
                    int _stage_rows_mb = gi + 5 * (int)((unsigned int)k % 2);
                    int _parity_rows_mb = (int)((unsigned int)k / 2 % 2);
                    mbarrier_wait(rows_mb_addr + (_stage_rows_mb) * 8, _parity_rows_mb);
                    float tot = dots[par];
                    tot = tot + dots[par + 1];
                    float cq = tot * 0.0009765625f;
                    float kx = rq * rq * rq * cq;
                    float nkx = -kx;
                    unsigned long long dx_base = (unsigned long long)row * 1024ull;
#if defined(_MSC_VER)
                    uint64_t _smem_raw_7[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_7[0]), "=l"(_smem_raw_7[1]) : "r"(sbase + (0 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_7;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(sbase + (0 + t) * 8 * 2));
#endif
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
#if defined(_MSC_VER)
                    uint64_t _smem_raw_8[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_8[0]), "=l"(_smem_raw_8[1]) : "r"(sbase + 2048 + (0 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_8;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(sbase + 2048 + (0 + t) * 8 * 2));
#endif
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    float rg0_0 = rq * g0_0;
                    float rg1_0 = rq * g1_0;
                    out_0[0] = __fmaf_rn(nkx, x0_0, rg0_0 * ww[0]);
                    out_0[1] = __fmaf_rn(nkx, x1_0, rg1_0 * ww[1]);
                    acc[0] = ((row < T) ? __fmaf_rn(rg0_0, x0_0, acc[0]) : acc[0]);
                    acc[1] = ((row < T) ? __fmaf_rn(rg1_0, x1_0, acc[1]) : acc[1]);
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    float rg0_1 = rq * g0_1;
                    float rg1_1 = rq * g1_1;
                    out_0[2] = __fmaf_rn(nkx, x0_1, rg0_1 * ww[2]);
                    out_0[3] = __fmaf_rn(nkx, x1_1, rg1_1 * ww[3]);
                    acc[2] = ((row < T) ? __fmaf_rn(rg0_1, x0_1, acc[2]) : acc[2]);
                    acc[3] = ((row < T) ? __fmaf_rn(rg1_1, x1_1, acc[3]) : acc[3]);
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    float rg0_2 = rq * g0_2;
                    float rg1_2 = rq * g1_2;
                    out_0[4] = __fmaf_rn(nkx, x0_2, rg0_2 * ww[4]);
                    out_0[5] = __fmaf_rn(nkx, x1_2, rg1_2 * ww[5]);
                    acc[4] = ((row < T) ? __fmaf_rn(rg0_2, x0_2, acc[4]) : acc[4]);
                    acc[5] = ((row < T) ? __fmaf_rn(rg1_2, x1_2, acc[5]) : acc[5]);
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    float rg0_3 = rq * g0_3;
                    float rg1_3 = rq * g1_3;
                    out_0[6] = __fmaf_rn(nkx, x0_3, rg0_3 * ww[6]);
                    out_0[7] = __fmaf_rn(nkx, x1_3, rg1_3 * ww[7]);
                    acc[6] = ((row < T) ? __fmaf_rn(rg0_3, x0_3, acc[6]) : acc[6]);
                    acc[7] = ((row < T) ? __fmaf_rn(rg1_3, x1_3, acc[7]) : acc[7]);
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (row < T) {
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((0 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    }
#if defined(_MSC_VER)
                    uint64_t _smem_raw_9[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_9[0]), "=l"(_smem_raw_9[1]) : "r"(sbase + (64 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_9;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(sbase + (64 + t) * 8 * 2));
#endif
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
#if defined(_MSC_VER)
                    uint64_t _smem_raw_10[2];
                    asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_10[0]), "=l"(_smem_raw_10[1]) : "r"(sbase + 2048 + (64 + t) * 8 * 2));
#else
                    __int128_t _smem_raw_10;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(sbase + 2048 + (64 + t) * 8 * 2));
#endif
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    float rg0_4 = rq * g0_4;
                    float rg1_4 = rq * g1_4;
                    out_1[0] = __fmaf_rn(nkx, x0_4, rg0_4 * ww[8]);
                    out_1[1] = __fmaf_rn(nkx, x1_4, rg1_4 * ww[9]);
                    acc[8] = ((row < T) ? __fmaf_rn(rg0_4, x0_4, acc[8]) : acc[8]);
                    acc[9] = ((row < T) ? __fmaf_rn(rg1_4, x1_4, acc[9]) : acc[9]);
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    float rg0_5 = rq * g0_5;
                    float rg1_5 = rq * g1_5;
                    out_1[2] = __fmaf_rn(nkx, x0_5, rg0_5 * ww[10]);
                    out_1[3] = __fmaf_rn(nkx, x1_5, rg1_5 * ww[11]);
                    acc[10] = ((row < T) ? __fmaf_rn(rg0_5, x0_5, acc[10]) : acc[10]);
                    acc[11] = ((row < T) ? __fmaf_rn(rg1_5, x1_5, acc[11]) : acc[11]);
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    float rg0_6 = rq * g0_6;
                    float rg1_6 = rq * g1_6;
                    out_1[4] = __fmaf_rn(nkx, x0_6, rg0_6 * ww[12]);
                    out_1[5] = __fmaf_rn(nkx, x1_6, rg1_6 * ww[13]);
                    acc[12] = ((row < T) ? __fmaf_rn(rg0_6, x0_6, acc[12]) : acc[12]);
                    acc[13] = ((row < T) ? __fmaf_rn(rg1_6, x1_6, acc[13]) : acc[13]);
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    float rg0_7 = rq * g0_7;
                    float rg1_7 = rq * g1_7;
                    out_1[6] = __fmaf_rn(nkx, x0_7, rg0_7 * ww[14]);
                    out_1[7] = __fmaf_rn(nkx, x1_7, rg1_7 * ww[15]);
                    acc[14] = ((row < T) ? __fmaf_rn(rg0_7, x0_7, acc[14]) : acc[14]);
                    acc[15] = ((row < T) ? __fmaf_rn(rg1_7, x1_7, acc[15]) : acc[15]);
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (row < T) {
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((64 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    }
                    mbarrier_arrive(empty_addr + (gi + 5 * (int)((unsigned int)k % 3)) * 8);
                    tile = tile + n_chunks;
                    k += 1;
                }
                asm volatile("barrier.sync 1, 320;" ::: "memory");
                if (gi != 0) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fold_addr + (unsigned int)(((gi - 1) * 1024 + (0 + t) * 8) * 4)), "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fold_addr + (unsigned int)(((gi - 1) * 1024 + (0 + t) * 8 + 4) * 4)), "f"(acc[4]), "f"(acc[5]), "f"(acc[6]), "f"(acc[7]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fold_addr + (unsigned int)(((gi - 1) * 1024 + (64 + t) * 8) * 4)), "f"(acc[8]), "f"(acc[9]), "f"(acc[10]), "f"(acc[11]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fold_addr + (unsigned int)(((gi - 1) * 1024 + (64 + t) * 8 + 4) * 4)), "f"(acc[12]), "f"(acc[13]), "f"(acc[14]), "f"(acc[15]) : "memory");
                }
                asm volatile("barrier.sync 1, 320;" ::: "memory");
                if (gi == 0) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((0 + t) * 8 * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)(((0 + t) * 8 + 4) * 4)));
                    acc[0] = acc[0] + part[0];
                    acc[1] = acc[1] + part[1];
                    acc[2] = acc[2] + part[2];
                    acc[3] = acc[3] + part[3];
                    acc[4] = acc[4] + part[4];
                    acc[5] = acc[5] + part[5];
                    acc[6] = acc[6] + part[6];
                    acc[7] = acc[7] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((64 + t) * 8 * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)(((64 + t) * 8 + 4) * 4)));
                    acc[8] = acc[8] + part[0];
                    acc[9] = acc[9] + part[1];
                    acc[10] = acc[10] + part[2];
                    acc[11] = acc[11] + part[3];
                    acc[12] = acc[12] + part[4];
                    acc[13] = acc[13] + part[5];
                    acc[14] = acc[14] + part[6];
                    acc[15] = acc[15] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((1024 + (0 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((1024 + (0 + t) * 8 + 4) * 4)));
                    acc[0] = acc[0] + part[0];
                    acc[1] = acc[1] + part[1];
                    acc[2] = acc[2] + part[2];
                    acc[3] = acc[3] + part[3];
                    acc[4] = acc[4] + part[4];
                    acc[5] = acc[5] + part[5];
                    acc[6] = acc[6] + part[6];
                    acc[7] = acc[7] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((1024 + (64 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((1024 + (64 + t) * 8 + 4) * 4)));
                    acc[8] = acc[8] + part[0];
                    acc[9] = acc[9] + part[1];
                    acc[10] = acc[10] + part[2];
                    acc[11] = acc[11] + part[3];
                    acc[12] = acc[12] + part[4];
                    acc[13] = acc[13] + part[5];
                    acc[14] = acc[14] + part[6];
                    acc[15] = acc[15] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((2048 + (0 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((2048 + (0 + t) * 8 + 4) * 4)));
                    acc[0] = acc[0] + part[0];
                    acc[1] = acc[1] + part[1];
                    acc[2] = acc[2] + part[2];
                    acc[3] = acc[3] + part[3];
                    acc[4] = acc[4] + part[4];
                    acc[5] = acc[5] + part[5];
                    acc[6] = acc[6] + part[6];
                    acc[7] = acc[7] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((2048 + (64 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((2048 + (64 + t) * 8 + 4) * 4)));
                    acc[8] = acc[8] + part[0];
                    acc[9] = acc[9] + part[1];
                    acc[10] = acc[10] + part[2];
                    acc[11] = acc[11] + part[3];
                    acc[12] = acc[12] + part[4];
                    acc[13] = acc[13] + part[5];
                    acc[14] = acc[14] + part[6];
                    acc[15] = acc[15] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((3072 + (0 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((3072 + (0 + t) * 8 + 4) * 4)));
                    acc[0] = acc[0] + part[0];
                    acc[1] = acc[1] + part[1];
                    acc[2] = acc[2] + part[2];
                    acc[3] = acc[3] + part[3];
                    acc[4] = acc[4] + part[4];
                    acc[5] = acc[5] + part[5];
                    acc[6] = acc[6] + part[6];
                    acc[7] = acc[7] + part[7];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                        : "r"(fold_addr + (unsigned int)((3072 + (64 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((3072 + (64 + t) * 8 + 4) * 4)));
                    acc[8] = acc[8] + part[0];
                    acc[9] = acc[9] + part[1];
                    acc[10] = acc[10] + part[2];
                    acc[11] = acc[11] + part[3];
                    acc[12] = acc[12] + part[4];
                    acc[13] = acc[13] + part[5];
                    acc[14] = acc[14] + part[6];
                    acc[15] = acc[15] + part[7];
                    unsigned long long pbase = (unsigned long long)slot * 1024ull;
                    {
                        unsigned _stv8_11_0 = __float_as_uint(acc[0 + 0]);
                        unsigned _stv8_11_1 = __float_as_uint(acc[0 + 1]);
                        unsigned _stv8_11_2 = __float_as_uint(acc[0 + 2]);
                        unsigned _stv8_11_3 = __float_as_uint(acc[0 + 3]);
                        unsigned _stv8_11_4 = __float_as_uint(acc[0 + 4]);
                        unsigned _stv8_11_5 = __float_as_uint(acc[0 + 5]);
                        unsigned _stv8_11_6 = __float_as_uint(acc[0 + 6]);
                        unsigned _stv8_11_7 = __float_as_uint(acc[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((0 + t) * 8)))), "r"(_stv8_11_0), "r"(_stv8_11_1), "r"(_stv8_11_2), "r"(_stv8_11_3), "r"(_stv8_11_4), "r"(_stv8_11_5), "r"(_stv8_11_6), "r"(_stv8_11_7) : "memory");
                    }
                    {
                        unsigned _stv8_12_0 = __float_as_uint(acc[8 + 0]);
                        unsigned _stv8_12_1 = __float_as_uint(acc[8 + 1]);
                        unsigned _stv8_12_2 = __float_as_uint(acc[8 + 2]);
                        unsigned _stv8_12_3 = __float_as_uint(acc[8 + 3]);
                        unsigned _stv8_12_4 = __float_as_uint(acc[8 + 4]);
                        unsigned _stv8_12_5 = __float_as_uint(acc[8 + 5]);
                        unsigned _stv8_12_6 = __float_as_uint(acc[8 + 6]);
                        unsigned _stv8_12_7 = __float_as_uint(acc[8 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((64 + t) * 8)))), "r"(_stv8_12_0), "r"(_stv8_12_1), "r"(_stv8_12_2), "r"(_stv8_12_3), "r"(_stv8_12_4), "r"(_stv8_12_5), "r"(_stv8_12_6), "r"(_stv8_12_7) : "memory");
                    }
                }
                slot = slot + (int)num_bids;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 10) {
        { // prod_main
            if (elect_sync()) {
                int q_ring = (int)ring_addr;
                int q_tiles = (T + 4) / 5;
                int qk = 0;
                int q_slot = (int)bid;
                while (q_slot < n_chunks) {
                    int q_tile = q_slot;
                    while (q_tile < q_tiles) {
                        int _min_0 = ((q_tile * 5) < (T - 1) ? (q_tile * 5) : (T - 1));
                        long long q_row_0 = (long long)_min_0;
                        int _stage_empty = 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (5 * (int)((unsigned int)qk % 3)) * 8, 4096);
                        cp_async_bulk_gmem2smem(q_ring + 5 * (qk % 3) * 4096, reinterpret_cast<const uint8_t*>(g + q_row_0 * g_stride_p0), 2048, full_addr + (5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + 5 * (qk % 3) * 4096 + 2048, reinterpret_cast<const uint8_t*>(x + q_row_0 * x_stride_p0), 2048, full_addr + (5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_1 = ((q_tile * 5 + 1) < (T - 1) ? (q_tile * 5 + 1) : (T - 1));
                        long long q_row_1 = (long long)_min_1;
                        int _stage_empty_1 = 1 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_1 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_1) * 8, _parity_empty_1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (1 + 5 * (int)((unsigned int)qk % 3)) * 8, 4096);
                        cp_async_bulk_gmem2smem(q_ring + (1 + 5 * (qk % 3)) * 4096, reinterpret_cast<const uint8_t*>(g + q_row_1 * g_stride_p0), 2048, full_addr + (1 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (1 + 5 * (qk % 3)) * 4096 + 2048, reinterpret_cast<const uint8_t*>(x + q_row_1 * x_stride_p0), 2048, full_addr + (1 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_2_1 = ((q_tile * 5 + 2) < (T - 1) ? (q_tile * 5 + 2) : (T - 1));
                        long long q_row_2 = (long long)_min_2_1;
                        int _stage_empty_2 = 2 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_2 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_2) * 8, _parity_empty_2);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (2 + 5 * (int)((unsigned int)qk % 3)) * 8, 4096);
                        cp_async_bulk_gmem2smem(q_ring + (2 + 5 * (qk % 3)) * 4096, reinterpret_cast<const uint8_t*>(g + q_row_2 * g_stride_p0), 2048, full_addr + (2 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (2 + 5 * (qk % 3)) * 4096 + 2048, reinterpret_cast<const uint8_t*>(x + q_row_2 * x_stride_p0), 2048, full_addr + (2 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_3 = ((q_tile * 5 + 3) < (T - 1) ? (q_tile * 5 + 3) : (T - 1));
                        long long q_row_3 = (long long)_min_3;
                        int _stage_empty_3 = 3 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_3 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_3) * 8, _parity_empty_3);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (3 + 5 * (int)((unsigned int)qk % 3)) * 8, 4096);
                        cp_async_bulk_gmem2smem(q_ring + (3 + 5 * (qk % 3)) * 4096, reinterpret_cast<const uint8_t*>(g + q_row_3 * g_stride_p0), 2048, full_addr + (3 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (3 + 5 * (qk % 3)) * 4096 + 2048, reinterpret_cast<const uint8_t*>(x + q_row_3 * x_stride_p0), 2048, full_addr + (3 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_4 = ((q_tile * 5 + 4) < (T - 1) ? (q_tile * 5 + 4) : (T - 1));
                        long long q_row_4 = (long long)_min_4;
                        int _stage_empty_4 = 4 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_4 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_4) * 8, _parity_empty_4);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (4 + 5 * (int)((unsigned int)qk % 3)) * 8, 4096);
                        cp_async_bulk_gmem2smem(q_ring + (4 + 5 * (qk % 3)) * 4096, reinterpret_cast<const uint8_t*>(g + q_row_4 * g_stride_p0), 2048, full_addr + (4 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (4 + 5 * (qk % 3)) * 4096 + 2048, reinterpret_cast<const uint8_t*>(x + q_row_4 * x_stride_p0), 2048, full_addr + (4 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        q_tile = q_tile + n_chunks;
                        qk += 1;
                    }
                    q_slot = q_slot + (int)num_bids;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
