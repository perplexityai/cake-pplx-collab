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

#define LAUNCH_MIN_BLOCKS 3

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

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_fdb7fdce98e515b40393(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* dots = reinterpret_cast<float*>(smem_raw + 1024);
    const int dots_addr = smem + 1024;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 1152);
    const int ring_addr = smem + 1152;
    float* folds = reinterpret_cast<float*>(smem_raw + 47232);
    const int fold_addr = smem + 47232;

    // Mbarrier init (18 barriers at smem_raw[0..144))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 9 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // empty: 9 barriers, init_count=32
            mbarrier_init(smem + 72, 32);
            mbarrier_init(smem + 80, 32);
            mbarrier_init(smem + 88, 32);
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 2) {
        { // compute_main
            int gi = (int)warp;
            int wg = (int)warp - gi;
            int t = (int)tid - gi * 32;
            int ring_at = (int)ring_addr;
            float ww[40];
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
                    : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((32 + t) * 8)))) : "memory");
                ww[8 + 0] = __uint_as_float(_ldv8_1_0);
                ww[8 + 1] = __uint_as_float(_ldv8_1_1);
                ww[8 + 2] = __uint_as_float(_ldv8_1_2);
                ww[8 + 3] = __uint_as_float(_ldv8_1_3);
                ww[8 + 4] = __uint_as_float(_ldv8_1_4);
                ww[8 + 5] = __uint_as_float(_ldv8_1_5);
                ww[8 + 6] = __uint_as_float(_ldv8_1_6);
                ww[8 + 7] = __uint_as_float(_ldv8_1_7);
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
                    : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + ((unsigned long long)((64 + t) * 8)))) : "memory");
                ww[16 + 0] = __uint_as_float(_ldv8_2_0);
                ww[16 + 1] = __uint_as_float(_ldv8_2_1);
                ww[16 + 2] = __uint_as_float(_ldv8_2_2);
                ww[16 + 3] = __uint_as_float(_ldv8_2_3);
                ww[16 + 4] = __uint_as_float(_ldv8_2_4);
                ww[16 + 5] = __uint_as_float(_ldv8_2_5);
                ww[16 + 6] = __uint_as_float(_ldv8_2_6);
                ww[16 + 7] = __uint_as_float(_ldv8_2_7);
            }
            {
                unsigned _ldv8_3_0;
                unsigned _ldv8_3_1;
                unsigned _ldv8_3_2;
                unsigned _ldv8_3_3;
                unsigned _ldv8_3_4;
                unsigned _ldv8_3_5;
                unsigned _ldv8_3_6;
                unsigned _ldv8_3_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(w + ((unsigned long long)((96 + t) * 8)))) : "memory");
                ww[24 + 0] = __uint_as_float(_ldv8_3_0);
                ww[24 + 1] = __uint_as_float(_ldv8_3_1);
                ww[24 + 2] = __uint_as_float(_ldv8_3_2);
                ww[24 + 3] = __uint_as_float(_ldv8_3_3);
                ww[24 + 4] = __uint_as_float(_ldv8_3_4);
                ww[24 + 5] = __uint_as_float(_ldv8_3_5);
                ww[24 + 6] = __uint_as_float(_ldv8_3_6);
                ww[24 + 7] = __uint_as_float(_ldv8_3_7);
            }
            {
                unsigned _ldv8_4_0;
                unsigned _ldv8_4_1;
                unsigned _ldv8_4_2;
                unsigned _ldv8_4_3;
                unsigned _ldv8_4_4;
                unsigned _ldv8_4_5;
                unsigned _ldv8_4_6;
                unsigned _ldv8_4_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(w + ((unsigned long long)((128 + t) * 8)))) : "memory");
                ww[32 + 0] = __uint_as_float(_ldv8_4_0);
                ww[32 + 1] = __uint_as_float(_ldv8_4_1);
                ww[32 + 2] = __uint_as_float(_ldv8_4_2);
                ww[32 + 3] = __uint_as_float(_ldv8_4_3);
                ww[32 + 4] = __uint_as_float(_ldv8_4_4);
                ww[32 + 5] = __uint_as_float(_ldv8_4_5);
                ww[32 + 6] = __uint_as_float(_ldv8_4_6);
                ww[32 + 7] = __uint_as_float(_ldv8_4_7);
            }
            float acc[40];
            float part[8];
            unsigned int gw[20];
            unsigned int xw[20];
            int tiles = (T + 2) / 3;
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
                acc[16] = 0.0f;
                acc[17] = 0.0f;
                acc[18] = 0.0f;
                acc[19] = 0.0f;
                acc[20] = 0.0f;
                acc[21] = 0.0f;
                acc[22] = 0.0f;
                acc[23] = 0.0f;
                acc[24] = 0.0f;
                acc[25] = 0.0f;
                acc[26] = 0.0f;
                acc[27] = 0.0f;
                acc[28] = 0.0f;
                acc[29] = 0.0f;
                acc[30] = 0.0f;
                acc[31] = 0.0f;
                acc[32] = 0.0f;
                acc[33] = 0.0f;
                acc[34] = 0.0f;
                acc[35] = 0.0f;
                acc[36] = 0.0f;
                acc[37] = 0.0f;
                acc[38] = 0.0f;
                acc[39] = 0.0f;
                int tile = slot;
                while (tile < tiles) {
                    int row = tile * 3 + gi;
                    int _min_5 = ((row) < (T - 1) ? (row) : (T - 1));
                    float rq = r[_min_5];
                    int _stage_full = gi + 3 * (int)((unsigned int)k % 3);
                    int _parity_full = (int)((unsigned int)k / 3 % 2);
                    mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                    int sbase = ring_at + (gi + 3 * (k % 3)) * 5120;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[0])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 3]))
                        : "r"(sbase + (0 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[0])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 3]))
                        : "r"(sbase + 2560 + (0 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[4])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4) + 3]))
                        : "r"(sbase + (32 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[4])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4) + 3]))
                        : "r"(sbase + 2560 + (32 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[8])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(8) + 3]))
                        : "r"(sbase + (64 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[8])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(8) + 3]))
                        : "r"(sbase + 2560 + (64 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[12])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(12) + 3]))
                        : "r"(sbase + (96 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[12])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(12) + 3]))
                        : "r"(sbase + 2560 + (96 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[16])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(16) + 3]))
                        : "r"(sbase + (128 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[16])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(16) + 3]))
                        : "r"(sbase + 2560 + (128 + t) * 8 * 2));
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
                    float ga_8 = __uint_as_float(gw[8] << 16);
                    float gb_8 = __uint_as_float(gw[8] & 4294901760u);
                    float xa_8 = __uint_as_float(xw[8] << 16);
                    float xb_8 = __uint_as_float(xw[8] & 4294901760u);
                    dq = __fmaf_rn(ga_8 * ww[16], xa_8, dq);
                    dq = __fmaf_rn(gb_8 * ww[17], xb_8, dq);
                    float ga_9 = __uint_as_float(gw[9] << 16);
                    float gb_9 = __uint_as_float(gw[9] & 4294901760u);
                    float xa_9 = __uint_as_float(xw[9] << 16);
                    float xb_9 = __uint_as_float(xw[9] & 4294901760u);
                    dq = __fmaf_rn(ga_9 * ww[18], xa_9, dq);
                    dq = __fmaf_rn(gb_9 * ww[19], xb_9, dq);
                    float ga_10 = __uint_as_float(gw[10] << 16);
                    float gb_10 = __uint_as_float(gw[10] & 4294901760u);
                    float xa_10 = __uint_as_float(xw[10] << 16);
                    float xb_10 = __uint_as_float(xw[10] & 4294901760u);
                    dq = __fmaf_rn(ga_10 * ww[20], xa_10, dq);
                    dq = __fmaf_rn(gb_10 * ww[21], xb_10, dq);
                    float ga_11 = __uint_as_float(gw[11] << 16);
                    float gb_11 = __uint_as_float(gw[11] & 4294901760u);
                    float xa_11 = __uint_as_float(xw[11] << 16);
                    float xb_11 = __uint_as_float(xw[11] & 4294901760u);
                    dq = __fmaf_rn(ga_11 * ww[22], xa_11, dq);
                    dq = __fmaf_rn(gb_11 * ww[23], xb_11, dq);
                    float ga_12 = __uint_as_float(gw[12] << 16);
                    float gb_12 = __uint_as_float(gw[12] & 4294901760u);
                    float xa_12 = __uint_as_float(xw[12] << 16);
                    float xb_12 = __uint_as_float(xw[12] & 4294901760u);
                    dq = __fmaf_rn(ga_12 * ww[24], xa_12, dq);
                    dq = __fmaf_rn(gb_12 * ww[25], xb_12, dq);
                    float ga_13 = __uint_as_float(gw[13] << 16);
                    float gb_13 = __uint_as_float(gw[13] & 4294901760u);
                    float xa_13 = __uint_as_float(xw[13] << 16);
                    float xb_13 = __uint_as_float(xw[13] & 4294901760u);
                    dq = __fmaf_rn(ga_13 * ww[26], xa_13, dq);
                    dq = __fmaf_rn(gb_13 * ww[27], xb_13, dq);
                    float ga_14 = __uint_as_float(gw[14] << 16);
                    float gb_14 = __uint_as_float(gw[14] & 4294901760u);
                    float xa_14 = __uint_as_float(xw[14] << 16);
                    float xb_14 = __uint_as_float(xw[14] & 4294901760u);
                    dq = __fmaf_rn(ga_14 * ww[28], xa_14, dq);
                    dq = __fmaf_rn(gb_14 * ww[29], xb_14, dq);
                    float ga_15 = __uint_as_float(gw[15] << 16);
                    float gb_15 = __uint_as_float(gw[15] & 4294901760u);
                    float xa_15 = __uint_as_float(xw[15] << 16);
                    float xb_15 = __uint_as_float(xw[15] & 4294901760u);
                    dq = __fmaf_rn(ga_15 * ww[30], xa_15, dq);
                    dq = __fmaf_rn(gb_15 * ww[31], xb_15, dq);
                    float ga_16 = __uint_as_float(gw[16] << 16);
                    float gb_16 = __uint_as_float(gw[16] & 4294901760u);
                    float xa_16 = __uint_as_float(xw[16] << 16);
                    float xb_16 = __uint_as_float(xw[16] & 4294901760u);
                    dq = __fmaf_rn(ga_16 * ww[32], xa_16, dq);
                    dq = __fmaf_rn(gb_16 * ww[33], xb_16, dq);
                    float ga_17 = __uint_as_float(gw[17] << 16);
                    float gb_17 = __uint_as_float(gw[17] & 4294901760u);
                    float xa_17 = __uint_as_float(xw[17] << 16);
                    float xb_17 = __uint_as_float(xw[17] & 4294901760u);
                    dq = __fmaf_rn(ga_17 * ww[34], xa_17, dq);
                    dq = __fmaf_rn(gb_17 * ww[35], xb_17, dq);
                    float ga_18 = __uint_as_float(gw[18] << 16);
                    float gb_18 = __uint_as_float(gw[18] & 4294901760u);
                    float xa_18 = __uint_as_float(xw[18] << 16);
                    float xb_18 = __uint_as_float(xw[18] & 4294901760u);
                    dq = __fmaf_rn(ga_18 * ww[36], xa_18, dq);
                    dq = __fmaf_rn(gb_18 * ww[37], xb_18, dq);
                    float ga_19 = __uint_as_float(gw[19] << 16);
                    float gb_19 = __uint_as_float(gw[19] & 4294901760u);
                    float xa_19 = __uint_as_float(xw[19] << 16);
                    float xb_19 = __uint_as_float(xw[19] & 4294901760u);
                    dq = __fmaf_rn(ga_19 * ww[38], xa_19, dq);
                    dq = __fmaf_rn(gb_19 * ww[39], xb_19, dq);
                    float dsum = dq;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsum += __shfl_xor_sync(0xFFFFFFFF, dsum, offset);
                    float tot = dsum;
                    __syncwarp();
                    float cq = tot * 0.00078125f;
                    float kx = rq * rq * rq * cq;
                    float nkx = -kx;
                    unsigned long long dx_base = (unsigned long long)row * 1280ull;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[0])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 3]))
                        : "r"(sbase + (0 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[0])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 3]))
                        : "r"(sbase + 2560 + (0 + t) * 8 * 2));
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
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[4])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(4) + 3]))
                        : "r"(sbase + (32 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[4])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(4) + 3]))
                        : "r"(sbase + 2560 + (32 + t) * 8 * 2));
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
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((32 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[8])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(8) + 3]))
                        : "r"(sbase + (64 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[8])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(8) + 3]))
                        : "r"(sbase + 2560 + (64 + t) * 8 * 2));
                    float out_2[8];
                    float g0_8 = __uint_as_float(gw[8] << 16);
                    float g1_8 = __uint_as_float(gw[8] & 4294901760u);
                    float x0_8 = __uint_as_float(xw[8] << 16);
                    float x1_8 = __uint_as_float(xw[8] & 4294901760u);
                    float rg0_8 = rq * g0_8;
                    float rg1_8 = rq * g1_8;
                    out_2[0] = __fmaf_rn(nkx, x0_8, rg0_8 * ww[16]);
                    out_2[1] = __fmaf_rn(nkx, x1_8, rg1_8 * ww[17]);
                    acc[16] = ((row < T) ? __fmaf_rn(rg0_8, x0_8, acc[16]) : acc[16]);
                    acc[17] = ((row < T) ? __fmaf_rn(rg1_8, x1_8, acc[17]) : acc[17]);
                    float g0_9 = __uint_as_float(gw[9] << 16);
                    float g1_9 = __uint_as_float(gw[9] & 4294901760u);
                    float x0_9 = __uint_as_float(xw[9] << 16);
                    float x1_9 = __uint_as_float(xw[9] & 4294901760u);
                    float rg0_9 = rq * g0_9;
                    float rg1_9 = rq * g1_9;
                    out_2[2] = __fmaf_rn(nkx, x0_9, rg0_9 * ww[18]);
                    out_2[3] = __fmaf_rn(nkx, x1_9, rg1_9 * ww[19]);
                    acc[18] = ((row < T) ? __fmaf_rn(rg0_9, x0_9, acc[18]) : acc[18]);
                    acc[19] = ((row < T) ? __fmaf_rn(rg1_9, x1_9, acc[19]) : acc[19]);
                    float g0_10 = __uint_as_float(gw[10] << 16);
                    float g1_10 = __uint_as_float(gw[10] & 4294901760u);
                    float x0_10 = __uint_as_float(xw[10] << 16);
                    float x1_10 = __uint_as_float(xw[10] & 4294901760u);
                    float rg0_10 = rq * g0_10;
                    float rg1_10 = rq * g1_10;
                    out_2[4] = __fmaf_rn(nkx, x0_10, rg0_10 * ww[20]);
                    out_2[5] = __fmaf_rn(nkx, x1_10, rg1_10 * ww[21]);
                    acc[20] = ((row < T) ? __fmaf_rn(rg0_10, x0_10, acc[20]) : acc[20]);
                    acc[21] = ((row < T) ? __fmaf_rn(rg1_10, x1_10, acc[21]) : acc[21]);
                    float g0_11 = __uint_as_float(gw[11] << 16);
                    float g1_11 = __uint_as_float(gw[11] & 4294901760u);
                    float x0_11 = __uint_as_float(xw[11] << 16);
                    float x1_11 = __uint_as_float(xw[11] & 4294901760u);
                    float rg0_11 = rq * g0_11;
                    float rg1_11 = rq * g1_11;
                    out_2[6] = __fmaf_rn(nkx, x0_11, rg0_11 * ww[22]);
                    out_2[7] = __fmaf_rn(nkx, x1_11, rg1_11 * ww[23]);
                    acc[22] = ((row < T) ? __fmaf_rn(rg0_11, x0_11, acc[22]) : acc[22]);
                    acc[23] = ((row < T) ? __fmaf_rn(rg1_11, x1_11, acc[23]) : acc[23]);
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (row < T) {
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((64 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[12])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(12) + 3]))
                        : "r"(sbase + (96 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[12])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(12) + 3]))
                        : "r"(sbase + 2560 + (96 + t) * 8 * 2));
                    float out_3[8];
                    float g0_12 = __uint_as_float(gw[12] << 16);
                    float g1_12 = __uint_as_float(gw[12] & 4294901760u);
                    float x0_12 = __uint_as_float(xw[12] << 16);
                    float x1_12 = __uint_as_float(xw[12] & 4294901760u);
                    float rg0_12 = rq * g0_12;
                    float rg1_12 = rq * g1_12;
                    out_3[0] = __fmaf_rn(nkx, x0_12, rg0_12 * ww[24]);
                    out_3[1] = __fmaf_rn(nkx, x1_12, rg1_12 * ww[25]);
                    acc[24] = ((row < T) ? __fmaf_rn(rg0_12, x0_12, acc[24]) : acc[24]);
                    acc[25] = ((row < T) ? __fmaf_rn(rg1_12, x1_12, acc[25]) : acc[25]);
                    float g0_13 = __uint_as_float(gw[13] << 16);
                    float g1_13 = __uint_as_float(gw[13] & 4294901760u);
                    float x0_13 = __uint_as_float(xw[13] << 16);
                    float x1_13 = __uint_as_float(xw[13] & 4294901760u);
                    float rg0_13 = rq * g0_13;
                    float rg1_13 = rq * g1_13;
                    out_3[2] = __fmaf_rn(nkx, x0_13, rg0_13 * ww[26]);
                    out_3[3] = __fmaf_rn(nkx, x1_13, rg1_13 * ww[27]);
                    acc[26] = ((row < T) ? __fmaf_rn(rg0_13, x0_13, acc[26]) : acc[26]);
                    acc[27] = ((row < T) ? __fmaf_rn(rg1_13, x1_13, acc[27]) : acc[27]);
                    float g0_14 = __uint_as_float(gw[14] << 16);
                    float g1_14 = __uint_as_float(gw[14] & 4294901760u);
                    float x0_14 = __uint_as_float(xw[14] << 16);
                    float x1_14 = __uint_as_float(xw[14] & 4294901760u);
                    float rg0_14 = rq * g0_14;
                    float rg1_14 = rq * g1_14;
                    out_3[4] = __fmaf_rn(nkx, x0_14, rg0_14 * ww[28]);
                    out_3[5] = __fmaf_rn(nkx, x1_14, rg1_14 * ww[29]);
                    acc[28] = ((row < T) ? __fmaf_rn(rg0_14, x0_14, acc[28]) : acc[28]);
                    acc[29] = ((row < T) ? __fmaf_rn(rg1_14, x1_14, acc[29]) : acc[29]);
                    float g0_15 = __uint_as_float(gw[15] << 16);
                    float g1_15 = __uint_as_float(gw[15] & 4294901760u);
                    float x0_15 = __uint_as_float(xw[15] << 16);
                    float x1_15 = __uint_as_float(xw[15] & 4294901760u);
                    float rg0_15 = rq * g0_15;
                    float rg1_15 = rq * g1_15;
                    out_3[6] = __fmaf_rn(nkx, x0_15, rg0_15 * ww[30]);
                    out_3[7] = __fmaf_rn(nkx, x1_15, rg1_15 * ww[31]);
                    acc[30] = ((row < T) ? __fmaf_rn(rg0_15, x0_15, acc[30]) : acc[30]);
                    acc[31] = ((row < T) ? __fmaf_rn(rg1_15, x1_15, acc[31]) : acc[31]);
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (row < T) {
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((96 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[16])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(16) + 3]))
                        : "r"(sbase + (128 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[16])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(16) + 3]))
                        : "r"(sbase + 2560 + (128 + t) * 8 * 2));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    float out_4[8];
                    float g0_16 = __uint_as_float(gw[16] << 16);
                    float g1_16 = __uint_as_float(gw[16] & 4294901760u);
                    float x0_16 = __uint_as_float(xw[16] << 16);
                    float x1_16 = __uint_as_float(xw[16] & 4294901760u);
                    float rg0_16 = rq * g0_16;
                    float rg1_16 = rq * g1_16;
                    out_4[0] = __fmaf_rn(nkx, x0_16, rg0_16 * ww[32]);
                    out_4[1] = __fmaf_rn(nkx, x1_16, rg1_16 * ww[33]);
                    acc[32] = ((row < T) ? __fmaf_rn(rg0_16, x0_16, acc[32]) : acc[32]);
                    acc[33] = ((row < T) ? __fmaf_rn(rg1_16, x1_16, acc[33]) : acc[33]);
                    float g0_17 = __uint_as_float(gw[17] << 16);
                    float g1_17 = __uint_as_float(gw[17] & 4294901760u);
                    float x0_17 = __uint_as_float(xw[17] << 16);
                    float x1_17 = __uint_as_float(xw[17] & 4294901760u);
                    float rg0_17 = rq * g0_17;
                    float rg1_17 = rq * g1_17;
                    out_4[2] = __fmaf_rn(nkx, x0_17, rg0_17 * ww[34]);
                    out_4[3] = __fmaf_rn(nkx, x1_17, rg1_17 * ww[35]);
                    acc[34] = ((row < T) ? __fmaf_rn(rg0_17, x0_17, acc[34]) : acc[34]);
                    acc[35] = ((row < T) ? __fmaf_rn(rg1_17, x1_17, acc[35]) : acc[35]);
                    float g0_18 = __uint_as_float(gw[18] << 16);
                    float g1_18 = __uint_as_float(gw[18] & 4294901760u);
                    float x0_18 = __uint_as_float(xw[18] << 16);
                    float x1_18 = __uint_as_float(xw[18] & 4294901760u);
                    float rg0_18 = rq * g0_18;
                    float rg1_18 = rq * g1_18;
                    out_4[4] = __fmaf_rn(nkx, x0_18, rg0_18 * ww[36]);
                    out_4[5] = __fmaf_rn(nkx, x1_18, rg1_18 * ww[37]);
                    acc[36] = ((row < T) ? __fmaf_rn(rg0_18, x0_18, acc[36]) : acc[36]);
                    acc[37] = ((row < T) ? __fmaf_rn(rg1_18, x1_18, acc[37]) : acc[37]);
                    float g0_19 = __uint_as_float(gw[19] << 16);
                    float g1_19 = __uint_as_float(gw[19] & 4294901760u);
                    float x0_19 = __uint_as_float(xw[19] << 16);
                    float x1_19 = __uint_as_float(xw[19] & 4294901760u);
                    float rg0_19 = rq * g0_19;
                    float rg1_19 = rq * g1_19;
                    out_4[6] = __fmaf_rn(nkx, x0_19, rg0_19 * ww[38]);
                    out_4[7] = __fmaf_rn(nkx, x1_19, rg1_19 * ww[39]);
                    acc[38] = ((row < T) ? __fmaf_rn(rg0_19, x0_19, acc[38]) : acc[38]);
                    acc[39] = ((row < T) ? __fmaf_rn(rg1_19, x1_19, acc[39]) : acc[39]);
                    unsigned int packed_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 0], out_4[_lp*2+1 + 0]));
                        packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (row < T) {
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((128 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_4)[0];
                    }
                    mbarrier_arrive(empty_addr + (gi + 3 * (int)((unsigned int)k % 3)) * 8);
                    tile = tile + n_chunks;
                    k += 1;
                }
                asm volatile("barrier.sync 1, 96;" ::: "memory");
                if (gi == 0) {
                    {
                        uint32_t _addr_6 = static_cast<uint32_t>(fold_addr + 2048 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(acc[16]) : "memory");
                    }
                    {
                        uint32_t _addr_7 = static_cast<uint32_t>(fold_addr + 2176 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(acc[17]) : "memory");
                    }
                    {
                        uint32_t _addr_8 = static_cast<uint32_t>(fold_addr + 2304 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(acc[18]) : "memory");
                    }
                    {
                        uint32_t _addr_9 = static_cast<uint32_t>(fold_addr + 2432 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(acc[19]) : "memory");
                    }
                    {
                        uint32_t _addr_10 = static_cast<uint32_t>(fold_addr + 2560 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(acc[20]) : "memory");
                    }
                    {
                        uint32_t _addr_11 = static_cast<uint32_t>(fold_addr + 2688 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(acc[21]) : "memory");
                    }
                    {
                        uint32_t _addr_12 = static_cast<uint32_t>(fold_addr + 2816 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(acc[22]) : "memory");
                    }
                    {
                        uint32_t _addr_13 = static_cast<uint32_t>(fold_addr + 2944 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(acc[23]) : "memory");
                    }
                    {
                        uint32_t _addr_14 = static_cast<uint32_t>(fold_addr + 3072 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(acc[24]) : "memory");
                    }
                    {
                        uint32_t _addr_15 = static_cast<uint32_t>(fold_addr + 3200 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(acc[25]) : "memory");
                    }
                    {
                        uint32_t _addr_16 = static_cast<uint32_t>(fold_addr + 3328 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(acc[26]) : "memory");
                    }
                    {
                        uint32_t _addr_17 = static_cast<uint32_t>(fold_addr + 3456 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(acc[27]) : "memory");
                    }
                    {
                        uint32_t _addr_18 = static_cast<uint32_t>(fold_addr + 3584 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(acc[28]) : "memory");
                    }
                    {
                        uint32_t _addr_19 = static_cast<uint32_t>(fold_addr + 3712 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(acc[29]) : "memory");
                    }
                    {
                        uint32_t _addr_20 = static_cast<uint32_t>(fold_addr + 3840 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(acc[30]) : "memory");
                    }
                    {
                        uint32_t _addr_21 = static_cast<uint32_t>(fold_addr + 3968 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(acc[31]) : "memory");
                    }
                    {
                        uint32_t _addr_22 = static_cast<uint32_t>(fold_addr + 4096 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(acc[32]) : "memory");
                    }
                    {
                        uint32_t _addr_23 = static_cast<uint32_t>(fold_addr + 4224 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(acc[33]) : "memory");
                    }
                    {
                        uint32_t _addr_24 = static_cast<uint32_t>(fold_addr + 4352 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(acc[34]) : "memory");
                    }
                    {
                        uint32_t _addr_25 = static_cast<uint32_t>(fold_addr + 4480 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(acc[35]) : "memory");
                    }
                    {
                        uint32_t _addr_26 = static_cast<uint32_t>(fold_addr + 4608 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(acc[36]) : "memory");
                    }
                    {
                        uint32_t _addr_27 = static_cast<uint32_t>(fold_addr + 4736 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(acc[37]) : "memory");
                    }
                    {
                        uint32_t _addr_28 = static_cast<uint32_t>(fold_addr + 4864 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(acc[38]) : "memory");
                    }
                    {
                        uint32_t _addr_29 = static_cast<uint32_t>(fold_addr + 4992 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(acc[39]) : "memory");
                    }
                }
                if (gi == 1) {
                    {
                        uint32_t _addr_30 = static_cast<uint32_t>(fold_addr + 0 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(acc[0]) : "memory");
                    }
                    {
                        uint32_t _addr_31 = static_cast<uint32_t>(fold_addr + 128 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(acc[1]) : "memory");
                    }
                    {
                        uint32_t _addr_32 = static_cast<uint32_t>(fold_addr + 256 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(acc[2]) : "memory");
                    }
                    {
                        uint32_t _addr_33 = static_cast<uint32_t>(fold_addr + 384 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(acc[3]) : "memory");
                    }
                    {
                        uint32_t _addr_34 = static_cast<uint32_t>(fold_addr + 512 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(acc[4]) : "memory");
                    }
                    {
                        uint32_t _addr_35 = static_cast<uint32_t>(fold_addr + 640 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(acc[5]) : "memory");
                    }
                    {
                        uint32_t _addr_36 = static_cast<uint32_t>(fold_addr + 768 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(acc[6]) : "memory");
                    }
                    {
                        uint32_t _addr_37 = static_cast<uint32_t>(fold_addr + 896 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(acc[7]) : "memory");
                    }
                    {
                        uint32_t _addr_38 = static_cast<uint32_t>(fold_addr + 1024 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(acc[8]) : "memory");
                    }
                    {
                        uint32_t _addr_39 = static_cast<uint32_t>(fold_addr + 1152 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(acc[9]) : "memory");
                    }
                    {
                        uint32_t _addr_40 = static_cast<uint32_t>(fold_addr + 1280 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(acc[10]) : "memory");
                    }
                    {
                        uint32_t _addr_41 = static_cast<uint32_t>(fold_addr + 1408 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(acc[11]) : "memory");
                    }
                    {
                        uint32_t _addr_42 = static_cast<uint32_t>(fold_addr + 1536 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(acc[12]) : "memory");
                    }
                    {
                        uint32_t _addr_43 = static_cast<uint32_t>(fold_addr + 1664 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(acc[13]) : "memory");
                    }
                    {
                        uint32_t _addr_44 = static_cast<uint32_t>(fold_addr + 1792 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(acc[14]) : "memory");
                    }
                    {
                        uint32_t _addr_45 = static_cast<uint32_t>(fold_addr + 1920 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(acc[15]) : "memory");
                    }
                    {
                        uint32_t _addr_46 = static_cast<uint32_t>(fold_addr + 9216 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(acc[32]) : "memory");
                    }
                    {
                        uint32_t _addr_47 = static_cast<uint32_t>(fold_addr + 9344 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(acc[33]) : "memory");
                    }
                    {
                        uint32_t _addr_48 = static_cast<uint32_t>(fold_addr + 9472 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(acc[34]) : "memory");
                    }
                    {
                        uint32_t _addr_49 = static_cast<uint32_t>(fold_addr + 9600 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(acc[35]) : "memory");
                    }
                    {
                        uint32_t _addr_50 = static_cast<uint32_t>(fold_addr + 9728 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(acc[36]) : "memory");
                    }
                    {
                        uint32_t _addr_51 = static_cast<uint32_t>(fold_addr + 9856 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(acc[37]) : "memory");
                    }
                    {
                        uint32_t _addr_52 = static_cast<uint32_t>(fold_addr + 9984 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(acc[38]) : "memory");
                    }
                    {
                        uint32_t _addr_53 = static_cast<uint32_t>(fold_addr + 10112 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(acc[39]) : "memory");
                    }
                }
                if (gi == 2) {
                    {
                        uint32_t _addr_54 = static_cast<uint32_t>(fold_addr + 5120 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(acc[0]) : "memory");
                    }
                    {
                        uint32_t _addr_55 = static_cast<uint32_t>(fold_addr + 5248 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(acc[1]) : "memory");
                    }
                    {
                        uint32_t _addr_56 = static_cast<uint32_t>(fold_addr + 5376 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(acc[2]) : "memory");
                    }
                    {
                        uint32_t _addr_57 = static_cast<uint32_t>(fold_addr + 5504 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(acc[3]) : "memory");
                    }
                    {
                        uint32_t _addr_58 = static_cast<uint32_t>(fold_addr + 5632 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(acc[4]) : "memory");
                    }
                    {
                        uint32_t _addr_59 = static_cast<uint32_t>(fold_addr + 5760 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(acc[5]) : "memory");
                    }
                    {
                        uint32_t _addr_60 = static_cast<uint32_t>(fold_addr + 5888 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(acc[6]) : "memory");
                    }
                    {
                        uint32_t _addr_61 = static_cast<uint32_t>(fold_addr + 6016 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(acc[7]) : "memory");
                    }
                    {
                        uint32_t _addr_62 = static_cast<uint32_t>(fold_addr + 6144 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(acc[8]) : "memory");
                    }
                    {
                        uint32_t _addr_63 = static_cast<uint32_t>(fold_addr + 6272 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(acc[9]) : "memory");
                    }
                    {
                        uint32_t _addr_64 = static_cast<uint32_t>(fold_addr + 6400 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(acc[10]) : "memory");
                    }
                    {
                        uint32_t _addr_65 = static_cast<uint32_t>(fold_addr + 6528 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(acc[11]) : "memory");
                    }
                    {
                        uint32_t _addr_66 = static_cast<uint32_t>(fold_addr + 6656 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(acc[12]) : "memory");
                    }
                    {
                        uint32_t _addr_67 = static_cast<uint32_t>(fold_addr + 6784 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(acc[13]) : "memory");
                    }
                    {
                        uint32_t _addr_68 = static_cast<uint32_t>(fold_addr + 6912 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(acc[14]) : "memory");
                    }
                    {
                        uint32_t _addr_69 = static_cast<uint32_t>(fold_addr + 7040 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(acc[15]) : "memory");
                    }
                    {
                        uint32_t _addr_70 = static_cast<uint32_t>(fold_addr + 7168 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(acc[16]) : "memory");
                    }
                    {
                        uint32_t _addr_71 = static_cast<uint32_t>(fold_addr + 7296 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_71), "f"(acc[17]) : "memory");
                    }
                    {
                        uint32_t _addr_72 = static_cast<uint32_t>(fold_addr + 7424 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(acc[18]) : "memory");
                    }
                    {
                        uint32_t _addr_73 = static_cast<uint32_t>(fold_addr + 7552 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(acc[19]) : "memory");
                    }
                    {
                        uint32_t _addr_74 = static_cast<uint32_t>(fold_addr + 7680 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(acc[20]) : "memory");
                    }
                    {
                        uint32_t _addr_75 = static_cast<uint32_t>(fold_addr + 7808 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(acc[21]) : "memory");
                    }
                    {
                        uint32_t _addr_76 = static_cast<uint32_t>(fold_addr + 7936 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(acc[22]) : "memory");
                    }
                    {
                        uint32_t _addr_77 = static_cast<uint32_t>(fold_addr + 8064 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(acc[23]) : "memory");
                    }
                    {
                        uint32_t _addr_78 = static_cast<uint32_t>(fold_addr + 8192 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(acc[24]) : "memory");
                    }
                    {
                        uint32_t _addr_79 = static_cast<uint32_t>(fold_addr + 8320 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(acc[25]) : "memory");
                    }
                    {
                        uint32_t _addr_80 = static_cast<uint32_t>(fold_addr + 8448 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(acc[26]) : "memory");
                    }
                    {
                        uint32_t _addr_81 = static_cast<uint32_t>(fold_addr + 8576 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(acc[27]) : "memory");
                    }
                    {
                        uint32_t _addr_82 = static_cast<uint32_t>(fold_addr + 8704 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(acc[28]) : "memory");
                    }
                    {
                        uint32_t _addr_83 = static_cast<uint32_t>(fold_addr + 8832 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(acc[29]) : "memory");
                    }
                    {
                        uint32_t _addr_84 = static_cast<uint32_t>(fold_addr + 8960 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(acc[30]) : "memory");
                    }
                    {
                        uint32_t _addr_85 = static_cast<uint32_t>(fold_addr + 9088 + (unsigned int)(lane * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(acc[31]) : "memory");
                    }
                }
                asm volatile("barrier.sync 1, 96;" ::: "memory");
                float pq[8];
                unsigned long long pbase = (unsigned long long)slot * 1280ull;
                if (gi == 0) {
                    pq[0] = acc[0];
                    pq[1] = acc[1];
                    pq[2] = acc[2];
                    pq[3] = acc[3];
                    pq[4] = acc[4];
                    pq[5] = acc[5];
                    pq[6] = acc[6];
                    pq[7] = acc[7];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 0 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 128 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 256 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 384 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 512 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 640 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 768 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 896 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 5120 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 5248 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 5376 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 5504 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 5632 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 5760 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 5888 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 6016 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    {
                        unsigned _stv8_86_0 = __float_as_uint(pq[0 + 0]);
                        unsigned _stv8_86_1 = __float_as_uint(pq[0 + 1]);
                        unsigned _stv8_86_2 = __float_as_uint(pq[0 + 2]);
                        unsigned _stv8_86_3 = __float_as_uint(pq[0 + 3]);
                        unsigned _stv8_86_4 = __float_as_uint(pq[0 + 4]);
                        unsigned _stv8_86_5 = __float_as_uint(pq[0 + 5]);
                        unsigned _stv8_86_6 = __float_as_uint(pq[0 + 6]);
                        unsigned _stv8_86_7 = __float_as_uint(pq[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((0 + t) * 8)))), "r"(_stv8_86_0), "r"(_stv8_86_1), "r"(_stv8_86_2), "r"(_stv8_86_3), "r"(_stv8_86_4), "r"(_stv8_86_5), "r"(_stv8_86_6), "r"(_stv8_86_7) : "memory");
                    }
                    pq[0] = acc[8];
                    pq[1] = acc[9];
                    pq[2] = acc[10];
                    pq[3] = acc[11];
                    pq[4] = acc[12];
                    pq[5] = acc[13];
                    pq[6] = acc[14];
                    pq[7] = acc[15];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 1024 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 1152 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 1280 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 1408 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 1536 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 1664 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 1792 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 1920 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 6144 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 6272 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 6400 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 6528 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 6656 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 6784 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 6912 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 7040 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    {
                        unsigned _stv8_87_0 = __float_as_uint(pq[0 + 0]);
                        unsigned _stv8_87_1 = __float_as_uint(pq[0 + 1]);
                        unsigned _stv8_87_2 = __float_as_uint(pq[0 + 2]);
                        unsigned _stv8_87_3 = __float_as_uint(pq[0 + 3]);
                        unsigned _stv8_87_4 = __float_as_uint(pq[0 + 4]);
                        unsigned _stv8_87_5 = __float_as_uint(pq[0 + 5]);
                        unsigned _stv8_87_6 = __float_as_uint(pq[0 + 6]);
                        unsigned _stv8_87_7 = __float_as_uint(pq[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((32 + t) * 8)))), "r"(_stv8_87_0), "r"(_stv8_87_1), "r"(_stv8_87_2), "r"(_stv8_87_3), "r"(_stv8_87_4), "r"(_stv8_87_5), "r"(_stv8_87_6), "r"(_stv8_87_7) : "memory");
                    }
                }
                if (gi == 1) {
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 2048 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 2176 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 2304 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 2432 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 2560 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 2688 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 2816 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 2944 + (unsigned int)(lane * 4)));
                    pq[0] = part[0];
                    pq[1] = part[1];
                    pq[2] = part[2];
                    pq[3] = part[3];
                    pq[4] = part[4];
                    pq[5] = part[5];
                    pq[6] = part[6];
                    pq[7] = part[7];
                    pq[0] = pq[0] + acc[16];
                    pq[1] = pq[1] + acc[17];
                    pq[2] = pq[2] + acc[18];
                    pq[3] = pq[3] + acc[19];
                    pq[4] = pq[4] + acc[20];
                    pq[5] = pq[5] + acc[21];
                    pq[6] = pq[6] + acc[22];
                    pq[7] = pq[7] + acc[23];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 7168 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 7296 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 7424 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 7552 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 7680 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 7808 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 7936 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 8064 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    {
                        unsigned _stv8_88_0 = __float_as_uint(pq[0 + 0]);
                        unsigned _stv8_88_1 = __float_as_uint(pq[0 + 1]);
                        unsigned _stv8_88_2 = __float_as_uint(pq[0 + 2]);
                        unsigned _stv8_88_3 = __float_as_uint(pq[0 + 3]);
                        unsigned _stv8_88_4 = __float_as_uint(pq[0 + 4]);
                        unsigned _stv8_88_5 = __float_as_uint(pq[0 + 5]);
                        unsigned _stv8_88_6 = __float_as_uint(pq[0 + 6]);
                        unsigned _stv8_88_7 = __float_as_uint(pq[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((64 + t) * 8)))), "r"(_stv8_88_0), "r"(_stv8_88_1), "r"(_stv8_88_2), "r"(_stv8_88_3), "r"(_stv8_88_4), "r"(_stv8_88_5), "r"(_stv8_88_6), "r"(_stv8_88_7) : "memory");
                    }
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 3072 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 3200 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 3328 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 3456 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 3584 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 3712 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 3840 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 3968 + (unsigned int)(lane * 4)));
                    pq[0] = part[0];
                    pq[1] = part[1];
                    pq[2] = part[2];
                    pq[3] = part[3];
                    pq[4] = part[4];
                    pq[5] = part[5];
                    pq[6] = part[6];
                    pq[7] = part[7];
                    pq[0] = pq[0] + acc[24];
                    pq[1] = pq[1] + acc[25];
                    pq[2] = pq[2] + acc[26];
                    pq[3] = pq[3] + acc[27];
                    pq[4] = pq[4] + acc[28];
                    pq[5] = pq[5] + acc[29];
                    pq[6] = pq[6] + acc[30];
                    pq[7] = pq[7] + acc[31];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 8192 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 8320 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 8448 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 8576 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 8704 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 8832 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 8960 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 9088 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    {
                        unsigned _stv8_89_0 = __float_as_uint(pq[0 + 0]);
                        unsigned _stv8_89_1 = __float_as_uint(pq[0 + 1]);
                        unsigned _stv8_89_2 = __float_as_uint(pq[0 + 2]);
                        unsigned _stv8_89_3 = __float_as_uint(pq[0 + 3]);
                        unsigned _stv8_89_4 = __float_as_uint(pq[0 + 4]);
                        unsigned _stv8_89_5 = __float_as_uint(pq[0 + 5]);
                        unsigned _stv8_89_6 = __float_as_uint(pq[0 + 6]);
                        unsigned _stv8_89_7 = __float_as_uint(pq[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((96 + t) * 8)))), "r"(_stv8_89_0), "r"(_stv8_89_1), "r"(_stv8_89_2), "r"(_stv8_89_3), "r"(_stv8_89_4), "r"(_stv8_89_5), "r"(_stv8_89_6), "r"(_stv8_89_7) : "memory");
                    }
                }
                if (gi == 2) {
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 4096 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 4224 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 4352 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 4480 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 4608 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 4736 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 4864 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 4992 + (unsigned int)(lane * 4)));
                    pq[0] = part[0];
                    pq[1] = part[1];
                    pq[2] = part[2];
                    pq[3] = part[3];
                    pq[4] = part[4];
                    pq[5] = part[5];
                    pq[6] = part[6];
                    pq[7] = part[7];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(fold_addr + 9216 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(fold_addr + 9344 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(fold_addr + 9472 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(fold_addr + 9600 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(fold_addr + 9728 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(fold_addr + 9856 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(fold_addr + 9984 + (unsigned int)(lane * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(fold_addr + 10112 + (unsigned int)(lane * 4)));
                    pq[0] = pq[0] + part[0];
                    pq[1] = pq[1] + part[1];
                    pq[2] = pq[2] + part[2];
                    pq[3] = pq[3] + part[3];
                    pq[4] = pq[4] + part[4];
                    pq[5] = pq[5] + part[5];
                    pq[6] = pq[6] + part[6];
                    pq[7] = pq[7] + part[7];
                    pq[0] = pq[0] + acc[32];
                    pq[1] = pq[1] + acc[33];
                    pq[2] = pq[2] + acc[34];
                    pq[3] = pq[3] + acc[35];
                    pq[4] = pq[4] + acc[36];
                    pq[5] = pq[5] + acc[37];
                    pq[6] = pq[6] + acc[38];
                    pq[7] = pq[7] + acc[39];
                    {
                        unsigned _stv8_90_0 = __float_as_uint(pq[0 + 0]);
                        unsigned _stv8_90_1 = __float_as_uint(pq[0 + 1]);
                        unsigned _stv8_90_2 = __float_as_uint(pq[0 + 2]);
                        unsigned _stv8_90_3 = __float_as_uint(pq[0 + 3]);
                        unsigned _stv8_90_4 = __float_as_uint(pq[0 + 4]);
                        unsigned _stv8_90_5 = __float_as_uint(pq[0 + 5]);
                        unsigned _stv8_90_6 = __float_as_uint(pq[0 + 6]);
                        unsigned _stv8_90_7 = __float_as_uint(pq[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(partial + (pbase + (unsigned long long)((128 + t) * 8)))), "r"(_stv8_90_0), "r"(_stv8_90_1), "r"(_stv8_90_2), "r"(_stv8_90_3), "r"(_stv8_90_4), "r"(_stv8_90_5), "r"(_stv8_90_6), "r"(_stv8_90_7) : "memory");
                    }
                }
                slot = slot + (int)num_bids;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 3) {
        { // prod_main
            if (elect_sync()) {
                int q_ring = (int)ring_addr;
                int q_tiles = (T + 2) / 3;
                int qk = 0;
                int q_slot = (int)bid;
                while (q_slot < n_chunks) {
                    int q_tile = q_slot;
                    while (q_tile < q_tiles) {
                        int _min_0 = ((q_tile * 3) < (T - 1) ? (q_tile * 3) : (T - 1));
                        long long q_row_0 = (long long)_min_0;
                        int _stage_empty = 3 * (int)((unsigned int)qk % 3);
                        int _parity_empty = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (3 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + 3 * (qk % 3) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_0 * g_stride_p0), 2560, full_addr + (3 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + 3 * (qk % 3) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_0 * x_stride_p0), 2560, full_addr + (3 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_1 = ((q_tile * 3 + 1) < (T - 1) ? (q_tile * 3 + 1) : (T - 1));
                        long long q_row_1 = (long long)_min_1;
                        int _stage_empty_1 = 1 + 3 * (int)((unsigned int)qk % 3);
                        int _parity_empty_1 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_1) * 8, _parity_empty_1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (1 + 3 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + (1 + 3 * (qk % 3)) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_1 * g_stride_p0), 2560, full_addr + (1 + 3 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (1 + 3 * (qk % 3)) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_1 * x_stride_p0), 2560, full_addr + (1 + 3 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_2 = ((q_tile * 3 + 2) < (T - 1) ? (q_tile * 3 + 2) : (T - 1));
                        long long q_row_2 = (long long)_min_2;
                        int _stage_empty_2 = 2 + 3 * (int)((unsigned int)qk % 3);
                        int _parity_empty_2 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_2) * 8, _parity_empty_2);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (2 + 3 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + (2 + 3 * (qk % 3)) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_2 * g_stride_p0), 2560, full_addr + (2 + 3 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (2 + 3 * (qk % 3)) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_2 * x_stride_p0), 2560, full_addr + (2 + 3 * (int)((unsigned int)qk % 3)) * 8);
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
