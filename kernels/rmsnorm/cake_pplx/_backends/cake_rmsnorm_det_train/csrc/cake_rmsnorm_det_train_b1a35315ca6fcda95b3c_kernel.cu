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



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(1024, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_b1a35315ca6fcda95b3c(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 1408);
    const int ring_addr = smem + 1408;
    float* folds = reinterpret_cast<float*>(smem_raw + 78208);
    const int fold_addr = smem + 78208;

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
            // empty: 15 barriers, init_count=160
            mbarrier_init(smem + 120, 160);
            mbarrier_init(smem + 128, 160);
            mbarrier_init(smem + 136, 160);
            mbarrier_init(smem + 144, 160);
            mbarrier_init(smem + 152, 160);
            mbarrier_init(smem + 160, 160);
            mbarrier_init(smem + 168, 160);
            mbarrier_init(smem + 176, 160);
            mbarrier_init(smem + 184, 160);
            mbarrier_init(smem + 192, 160);
            mbarrier_init(smem + 200, 160);
            mbarrier_init(smem + 208, 160);
            mbarrier_init(smem + 216, 160);
            mbarrier_init(smem + 224, 160);
            mbarrier_init(smem + 232, 160);
            // rows_mb: 10 barriers, init_count=160
            mbarrier_init(smem + 240, 160);
            mbarrier_init(smem + 248, 160);
            mbarrier_init(smem + 256, 160);
            mbarrier_init(smem + 264, 160);
            mbarrier_init(smem + 272, 160);
            mbarrier_init(smem + 280, 160);
            mbarrier_init(smem + 288, 160);
            mbarrier_init(smem + 296, 160);
            mbarrier_init(smem + 304, 160);
            mbarrier_init(smem + 312, 160);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 24) {
        { // compute_main
            int gi = (int)warp / 5;
            int wg = (int)warp - gi * 5;
            int t = (int)tid - gi * 160;
            int ring_at = (int)ring_addr;
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
            float acc[8];
            float part[8];
            unsigned int gw[4];
            unsigned int xw[4];
            float dv[5];
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
                int tile = slot;
                while (tile < tiles) {
                    int row = tile * 5 + gi;
                    int _min_1 = ((row) < (T - 1) ? (row) : (T - 1));
                    float rq = r[_min_1];
                    int _stage_full = gi + 5 * (int)((unsigned int)k % 3);
                    int _parity_full = (int)((unsigned int)k / 3 % 2);
                    mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                    int sbase = ring_at + (gi + 5 * (k % 3)) * 5120;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gw[0])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gw[(0) + 3]))
                        : "r"(sbase + (0 + t) * 8 * 2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xw[0])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xw[(0) + 3]))
                        : "r"(sbase + 2560 + (0 + t) * 8 * 2));
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
                    float dsum = dq;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsum += __shfl_xor_sync(0xFFFFFFFF, dsum, offset);
                    int par = gi * 16 + (k & 1) * 8;
                    if (lane == 0) {
                        dots[par + wg] = dsum;
                    }
                    mbarrier_arrive(rows_mb_addr + (gi + 5 * (int)((unsigned int)k % 2)) * 8);
                    int _stage_rows_mb = gi + 5 * (int)((unsigned int)k % 2);
                    int _parity_rows_mb = (int)((unsigned int)k / 2 % 2);
                    mbarrier_wait(rows_mb_addr + (_stage_rows_mb) * 8, _parity_rows_mb);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dv[0])), "=r"(*reinterpret_cast<uint32_t*>(&dv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dv[(0) + 3]))
                        : "r"(dots_addr + (unsigned int)(par * 4)));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&dv[4])) : "r"(dots_addr + (unsigned int)((par + 5 - 1) * 4)));
                    float tot = dv[0];
                    tot = tot + dv[1];
                    tot = tot + dv[2];
                    tot = tot + dv[3];
                    tot = tot + dv[4];
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(empty_addr + (gi + 5 * (int)((unsigned int)k % 3)) * 8);
                    float cq = tot * 0.00078125f;
                    float kx = rq * rq * rq * cq;
                    float nkx = -kx;
                    unsigned long long dx_base = (unsigned long long)row * 1280ull;
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
                    tile = tile + n_chunks;
                    k += 1;
                }
                asm volatile("barrier.sync 1, 800;" ::: "memory");
                if ((unsigned int)(gi - 1) < 2) {
                    {
                        uint32_t _addr_2 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(acc[0]) : "memory");
                    }
                    {
                        uint32_t _addr_3 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 1) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(acc[1]) : "memory");
                    }
                    {
                        uint32_t _addr_4 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 2) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(acc[2]) : "memory");
                    }
                    {
                        uint32_t _addr_5 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 3) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(acc[3]) : "memory");
                    }
                    {
                        uint32_t _addr_6 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 4) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(acc[4]) : "memory");
                    }
                    {
                        uint32_t _addr_7 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 5) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(acc[5]) : "memory");
                    }
                    {
                        uint32_t _addr_8 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 6) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(acc[6]) : "memory");
                    }
                    {
                        uint32_t _addr_9 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 1) * 1280 + (0 + t) * 8 + 7) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(acc[7]) : "memory");
                    }
                }
                asm volatile("barrier.sync 1, 800;" ::: "memory");
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
                        : "r"(fold_addr + (unsigned int)((1280 + (0 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((1280 + (0 + t) * 8 + 4) * 4)));
                    acc[0] = acc[0] + part[0];
                    acc[1] = acc[1] + part[1];
                    acc[2] = acc[2] + part[2];
                    acc[3] = acc[3] + part[3];
                    acc[4] = acc[4] + part[4];
                    acc[5] = acc[5] + part[5];
                    acc[6] = acc[6] + part[6];
                    acc[7] = acc[7] + part[7];
                }
                asm volatile("barrier.sync 1, 800;" ::: "memory");
                if (gi > 2) {
                    {
                        uint32_t _addr_10 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(acc[0]) : "memory");
                    }
                    {
                        uint32_t _addr_11 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 1) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(acc[1]) : "memory");
                    }
                    {
                        uint32_t _addr_12 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 2) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(acc[2]) : "memory");
                    }
                    {
                        uint32_t _addr_13 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 3) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(acc[3]) : "memory");
                    }
                    {
                        uint32_t _addr_14 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 4) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(acc[4]) : "memory");
                    }
                    {
                        uint32_t _addr_15 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 5) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(acc[5]) : "memory");
                    }
                    {
                        uint32_t _addr_16 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 6) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(acc[6]) : "memory");
                    }
                    {
                        uint32_t _addr_17 = static_cast<uint32_t>(fold_addr + (unsigned int)(((gi - 3) * 1280 + (0 + t) * 8 + 7) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(acc[7]) : "memory");
                    }
                }
                asm volatile("barrier.sync 1, 800;" ::: "memory");
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
                        : "r"(fold_addr + (unsigned int)((1280 + (0 + t) * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
                        : "r"(fold_addr + (unsigned int)((1280 + (0 + t) * 8 + 4) * 4)));
                    acc[0] = acc[0] + part[0];
                    acc[1] = acc[1] + part[1];
                    acc[2] = acc[2] + part[2];
                    acc[3] = acc[3] + part[3];
                    acc[4] = acc[4] + part[4];
                    acc[5] = acc[5] + part[5];
                    acc[6] = acc[6] + part[6];
                    acc[7] = acc[7] + part[7];
                    unsigned long long pbase = (unsigned long long)slot * 1280ull;
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8))) = acc[0];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 1))) = acc[1];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 2))) = acc[2];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 3))) = acc[3];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 4))) = acc[4];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 5))) = acc[5];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 6))) = acc[6];
                    *(reinterpret_cast<float*>(partial) + (pbase + (unsigned long long)((0 + t) * 8 + 7))) = acc[7];
                }
                slot = slot + (int)num_bids;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 25) {
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
                        mbarrier_arrive_expect_tx(full_addr + (5 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + 5 * (qk % 3) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_0 * g_stride_p0), 2560, full_addr + (5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + 5 * (qk % 3) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_0 * x_stride_p0), 2560, full_addr + (5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_1_1 = ((q_tile * 5 + 1) < (T - 1) ? (q_tile * 5 + 1) : (T - 1));
                        long long q_row_1 = (long long)_min_1_1;
                        int _stage_empty_1 = 1 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_1 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_1) * 8, _parity_empty_1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (1 + 5 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + (1 + 5 * (qk % 3)) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_1 * g_stride_p0), 2560, full_addr + (1 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (1 + 5 * (qk % 3)) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_1 * x_stride_p0), 2560, full_addr + (1 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_2 = ((q_tile * 5 + 2) < (T - 1) ? (q_tile * 5 + 2) : (T - 1));
                        long long q_row_2 = (long long)_min_2;
                        int _stage_empty_2 = 2 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_2 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_2) * 8, _parity_empty_2);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (2 + 5 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + (2 + 5 * (qk % 3)) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_2 * g_stride_p0), 2560, full_addr + (2 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (2 + 5 * (qk % 3)) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_2 * x_stride_p0), 2560, full_addr + (2 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_3 = ((q_tile * 5 + 3) < (T - 1) ? (q_tile * 5 + 3) : (T - 1));
                        long long q_row_3 = (long long)_min_3;
                        int _stage_empty_3 = 3 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_3 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_3) * 8, _parity_empty_3);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (3 + 5 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + (3 + 5 * (qk % 3)) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_3 * g_stride_p0), 2560, full_addr + (3 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (3 + 5 * (qk % 3)) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_3 * x_stride_p0), 2560, full_addr + (3 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        int _min_4 = ((q_tile * 5 + 4) < (T - 1) ? (q_tile * 5 + 4) : (T - 1));
                        long long q_row_4 = (long long)_min_4;
                        int _stage_empty_4 = 4 + 5 * (int)((unsigned int)qk % 3);
                        int _parity_empty_4 = (int)((unsigned int)qk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty_4) * 8, _parity_empty_4);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + (4 + 5 * (int)((unsigned int)qk % 3)) * 8, 5120);
                        cp_async_bulk_gmem2smem(q_ring + (4 + 5 * (qk % 3)) * 5120, reinterpret_cast<const uint8_t*>(g + q_row_4 * g_stride_p0), 2560, full_addr + (4 + 5 * (int)((unsigned int)qk % 3)) * 8);
                        cp_async_bulk_gmem2smem(q_ring + (4 + 5 * (qk % 3)) * 5120 + 2560, reinterpret_cast<const uint8_t*>(x + q_row_4 * x_stride_p0), 2560, full_addr + (4 + 5 * (int)((unsigned int)qk % 3)) * 8);
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
