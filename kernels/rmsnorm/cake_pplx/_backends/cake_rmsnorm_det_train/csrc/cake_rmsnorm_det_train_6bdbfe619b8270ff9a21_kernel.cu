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

#define LAUNCH_MIN_BLOCKS 5

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
kernel_cake_rmsnorm_det_train_6bdbfe619b8270ff9a21(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* fold = reinterpret_cast<float*>(smem_raw + 1024);
    const int fold_addr = smem + 1024;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int ring_addr = smem + 5120;

    // Mbarrier init (6 barriers at smem_raw[0..48))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // empty: 3 barriers, init_count=128
            mbarrier_init(smem + 24, 128);
            mbarrier_init(smem + 32, 128);
            mbarrier_init(smem + 40, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int t = tid;
            int wr = warp * 2;
            int ring_at = (int)ring_addr;
            unsigned int ww[4];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)((0 + lane) * 8));
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
            float acc[8];
            unsigned int gw[8];
            unsigned int xw[8];
            float rq[2];
            float rn[2];
            float part[8];
            int n_tiles = (T + 7) / 8;
            int k = 0;
            #pragma unroll 1
            for (int slot = bid; slot < n_chunks; slot += num_bids) {
                acc[0] = 0.0f;
                acc[1] = 0.0f;
                acc[2] = 0.0f;
                acc[3] = 0.0f;
                acc[4] = 0.0f;
                acc[5] = 0.0f;
                acc[6] = 0.0f;
                acc[7] = 0.0f;
                int _min_1 = ((slot * 8 + wr) < (T - 1) ? (slot * 8 + wr) : (T - 1));
                rn[0] = r[_min_1];
                int _min_2 = ((slot * 8 + wr + 1) < (T - 1) ? (slot * 8 + wr + 1) : (T - 1));
                rn[1] = r[_min_2];
                #pragma unroll 1
                for (int tile = slot; tile < n_tiles; tile += n_chunks) {
                    rq[0] = rn[0];
                    rq[1] = rn[1];
                    int _stage_full = (int)((unsigned int)k % 3);
                    int _parity_full = (int)((unsigned int)k / 3 % 2);
                    mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                    int sbase = ring_at + k % 3 * 8192 + wr * 512;
                    __int128_t _smem_raw_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(sbase + (0 + lane) * 8 * 2));
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(sbase + 4096 + (0 + lane) * 8 * 2));
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(sbase + 512 + (0 + lane) * 8 * 2));
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    __int128_t _smem_raw_6;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(sbase + 4608 + (0 + lane) * 8 * 2));
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                    mbarrier_arrive(empty_addr + ((int)((unsigned int)k % 3)) * 8);
                    int _min_7 = (((tile + n_chunks) * 8 + wr) < (T - 1) ? ((tile + n_chunks) * 8 + wr) : (T - 1));
                    rn[0] = r[_min_7];
                    int _min_8 = (((tile + n_chunks) * 8 + wr + 1) < (T - 1) ? ((tile + n_chunks) * 8 + wr + 1) : (T - 1));
                    rn[1] = r[_min_8];
                    float dq_0 = 0.0f;
                    float ga_0 = __uint_as_float(gw[0] << 16);
                    float gb_0 = __uint_as_float(gw[0] & 4294901760u);
                    float xa_0 = __uint_as_float(xw[0] << 16);
                    float xb_0 = __uint_as_float(xw[0] & 4294901760u);
                    dq_0 = __fmaf_rn(ga_0 * __uint_as_float(ww[0] << 16), xa_0, dq_0);
                    dq_0 = __fmaf_rn(gb_0 * __uint_as_float(ww[0] & 4294901760u), xb_0, dq_0);
                    float ga_1 = __uint_as_float(gw[1] << 16);
                    float gb_1 = __uint_as_float(gw[1] & 4294901760u);
                    float xa_1 = __uint_as_float(xw[1] << 16);
                    float xb_1 = __uint_as_float(xw[1] & 4294901760u);
                    dq_0 = __fmaf_rn(ga_1 * __uint_as_float(ww[1] << 16), xa_1, dq_0);
                    dq_0 = __fmaf_rn(gb_1 * __uint_as_float(ww[1] & 4294901760u), xb_1, dq_0);
                    float ga_2 = __uint_as_float(gw[2] << 16);
                    float gb_2 = __uint_as_float(gw[2] & 4294901760u);
                    float xa_2 = __uint_as_float(xw[2] << 16);
                    float xb_2 = __uint_as_float(xw[2] & 4294901760u);
                    dq_0 = __fmaf_rn(ga_2 * __uint_as_float(ww[2] << 16), xa_2, dq_0);
                    dq_0 = __fmaf_rn(gb_2 * __uint_as_float(ww[2] & 4294901760u), xb_2, dq_0);
                    float ga_3 = __uint_as_float(gw[3] << 16);
                    float gb_3 = __uint_as_float(gw[3] & 4294901760u);
                    float xa_3 = __uint_as_float(xw[3] << 16);
                    float xb_3 = __uint_as_float(xw[3] & 4294901760u);
                    dq_0 = __fmaf_rn(ga_3 * __uint_as_float(ww[3] << 16), xa_3, dq_0);
                    dq_0 = __fmaf_rn(gb_3 * __uint_as_float(ww[3] & 4294901760u), xb_3, dq_0);
                    float dsum_0 = dq_0;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsum_0 += __shfl_xor_sync(0xFFFFFFFF, dsum_0, offset);
                    float cq_0 = dsum_0 * 0.00390625f;
                    float kx_0 = rq[0] * rq[0] * rq[0] * cq_0;
                    float nkx_0 = -kx_0;
                    int row_0 = tile * 8 + wr;
                    if (row_0 < T) {
                        unsigned long long dx_base_0 = (unsigned long long)row_0 * 256ull;
                        float out_0[8];
                        float g0_0 = __uint_as_float(gw[0] << 16);
                        float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                        float x0_0 = __uint_as_float(xw[0] << 16);
                        float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                        out_0[0] = __fmaf_rn(nkx_0, x0_0, rq[0] * (g0_0 * __uint_as_float(ww[0] << 16)));
                        out_0[1] = __fmaf_rn(nkx_0, x1_0, rq[0] * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
                        acc[0] = __fmaf_rn(g0_0 * x0_0, rq[0], acc[0]);
                        acc[1] = __fmaf_rn(g1_0 * x1_0, rq[0], acc[1]);
                        float g0_1 = __uint_as_float(gw[1] << 16);
                        float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                        float x0_1 = __uint_as_float(xw[1] << 16);
                        float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                        out_0[2] = __fmaf_rn(nkx_0, x0_1, rq[0] * (g0_1 * __uint_as_float(ww[1] << 16)));
                        out_0[3] = __fmaf_rn(nkx_0, x1_1, rq[0] * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
                        acc[2] = __fmaf_rn(g0_1 * x0_1, rq[0], acc[2]);
                        acc[3] = __fmaf_rn(g1_1 * x1_1, rq[0], acc[3]);
                        float g0_2 = __uint_as_float(gw[2] << 16);
                        float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                        float x0_2 = __uint_as_float(xw[2] << 16);
                        float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                        out_0[4] = __fmaf_rn(nkx_0, x0_2, rq[0] * (g0_2 * __uint_as_float(ww[2] << 16)));
                        out_0[5] = __fmaf_rn(nkx_0, x1_2, rq[0] * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
                        acc[4] = __fmaf_rn(g0_2 * x0_2, rq[0], acc[4]);
                        acc[5] = __fmaf_rn(g1_2 * x1_2, rq[0], acc[5]);
                        float g0_3 = __uint_as_float(gw[3] << 16);
                        float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                        float x0_3 = __uint_as_float(xw[3] << 16);
                        float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                        out_0[6] = __fmaf_rn(nkx_0, x0_3, rq[0] * (g0_3 * __uint_as_float(ww[3] << 16)));
                        out_0[7] = __fmaf_rn(nkx_0, x1_3, rq[0] * (g1_3 * __uint_as_float(ww[3] & 4294901760u)));
                        acc[6] = __fmaf_rn(g0_3 * x0_3, rq[0], acc[6]);
                        acc[7] = __fmaf_rn(g1_3 * x1_3, rq[0], acc[7]);
                        unsigned int packed_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                            packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (dx_base_0 + (unsigned long long)((0 + lane) * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    }
                    float dq_1 = 0.0f;
                    float ga_4 = __uint_as_float(gw[4] << 16);
                    float gb_4 = __uint_as_float(gw[4] & 4294901760u);
                    float xa_4 = __uint_as_float(xw[4] << 16);
                    float xb_4 = __uint_as_float(xw[4] & 4294901760u);
                    dq_1 = __fmaf_rn(ga_4 * __uint_as_float(ww[0] << 16), xa_4, dq_1);
                    dq_1 = __fmaf_rn(gb_4 * __uint_as_float(ww[0] & 4294901760u), xb_4, dq_1);
                    float ga_5 = __uint_as_float(gw[5] << 16);
                    float gb_5 = __uint_as_float(gw[5] & 4294901760u);
                    float xa_5 = __uint_as_float(xw[5] << 16);
                    float xb_5 = __uint_as_float(xw[5] & 4294901760u);
                    dq_1 = __fmaf_rn(ga_5 * __uint_as_float(ww[1] << 16), xa_5, dq_1);
                    dq_1 = __fmaf_rn(gb_5 * __uint_as_float(ww[1] & 4294901760u), xb_5, dq_1);
                    float ga_6 = __uint_as_float(gw[6] << 16);
                    float gb_6 = __uint_as_float(gw[6] & 4294901760u);
                    float xa_6 = __uint_as_float(xw[6] << 16);
                    float xb_6 = __uint_as_float(xw[6] & 4294901760u);
                    dq_1 = __fmaf_rn(ga_6 * __uint_as_float(ww[2] << 16), xa_6, dq_1);
                    dq_1 = __fmaf_rn(gb_6 * __uint_as_float(ww[2] & 4294901760u), xb_6, dq_1);
                    float ga_7 = __uint_as_float(gw[7] << 16);
                    float gb_7 = __uint_as_float(gw[7] & 4294901760u);
                    float xa_7 = __uint_as_float(xw[7] << 16);
                    float xb_7 = __uint_as_float(xw[7] & 4294901760u);
                    dq_1 = __fmaf_rn(ga_7 * __uint_as_float(ww[3] << 16), xa_7, dq_1);
                    dq_1 = __fmaf_rn(gb_7 * __uint_as_float(ww[3] & 4294901760u), xb_7, dq_1);
                    float dsum_1 = dq_1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsum_1 += __shfl_xor_sync(0xFFFFFFFF, dsum_1, offset);
                    float cq_1 = dsum_1 * 0.00390625f;
                    float kx_1 = rq[1] * rq[1] * rq[1] * cq_1;
                    float nkx_1 = -kx_1;
                    int row_1 = tile * 8 + wr + 1;
                    if (row_1 < T) {
                        unsigned long long dx_base_1 = (unsigned long long)row_1 * 256ull;
                        float out_1[8];
                        float g0_4 = __uint_as_float(gw[4] << 16);
                        float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                        float x0_4 = __uint_as_float(xw[4] << 16);
                        float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                        out_1[0] = __fmaf_rn(nkx_1, x0_4, rq[1] * (g0_4 * __uint_as_float(ww[0] << 16)));
                        out_1[1] = __fmaf_rn(nkx_1, x1_4, rq[1] * (g1_4 * __uint_as_float(ww[0] & 4294901760u)));
                        acc[0] = __fmaf_rn(g0_4 * x0_4, rq[1], acc[0]);
                        acc[1] = __fmaf_rn(g1_4 * x1_4, rq[1], acc[1]);
                        float g0_5 = __uint_as_float(gw[5] << 16);
                        float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                        float x0_5 = __uint_as_float(xw[5] << 16);
                        float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                        out_1[2] = __fmaf_rn(nkx_1, x0_5, rq[1] * (g0_5 * __uint_as_float(ww[1] << 16)));
                        out_1[3] = __fmaf_rn(nkx_1, x1_5, rq[1] * (g1_5 * __uint_as_float(ww[1] & 4294901760u)));
                        acc[2] = __fmaf_rn(g0_5 * x0_5, rq[1], acc[2]);
                        acc[3] = __fmaf_rn(g1_5 * x1_5, rq[1], acc[3]);
                        float g0_6 = __uint_as_float(gw[6] << 16);
                        float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                        float x0_6 = __uint_as_float(xw[6] << 16);
                        float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                        out_1[4] = __fmaf_rn(nkx_1, x0_6, rq[1] * (g0_6 * __uint_as_float(ww[2] << 16)));
                        out_1[5] = __fmaf_rn(nkx_1, x1_6, rq[1] * (g1_6 * __uint_as_float(ww[2] & 4294901760u)));
                        acc[4] = __fmaf_rn(g0_6 * x0_6, rq[1], acc[4]);
                        acc[5] = __fmaf_rn(g1_6 * x1_6, rq[1], acc[5]);
                        float g0_7 = __uint_as_float(gw[7] << 16);
                        float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                        float x0_7 = __uint_as_float(xw[7] << 16);
                        float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                        out_1[6] = __fmaf_rn(nkx_1, x0_7, rq[1] * (g0_7 * __uint_as_float(ww[3] << 16)));
                        out_1[7] = __fmaf_rn(nkx_1, x1_7, rq[1] * (g1_7 * __uint_as_float(ww[3] & 4294901760u)));
                        acc[6] = __fmaf_rn(g0_7 * x0_7, rq[1], acc[6]);
                        acc[7] = __fmaf_rn(g1_7 * x1_7, rq[1], acc[7]);
                        unsigned int packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (dx_base_1 + (unsigned long long)((0 + lane) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    }
                    k += 1;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fold_addr + (unsigned int)((warp * 256 + (0 + lane) * 8) * 4)), "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fold_addr + (unsigned int)((warp * 256 + (0 + lane) * 8 + 4) * 4)), "f"(acc[4]), "f"(acc[5]), "f"(acc[6]), "f"(acc[7]) : "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1]))
                    : "r"(fold_addr + (unsigned int)(t * 2 * 4)));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(2) + 1]))
                    : "r"(fold_addr + (unsigned int)((256 + t * 2) * 4)));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1]))
                    : "r"(fold_addr + (unsigned int)((512 + t * 2) * 4)));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[6])), "=r"(*reinterpret_cast<uint32_t*>(&part[(6) + 1]))
                    : "r"(fold_addr + (unsigned int)((768 + t * 2) * 4)));
                part[0] = part[0] + part[2];
                part[1] = part[1] + part[3];
                part[0] = part[0] + part[4];
                part[1] = part[1] + part[5];
                part[0] = part[0] + part[6];
                part[1] = part[1] + part[7];
                {
                    float2 _v2 = make_float2(part[0 + 0], part[0 + 1]);
                    *reinterpret_cast<float2*>(partial + (unsigned long long)slot * 256ull + (unsigned long long)(t * 2)) = _v2;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 4) {
        { // prod_main
            if (elect_sync()) {
                int p_ring = (int)ring_addr;
                int p_tiles = (T + 7) / 8;
                int pk = 0;
                #pragma unroll 1
                for (int p_slot = bid; p_slot < n_chunks; p_slot += num_bids) {
                    #pragma unroll 1
                    for (int p_tile = p_slot; p_tile < p_tiles; p_tile += n_chunks) {
                        int _stage_empty = (int)((unsigned int)pk % 3);
                        int _parity_empty = (int)((unsigned int)pk / 3 % 2) ^ 1;
                        mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 3)) * 8, 8192);
                        int sb = p_ring + pk % 3 * 8192;
                        int _min_0 = ((p_tile * 8) < (T - 1) ? (p_tile * 8) : (T - 1));
                        long long src_0 = (long long)_min_0;
                        cp_async_bulk_gmem2smem(sb, reinterpret_cast<const uint8_t*>(g + src_0 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 4096, reinterpret_cast<const uint8_t*>(x + src_0 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_1_1 = ((p_tile * 8 + 1) < (T - 1) ? (p_tile * 8 + 1) : (T - 1));
                        long long src_1 = (long long)_min_1_1;
                        cp_async_bulk_gmem2smem(sb + 512, reinterpret_cast<const uint8_t*>(g + src_1 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 4608, reinterpret_cast<const uint8_t*>(x + src_1 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_2_1 = ((p_tile * 8 + 2) < (T - 1) ? (p_tile * 8 + 2) : (T - 1));
                        long long src_2 = (long long)_min_2_1;
                        cp_async_bulk_gmem2smem(sb + 1024, reinterpret_cast<const uint8_t*>(g + src_2 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 5120, reinterpret_cast<const uint8_t*>(x + src_2 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_3 = ((p_tile * 8 + 3) < (T - 1) ? (p_tile * 8 + 3) : (T - 1));
                        long long src_3 = (long long)_min_3;
                        cp_async_bulk_gmem2smem(sb + 1536, reinterpret_cast<const uint8_t*>(g + src_3 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 5632, reinterpret_cast<const uint8_t*>(x + src_3 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_4 = ((p_tile * 8 + 4) < (T - 1) ? (p_tile * 8 + 4) : (T - 1));
                        long long src_4 = (long long)_min_4;
                        cp_async_bulk_gmem2smem(sb + 2048, reinterpret_cast<const uint8_t*>(g + src_4 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 6144, reinterpret_cast<const uint8_t*>(x + src_4 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_5 = ((p_tile * 8 + 5) < (T - 1) ? (p_tile * 8 + 5) : (T - 1));
                        long long src_5 = (long long)_min_5;
                        cp_async_bulk_gmem2smem(sb + 2560, reinterpret_cast<const uint8_t*>(g + src_5 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 6656, reinterpret_cast<const uint8_t*>(x + src_5 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_6 = ((p_tile * 8 + 6) < (T - 1) ? (p_tile * 8 + 6) : (T - 1));
                        long long src_6 = (long long)_min_6;
                        cp_async_bulk_gmem2smem(sb + 3072, reinterpret_cast<const uint8_t*>(g + src_6 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 7168, reinterpret_cast<const uint8_t*>(x + src_6 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        int _min_7_1 = ((p_tile * 8 + 7) < (T - 1) ? (p_tile * 8 + 7) : (T - 1));
                        long long src_7 = (long long)_min_7_1;
                        cp_async_bulk_gmem2smem(sb + 3584, reinterpret_cast<const uint8_t*>(g + src_7 * g_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        cp_async_bulk_gmem2smem(sb + 7680, reinterpret_cast<const uint8_t*>(x + src_7 * x_stride_p0), 512, full_addr + ((int)((unsigned int)pk % 3)) * 8);
                        pk += 1;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
