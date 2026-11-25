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

__global__ __launch_bounds__(416, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_f5453c09151f9d441744(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    float* sums = reinterpret_cast<float*>(smem_raw + 1184);
    const int sums_addr = smem + 1184;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1184);
    const int dwbuf_addr = smem + 1184;
    int* meta = reinterpret_cast<int*>(smem_raw + 1152);
    const int meta_addr = smem + 1152;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 25856);
    const int ring_addr = smem + 25856;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=384
            mbarrier_init(smem + 32, 384);
            mbarrier_init(smem + 40, 384);
            mbarrier_init(smem + 48, 384);
            mbarrier_init(smem + 56, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 11) {
        { // compute_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int nxt = 0;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            long long g_stride = g_stride_p0;
            long long x_stride = x_stride_p0;
            long long gh_stride = g_h_stride_p0;
            int rg = warp / 12;
            int tir = tid - rg * 384;
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
                    : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((384 + tir) * 8)))) : "memory");
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
            sums[rg * 6144 + tir] = 0.0f;
            sums[rg * 6144 + 384 + tir] = 0.0f;
            sums[rg * 6144 + 768 + tir] = 0.0f;
            sums[rg * 6144 + 1152 + tir] = 0.0f;
            sums[rg * 6144 + 1536 + tir] = 0.0f;
            sums[rg * 6144 + 1920 + tir] = 0.0f;
            sums[rg * 6144 + 2304 + tir] = 0.0f;
            sums[rg * 6144 + 2688 + tir] = 0.0f;
            sums[rg * 6144 + 3072 + tir] = 0.0f;
            sums[rg * 6144 + 3456 + tir] = 0.0f;
            sums[rg * 6144 + 3840 + tir] = 0.0f;
            sums[rg * 6144 + 4224 + tir] = 0.0f;
            sums[rg * 6144 + 4608 + tir] = 0.0f;
            sums[rg * 6144 + 4992 + tir] = 0.0f;
            sums[rg * 6144 + 5376 + tir] = 0.0f;
            sums[rg * 6144 + 5760 + tir] = 0.0f;
            int ring_at = (int)ring_addr;
            unsigned int gw[8];
            unsigned int xw[8];
            unsigned int hw[8];
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
                int rbase = ring_at + k % 4 * 36864 + rg * 12288;
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
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_3[0]), "=l"(_smem_raw_3[1]) : "r"(rbase + 12288 + tir * 8 * 2));
#else
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(rbase + 12288 + tir * 8 * 2));
#endif
                xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_4[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_4[0]), "=l"(_smem_raw_4[1]) : "r"(rbase + 24576 + tir * 8 * 2));
#else
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(rbase + 24576 + tir * 8 * 2));
#endif
                hw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                hw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                hw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                hw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_5[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_5[0]), "=l"(_smem_raw_5[1]) : "r"(rbase + (384 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_5;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(rbase + (384 + tir) * 8 * 2));
#endif
                gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_6[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_6[0]), "=l"(_smem_raw_6[1]) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
#endif
                xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
#if defined(_MSC_VER)
                uint64_t _smem_raw_7[2];
                asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_7[0]), "=l"(_smem_raw_7[1]) : "r"(rbase + 24576 + (384 + tir) * 8 * 2));
#else
                __int128_t _smem_raw_7;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(rbase + 24576 + (384 + tir) * 8 * 2));
#endif
                hw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                hw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                hw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                hw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
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
                float _warp_reduce_8 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
                dot[0] = _warp_reduce_8;
                if (lane == 0) {
                    dots[(k & 1) * 12 + warp] = dot[0];
                }
                asm volatile("barrier.sync 1, 384;" ::: "memory");
                float tot = 0.0f;
                tot += dots[(k & 1) * 12 + rg * 12];
                tot += dots[(k & 1) * 12 + rg * 12 + 1];
                tot += dots[(k & 1) * 12 + rg * 12 + 2];
                tot += dots[(k & 1) * 12 + rg * 12 + 3];
                tot += dots[(k & 1) * 12 + rg * 12 + 4];
                tot += dots[(k & 1) * 12 + rg * 12 + 5];
                tot += dots[(k & 1) * 12 + rg * 12 + 6];
                tot += dots[(k & 1) * 12 + rg * 12 + 7];
                tot += dots[(k & 1) * 12 + rg * 12 + 8];
                tot += dots[(k & 1) * 12 + rg * 12 + 9];
                tot += dots[(k & 1) * 12 + rg * 12 + 10];
                tot += dots[(k & 1) * 12 + rg * 12 + 11];
                dot[0] = tot;
                float rq = rr[0];
                float cq = __fdiv_rn(dot[0], 6144.0f);
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
                    out_0[0] = y0_0 + __uint_as_float(hw[0] << 16);
                    out_0[1] = y1_0 + __uint_as_float(hw[0] & 4294901760u);
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_1 * x0_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(g1_1 * x1_1, rq, acc[3]);
                    float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * ww[2]));
                    float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * ww[3]));
                    out_0[2] = y0_1 + __uint_as_float(hw[1] << 16);
                    out_0[3] = y1_1 + __uint_as_float(hw[1] & 4294901760u);
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_2 * x0_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(g1_2 * x1_2, rq, acc[5]);
                    float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * ww[4]));
                    float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * ww[5]));
                    out_0[4] = y0_2 + __uint_as_float(hw[2] << 16);
                    out_0[5] = y1_2 + __uint_as_float(hw[2] & 4294901760u);
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_3 * x0_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(g1_3 * x1_3, rq, acc[7]);
                    float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * ww[6]));
                    float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * ww[7]));
                    out_0[6] = y0_3 + __uint_as_float(hw[3] << 16);
                    out_0[7] = y1_3 + __uint_as_float(hw[3] & 4294901760u);
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    acc[8] = __fmaf_rn(g0_4 * x0_4, rq, acc[8]);
                    acc[9] = __fmaf_rn(g1_4 * x1_4, rq, acc[9]);
                    float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * ww[8]));
                    float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * ww[9]));
                    out_1[0] = y0_4 + __uint_as_float(hw[4] << 16);
                    out_1[1] = y1_4 + __uint_as_float(hw[4] & 4294901760u);
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    acc[10] = __fmaf_rn(g0_5 * x0_5, rq, acc[10]);
                    acc[11] = __fmaf_rn(g1_5 * x1_5, rq, acc[11]);
                    float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * ww[10]));
                    float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * ww[11]));
                    out_1[2] = y0_5 + __uint_as_float(hw[5] << 16);
                    out_1[3] = y1_5 + __uint_as_float(hw[5] & 4294901760u);
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    acc[12] = __fmaf_rn(g0_6 * x0_6, rq, acc[12]);
                    acc[13] = __fmaf_rn(g1_6 * x1_6, rq, acc[13]);
                    float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * ww[12]));
                    float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * ww[13]));
                    out_1[4] = y0_6 + __uint_as_float(hw[6] << 16);
                    out_1[5] = y1_6 + __uint_as_float(hw[6] & 4294901760u);
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    acc[14] = __fmaf_rn(g0_7 * x0_7, rq, acc[14]);
                    acc[15] = __fmaf_rn(g1_7 * x1_7, rq, acc[15]);
                    float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * ww[14]));
                    float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * ww[15]));
                    out_1[6] = y0_7 + __uint_as_float(hw[7] << 16);
                    out_1[7] = y1_7 + __uint_as_float(hw[7] & 4294901760u);
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
                if ((k + 1 & 7) == 0) {
                    sums[rg * 6144 + tir] = sums[rg * 6144 + tir] + acc[0];
                    acc[0] = 0.0f;
                    sums[rg * 6144 + 384 + tir] = sums[rg * 6144 + 384 + tir] + acc[1];
                    acc[1] = 0.0f;
                    sums[rg * 6144 + 768 + tir] = sums[rg * 6144 + 768 + tir] + acc[2];
                    acc[2] = 0.0f;
                    sums[rg * 6144 + 1152 + tir] = sums[rg * 6144 + 1152 + tir] + acc[3];
                    acc[3] = 0.0f;
                    sums[rg * 6144 + 1536 + tir] = sums[rg * 6144 + 1536 + tir] + acc[4];
                    acc[4] = 0.0f;
                    sums[rg * 6144 + 1920 + tir] = sums[rg * 6144 + 1920 + tir] + acc[5];
                    acc[5] = 0.0f;
                    sums[rg * 6144 + 2304 + tir] = sums[rg * 6144 + 2304 + tir] + acc[6];
                    acc[6] = 0.0f;
                    sums[rg * 6144 + 2688 + tir] = sums[rg * 6144 + 2688 + tir] + acc[7];
                    acc[7] = 0.0f;
                    sums[rg * 6144 + 3072 + tir] = sums[rg * 6144 + 3072 + tir] + acc[8];
                    acc[8] = 0.0f;
                    sums[rg * 6144 + 3456 + tir] = sums[rg * 6144 + 3456 + tir] + acc[9];
                    acc[9] = 0.0f;
                    sums[rg * 6144 + 3840 + tir] = sums[rg * 6144 + 3840 + tir] + acc[10];
                    acc[10] = 0.0f;
                    sums[rg * 6144 + 4224 + tir] = sums[rg * 6144 + 4224 + tir] + acc[11];
                    acc[11] = 0.0f;
                    sums[rg * 6144 + 4608 + tir] = sums[rg * 6144 + 4608 + tir] + acc[12];
                    acc[12] = 0.0f;
                    sums[rg * 6144 + 4992 + tir] = sums[rg * 6144 + 4992 + tir] + acc[13];
                    acc[13] = 0.0f;
                    sums[rg * 6144 + 5376 + tir] = sums[rg * 6144 + 5376 + tir] + acc[14];
                    acc[14] = 0.0f;
                    sums[rg * 6144 + 5760 + tir] = sums[rg * 6144 + 5760 + tir] + acc[15];
                    acc[15] = 0.0f;
                }
            }
            sums[rg * 6144 + tir] = sums[rg * 6144 + tir] + acc[0];
            acc[0] = 0.0f;
            sums[rg * 6144 + 384 + tir] = sums[rg * 6144 + 384 + tir] + acc[1];
            acc[1] = 0.0f;
            sums[rg * 6144 + 768 + tir] = sums[rg * 6144 + 768 + tir] + acc[2];
            acc[2] = 0.0f;
            sums[rg * 6144 + 1152 + tir] = sums[rg * 6144 + 1152 + tir] + acc[3];
            acc[3] = 0.0f;
            sums[rg * 6144 + 1536 + tir] = sums[rg * 6144 + 1536 + tir] + acc[4];
            acc[4] = 0.0f;
            sums[rg * 6144 + 1920 + tir] = sums[rg * 6144 + 1920 + tir] + acc[5];
            acc[5] = 0.0f;
            sums[rg * 6144 + 2304 + tir] = sums[rg * 6144 + 2304 + tir] + acc[6];
            acc[6] = 0.0f;
            sums[rg * 6144 + 2688 + tir] = sums[rg * 6144 + 2688 + tir] + acc[7];
            acc[7] = 0.0f;
            sums[rg * 6144 + 3072 + tir] = sums[rg * 6144 + 3072 + tir] + acc[8];
            acc[8] = 0.0f;
            sums[rg * 6144 + 3456 + tir] = sums[rg * 6144 + 3456 + tir] + acc[9];
            acc[9] = 0.0f;
            sums[rg * 6144 + 3840 + tir] = sums[rg * 6144 + 3840 + tir] + acc[10];
            acc[10] = 0.0f;
            sums[rg * 6144 + 4224 + tir] = sums[rg * 6144 + 4224 + tir] + acc[11];
            acc[11] = 0.0f;
            sums[rg * 6144 + 4608 + tir] = sums[rg * 6144 + 4608 + tir] + acc[12];
            acc[12] = 0.0f;
            sums[rg * 6144 + 4992 + tir] = sums[rg * 6144 + 4992 + tir] + acc[13];
            acc[13] = 0.0f;
            sums[rg * 6144 + 5376 + tir] = sums[rg * 6144 + 5376 + tir] + acc[14];
            acc[14] = 0.0f;
            sums[rg * 6144 + 5760 + tir] = sums[rg * 6144 + 5760 + tir] + acc[15];
            acc[15] = 0.0f;
            int _min_9 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
            int nrep = ((_min_9) < (4) ? (_min_9) : (4));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (tid == 0) {
                if (first_c < n_chunks) {
                    {
                        void* _cpred_dst_10 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 6144));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_10), "r"(sums_addr + 0), "r"((uint32_t)(6144))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_11 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 6144 + 1536));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_11), "r"(sums_addr + 6144), "r"((uint32_t)(6144))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_12 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 6144 + 3072));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_12), "r"(sums_addr + 12288), "r"((uint32_t)(6144))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_13 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 6144 + 4608));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_13), "r"(sums_addr + 18432), "r"((uint32_t)(6144))
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
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (flags[2] != 0) {
                float buf[32];
                buf[0] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_14;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_14) : "l"((const void*)(partial + (tid))) : "memory");
                        buf[0] = __uint_as_float(_scalar_bits_14);
                    }
                }
                buf[1] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_15;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_15) : "l"((const void*)(partial + (6144 + tid))) : "memory");
                        buf[1] = __uint_as_float(_scalar_bits_15);
                    }
                }
                buf[2] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_16;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_16) : "l"((const void*)(partial + (12288 + tid))) : "memory");
                        buf[2] = __uint_as_float(_scalar_bits_16);
                    }
                }
                buf[3] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_17;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_17) : "l"((const void*)(partial + (18432 + tid))) : "memory");
                        buf[3] = __uint_as_float(_scalar_bits_17);
                    }
                }
                buf[4] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_18;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_18) : "l"((const void*)(partial + (384 + tid))) : "memory");
                        buf[4] = __uint_as_float(_scalar_bits_18);
                    }
                }
                buf[5] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_19;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_19) : "l"((const void*)(partial + (6528 + tid))) : "memory");
                        buf[5] = __uint_as_float(_scalar_bits_19);
                    }
                }
                buf[6] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_20;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_20) : "l"((const void*)(partial + (12672 + tid))) : "memory");
                        buf[6] = __uint_as_float(_scalar_bits_20);
                    }
                }
                buf[7] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_21;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_21) : "l"((const void*)(partial + (18816 + tid))) : "memory");
                        buf[7] = __uint_as_float(_scalar_bits_21);
                    }
                }
                buf[8] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_22;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_22) : "l"((const void*)(partial + (768 + tid))) : "memory");
                        buf[8] = __uint_as_float(_scalar_bits_22);
                    }
                }
                buf[9] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_23;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_23) : "l"((const void*)(partial + (6912 + tid))) : "memory");
                        buf[9] = __uint_as_float(_scalar_bits_23);
                    }
                }
                buf[10] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_24;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_24) : "l"((const void*)(partial + (13056 + tid))) : "memory");
                        buf[10] = __uint_as_float(_scalar_bits_24);
                    }
                }
                buf[11] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_25;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_25) : "l"((const void*)(partial + (19200 + tid))) : "memory");
                        buf[11] = __uint_as_float(_scalar_bits_25);
                    }
                }
                buf[12] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_26;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_26) : "l"((const void*)(partial + (1152 + tid))) : "memory");
                        buf[12] = __uint_as_float(_scalar_bits_26);
                    }
                }
                buf[13] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_27;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_27) : "l"((const void*)(partial + (7296 + tid))) : "memory");
                        buf[13] = __uint_as_float(_scalar_bits_27);
                    }
                }
                buf[14] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_28;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_28) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                        buf[14] = __uint_as_float(_scalar_bits_28);
                    }
                }
                buf[15] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_29;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_29) : "l"((const void*)(partial + (19584 + tid))) : "memory");
                        buf[15] = __uint_as_float(_scalar_bits_29);
                    }
                }
                buf[16] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_30;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_30) : "l"((const void*)(partial + (1536 + tid))) : "memory");
                        buf[16] = __uint_as_float(_scalar_bits_30);
                    }
                }
                buf[17] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_31;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_31) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                        buf[17] = __uint_as_float(_scalar_bits_31);
                    }
                }
                buf[18] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_32;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_32) : "l"((const void*)(partial + (13824 + tid))) : "memory");
                        buf[18] = __uint_as_float(_scalar_bits_32);
                    }
                }
                buf[19] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_33;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_33) : "l"((const void*)(partial + (19968 + tid))) : "memory");
                        buf[19] = __uint_as_float(_scalar_bits_33);
                    }
                }
                buf[20] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_34;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_34) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                        buf[20] = __uint_as_float(_scalar_bits_34);
                    }
                }
                buf[21] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_35;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_35) : "l"((const void*)(partial + (8064 + tid))) : "memory");
                        buf[21] = __uint_as_float(_scalar_bits_35);
                    }
                }
                buf[22] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_36;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_36) : "l"((const void*)(partial + (14208 + tid))) : "memory");
                        buf[22] = __uint_as_float(_scalar_bits_36);
                    }
                }
                buf[23] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_37;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_37) : "l"((const void*)(partial + (20352 + tid))) : "memory");
                        buf[23] = __uint_as_float(_scalar_bits_37);
                    }
                }
                buf[24] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_38;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_38) : "l"((const void*)(partial + (2304 + tid))) : "memory");
                        buf[24] = __uint_as_float(_scalar_bits_38);
                    }
                }
                buf[25] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_39;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_39) : "l"((const void*)(partial + (8448 + tid))) : "memory");
                        buf[25] = __uint_as_float(_scalar_bits_39);
                    }
                }
                buf[26] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_40;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_40) : "l"((const void*)(partial + (14592 + tid))) : "memory");
                        buf[26] = __uint_as_float(_scalar_bits_40);
                    }
                }
                buf[27] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_41;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_41) : "l"((const void*)(partial + (20736 + tid))) : "memory");
                        buf[27] = __uint_as_float(_scalar_bits_41);
                    }
                }
                buf[28] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_42;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_42) : "l"((const void*)(partial + (2688 + tid))) : "memory");
                        buf[28] = __uint_as_float(_scalar_bits_42);
                    }
                }
                buf[29] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_43;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_43) : "l"((const void*)(partial + (8832 + tid))) : "memory");
                        buf[29] = __uint_as_float(_scalar_bits_43);
                    }
                }
                buf[30] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_44;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_44) : "l"((const void*)(partial + (14976 + tid))) : "memory");
                        buf[30] = __uint_as_float(_scalar_bits_44);
                    }
                }
                buf[31] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_45;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_45) : "l"((const void*)(partial + (21120 + tid))) : "memory");
                        buf[31] = __uint_as_float(_scalar_bits_45);
                    }
                }
                buf[0] = buf[0] + buf[1];
                buf[2] = buf[2] + buf[3];
                buf[0] = buf[0] + buf[2];
                int at_w_0 = tid;
                dwbuf[(at_w_0 / 384 / 8 * 384 + at_w_0 % 384) * 8 + at_w_0 / 384 % 8] = buf[0];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_0)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_0)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_0)) = 0.0f;
                }
                buf[4] = buf[4] + buf[5];
                buf[6] = buf[6] + buf[7];
                buf[4] = buf[4] + buf[6];
                int at_w_1 = 384 + tid;
                dwbuf[(at_w_1 / 384 / 8 * 384 + at_w_1 % 384) * 8 + at_w_1 / 384 % 8] = buf[4];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_1)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_1)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_1)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_1)) = 0.0f;
                }
                buf[8] = buf[8] + buf[9];
                buf[10] = buf[10] + buf[11];
                buf[8] = buf[8] + buf[10];
                int at_w_2 = 768 + tid;
                dwbuf[(at_w_2 / 384 / 8 * 384 + at_w_2 % 384) * 8 + at_w_2 / 384 % 8] = buf[8];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_2)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_2)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_2)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_2)) = 0.0f;
                }
                buf[12] = buf[12] + buf[13];
                buf[14] = buf[14] + buf[15];
                buf[12] = buf[12] + buf[14];
                int at_w_3 = 1152 + tid;
                dwbuf[(at_w_3 / 384 / 8 * 384 + at_w_3 % 384) * 8 + at_w_3 / 384 % 8] = buf[12];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_3)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_3)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_3)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_3)) = 0.0f;
                }
                buf[0] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_46;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_46) : "l"((const void*)(partial + (3072 + tid))) : "memory");
                        buf[0] = __uint_as_float(_scalar_bits_46);
                    }
                }
                buf[1] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_47;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_47) : "l"((const void*)(partial + (9216 + tid))) : "memory");
                        buf[1] = __uint_as_float(_scalar_bits_47);
                    }
                }
                buf[2] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_48;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_48) : "l"((const void*)(partial + (15360 + tid))) : "memory");
                        buf[2] = __uint_as_float(_scalar_bits_48);
                    }
                }
                buf[3] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_49;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_49) : "l"((const void*)(partial + (21504 + tid))) : "memory");
                        buf[3] = __uint_as_float(_scalar_bits_49);
                    }
                }
                buf[4] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_50;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_50) : "l"((const void*)(partial + (3456 + tid))) : "memory");
                        buf[4] = __uint_as_float(_scalar_bits_50);
                    }
                }
                buf[5] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_51;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_51) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                        buf[5] = __uint_as_float(_scalar_bits_51);
                    }
                }
                buf[6] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_52;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_52) : "l"((const void*)(partial + (15744 + tid))) : "memory");
                        buf[6] = __uint_as_float(_scalar_bits_52);
                    }
                }
                buf[7] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_53;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_53) : "l"((const void*)(partial + (21888 + tid))) : "memory");
                        buf[7] = __uint_as_float(_scalar_bits_53);
                    }
                }
                buf[8] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_54;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_54) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                        buf[8] = __uint_as_float(_scalar_bits_54);
                    }
                }
                buf[9] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_55;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_55) : "l"((const void*)(partial + (9984 + tid))) : "memory");
                        buf[9] = __uint_as_float(_scalar_bits_55);
                    }
                }
                buf[10] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_56;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_56) : "l"((const void*)(partial + (16128 + tid))) : "memory");
                        buf[10] = __uint_as_float(_scalar_bits_56);
                    }
                }
                buf[11] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_57;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (22272 + tid))) : "memory");
                        buf[11] = __uint_as_float(_scalar_bits_57);
                    }
                }
                buf[12] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_58;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (4224 + tid))) : "memory");
                        buf[12] = __uint_as_float(_scalar_bits_58);
                    }
                }
                buf[13] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_59;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (10368 + tid))) : "memory");
                        buf[13] = __uint_as_float(_scalar_bits_59);
                    }
                }
                buf[14] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_60;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (16512 + tid))) : "memory");
                        buf[14] = __uint_as_float(_scalar_bits_60);
                    }
                }
                buf[15] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_61;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (22656 + tid))) : "memory");
                        buf[15] = __uint_as_float(_scalar_bits_61);
                    }
                }
                buf[16] = buf[16] + buf[17];
                buf[18] = buf[18] + buf[19];
                buf[16] = buf[16] + buf[18];
                int at_w_4 = 1536 + tid;
                dwbuf[(at_w_4 / 384 / 8 * 384 + at_w_4 % 384) * 8 + at_w_4 / 384 % 8] = buf[16];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_4)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_4)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_4)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_4)) = 0.0f;
                }
                buf[20] = buf[20] + buf[21];
                buf[22] = buf[22] + buf[23];
                buf[20] = buf[20] + buf[22];
                int at_w_5 = 1920 + tid;
                dwbuf[(at_w_5 / 384 / 8 * 384 + at_w_5 % 384) * 8 + at_w_5 / 384 % 8] = buf[20];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_5)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_5)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_5)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_5)) = 0.0f;
                }
                buf[24] = buf[24] + buf[25];
                buf[26] = buf[26] + buf[27];
                buf[24] = buf[24] + buf[26];
                int at_w_6 = 2304 + tid;
                dwbuf[(at_w_6 / 384 / 8 * 384 + at_w_6 % 384) * 8 + at_w_6 / 384 % 8] = buf[24];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_6)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_6)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_6)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_6)) = 0.0f;
                }
                buf[28] = buf[28] + buf[29];
                buf[30] = buf[30] + buf[31];
                buf[28] = buf[28] + buf[30];
                int at_w_7 = 2688 + tid;
                dwbuf[(at_w_7 / 384 / 8 * 384 + at_w_7 % 384) * 8 + at_w_7 / 384 % 8] = buf[28];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_7)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_7)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_7)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_7)) = 0.0f;
                }
                buf[16] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_62;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (4608 + tid))) : "memory");
                        buf[16] = __uint_as_float(_scalar_bits_62);
                    }
                }
                buf[17] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_63;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (10752 + tid))) : "memory");
                        buf[17] = __uint_as_float(_scalar_bits_63);
                    }
                }
                buf[18] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_64;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (16896 + tid))) : "memory");
                        buf[18] = __uint_as_float(_scalar_bits_64);
                    }
                }
                buf[19] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_65;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (23040 + tid))) : "memory");
                        buf[19] = __uint_as_float(_scalar_bits_65);
                    }
                }
                buf[20] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_66;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (4992 + tid))) : "memory");
                        buf[20] = __uint_as_float(_scalar_bits_66);
                    }
                }
                buf[21] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_67;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (11136 + tid))) : "memory");
                        buf[21] = __uint_as_float(_scalar_bits_67);
                    }
                }
                buf[22] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_68;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (17280 + tid))) : "memory");
                        buf[22] = __uint_as_float(_scalar_bits_68);
                    }
                }
                buf[23] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_69;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (23424 + tid))) : "memory");
                        buf[23] = __uint_as_float(_scalar_bits_69);
                    }
                }
                buf[24] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_70;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (5376 + tid))) : "memory");
                        buf[24] = __uint_as_float(_scalar_bits_70);
                    }
                }
                buf[25] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_71;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (11520 + tid))) : "memory");
                        buf[25] = __uint_as_float(_scalar_bits_71);
                    }
                }
                buf[26] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_72;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (17664 + tid))) : "memory");
                        buf[26] = __uint_as_float(_scalar_bits_72);
                    }
                }
                buf[27] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_73;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (23808 + tid))) : "memory");
                        buf[27] = __uint_as_float(_scalar_bits_73);
                    }
                }
                buf[28] = 0.0f;
                if (0 < nrep) {
                    {
                        uint32_t _scalar_bits_74;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (5760 + tid))) : "memory");
                        buf[28] = __uint_as_float(_scalar_bits_74);
                    }
                }
                buf[29] = 0.0f;
                if (1 < nrep) {
                    {
                        uint32_t _scalar_bits_75;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (11904 + tid))) : "memory");
                        buf[29] = __uint_as_float(_scalar_bits_75);
                    }
                }
                buf[30] = 0.0f;
                if (2 < nrep) {
                    {
                        uint32_t _scalar_bits_76;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (18048 + tid))) : "memory");
                        buf[30] = __uint_as_float(_scalar_bits_76);
                    }
                }
                buf[31] = 0.0f;
                if (3 < nrep) {
                    {
                        uint32_t _scalar_bits_77;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (24192 + tid))) : "memory");
                        buf[31] = __uint_as_float(_scalar_bits_77);
                    }
                }
                buf[0] = buf[0] + buf[1];
                buf[2] = buf[2] + buf[3];
                buf[0] = buf[0] + buf[2];
                int at_w_8 = 3072 + tid;
                dwbuf[(at_w_8 / 384 / 8 * 384 + at_w_8 % 384) * 8 + at_w_8 / 384 % 8] = buf[0];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_8)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_8)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_8)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_8)) = 0.0f;
                }
                buf[4] = buf[4] + buf[5];
                buf[6] = buf[6] + buf[7];
                buf[4] = buf[4] + buf[6];
                int at_w_9 = 3456 + tid;
                dwbuf[(at_w_9 / 384 / 8 * 384 + at_w_9 % 384) * 8 + at_w_9 / 384 % 8] = buf[4];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_9)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_9)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_9)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_9)) = 0.0f;
                }
                buf[8] = buf[8] + buf[9];
                buf[10] = buf[10] + buf[11];
                buf[8] = buf[8] + buf[10];
                int at_w_10 = 3840 + tid;
                dwbuf[(at_w_10 / 384 / 8 * 384 + at_w_10 % 384) * 8 + at_w_10 / 384 % 8] = buf[8];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_10)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_10)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_10)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_10)) = 0.0f;
                }
                buf[12] = buf[12] + buf[13];
                buf[14] = buf[14] + buf[15];
                buf[12] = buf[12] + buf[14];
                int at_w_11 = 4224 + tid;
                dwbuf[(at_w_11 / 384 / 8 * 384 + at_w_11 % 384) * 8 + at_w_11 / 384 % 8] = buf[12];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_11)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_11)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_11)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_11)) = 0.0f;
                }
                buf[16] = buf[16] + buf[17];
                buf[18] = buf[18] + buf[19];
                buf[16] = buf[16] + buf[18];
                int at_w_12 = 4608 + tid;
                dwbuf[(at_w_12 / 384 / 8 * 384 + at_w_12 % 384) * 8 + at_w_12 / 384 % 8] = buf[16];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_12)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_12)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_12)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_12)) = 0.0f;
                }
                buf[20] = buf[20] + buf[21];
                buf[22] = buf[22] + buf[23];
                buf[20] = buf[20] + buf[22];
                int at_w_13 = 4992 + tid;
                dwbuf[(at_w_13 / 384 / 8 * 384 + at_w_13 % 384) * 8 + at_w_13 / 384 % 8] = buf[20];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_13)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_13)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_13)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_13)) = 0.0f;
                }
                buf[24] = buf[24] + buf[25];
                buf[26] = buf[26] + buf[27];
                buf[24] = buf[24] + buf[26];
                int at_w_14 = 5376 + tid;
                dwbuf[(at_w_14 / 384 / 8 * 384 + at_w_14 % 384) * 8 + at_w_14 / 384 % 8] = buf[24];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_14)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_14)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_14)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_14)) = 0.0f;
                }
                buf[28] = buf[28] + buf[29];
                buf[30] = buf[30] + buf[31];
                buf[28] = buf[28] + buf[30];
                int at_w_15 = 5760 + tid;
                dwbuf[(at_w_15 / 384 / 8 * 384 + at_w_15 % 384) * 8 + at_w_15 / 384 % 8] = buf[28];
                if (0 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (at_w_15)) = 0.0f;
                }
                if (1 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (6144 + at_w_15)) = 0.0f;
                }
                if (2 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (12288 + at_w_15)) = 0.0f;
                }
                if (3 < nrep) {
                    *(reinterpret_cast<float*>(partial) + (18432 + at_w_15)) = 0.0f;
                }
                asm volatile("barrier.sync 1, 384;" ::: "memory");
                int col_0 = tid * 4;
                if (col_0 < 6144) {
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
                int col_1 = (384 + tid) * 4;
                if (col_1 < 6144) {
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
                int col_2 = (768 + tid) * 4;
                if (col_2 < 6144) {
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
                int col_3 = (1152 + tid) * 4;
                if (col_3 < 6144) {
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
    if (warp == 12) {
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
                        int _min_7 = ((p_end - p_row) < (1) ? (p_end - p_row) : (1));
                        nval_1 = _min_7;
                    }
                    if (nval_1 > 0) {
                        meta[2 * slot] = p_row;
                        meta[2 * slot + 1] = nval_1;
                        mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 4)) * 8, 36864);
                        int sbase = p_ring + slot * 36864;
                        int _min_8 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
                        long long row_i_0 = (long long)(p_row + _min_8);
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + row_i_0 * g_h_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
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
