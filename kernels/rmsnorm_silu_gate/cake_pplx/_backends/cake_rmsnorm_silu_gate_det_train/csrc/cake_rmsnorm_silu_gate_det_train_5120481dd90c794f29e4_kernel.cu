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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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

#define LAUNCH_MIN_BLOCKS 6

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int group = tid >> 3;
    int col = (tid & 7) * 16;
    float wv[16];
    int nblocks = (rows + 15) / 16;
    unsigned int xw[8];
    unsigned int zw[8];
    int src = ((bid * 16 + group) < (rows - 1) ? (bid * 16 + group) : (rows - 1));
    uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
    uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
    long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
    long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
    {
        uint32_t _iv_0_0;
        uint32_t _iv_0_1;
        uint32_t _iv_0_2;
        uint32_t _iv_0_3;
        uint32_t _iv_0_4;
        uint32_t _iv_0_5;
        uint32_t _iv_0_6;
        uint32_t _iv_0_7;
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        xw[0 + 0] = (uint32_t)_iv_0_0;
        xw[0 + 1] = (uint32_t)_iv_0_1;
        xw[0 + 2] = (uint32_t)_iv_0_2;
        xw[0 + 3] = (uint32_t)_iv_0_3;
        xw[0 + 4] = (uint32_t)_iv_0_4;
        xw[0 + 5] = (uint32_t)_iv_0_5;
        xw[0 + 6] = (uint32_t)_iv_0_6;
        xw[0 + 7] = (uint32_t)_iv_0_7;
    }
    {
        uint32_t _iv_1_0;
        uint32_t _iv_1_1;
        uint32_t _iv_1_2;
        uint32_t _iv_1_3;
        uint32_t _iv_1_4;
        uint32_t _iv_1_5;
        uint32_t _iv_1_6;
        uint32_t _iv_1_7;
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff) + (0))), "l"(0x12F0000000000000ULL) : "memory");
        zw[0 + 0] = (uint32_t)_iv_1_0;
        zw[0 + 1] = (uint32_t)_iv_1_1;
        zw[0 + 2] = (uint32_t)_iv_1_2;
        zw[0 + 3] = (uint32_t)_iv_1_3;
        zw[0 + 4] = (uint32_t)_iv_1_4;
        zw[0 + 5] = (uint32_t)_iv_1_5;
        zw[0 + 6] = (uint32_t)_iv_1_6;
        zw[0 + 7] = (uint32_t)_iv_1_7;
    }
    {
        const void* _v8p_2 = (const void*)(w + col + (0));
        uint32_t _v8_2_0[8];
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const void*)((const char*)_v8p_2 + 0)), "l"(0x14F0000000000000ULL) : "memory");
        (&wv[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
        (&wv[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
        (&wv[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
        (&wv[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
        (&wv[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
        (&wv[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
        (&wv[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
        (&wv[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
        (&wv[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
        (&wv[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
        (&wv[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
        (&wv[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
        (&wv[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
        (&wv[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
        (&wv[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
        (&wv[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
    }
    int x_mode = 0;
    int z_mode = 0;
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    if (x_head == 128ll) {
        x_mode = 1;
        if (x_tok == heads64 * 128ll) {
            x_mode = 2;
        }
    }
    if (z_head == 128ll) {
        z_mode = 1;
        if (z_tok == heads64 * 128ll) {
            z_mode = 2;
        }
    }
    if (x_mode != 2 || z_mode != 2) {
        int oblk = nblocks;
        uint32_t _read_special_reg_3;
        asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_3));
        oblk = bid + 6 * (int)_read_special_reg_3;
        if (bid + num_bids < nblocks || oblk < num_bids && oblk < nblocks) {
            if (tid < 32) {
                int pblk = bid + num_bids;
                if (pblk < nblocks) {
                    int pfirst = pblk * 16;
                    int pn = ((rows - pfirst) < (16) ? (rows - pfirst) : (16));
                    if (x_mode == 2) {
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst * 128ll), "r"((uint32_t)(pn * 256)) : "memory");
                        }
                    }
                    if (x_mode == 1) {
                        if (lane == 0) {
                            int h = (int)((uint32_t)(heads.divisor));
                            uint32_t q0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)pfirst), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)pfirst);
                            uint32_t _h0 = (uint32_t)((unsigned int)pfirst) - q0 * (uint32_t)(heads.divisor);
                            uint32_t q1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(pfirst + pn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(pfirst + pn - 1));
                            uint32_t _h1 = (uint32_t)((unsigned int)(pfirst + pn - 1)) - q1 * (uint32_t)(heads.divisor);
                            #pragma unroll 1
                            for (int q = (int)q0; q < (int)q1 + 1; q++) {
                                int lo = ((pfirst) > (q * h) ? (pfirst) : (q * h));
                                int hi = ((pfirst + pn) < ((q + 1) * h) ? (pfirst + pn) : ((q + 1) * h));
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)q * x_tok + (long long)(lo - q * h) * 128ll)), "r"((uint32_t)((hi - lo) * 256)) : "memory");
                            }
                        }
                    }
                    if (x_mode == 0) {
                        if (lane < pn) {
                            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(pfirst + lane)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(pfirst + lane));
                            uint32_t th_0 = (uint32_t)((unsigned int)(pfirst + lane)) - tq_0 * (uint32_t)(heads.divisor);
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)tq_0 * x_tok + (long long)th_0 * x_head)), "r"((uint32_t)(256)) : "memory");
                        }
                    }
                    if (z_mode == 2) {
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst * 128ll), "r"((uint32_t)(pn * 256)) : "memory");
                        }
                    }
                    if (z_mode == 1) {
                        if (lane == 0) {
                            int h_1 = (int)((uint32_t)(heads.divisor));
                            uint32_t q0_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)pfirst), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)pfirst);
                            uint32_t _h0_1 = (uint32_t)((unsigned int)pfirst) - q0_1 * (uint32_t)(heads.divisor);
                            uint32_t q1_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(pfirst + pn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(pfirst + pn - 1));
                            uint32_t _h1_1 = (uint32_t)((unsigned int)(pfirst + pn - 1)) - q1_1 * (uint32_t)(heads.divisor);
                            #pragma unroll 1
                            for (int q_1 = (int)q0_1; q_1 < (int)q1_1 + 1; q_1++) {
                                int lo_1 = ((pfirst) > (q_1 * h_1) ? (pfirst) : (q_1 * h_1));
                                int hi_1 = ((pfirst + pn) < ((q_1 + 1) * h_1) ? (pfirst + pn) : ((q_1 + 1) * h_1));
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)q_1 * z_tok + (long long)(lo_1 - q_1 * h_1) * 128ll)), "r"((uint32_t)((hi_1 - lo_1) * 256)) : "memory");
                            }
                        }
                    }
                    if (z_mode == 0) {
                        if (lane < pn) {
                            uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(pfirst + lane)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(pfirst + lane));
                            uint32_t th_1 = (uint32_t)((unsigned int)(pfirst + lane)) - tq_1 * (uint32_t)(heads.divisor);
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)tq_1 * z_tok + (long long)th_1 * z_head)), "r"((uint32_t)(256)) : "memory");
                        }
                    }
                }
                uint32_t _read_special_reg_4;
                asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_4));
                int wblk = bid + 6 * (int)_read_special_reg_4;
                if (wblk < num_bids) {
                    if (wblk < nblocks) {
                        int wfirst = wblk * 16;
                        int wn = ((rows - wfirst) < (16) ? (rows - wfirst) : (16));
                        if (x_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst * 128ll), "r"((uint32_t)(wn * 256)) : "memory");
                            }
                        }
                        if (x_mode == 1) {
                            if (lane == 0) {
                                int h_2 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_2 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)wfirst), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)wfirst);
                                uint32_t _h0_2 = (uint32_t)((unsigned int)wfirst) - q0_2 * (uint32_t)(heads.divisor);
                                uint32_t q1_2 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(wfirst + wn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(wfirst + wn - 1));
                                uint32_t _h1_2 = (uint32_t)((unsigned int)(wfirst + wn - 1)) - q1_2 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_2 = (int)q0_2; q_2 < (int)q1_2 + 1; q_2++) {
                                    int lo_2 = ((wfirst) > (q_2 * h_2) ? (wfirst) : (q_2 * h_2));
                                    int hi_2 = ((wfirst + wn) < ((q_2 + 1) * h_2) ? (wfirst + wn) : ((q_2 + 1) * h_2));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)q_2 * x_tok + (long long)(lo_2 - q_2 * h_2) * 128ll)), "r"((uint32_t)((hi_2 - lo_2) * 256)) : "memory");
                                }
                            }
                        }
                        if (x_mode == 0) {
                            if (lane < wn) {
                                uint32_t tq_2 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(wfirst + lane)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(wfirst + lane));
                                uint32_t th_2 = (uint32_t)((unsigned int)(wfirst + lane)) - tq_2 * (uint32_t)(heads.divisor);
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)tq_2 * x_tok + (long long)th_2 * x_head)), "r"((uint32_t)(256)) : "memory");
                            }
                        }
                        if (z_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst * 128ll), "r"((uint32_t)(wn * 256)) : "memory");
                            }
                        }
                        if (z_mode == 1) {
                            if (lane == 0) {
                                int h_3 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_3 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)wfirst), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)wfirst);
                                uint32_t _h0_3 = (uint32_t)((unsigned int)wfirst) - q0_3 * (uint32_t)(heads.divisor);
                                uint32_t q1_3 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(wfirst + wn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(wfirst + wn - 1));
                                uint32_t _h1_3 = (uint32_t)((unsigned int)(wfirst + wn - 1)) - q1_3 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_3 = (int)q0_3; q_3 < (int)q1_3 + 1; q_3++) {
                                    int lo_3 = ((wfirst) > (q_3 * h_3) ? (wfirst) : (q_3 * h_3));
                                    int hi_3 = ((wfirst + wn) < ((q_3 + 1) * h_3) ? (wfirst + wn) : ((q_3 + 1) * h_3));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)q_3 * z_tok + (long long)(lo_3 - q_3 * h_3) * 128ll)), "r"((uint32_t)((hi_3 - lo_3) * 256)) : "memory");
                                }
                            }
                        }
                        if (z_mode == 0) {
                            if (lane < wn) {
                                uint32_t tq_3 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(wfirst + lane)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(wfirst + lane));
                                uint32_t th_3 = (uint32_t)((unsigned int)(wfirst + lane)) - tq_3 * (uint32_t)(heads.divisor);
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)tq_3 * z_tok + (long long)th_3 * z_head)), "r"((uint32_t)(256)) : "memory");
                            }
                        }
                    }
                }
            }
            #pragma unroll 1
            for (int blk = bid; blk < nblocks; blk += num_bids) {
                float xs[16];
                float zs[16];
                xs[0] = __uint_as_float(xw[0] << 16);
                xs[1] = __uint_as_float(xw[0] & 4294901760u);
                zs[0] = __uint_as_float(zw[0] << 16);
                zs[1] = __uint_as_float(zw[0] & 4294901760u);
                xs[2] = __uint_as_float(xw[1] << 16);
                xs[3] = __uint_as_float(xw[1] & 4294901760u);
                zs[2] = __uint_as_float(zw[1] << 16);
                zs[3] = __uint_as_float(zw[1] & 4294901760u);
                xs[4] = __uint_as_float(xw[2] << 16);
                xs[5] = __uint_as_float(xw[2] & 4294901760u);
                zs[4] = __uint_as_float(zw[2] << 16);
                zs[5] = __uint_as_float(zw[2] & 4294901760u);
                xs[6] = __uint_as_float(xw[3] << 16);
                xs[7] = __uint_as_float(xw[3] & 4294901760u);
                zs[6] = __uint_as_float(zw[3] << 16);
                zs[7] = __uint_as_float(zw[3] & 4294901760u);
                xs[8] = __uint_as_float(xw[4] << 16);
                xs[9] = __uint_as_float(xw[4] & 4294901760u);
                zs[8] = __uint_as_float(zw[4] << 16);
                zs[9] = __uint_as_float(zw[4] & 4294901760u);
                xs[10] = __uint_as_float(xw[5] << 16);
                xs[11] = __uint_as_float(xw[5] & 4294901760u);
                zs[10] = __uint_as_float(zw[5] << 16);
                zs[11] = __uint_as_float(zw[5] & 4294901760u);
                xs[12] = __uint_as_float(xw[6] << 16);
                xs[13] = __uint_as_float(xw[6] & 4294901760u);
                zs[12] = __uint_as_float(zw[6] << 16);
                zs[13] = __uint_as_float(zw[6] & 4294901760u);
                xs[14] = __uint_as_float(xw[7] << 16);
                xs[15] = __uint_as_float(xw[7] & 4294901760u);
                zs[14] = __uint_as_float(zw[7] << 16);
                zs[15] = __uint_as_float(zw[7] & 4294901760u);
                if (blk + num_bids < nblocks) {
                    int src_0 = (((blk + num_bids) * 16 + group) < (rows - 1) ? ((blk + num_bids) * 16 + group) : (rows - 1));
                    uint32_t tq_4 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
                    uint32_t th_4 = (uint32_t)((unsigned int)src_0) - tq_4 * (uint32_t)(heads.divisor);
                    long long xoff_0 = (long long)tq_4 * x_tok + (long long)th_4 * x_head + (long long)col;
                    long long zoff_0 = (long long)tq_4 * z_tok + (long long)th_4 * z_head + (long long)col;
                    {
                        uint32_t _iv_5_0;
                        uint32_t _iv_5_1;
                        uint32_t _iv_5_2;
                        uint32_t _iv_5_3;
                        uint32_t _iv_5_4;
                        uint32_t _iv_5_5;
                        uint32_t _iv_5_6;
                        uint32_t _iv_5_7;
                        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                        xw[0 + 0] = (uint32_t)_iv_5_0;
                        xw[0 + 1] = (uint32_t)_iv_5_1;
                        xw[0 + 2] = (uint32_t)_iv_5_2;
                        xw[0 + 3] = (uint32_t)_iv_5_3;
                        xw[0 + 4] = (uint32_t)_iv_5_4;
                        xw[0 + 5] = (uint32_t)_iv_5_5;
                        xw[0 + 6] = (uint32_t)_iv_5_6;
                        xw[0 + 7] = (uint32_t)_iv_5_7;
                    }
                    {
                        uint32_t _iv_6_0;
                        uint32_t _iv_6_1;
                        uint32_t _iv_6_2;
                        uint32_t _iv_6_3;
                        uint32_t _iv_6_4;
                        uint32_t _iv_6_5;
                        uint32_t _iv_6_6;
                        uint32_t _iv_6_7;
                        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                        zw[0 + 0] = (uint32_t)_iv_6_0;
                        zw[0 + 1] = (uint32_t)_iv_6_1;
                        zw[0 + 2] = (uint32_t)_iv_6_2;
                        zw[0 + 3] = (uint32_t)_iv_6_3;
                        zw[0 + 4] = (uint32_t)_iv_6_4;
                        zw[0 + 5] = (uint32_t)_iv_6_5;
                        zw[0 + 6] = (uint32_t)_iv_6_6;
                        zw[0 + 7] = (uint32_t)_iv_6_7;
                    }
                }
                if (tid < 32) {
                    int nblk = blk + 2 * num_bids;
                    if (nblk < nblocks) {
                        int nfirst = nblk * 16;
                        int nn = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                        if (x_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nn * 256)) : "memory");
                            }
                        }
                        if (x_mode == 1) {
                            if (lane == 0) {
                                int h_4 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_4 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nfirst), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nfirst);
                                uint32_t _h0_4 = (uint32_t)((unsigned int)nfirst) - q0_4 * (uint32_t)(heads.divisor);
                                uint32_t q1_4 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(nfirst + nn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(nfirst + nn - 1));
                                uint32_t _h1_4 = (uint32_t)((unsigned int)(nfirst + nn - 1)) - q1_4 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_4 = (int)q0_4; q_4 < (int)q1_4 + 1; q_4++) {
                                    int lo_4 = ((nfirst) > (q_4 * h_4) ? (nfirst) : (q_4 * h_4));
                                    int hi_4 = ((nfirst + nn) < ((q_4 + 1) * h_4) ? (nfirst + nn) : ((q_4 + 1) * h_4));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)q_4 * x_tok + (long long)(lo_4 - q_4 * h_4) * 128ll)), "r"((uint32_t)((hi_4 - lo_4) * 256)) : "memory");
                                }
                            }
                        }
                        if (x_mode == 0) {
                            if (lane < nn) {
                                uint32_t tq_5 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(nfirst + lane)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(nfirst + lane));
                                uint32_t th_5 = (uint32_t)((unsigned int)(nfirst + lane)) - tq_5 * (uint32_t)(heads.divisor);
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + ((long long)tq_5 * x_tok + (long long)th_5 * x_head)), "r"((uint32_t)(256)) : "memory");
                            }
                        }
                        if (z_mode == 2) {
                            if (lane == 0) {
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nn * 256)) : "memory");
                            }
                        }
                        if (z_mode == 1) {
                            if (lane == 0) {
                                int h_5 = (int)((uint32_t)(heads.divisor));
                                uint32_t q0_5 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nfirst), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nfirst);
                                uint32_t _h0_5 = (uint32_t)((unsigned int)nfirst) - q0_5 * (uint32_t)(heads.divisor);
                                uint32_t q1_5 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(nfirst + nn - 1)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(nfirst + nn - 1));
                                uint32_t _h1_5 = (uint32_t)((unsigned int)(nfirst + nn - 1)) - q1_5 * (uint32_t)(heads.divisor);
                                #pragma unroll 1
                                for (int q_5 = (int)q0_5; q_5 < (int)q1_5 + 1; q_5++) {
                                    int lo_5 = ((nfirst) > (q_5 * h_5) ? (nfirst) : (q_5 * h_5));
                                    int hi_5 = ((nfirst + nn) < ((q_5 + 1) * h_5) ? (nfirst + nn) : ((q_5 + 1) * h_5));
                                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)q_5 * z_tok + (long long)(lo_5 - q_5 * h_5) * 128ll)), "r"((uint32_t)((hi_5 - lo_5) * 256)) : "memory");
                                }
                            }
                        }
                        if (z_mode == 0) {
                            if (lane < nn) {
                                uint32_t tq_6 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(nfirst + lane)), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)(nfirst + lane));
                                uint32_t th_6 = (uint32_t)((unsigned int)(nfirst + lane)) - tq_6 * (uint32_t)(heads.divisor);
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + ((long long)tq_6 * z_tok + (long long)th_6 * z_head)), "r"((uint32_t)(256)) : "memory");
                            }
                        }
                    }
                }
                float gs[16];
                float _exp2_8 = approx_exp2(zs[0] * -1.442695f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_8);
                gs[0] = zs[0] * _rcp_7;
                float _exp2_10 = approx_exp2(zs[1] * -1.442695f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_10);
                gs[1] = zs[1] * _rcp_9;
                float _exp2_12 = approx_exp2(zs[2] * -1.442695f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_12);
                gs[2] = zs[2] * _rcp_11;
                float _exp2_14 = approx_exp2(zs[3] * -1.442695f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_14);
                gs[3] = zs[3] * _rcp_13;
                float _exp2_16 = approx_exp2(zs[4] * -1.442695f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_16);
                gs[4] = zs[4] * _rcp_15;
                float _exp2_18 = approx_exp2(zs[5] * -1.442695f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_18);
                gs[5] = zs[5] * _rcp_17;
                float _exp2_20 = approx_exp2(zs[6] * -1.442695f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_20);
                gs[6] = zs[6] * _rcp_19;
                float _exp2_22 = approx_exp2(zs[7] * -1.442695f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_22);
                gs[7] = zs[7] * _rcp_21;
                float _exp2_24 = approx_exp2(zs[8] * -1.442695f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_24);
                gs[8] = zs[8] * _rcp_23;
                float _exp2_26 = approx_exp2(zs[9] * -1.442695f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_26);
                gs[9] = zs[9] * _rcp_25;
                float _exp2_28 = approx_exp2(zs[10] * -1.442695f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_28);
                gs[10] = zs[10] * _rcp_27;
                float _exp2_30 = approx_exp2(zs[11] * -1.442695f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_30);
                gs[11] = zs[11] * _rcp_29;
                float _exp2_32 = approx_exp2(zs[12] * -1.442695f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_32);
                gs[12] = zs[12] * _rcp_31;
                float _exp2_34 = approx_exp2(zs[13] * -1.442695f);
                float _rcp_33 = approx_rcp(1.0f + _exp2_34);
                gs[13] = zs[13] * _rcp_33;
                float _exp2_36 = approx_exp2(zs[14] * -1.442695f);
                float _rcp_35 = approx_rcp(1.0f + _exp2_36);
                gs[14] = zs[14] * _rcp_35;
                float _exp2_38 = approx_exp2(zs[15] * -1.442695f);
                float _rcp_37 = approx_rcp(1.0f + _exp2_38);
                gs[15] = zs[15] * _rcp_37;
                float total = 0.0f;
                float2 lo_6 = make_float2(0.0f, 0.0f);
                float2 hi_6 = make_float2(0.0f, 0.0f);
                float2 xlo_0 = make_float2(xs[0], xs[1]);
                float2 xhi_0 = make_float2(xs[2], xs[3]);
                lo_6 = fma_f32x2_rn_noftz(xlo_0, xlo_0, lo_6);
                hi_6 = fma_f32x2_rn_noftz(xhi_0, xhi_0, hi_6);
                float2 xlo_1 = make_float2(xs[4], xs[5]);
                float2 xhi_1 = make_float2(xs[6], xs[7]);
                lo_6 = fma_f32x2_rn_noftz(xlo_1, xlo_1, lo_6);
                hi_6 = fma_f32x2_rn_noftz(xhi_1, xhi_1, hi_6);
                float2 xlo_2 = make_float2(xs[8], xs[9]);
                float2 xhi_2 = make_float2(xs[10], xs[11]);
                lo_6 = fma_f32x2_rn_noftz(xlo_2, xlo_2, lo_6);
                hi_6 = fma_f32x2_rn_noftz(xhi_2, xhi_2, hi_6);
                float2 xlo_3 = make_float2(xs[12], xs[13]);
                float2 xhi_3 = make_float2(xs[14], xs[15]);
                lo_6 = fma_f32x2_rn_noftz(xlo_3, xlo_3, lo_6);
                hi_6 = fma_f32x2_rn_noftz(xhi_3, xhi_3, hi_6);
                total = lo_6.x + lo_6.y + (hi_6.x + hi_6.y);
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
                total = total + _shfl_xor_sync_39;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
                total = total + _shfl_xor_sync_40;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
                total = total + _shfl_xor_sync_41;
                float rstd;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
                float2 rstd2 = make_float2(rstd, rstd);
                float2 _f32x2_mul_a_42 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
                float2 _f32x2_mul_b_43 = make_float2(gs[0], gs[1]);
                float2 y2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                xs[0] = y2_0.x;
                xs[1] = y2_0.y;
                float2 _f32x2_mul_a_44 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
                float2 _f32x2_mul_b_45 = make_float2(gs[2], gs[3]);
                float2 y2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
                xs[2] = y2_1.x;
                xs[3] = y2_1.y;
                float2 _f32x2_mul_a_46 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
                float2 _f32x2_mul_b_47 = make_float2(gs[4], gs[5]);
                float2 y2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                xs[4] = y2_2.x;
                xs[5] = y2_2.y;
                float2 _f32x2_mul_a_48 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
                float2 _f32x2_mul_b_49 = make_float2(gs[6], gs[7]);
                float2 y2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                xs[6] = y2_3.x;
                xs[7] = y2_3.y;
                float2 _f32x2_mul_a_50 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
                float2 _f32x2_mul_b_51 = make_float2(gs[8], gs[9]);
                float2 y2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
                xs[8] = y2_4.x;
                xs[9] = y2_4.y;
                float2 _f32x2_mul_a_52 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
                float2 _f32x2_mul_b_53 = make_float2(gs[10], gs[11]);
                float2 y2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
                xs[10] = y2_5.x;
                xs[11] = y2_5.y;
                float2 _f32x2_mul_a_54 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
                float2 _f32x2_mul_b_55 = make_float2(gs[12], gs[13]);
                float2 y2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
                xs[12] = y2_6.x;
                xs[13] = y2_6.y;
                float2 _f32x2_mul_a_56 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
                float2 _f32x2_mul_b_57 = make_float2(gs[14], gs[15]);
                float2 y2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
                xs[14] = y2_7.x;
                xs[15] = y2_7.y;
                unsigned int packed[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                    packed[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                    packed[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                    packed[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 8], xs[_lp*2+1 + 8]));
                    packed[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 10], xs[_lp*2+1 + 10]));
                    packed[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 12], xs[_lp*2+1 + 12]));
                    packed[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 14], xs[_lp*2+1 + 14]));
                    packed[_lp + 7] = *(uint32_t*)&_bf2;
                }
                if (blk * 16 + group < rows) {
                    reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
                    reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed + (4))[0];
                }
            }
        } else if (bid < nblocks) {
            float oxs[16];
            float ozs[16];
            oxs[0] = __uint_as_float(xw[0] << 16);
            oxs[1] = __uint_as_float(xw[0] & 4294901760u);
            ozs[0] = __uint_as_float(zw[0] << 16);
            ozs[1] = __uint_as_float(zw[0] & 4294901760u);
            oxs[2] = __uint_as_float(xw[1] << 16);
            oxs[3] = __uint_as_float(xw[1] & 4294901760u);
            ozs[2] = __uint_as_float(zw[1] << 16);
            ozs[3] = __uint_as_float(zw[1] & 4294901760u);
            oxs[4] = __uint_as_float(xw[2] << 16);
            oxs[5] = __uint_as_float(xw[2] & 4294901760u);
            ozs[4] = __uint_as_float(zw[2] << 16);
            ozs[5] = __uint_as_float(zw[2] & 4294901760u);
            oxs[6] = __uint_as_float(xw[3] << 16);
            oxs[7] = __uint_as_float(xw[3] & 4294901760u);
            ozs[6] = __uint_as_float(zw[3] << 16);
            ozs[7] = __uint_as_float(zw[3] & 4294901760u);
            oxs[8] = __uint_as_float(xw[4] << 16);
            oxs[9] = __uint_as_float(xw[4] & 4294901760u);
            ozs[8] = __uint_as_float(zw[4] << 16);
            ozs[9] = __uint_as_float(zw[4] & 4294901760u);
            oxs[10] = __uint_as_float(xw[5] << 16);
            oxs[11] = __uint_as_float(xw[5] & 4294901760u);
            ozs[10] = __uint_as_float(zw[5] << 16);
            ozs[11] = __uint_as_float(zw[5] & 4294901760u);
            oxs[12] = __uint_as_float(xw[6] << 16);
            oxs[13] = __uint_as_float(xw[6] & 4294901760u);
            ozs[12] = __uint_as_float(zw[6] << 16);
            ozs[13] = __uint_as_float(zw[6] & 4294901760u);
            oxs[14] = __uint_as_float(xw[7] << 16);
            oxs[15] = __uint_as_float(xw[7] & 4294901760u);
            ozs[14] = __uint_as_float(zw[7] << 16);
            ozs[15] = __uint_as_float(zw[7] & 4294901760u);
            float gs_1[16];
            float _exp2_59 = approx_exp2(ozs[0] * -1.442695f);
            float _rcp_58 = approx_rcp(1.0f + _exp2_59);
            gs_1[0] = ozs[0] * _rcp_58;
            float _exp2_61 = approx_exp2(ozs[1] * -1.442695f);
            float _rcp_60 = approx_rcp(1.0f + _exp2_61);
            gs_1[1] = ozs[1] * _rcp_60;
            float _exp2_63 = approx_exp2(ozs[2] * -1.442695f);
            float _rcp_62 = approx_rcp(1.0f + _exp2_63);
            gs_1[2] = ozs[2] * _rcp_62;
            float _exp2_65 = approx_exp2(ozs[3] * -1.442695f);
            float _rcp_64 = approx_rcp(1.0f + _exp2_65);
            gs_1[3] = ozs[3] * _rcp_64;
            float _exp2_67 = approx_exp2(ozs[4] * -1.442695f);
            float _rcp_66 = approx_rcp(1.0f + _exp2_67);
            gs_1[4] = ozs[4] * _rcp_66;
            float _exp2_69 = approx_exp2(ozs[5] * -1.442695f);
            float _rcp_68 = approx_rcp(1.0f + _exp2_69);
            gs_1[5] = ozs[5] * _rcp_68;
            float _exp2_71 = approx_exp2(ozs[6] * -1.442695f);
            float _rcp_70 = approx_rcp(1.0f + _exp2_71);
            gs_1[6] = ozs[6] * _rcp_70;
            float _exp2_73 = approx_exp2(ozs[7] * -1.442695f);
            float _rcp_72 = approx_rcp(1.0f + _exp2_73);
            gs_1[7] = ozs[7] * _rcp_72;
            float _exp2_75 = approx_exp2(ozs[8] * -1.442695f);
            float _rcp_74 = approx_rcp(1.0f + _exp2_75);
            gs_1[8] = ozs[8] * _rcp_74;
            float _exp2_77 = approx_exp2(ozs[9] * -1.442695f);
            float _rcp_76 = approx_rcp(1.0f + _exp2_77);
            gs_1[9] = ozs[9] * _rcp_76;
            float _exp2_79 = approx_exp2(ozs[10] * -1.442695f);
            float _rcp_78 = approx_rcp(1.0f + _exp2_79);
            gs_1[10] = ozs[10] * _rcp_78;
            float _exp2_81 = approx_exp2(ozs[11] * -1.442695f);
            float _rcp_80 = approx_rcp(1.0f + _exp2_81);
            gs_1[11] = ozs[11] * _rcp_80;
            float _exp2_83 = approx_exp2(ozs[12] * -1.442695f);
            float _rcp_82 = approx_rcp(1.0f + _exp2_83);
            gs_1[12] = ozs[12] * _rcp_82;
            float _exp2_85 = approx_exp2(ozs[13] * -1.442695f);
            float _rcp_84 = approx_rcp(1.0f + _exp2_85);
            gs_1[13] = ozs[13] * _rcp_84;
            float _exp2_87 = approx_exp2(ozs[14] * -1.442695f);
            float _rcp_86 = approx_rcp(1.0f + _exp2_87);
            gs_1[14] = ozs[14] * _rcp_86;
            float _exp2_89 = approx_exp2(ozs[15] * -1.442695f);
            float _rcp_88 = approx_rcp(1.0f + _exp2_89);
            gs_1[15] = ozs[15] * _rcp_88;
            float total_1 = 0.0f;
            float2 lo_7 = make_float2(0.0f, 0.0f);
            float2 hi_7 = make_float2(0.0f, 0.0f);
            float2 xlo_4 = make_float2(oxs[0], oxs[1]);
            float2 xhi_4 = make_float2(oxs[2], oxs[3]);
            lo_7 = fma_f32x2_rn_noftz(xlo_4, xlo_4, lo_7);
            hi_7 = fma_f32x2_rn_noftz(xhi_4, xhi_4, hi_7);
            float2 xlo_5 = make_float2(oxs[4], oxs[5]);
            float2 xhi_5 = make_float2(oxs[6], oxs[7]);
            lo_7 = fma_f32x2_rn_noftz(xlo_5, xlo_5, lo_7);
            hi_7 = fma_f32x2_rn_noftz(xhi_5, xhi_5, hi_7);
            float2 xlo_6 = make_float2(oxs[8], oxs[9]);
            float2 xhi_6 = make_float2(oxs[10], oxs[11]);
            lo_7 = fma_f32x2_rn_noftz(xlo_6, xlo_6, lo_7);
            hi_7 = fma_f32x2_rn_noftz(xhi_6, xhi_6, hi_7);
            float2 xlo_7 = make_float2(oxs[12], oxs[13]);
            float2 xhi_7 = make_float2(oxs[14], oxs[15]);
            lo_7 = fma_f32x2_rn_noftz(xlo_7, xlo_7, lo_7);
            hi_7 = fma_f32x2_rn_noftz(xhi_7, xhi_7, hi_7);
            total_1 = lo_7.x + lo_7.y + (hi_7.x + hi_7.y);
            float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
            total_1 = total_1 + _shfl_xor_sync_90;
            float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
            total_1 = total_1 + _shfl_xor_sync_91;
            float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
            total_1 = total_1 + _shfl_xor_sync_92;
            float rstd_1;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
            float2 rstd2_1 = make_float2(rstd_1, rstd_1);
            float2 _f32x2_mul_a_93 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[0], oxs[1]), rstd2_1), make_float2(wv[0], wv[1]));
            float2 _f32x2_mul_b_94 = make_float2(gs_1[0], gs_1[1]);
            float2 y2_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
            oxs[0] = y2_8.x;
            oxs[1] = y2_8.y;
            float2 _f32x2_mul_a_95 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[2], oxs[3]), rstd2_1), make_float2(wv[2], wv[3]));
            float2 _f32x2_mul_b_96 = make_float2(gs_1[2], gs_1[3]);
            float2 y2_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_96));
            oxs[2] = y2_9.x;
            oxs[3] = y2_9.y;
            float2 _f32x2_mul_a_97 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[4], oxs[5]), rstd2_1), make_float2(wv[4], wv[5]));
            float2 _f32x2_mul_b_98 = make_float2(gs_1[4], gs_1[5]);
            float2 y2_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
            oxs[4] = y2_10.x;
            oxs[5] = y2_10.y;
            float2 _f32x2_mul_a_99 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[6], oxs[7]), rstd2_1), make_float2(wv[6], wv[7]));
            float2 _f32x2_mul_b_100 = make_float2(gs_1[6], gs_1[7]);
            float2 y2_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_100));
            oxs[6] = y2_11.x;
            oxs[7] = y2_11.y;
            float2 _f32x2_mul_a_101 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[8], oxs[9]), rstd2_1), make_float2(wv[8], wv[9]));
            float2 _f32x2_mul_b_102 = make_float2(gs_1[8], gs_1[9]);
            float2 y2_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
            oxs[8] = y2_12.x;
            oxs[9] = y2_12.y;
            float2 _f32x2_mul_a_103 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[10], oxs[11]), rstd2_1), make_float2(wv[10], wv[11]));
            float2 _f32x2_mul_b_104 = make_float2(gs_1[10], gs_1[11]);
            float2 y2_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_104));
            oxs[10] = y2_13.x;
            oxs[11] = y2_13.y;
            float2 _f32x2_mul_a_105 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[12], oxs[13]), rstd2_1), make_float2(wv[12], wv[13]));
            float2 _f32x2_mul_b_106 = make_float2(gs_1[12], gs_1[13]);
            float2 y2_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
            oxs[12] = y2_14.x;
            oxs[13] = y2_14.y;
            float2 _f32x2_mul_a_107 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(oxs[14], oxs[15]), rstd2_1), make_float2(wv[14], wv[15]));
            float2 _f32x2_mul_b_108 = make_float2(gs_1[14], gs_1[15]);
            float2 y2_15;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_108));
            oxs[14] = y2_15.x;
            oxs[15] = y2_15.y;
            unsigned int packed_1[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 0], oxs[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 2], oxs[_lp*2+1 + 2]));
                packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 4], oxs[_lp*2+1 + 4]));
                packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 6], oxs[_lp*2+1 + 6]));
                packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 8], oxs[_lp*2+1 + 8]));
                packed_1[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 10], oxs[_lp*2+1 + 10]));
                packed_1[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 12], oxs[_lp*2+1 + 12]));
                packed_1[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(oxs[_lp*2 + 14], oxs[_lp*2+1 + 14]));
                packed_1[_lp + 7] = *(uint32_t*)&_bf2;
            }
            if (bid * 16 + group < rows) {
                reinterpret_cast<int4*>(y + ((long long)(bid * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
                reinterpret_cast<int4*>(y + ((long long)(bid * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_1 + (4))[0];
            }
        }
    } else {
        if (tid == 0) {
            int pblk_1 = bid + num_bids;
            if (pblk_1 < nblocks) {
                int pfirst_1 = pblk_1 * 16;
                int _min_109 = ((rows - pfirst_1) < (16) ? (rows - pfirst_1) : (16));
                int pbytes = _min_109 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst_1 * 128ll), "r"((uint32_t)(pbytes)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst_1 * 128ll), "r"((uint32_t)(pbytes)) : "memory");
            }
            uint32_t _read_special_reg_110;
            asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_110));
            int wblk_1 = bid + 6 * (int)_read_special_reg_110;
            if (wblk_1 < num_bids) {
                if (wblk_1 < nblocks) {
                    int wfirst_1 = wblk_1 * 16;
                    int _min_111 = ((rows - wfirst_1) < (16) ? (rows - wfirst_1) : (16));
                    int wbytes = _min_111 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst_1 * 128ll), "r"((uint32_t)(wbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst_1 * 128ll), "r"((uint32_t)(wbytes)) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int blk_1 = bid; blk_1 < nblocks; blk_1 += num_bids) {
            float xs_1[16];
            float zs_1[16];
            xs_1[0] = __uint_as_float(xw[0] << 16);
            xs_1[1] = __uint_as_float(xw[0] & 4294901760u);
            zs_1[0] = __uint_as_float(zw[0] << 16);
            zs_1[1] = __uint_as_float(zw[0] & 4294901760u);
            xs_1[2] = __uint_as_float(xw[1] << 16);
            xs_1[3] = __uint_as_float(xw[1] & 4294901760u);
            zs_1[2] = __uint_as_float(zw[1] << 16);
            zs_1[3] = __uint_as_float(zw[1] & 4294901760u);
            xs_1[4] = __uint_as_float(xw[2] << 16);
            xs_1[5] = __uint_as_float(xw[2] & 4294901760u);
            zs_1[4] = __uint_as_float(zw[2] << 16);
            zs_1[5] = __uint_as_float(zw[2] & 4294901760u);
            xs_1[6] = __uint_as_float(xw[3] << 16);
            xs_1[7] = __uint_as_float(xw[3] & 4294901760u);
            zs_1[6] = __uint_as_float(zw[3] << 16);
            zs_1[7] = __uint_as_float(zw[3] & 4294901760u);
            xs_1[8] = __uint_as_float(xw[4] << 16);
            xs_1[9] = __uint_as_float(xw[4] & 4294901760u);
            zs_1[8] = __uint_as_float(zw[4] << 16);
            zs_1[9] = __uint_as_float(zw[4] & 4294901760u);
            xs_1[10] = __uint_as_float(xw[5] << 16);
            xs_1[11] = __uint_as_float(xw[5] & 4294901760u);
            zs_1[10] = __uint_as_float(zw[5] << 16);
            zs_1[11] = __uint_as_float(zw[5] & 4294901760u);
            xs_1[12] = __uint_as_float(xw[6] << 16);
            xs_1[13] = __uint_as_float(xw[6] & 4294901760u);
            zs_1[12] = __uint_as_float(zw[6] << 16);
            zs_1[13] = __uint_as_float(zw[6] & 4294901760u);
            xs_1[14] = __uint_as_float(xw[7] << 16);
            xs_1[15] = __uint_as_float(xw[7] & 4294901760u);
            zs_1[14] = __uint_as_float(zw[7] << 16);
            zs_1[15] = __uint_as_float(zw[7] & 4294901760u);
            if (blk_1 + num_bids < nblocks) {
                int src_1 = (((blk_1 + num_bids) * 16 + group) < (rows - 1) ? ((blk_1 + num_bids) * 16 + group) : (rows - 1));
                uint32_t tq_7 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
                uint32_t th_7 = (uint32_t)((unsigned int)src_1) - tq_7 * (uint32_t)(heads.divisor);
                long long xoff_1 = (long long)tq_7 * x_tok + (long long)th_7 * x_head + (long long)col;
                long long zoff_1 = (long long)tq_7 * z_tok + (long long)th_7 * z_head + (long long)col;
                {
                    uint32_t _iv_112_0;
                    uint32_t _iv_112_1;
                    uint32_t _iv_112_2;
                    uint32_t _iv_112_3;
                    uint32_t _iv_112_4;
                    uint32_t _iv_112_5;
                    uint32_t _iv_112_6;
                    uint32_t _iv_112_7;
                    asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                        : "=r"(_iv_112_0), "=r"(_iv_112_1), "=r"(_iv_112_2), "=r"(_iv_112_3), "=r"(_iv_112_4), "=r"(_iv_112_5), "=r"(_iv_112_6), "=r"(_iv_112_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_1) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                    xw[0 + 0] = (uint32_t)_iv_112_0;
                    xw[0 + 1] = (uint32_t)_iv_112_1;
                    xw[0 + 2] = (uint32_t)_iv_112_2;
                    xw[0 + 3] = (uint32_t)_iv_112_3;
                    xw[0 + 4] = (uint32_t)_iv_112_4;
                    xw[0 + 5] = (uint32_t)_iv_112_5;
                    xw[0 + 6] = (uint32_t)_iv_112_6;
                    xw[0 + 7] = (uint32_t)_iv_112_7;
                }
                {
                    uint32_t _iv_113_0;
                    uint32_t _iv_113_1;
                    uint32_t _iv_113_2;
                    uint32_t _iv_113_3;
                    uint32_t _iv_113_4;
                    uint32_t _iv_113_5;
                    uint32_t _iv_113_6;
                    uint32_t _iv_113_7;
                    asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                        : "=r"(_iv_113_0), "=r"(_iv_113_1), "=r"(_iv_113_2), "=r"(_iv_113_3), "=r"(_iv_113_4), "=r"(_iv_113_5), "=r"(_iv_113_6), "=r"(_iv_113_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_1) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                    zw[0 + 0] = (uint32_t)_iv_113_0;
                    zw[0 + 1] = (uint32_t)_iv_113_1;
                    zw[0 + 2] = (uint32_t)_iv_113_2;
                    zw[0 + 3] = (uint32_t)_iv_113_3;
                    zw[0 + 4] = (uint32_t)_iv_113_4;
                    zw[0 + 5] = (uint32_t)_iv_113_5;
                    zw[0 + 6] = (uint32_t)_iv_113_6;
                    zw[0 + 7] = (uint32_t)_iv_113_7;
                }
            }
            if (tid == 0) {
                int nblk_1 = blk_1 + 2 * num_bids;
                if (nblk_1 < nblocks) {
                    int nfirst_1 = nblk_1 * 16;
                    int _min_114 = ((rows - nfirst_1) < (16) ? (rows - nfirst_1) : (16));
                    int nbytes = _min_114 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                }
            }
            float gs_2[16];
            float _exp2_116 = approx_exp2(zs_1[0] * -1.442695f);
            float _rcp_115 = approx_rcp(1.0f + _exp2_116);
            gs_2[0] = zs_1[0] * _rcp_115;
            float _exp2_118 = approx_exp2(zs_1[1] * -1.442695f);
            float _rcp_117 = approx_rcp(1.0f + _exp2_118);
            gs_2[1] = zs_1[1] * _rcp_117;
            float _exp2_120 = approx_exp2(zs_1[2] * -1.442695f);
            float _rcp_119 = approx_rcp(1.0f + _exp2_120);
            gs_2[2] = zs_1[2] * _rcp_119;
            float _exp2_122 = approx_exp2(zs_1[3] * -1.442695f);
            float _rcp_121 = approx_rcp(1.0f + _exp2_122);
            gs_2[3] = zs_1[3] * _rcp_121;
            float _exp2_124 = approx_exp2(zs_1[4] * -1.442695f);
            float _rcp_123 = approx_rcp(1.0f + _exp2_124);
            gs_2[4] = zs_1[4] * _rcp_123;
            float _exp2_126 = approx_exp2(zs_1[5] * -1.442695f);
            float _rcp_125 = approx_rcp(1.0f + _exp2_126);
            gs_2[5] = zs_1[5] * _rcp_125;
            float _exp2_128 = approx_exp2(zs_1[6] * -1.442695f);
            float _rcp_127 = approx_rcp(1.0f + _exp2_128);
            gs_2[6] = zs_1[6] * _rcp_127;
            float _exp2_130 = approx_exp2(zs_1[7] * -1.442695f);
            float _rcp_129 = approx_rcp(1.0f + _exp2_130);
            gs_2[7] = zs_1[7] * _rcp_129;
            float _exp2_132 = approx_exp2(zs_1[8] * -1.442695f);
            float _rcp_131 = approx_rcp(1.0f + _exp2_132);
            gs_2[8] = zs_1[8] * _rcp_131;
            float _exp2_134 = approx_exp2(zs_1[9] * -1.442695f);
            float _rcp_133 = approx_rcp(1.0f + _exp2_134);
            gs_2[9] = zs_1[9] * _rcp_133;
            float _exp2_136 = approx_exp2(zs_1[10] * -1.442695f);
            float _rcp_135 = approx_rcp(1.0f + _exp2_136);
            gs_2[10] = zs_1[10] * _rcp_135;
            float _exp2_138 = approx_exp2(zs_1[11] * -1.442695f);
            float _rcp_137 = approx_rcp(1.0f + _exp2_138);
            gs_2[11] = zs_1[11] * _rcp_137;
            float _exp2_140 = approx_exp2(zs_1[12] * -1.442695f);
            float _rcp_139 = approx_rcp(1.0f + _exp2_140);
            gs_2[12] = zs_1[12] * _rcp_139;
            float _exp2_142 = approx_exp2(zs_1[13] * -1.442695f);
            float _rcp_141 = approx_rcp(1.0f + _exp2_142);
            gs_2[13] = zs_1[13] * _rcp_141;
            float _exp2_144 = approx_exp2(zs_1[14] * -1.442695f);
            float _rcp_143 = approx_rcp(1.0f + _exp2_144);
            gs_2[14] = zs_1[14] * _rcp_143;
            float _exp2_146 = approx_exp2(zs_1[15] * -1.442695f);
            float _rcp_145 = approx_rcp(1.0f + _exp2_146);
            gs_2[15] = zs_1[15] * _rcp_145;
            float total_2 = 0.0f;
            float2 lo_8 = make_float2(0.0f, 0.0f);
            float2 hi_8 = make_float2(0.0f, 0.0f);
            float2 xlo_8 = make_float2(xs_1[0], xs_1[1]);
            float2 xhi_8 = make_float2(xs_1[2], xs_1[3]);
            lo_8 = fma_f32x2_rn_noftz(xlo_8, xlo_8, lo_8);
            hi_8 = fma_f32x2_rn_noftz(xhi_8, xhi_8, hi_8);
            float2 xlo_9 = make_float2(xs_1[4], xs_1[5]);
            float2 xhi_9 = make_float2(xs_1[6], xs_1[7]);
            lo_8 = fma_f32x2_rn_noftz(xlo_9, xlo_9, lo_8);
            hi_8 = fma_f32x2_rn_noftz(xhi_9, xhi_9, hi_8);
            float2 xlo_10 = make_float2(xs_1[8], xs_1[9]);
            float2 xhi_10 = make_float2(xs_1[10], xs_1[11]);
            lo_8 = fma_f32x2_rn_noftz(xlo_10, xlo_10, lo_8);
            hi_8 = fma_f32x2_rn_noftz(xhi_10, xhi_10, hi_8);
            float2 xlo_11 = make_float2(xs_1[12], xs_1[13]);
            float2 xhi_11 = make_float2(xs_1[14], xs_1[15]);
            lo_8 = fma_f32x2_rn_noftz(xlo_11, xlo_11, lo_8);
            hi_8 = fma_f32x2_rn_noftz(xhi_11, xhi_11, hi_8);
            total_2 = lo_8.x + lo_8.y + (hi_8.x + hi_8.y);
            float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, total_2, 1);
            total_2 = total_2 + _shfl_xor_sync_147;
            float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, total_2, 2);
            total_2 = total_2 + _shfl_xor_sync_148;
            float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, total_2, 4);
            total_2 = total_2 + _shfl_xor_sync_149;
            float rstd_2;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_2) : "f"(total_2 * 0.0078125f + eps));
            float2 rstd2_2 = make_float2(rstd_2, rstd_2);
            float2 _f32x2_mul_a_150 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[0], xs_1[1]), rstd2_2), make_float2(wv[0], wv[1]));
            float2 _f32x2_mul_b_151 = make_float2(gs_2[0], gs_2[1]);
            float2 y2_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
            xs_1[0] = y2_16.x;
            xs_1[1] = y2_16.y;
            float2 _f32x2_mul_a_152 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[2], xs_1[3]), rstd2_2), make_float2(wv[2], wv[3]));
            float2 _f32x2_mul_b_153 = make_float2(gs_2[2], gs_2[3]);
            float2 y2_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
            xs_1[2] = y2_17.x;
            xs_1[3] = y2_17.y;
            float2 _f32x2_mul_a_154 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[4], xs_1[5]), rstd2_2), make_float2(wv[4], wv[5]));
            float2 _f32x2_mul_b_155 = make_float2(gs_2[4], gs_2[5]);
            float2 y2_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
            xs_1[4] = y2_18.x;
            xs_1[5] = y2_18.y;
            float2 _f32x2_mul_a_156 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[6], xs_1[7]), rstd2_2), make_float2(wv[6], wv[7]));
            float2 _f32x2_mul_b_157 = make_float2(gs_2[6], gs_2[7]);
            float2 y2_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
            xs_1[6] = y2_19.x;
            xs_1[7] = y2_19.y;
            float2 _f32x2_mul_a_158 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[8], xs_1[9]), rstd2_2), make_float2(wv[8], wv[9]));
            float2 _f32x2_mul_b_159 = make_float2(gs_2[8], gs_2[9]);
            float2 y2_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
            xs_1[8] = y2_20.x;
            xs_1[9] = y2_20.y;
            float2 _f32x2_mul_a_160 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[10], xs_1[11]), rstd2_2), make_float2(wv[10], wv[11]));
            float2 _f32x2_mul_b_161 = make_float2(gs_2[10], gs_2[11]);
            float2 y2_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
            xs_1[10] = y2_21.x;
            xs_1[11] = y2_21.y;
            float2 _f32x2_mul_a_162 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[12], xs_1[13]), rstd2_2), make_float2(wv[12], wv[13]));
            float2 _f32x2_mul_b_163 = make_float2(gs_2[12], gs_2[13]);
            float2 y2_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
            xs_1[12] = y2_22.x;
            xs_1[13] = y2_22.y;
            float2 _f32x2_mul_a_164 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_1[14], xs_1[15]), rstd2_2), make_float2(wv[14], wv[15]));
            float2 _f32x2_mul_b_165 = make_float2(gs_2[14], gs_2[15]);
            float2 y2_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
            xs_1[14] = y2_23.x;
            xs_1[15] = y2_23.y;
            unsigned int packed_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 2], xs_1[_lp*2+1 + 2]));
                packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 4], xs_1[_lp*2+1 + 4]));
                packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 6], xs_1[_lp*2+1 + 6]));
                packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 8], xs_1[_lp*2+1 + 8]));
                packed_2[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 10], xs_1[_lp*2+1 + 10]));
                packed_2[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 12], xs_1[_lp*2+1 + 12]));
                packed_2[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 14], xs_1[_lp*2+1 + 14]));
                packed_2[_lp + 7] = *(uint32_t*)&_bf2;
            }
            if (blk_1 * 16 + group < rows) {
                reinterpret_cast<int4*>(y + ((long long)(blk_1 * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
                reinterpret_cast<int4*>(y + ((long long)(blk_1 * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_2 + (4))[0];
            }
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
