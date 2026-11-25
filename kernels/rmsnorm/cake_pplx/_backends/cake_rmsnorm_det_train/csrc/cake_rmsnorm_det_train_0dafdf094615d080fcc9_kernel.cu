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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_0dafdf094615d080fcc9(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)

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
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 13568);
    const int ring_addr = smem + 13568;

    // Mbarrier init (3 barriers at smem_raw[0..24))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

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
    int rg = warp / 12;
    int tir = tid - rg * 384;
    unsigned int ww[8];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((384 + tir) * 8));
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
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
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[1];
    float rr[2];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        flags[4] = (int)nxt;
    }
    __syncthreads();
    int tk = flags[4];
    #pragma unroll 1
    for (int ci = 0; ci < 96; ci++) {
        if (tk >= 96) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_3;
        }
        int vl = tid % 8;
        int rl = tid / 8;
        unsigned int gw_0[16];
        unsigned int xw_0[16];
        unsigned int hw_0[1];
        float rr_0[4];
        float acc_0[8];
        acc_0[0] = 0.0f;
        acc_0[1] = 0.0f;
        acc_0[2] = 0.0f;
        acc_0[3] = 0.0f;
        acc_0[4] = 0.0f;
        acc_0[5] = 0.0f;
        acc_0[6] = 0.0f;
        acc_0[7] = 0.0f;
        #pragma unroll 1
        for (int t0 = rl; t0 < T; t0 += 192) {
            if (t0 < T) {
                unsigned long long g_base = (unsigned long long)t0 * (unsigned long long)g_stride;
                unsigned long long x_base = (unsigned long long)t0 * (unsigned long long)x_stride;
                rr_0[0] = r[t0];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&gw_0[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&xw_0[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 48 < T) {
                unsigned long long g_base_1 = (unsigned long long)(t0 + 48) * (unsigned long long)g_stride;
                unsigned long long x_base_1 = (unsigned long long)(t0 + 48) * (unsigned long long)x_stride;
                rr_0[1] = r[t0 + 48];
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&gw_0[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&xw_0[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 96 < T) {
                unsigned long long g_base_2 = (unsigned long long)(t0 + 96) * (unsigned long long)g_stride;
                unsigned long long x_base_2 = (unsigned long long)(t0 + 96) * (unsigned long long)x_stride;
                rr_0[2] = r[t0 + 96];
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&gw_0[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_9 = reinterpret_cast<uint4*>(&xw_0[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 144 < T) {
                unsigned long long g_base_3 = (unsigned long long)(t0 + 144) * (unsigned long long)g_stride;
                unsigned long long x_base_3 = (unsigned long long)(t0 + 144) * (unsigned long long)x_stride;
                rr_0[3] = r[t0 + 144];
                {
                    const uint4* _vptr_10 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_10 = reinterpret_cast<uint4*>(&gw_0[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_11 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_11 = reinterpret_cast<uint4*>(&xw_0[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 < T) {
                float2 rr2_0 = make_float2(rr_0[0], rr_0[0]);
                float2 _f32x2_mul_a_12 = make_float2(__uint_as_float(gw_0[0] << 16), __uint_as_float(gw_0[0] & 4294901760u));
                float2 _f32x2_mul_b_13 = make_float2(__uint_as_float(xw_0[0] << 16), __uint_as_float(xw_0[0] & 4294901760u));
                float2 gx;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
                float2 a = fma_f32x2_rn_noftz(gx, rr2_0, make_float2(acc_0[0], acc_0[1]));
                acc_0[0] = a.x;
                acc_0[1] = a.y;
                float2 _f32x2_mul_a_14 = make_float2(__uint_as_float(gw_0[1] << 16), __uint_as_float(gw_0[1] & 4294901760u));
                float2 _f32x2_mul_b_15 = make_float2(__uint_as_float(xw_0[1] << 16), __uint_as_float(xw_0[1] & 4294901760u));
                float2 gx_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
                float2 a_0 = fma_f32x2_rn_noftz(gx_0, rr2_0, make_float2(acc_0[2], acc_0[3]));
                acc_0[2] = a_0.x;
                acc_0[3] = a_0.y;
                float2 _f32x2_mul_a_16 = make_float2(__uint_as_float(gw_0[2] << 16), __uint_as_float(gw_0[2] & 4294901760u));
                float2 _f32x2_mul_b_17 = make_float2(__uint_as_float(xw_0[2] << 16), __uint_as_float(xw_0[2] & 4294901760u));
                float2 gx_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
                float2 a_1 = fma_f32x2_rn_noftz(gx_1, rr2_0, make_float2(acc_0[4], acc_0[5]));
                acc_0[4] = a_1.x;
                acc_0[5] = a_1.y;
                float2 _f32x2_mul_a_18 = make_float2(__uint_as_float(gw_0[3] << 16), __uint_as_float(gw_0[3] & 4294901760u));
                float2 _f32x2_mul_b_19 = make_float2(__uint_as_float(xw_0[3] << 16), __uint_as_float(xw_0[3] & 4294901760u));
                float2 gx_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
                float2 a_2 = fma_f32x2_rn_noftz(gx_2, rr2_0, make_float2(acc_0[6], acc_0[7]));
                acc_0[6] = a_2.x;
                acc_0[7] = a_2.y;
            }
            if (t0 + 48 < T) {
                float2 rr2_1 = make_float2(rr_0[1], rr_0[1]);
                float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(gw_0[4] << 16), __uint_as_float(gw_0[4] & 4294901760u));
                float2 _f32x2_mul_b_21 = make_float2(__uint_as_float(xw_0[4] << 16), __uint_as_float(xw_0[4] & 4294901760u));
                float2 gx_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                float2 a_3 = fma_f32x2_rn_noftz(gx_3, rr2_1, make_float2(acc_0[0], acc_0[1]));
                acc_0[0] = a_3.x;
                acc_0[1] = a_3.y;
                float2 _f32x2_mul_a_22 = make_float2(__uint_as_float(gw_0[5] << 16), __uint_as_float(gw_0[5] & 4294901760u));
                float2 _f32x2_mul_b_23 = make_float2(__uint_as_float(xw_0[5] << 16), __uint_as_float(xw_0[5] & 4294901760u));
                float2 gx_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
                float2 a_3_1 = fma_f32x2_rn_noftz(gx_3_1, rr2_1, make_float2(acc_0[2], acc_0[3]));
                acc_0[2] = a_3_1.x;
                acc_0[3] = a_3_1.y;
                float2 _f32x2_mul_a_24 = make_float2(__uint_as_float(gw_0[6] << 16), __uint_as_float(gw_0[6] & 4294901760u));
                float2 _f32x2_mul_b_25 = make_float2(__uint_as_float(xw_0[6] << 16), __uint_as_float(xw_0[6] & 4294901760u));
                float2 gx_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
                float2 a_4 = fma_f32x2_rn_noftz(gx_4, rr2_1, make_float2(acc_0[4], acc_0[5]));
                acc_0[4] = a_4.x;
                acc_0[5] = a_4.y;
                float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(gw_0[7] << 16), __uint_as_float(gw_0[7] & 4294901760u));
                float2 _f32x2_mul_b_27 = make_float2(__uint_as_float(xw_0[7] << 16), __uint_as_float(xw_0[7] & 4294901760u));
                float2 gx_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
                float2 a_5 = fma_f32x2_rn_noftz(gx_5, rr2_1, make_float2(acc_0[6], acc_0[7]));
                acc_0[6] = a_5.x;
                acc_0[7] = a_5.y;
            }
            if (t0 + 96 < T) {
                float2 rr2_2 = make_float2(rr_0[2], rr_0[2]);
                float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(gw_0[8] << 16), __uint_as_float(gw_0[8] & 4294901760u));
                float2 _f32x2_mul_b_29 = make_float2(__uint_as_float(xw_0[8] << 16), __uint_as_float(xw_0[8] & 4294901760u));
                float2 gx_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                float2 a_6 = fma_f32x2_rn_noftz(gx_6, rr2_2, make_float2(acc_0[0], acc_0[1]));
                acc_0[0] = a_6.x;
                acc_0[1] = a_6.y;
                float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(gw_0[9] << 16), __uint_as_float(gw_0[9] & 4294901760u));
                float2 _f32x2_mul_b_31 = make_float2(__uint_as_float(xw_0[9] << 16), __uint_as_float(xw_0[9] & 4294901760u));
                float2 gx_6_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                float2 a_6_1 = fma_f32x2_rn_noftz(gx_6_1, rr2_2, make_float2(acc_0[2], acc_0[3]));
                acc_0[2] = a_6_1.x;
                acc_0[3] = a_6_1.y;
                float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(gw_0[10] << 16), __uint_as_float(gw_0[10] & 4294901760u));
                float2 _f32x2_mul_b_33 = make_float2(__uint_as_float(xw_0[10] << 16), __uint_as_float(xw_0[10] & 4294901760u));
                float2 gx_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
                float2 a_7 = fma_f32x2_rn_noftz(gx_7, rr2_2, make_float2(acc_0[4], acc_0[5]));
                acc_0[4] = a_7.x;
                acc_0[5] = a_7.y;
                float2 _f32x2_mul_a_34 = make_float2(__uint_as_float(gw_0[11] << 16), __uint_as_float(gw_0[11] & 4294901760u));
                float2 _f32x2_mul_b_35 = make_float2(__uint_as_float(xw_0[11] << 16), __uint_as_float(xw_0[11] & 4294901760u));
                float2 gx_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                float2 a_8 = fma_f32x2_rn_noftz(gx_8, rr2_2, make_float2(acc_0[6], acc_0[7]));
                acc_0[6] = a_8.x;
                acc_0[7] = a_8.y;
            }
            if (t0 + 144 < T) {
                float2 rr2_3 = make_float2(rr_0[3], rr_0[3]);
                float2 _f32x2_mul_a_36 = make_float2(__uint_as_float(gw_0[12] << 16), __uint_as_float(gw_0[12] & 4294901760u));
                float2 _f32x2_mul_b_37 = make_float2(__uint_as_float(xw_0[12] << 16), __uint_as_float(xw_0[12] & 4294901760u));
                float2 gx_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                float2 a_9 = fma_f32x2_rn_noftz(gx_9, rr2_3, make_float2(acc_0[0], acc_0[1]));
                acc_0[0] = a_9.x;
                acc_0[1] = a_9.y;
                float2 _f32x2_mul_a_38 = make_float2(__uint_as_float(gw_0[13] << 16), __uint_as_float(gw_0[13] & 4294901760u));
                float2 _f32x2_mul_b_39 = make_float2(__uint_as_float(xw_0[13] << 16), __uint_as_float(xw_0[13] & 4294901760u));
                float2 gx_9_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
                float2 a_9_1 = fma_f32x2_rn_noftz(gx_9_1, rr2_3, make_float2(acc_0[2], acc_0[3]));
                acc_0[2] = a_9_1.x;
                acc_0[3] = a_9_1.y;
                float2 _f32x2_mul_a_40 = make_float2(__uint_as_float(gw_0[14] << 16), __uint_as_float(gw_0[14] & 4294901760u));
                float2 _f32x2_mul_b_41 = make_float2(__uint_as_float(xw_0[14] << 16), __uint_as_float(xw_0[14] & 4294901760u));
                float2 gx_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
                float2 a_10 = fma_f32x2_rn_noftz(gx_10, rr2_3, make_float2(acc_0[4], acc_0[5]));
                acc_0[4] = a_10.x;
                acc_0[5] = a_10.y;
                float2 _f32x2_mul_a_42 = make_float2(__uint_as_float(gw_0[15] << 16), __uint_as_float(gw_0[15] & 4294901760u));
                float2 _f32x2_mul_b_43 = make_float2(__uint_as_float(xw_0[15] << 16), __uint_as_float(xw_0[15] & 4294901760u));
                float2 gx_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                float2 a_11 = fma_f32x2_rn_noftz(gx_11, rr2_3, make_float2(acc_0[6], acc_0[7]));
                acc_0[6] = a_11.x;
                acc_0[7] = a_11.y;
            }
        }
        sums[rl * 64 + vl * 8] = acc_0[0];
        sums[rl * 64 + vl * 8 + 1] = acc_0[1];
        sums[rl * 64 + vl * 8 + 2] = acc_0[2];
        sums[rl * 64 + vl * 8 + 3] = acc_0[3];
        sums[rl * 64 + vl * 8 + 4] = acc_0[4];
        sums[rl * 64 + vl * 8 + 5] = acc_0[5];
        sums[rl * 64 + vl * 8 + 6] = acc_0[6];
        sums[rl * 64 + vl * 8 + 7] = acc_0[7];
        __syncthreads();
        int cl = tid % 64;
        int s8 = tid / 64 * 8;
        float lv8[8];
        lv8[0] = sums[s8 * 64 + cl];
        lv8[1] = sums[(s8 + 1) * 64 + cl];
        lv8[2] = sums[(s8 + 2) * 64 + cl];
        lv8[3] = sums[(s8 + 3) * 64 + cl];
        lv8[4] = sums[(s8 + 4) * 64 + cl];
        lv8[5] = sums[(s8 + 5) * 64 + cl];
        lv8[6] = sums[(s8 + 6) * 64 + cl];
        lv8[7] = sums[(s8 + 7) * 64 + cl];
        lv8[0] = lv8[0] + lv8[1];
        lv8[2] = lv8[2] + lv8[3];
        lv8[4] = lv8[4] + lv8[5];
        lv8[6] = lv8[6] + lv8[7];
        lv8[0] = lv8[0] + lv8[2];
        lv8[4] = lv8[4] + lv8[6];
        lv8[0] = lv8[0] + lv8[4];
        sums[s8 * 64 + cl] = lv8[0];
        __syncthreads();
        if (tid < 64) {
            float gv[6];
            gv[0] = sums[cl];
            gv[1] = sums[512 + cl];
            gv[2] = sums[1024 + cl];
            gv[3] = sums[1536 + cl];
            gv[4] = sums[2048 + cl];
            gv[5] = sums[2560 + cl];
            gv[0] = gv[0] + gv[1];
            gv[2] = gv[2] + gv[3];
            gv[4] = gv[4] + gv[5];
            gv[0] = gv[0] + gv[2];
            gv[0] = gv[0] + gv[4];
            *(reinterpret_cast<float*>(dw) + (tk * 64 + cl)) = gv[0];
        }
        if (tid == 0) {
            flags[5 + (ci & 1)] = (int)nxt;
        }
        __syncthreads();
        tk = flags[5 + (ci & 1)];
    }
    if (tid == 0) {
        pc = (int)nxt - 96;
        first_c = pc;
        if (pc < n_chunks) {
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
            for (int k = 0; k < 5; k++) {
                if (halving != 0) {
                    if (size > tail) {
                        if (rem >= sl * size) {
                            if (pc < first + sl) {
                                span = size;
                                begin_it = base + (pc - first) * size;
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
                if (pc < first + extra) {
                    span = quot + 1;
                    begin_it = base + (pc - first) * (quot + 1);
                } else if (pc < first + sl) {
                    span = quot;
                    begin_it = base + extra * (quot + 1) + (pc - first - extra) * quot;
                } else {
                    begin_it = stages;
                }
            }
            int _min_44 = ((begin_it) < (T) ? (begin_it) : (T));
            prow = _min_44;
            int _min_45 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
            pend = _min_45;
            unsigned int _atomic_fetch_add_46 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_46;
        }
        int slot = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 96;
                if (pc < n_chunks) {
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
                    for (int k_1 = 0; k_1 < 5; k_1++) {
                        if (halving_1 != 0) {
                            if (size_1 > tail_1) {
                                if (rem_1 >= sl_1 * size_1) {
                                    if (pc < first_1 + sl_1) {
                                        span_1 = size_1;
                                        begin_it_1 = base_1 + (pc - first_1) * size_1;
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
                        if (pc < first_1 + extra_1) {
                            span_1 = quot_1 + 1;
                            begin_it_1 = base_1 + (pc - first_1) * (quot_1 + 1);
                        } else if (pc < first_1 + sl_1) {
                            span_1 = quot_1;
                            begin_it_1 = base_1 + extra_1 * (quot_1 + 1) + (pc - first_1 - extra_1) * quot_1;
                        } else {
                            begin_it_1 = stages_1;
                        }
                    }
                    int _min_47 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                    prow = _min_47;
                    int _min_48 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                    pend = _min_48;
                    unsigned int _atomic_fetch_add_49 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_49;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_50 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval = _min_50;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 24576);
            int sbase = ring_at + slot * 24576;
            int _min_51 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_51);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr);
        }
        prow = prow + nval;
        int slot_0 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 96;
                if (pc < n_chunks) {
                    int stages_2 = T;
                    int sl_2 = rows_per_chunk;
                    if (sl_2 > stages_2) {
                        sl_2 = stages_2;
                    }
                    if (sl_2 < 1) {
                        sl_2 = 1;
                    }
                    int share_2 = (stages_2 + sl_2 - 1) / sl_2;
                    int tail_2 = (share_2 + 11) / 12;
                    if (tail_2 < 1) {
                        tail_2 = 1;
                    }
                    int rem_2 = stages_2;
                    int size_2 = (share_2 + 1) / 2;
                    int first_2 = 0;
                    int base_2 = 0;
                    int span_2 = 0;
                    int halving_2 = 1;
                    int begin_it_2 = 0;
                    #pragma unroll 1
                    for (int k_2 = 0; k_2 < 5; k_2++) {
                        if (halving_2 != 0) {
                            if (size_2 > tail_2) {
                                if (rem_2 >= sl_2 * size_2) {
                                    if (pc < first_2 + sl_2) {
                                        span_2 = size_2;
                                        begin_it_2 = base_2 + (pc - first_2) * size_2;
                                        halving_2 = 0;
                                    } else {
                                        first_2 = first_2 + sl_2;
                                        base_2 = base_2 + sl_2 * size_2;
                                        rem_2 = rem_2 - sl_2 * size_2;
                                        size_2 = (size_2 + 1) / 2;
                                    }
                                } else {
                                    halving_2 = 0;
                                }
                            } else {
                                halving_2 = 0;
                            }
                        }
                    }
                    if (span_2 == 0) {
                        int quot_2 = rem_2 / sl_2;
                        int extra_2 = rem_2 - quot_2 * sl_2;
                        if (pc < first_2 + extra_2) {
                            span_2 = quot_2 + 1;
                            begin_it_2 = base_2 + (pc - first_2) * (quot_2 + 1);
                        } else if (pc < first_2 + sl_2) {
                            span_2 = quot_2;
                            begin_it_2 = base_2 + extra_2 * (quot_2 + 1) + (pc - first_2 - extra_2) * quot_2;
                        } else {
                            begin_it_2 = stages_2;
                        }
                    }
                    int _min_52 = ((begin_it_2) < (T) ? (begin_it_2) : (T));
                    prow = _min_52;
                    int _min_53 = ((begin_it_2 + span_2) < (T) ? (begin_it_2 + span_2) : (T));
                    pend = _min_53;
                    unsigned int _atomic_fetch_add_54 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_54;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_55 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_0 = _min_55;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 24576);
            int sbase_1 = ring_at + slot_0 * 24576;
            int _min_56 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_1 = (long long)(prow + _min_56);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr + 8);
        }
        prow = prow + nval_0;
        int slot_1 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 96;
                if (pc < n_chunks) {
                    int stages_3 = T;
                    int sl_3 = rows_per_chunk;
                    if (sl_3 > stages_3) {
                        sl_3 = stages_3;
                    }
                    if (sl_3 < 1) {
                        sl_3 = 1;
                    }
                    int share_3 = (stages_3 + sl_3 - 1) / sl_3;
                    int tail_3 = (share_3 + 11) / 12;
                    if (tail_3 < 1) {
                        tail_3 = 1;
                    }
                    int rem_3 = stages_3;
                    int size_3 = (share_3 + 1) / 2;
                    int first_3 = 0;
                    int base_3 = 0;
                    int span_3 = 0;
                    int halving_3 = 1;
                    int begin_it_3 = 0;
                    #pragma unroll 1
                    for (int k_3 = 0; k_3 < 5; k_3++) {
                        if (halving_3 != 0) {
                            if (size_3 > tail_3) {
                                if (rem_3 >= sl_3 * size_3) {
                                    if (pc < first_3 + sl_3) {
                                        span_3 = size_3;
                                        begin_it_3 = base_3 + (pc - first_3) * size_3;
                                        halving_3 = 0;
                                    } else {
                                        first_3 = first_3 + sl_3;
                                        base_3 = base_3 + sl_3 * size_3;
                                        rem_3 = rem_3 - sl_3 * size_3;
                                        size_3 = (size_3 + 1) / 2;
                                    }
                                } else {
                                    halving_3 = 0;
                                }
                            } else {
                                halving_3 = 0;
                            }
                        }
                    }
                    if (span_3 == 0) {
                        int quot_3 = rem_3 / sl_3;
                        int extra_3 = rem_3 - quot_3 * sl_3;
                        if (pc < first_3 + extra_3) {
                            span_3 = quot_3 + 1;
                            begin_it_3 = base_3 + (pc - first_3) * (quot_3 + 1);
                        } else if (pc < first_3 + sl_3) {
                            span_3 = quot_3;
                            begin_it_3 = base_3 + extra_3 * (quot_3 + 1) + (pc - first_3 - extra_3) * quot_3;
                        } else {
                            begin_it_3 = stages_3;
                        }
                    }
                    int _min_57 = ((begin_it_3) < (T) ? (begin_it_3) : (T));
                    prow = _min_57;
                    int _min_58 = ((begin_it_3 + span_3) < (T) ? (begin_it_3 + span_3) : (T));
                    pend = _min_58;
                    unsigned int _atomic_fetch_add_59 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_59;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_60 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_1 = _min_60;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 24576);
            int sbase_2 = ring_at + slot_1 * 24576;
            int _min_61 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_61);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr + 16);
        }
        prow = prow + nval_1;
    }
    __syncthreads();
    int row0 = meta[0];
    int nval_2 = meta[1];
    rr[0] = 0.0f;
    if (nval_2 > rg) {
        rr[0] = r[row0 + rg];
    }
    float dot[1];
    #pragma unroll 1
    for (int k_4 = 0; k_4 < T + n_chunks; k_4++) {
        if (nval_2 < 1) {
            break;
        }
        row0_n = meta[2 * ((k_4 + 1) % 3)];
        nval_n = meta[2 * ((k_4 + 1) % 3) + 1];
        if (nval_n > rg) {
            rr[1] = r[row0_n + rg];
        }
        int _stage_full = (int)((unsigned int)k_4 % 3);
        int _parity_full = (int)((unsigned int)k_4 / 3 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_4 % 3 * 24576 + rg * 12288;
#if defined(_MSC_VER)
        uint64_t _smem_raw_62[2];
        asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_62[0]), "=l"(_smem_raw_62[1]) : "r"(rbase + tir * 8 * 2));
#else
        __int128_t _smem_raw_62;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_62) : "r"(rbase + tir * 8 * 2));
#endif
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[3];
#if defined(_MSC_VER)
        uint64_t _smem_raw_63[2];
        asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_63[0]), "=l"(_smem_raw_63[1]) : "r"(rbase + 12288 + tir * 8 * 2));
#else
        __int128_t _smem_raw_63;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_63) : "r"(rbase + 12288 + tir * 8 * 2));
#endif
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[3];
#if defined(_MSC_VER)
        uint64_t _smem_raw_64[2];
        asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_64[0]), "=l"(_smem_raw_64[1]) : "r"(rbase + (384 + tir) * 8 * 2));
#else
        __int128_t _smem_raw_64;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_64) : "r"(rbase + (384 + tir) * 8 * 2));
#endif
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[3];
#if defined(_MSC_VER)
        uint64_t _smem_raw_65[2];
        asm volatile("{ .reg .b128 b; ld.shared.b128 b, [%2]; mov.b128 {%0, %1}, b; }" : "=l"(_smem_raw_65[0]), "=l"(_smem_raw_65[1]) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
#else
        __int128_t _smem_raw_65;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_65) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
#endif
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_65)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_65)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_65)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_65)[3];
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        float dq = 0.0f;
        dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[0] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[0] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[1] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[1] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[2] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[2] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[3] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[3] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * __uint_as_float(ww[4] << 16), __uint_as_float(xw[4] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * __uint_as_float(ww[4] & 4294901760u), __uint_as_float(xw[4] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * __uint_as_float(ww[5] << 16), __uint_as_float(xw[5] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * __uint_as_float(ww[5] & 4294901760u), __uint_as_float(xw[5] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * __uint_as_float(ww[6] << 16), __uint_as_float(xw[6] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * __uint_as_float(ww[6] & 4294901760u), __uint_as_float(xw[6] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * __uint_as_float(ww[7] << 16), __uint_as_float(xw[7] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * __uint_as_float(ww[7] & 4294901760u), __uint_as_float(xw[7] & 4294901760u), dq);
        float _warp_reduce_66 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_66 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_66, offset);
        dot[0] = _warp_reduce_66;
        if (lane == 0) {
            dots[(k_4 & 1) * 12 + warp] = dot[0];
        }
        __syncthreads();
        float tot = 0.0f;
        tot += dots[(k_4 & 1) * 12 + rg * 12];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 1];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 2];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 3];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 4];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 5];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 6];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 7];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 8];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 9];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 10];
        tot += dots[(k_4 & 1) * 12 + rg * 12 + 11];
        dot[0] = tot;
        if (tid == 0) {
            int slot_2 = (k_4 + 3) % 3;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt - 96;
                    if (pc < n_chunks) {
                        int stages_4 = T;
                        int sl_4 = rows_per_chunk;
                        if (sl_4 > stages_4) {
                            sl_4 = stages_4;
                        }
                        if (sl_4 < 1) {
                            sl_4 = 1;
                        }
                        int share_4 = (stages_4 + sl_4 - 1) / sl_4;
                        int tail_4 = (share_4 + 11) / 12;
                        if (tail_4 < 1) {
                            tail_4 = 1;
                        }
                        int rem_4 = stages_4;
                        int size_4 = (share_4 + 1) / 2;
                        int first_4 = 0;
                        int base_4 = 0;
                        int span_4 = 0;
                        int halving_4 = 1;
                        int begin_it_4 = 0;
                        #pragma unroll 1
                        for (int k_0 = 0; k_0 < 5; k_0++) {
                            if (halving_4 != 0) {
                                if (size_4 > tail_4) {
                                    if (rem_4 >= sl_4 * size_4) {
                                        if (pc < first_4 + sl_4) {
                                            span_4 = size_4;
                                            begin_it_4 = base_4 + (pc - first_4) * size_4;
                                            halving_4 = 0;
                                        } else {
                                            first_4 = first_4 + sl_4;
                                            base_4 = base_4 + sl_4 * size_4;
                                            rem_4 = rem_4 - sl_4 * size_4;
                                            size_4 = (size_4 + 1) / 2;
                                        }
                                    } else {
                                        halving_4 = 0;
                                    }
                                } else {
                                    halving_4 = 0;
                                }
                            }
                        }
                        if (span_4 == 0) {
                            int quot_4 = rem_4 / sl_4;
                            int extra_4 = rem_4 - quot_4 * sl_4;
                            if (pc < first_4 + extra_4) {
                                span_4 = quot_4 + 1;
                                begin_it_4 = base_4 + (pc - first_4) * (quot_4 + 1);
                            } else if (pc < first_4 + sl_4) {
                                span_4 = quot_4;
                                begin_it_4 = base_4 + extra_4 * (quot_4 + 1) + (pc - first_4 - extra_4) * quot_4;
                            } else {
                                begin_it_4 = stages_4;
                            }
                        }
                        int _min_67 = ((begin_it_4) < (T) ? (begin_it_4) : (T));
                        prow = _min_67;
                        int _min_68 = ((begin_it_4 + span_4) < (T) ? (begin_it_4 + span_4) : (T));
                        pend = _min_68;
                        unsigned int _atomic_fetch_add_69 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_69;
                    }
                }
            }
            int nval_2_1 = 0;
            if (prow < pend) {
                int _min_70 = ((pend - prow) < (1) ? (pend - prow) : (1));
                nval_2_1 = _min_70;
            }
            meta[2 * slot_2] = prow;
            meta[2 * slot_2 + 1] = nval_2_1;
            if (nval_2_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8, 24576);
                int sbase_3 = ring_at + slot_2 * 24576;
                int _min_71 = ((nval_2_1 - 1) < (0) ? (nval_2_1 - 1) : (0));
                long long row_i_3 = (long long)(prow + _min_71);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
            } else {
                mbarrier_arrive(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8);
            }
            prow = prow + nval_2_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 6144.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row0 + rg < row0 + nval_2) {
            float out_0[8];
            float g0_0 = __uint_as_float(gw[0] << 16);
            float g1_0 = __uint_as_float(gw[0] & 4294901760u);
            float x0_0 = __uint_as_float(xw[0] << 16);
            float x1_0 = __uint_as_float(xw[0] & 4294901760u);
            float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
            float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
            out_0[0] = y0_0;
            out_0[1] = y1_0;
            float g0_1 = __uint_as_float(gw[1] << 16);
            float g1_1 = __uint_as_float(gw[1] & 4294901760u);
            float x0_1 = __uint_as_float(xw[1] << 16);
            float x1_1 = __uint_as_float(xw[1] & 4294901760u);
            float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
            float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
            out_0[2] = y0_1;
            out_0[3] = y1_1;
            float g0_2 = __uint_as_float(gw[2] << 16);
            float g1_2 = __uint_as_float(gw[2] & 4294901760u);
            float x0_2 = __uint_as_float(xw[2] << 16);
            float x1_2 = __uint_as_float(xw[2] & 4294901760u);
            float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
            float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
            out_0[4] = y0_2;
            out_0[5] = y1_2;
            float g0_3 = __uint_as_float(gw[3] << 16);
            float g1_3 = __uint_as_float(gw[3] & 4294901760u);
            float x0_3 = __uint_as_float(xw[3] << 16);
            float x1_3 = __uint_as_float(xw[3] & 4294901760u);
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
            float out_1[8];
            float g0_4 = __uint_as_float(gw[4] << 16);
            float g1_4 = __uint_as_float(gw[4] & 4294901760u);
            float x0_4 = __uint_as_float(xw[4] << 16);
            float x1_4 = __uint_as_float(xw[4] & 4294901760u);
            float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(ww[4] << 16)));
            float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(ww[4] & 4294901760u)));
            out_1[0] = y0_4;
            out_1[1] = y1_4;
            float g0_5 = __uint_as_float(gw[5] << 16);
            float g1_5 = __uint_as_float(gw[5] & 4294901760u);
            float x0_5 = __uint_as_float(xw[5] << 16);
            float x1_5 = __uint_as_float(xw[5] & 4294901760u);
            float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(ww[5] << 16)));
            float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(ww[5] & 4294901760u)));
            out_1[2] = y0_5;
            out_1[3] = y1_5;
            float g0_6 = __uint_as_float(gw[6] << 16);
            float g1_6 = __uint_as_float(gw[6] & 4294901760u);
            float x0_6 = __uint_as_float(xw[6] << 16);
            float x1_6 = __uint_as_float(xw[6] & 4294901760u);
            float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(ww[6] << 16)));
            float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(ww[6] & 4294901760u)));
            out_1[4] = y0_6;
            out_1[5] = y1_6;
            float g0_7 = __uint_as_float(gw[7] << 16);
            float g1_7 = __uint_as_float(gw[7] & 4294901760u);
            float x0_7 = __uint_as_float(xw[7] << 16);
            float x1_7 = __uint_as_float(xw[7] & 4294901760u);
            float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * __uint_as_float(ww[7] << 16)));
            float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * __uint_as_float(ww[7] & 4294901760u)));
            out_1[6] = y0_7;
            out_1[7] = y1_7;
            unsigned int packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        row0 = row0_n;
        nval_2 = nval_n;
        rr[0] = rr[1];
    }
    c = pc;
    if (tid == 0) {
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
