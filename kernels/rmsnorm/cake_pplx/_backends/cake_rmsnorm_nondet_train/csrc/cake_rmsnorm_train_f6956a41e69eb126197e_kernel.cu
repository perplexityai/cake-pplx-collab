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



extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_f6956a41e69eb126197e(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    float* sums = reinterpret_cast<float*>(smem_raw + 1120);
    const int sums_addr = smem + 1120;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1120);
    const int dwbuf_addr = smem + 1120;
    int* meta = reinterpret_cast<int*>(smem_raw + 1088);
    const int meta_addr = smem + 1088;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 21632);
    const int ring_addr = smem + 21632;

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
    int rg = warp;
    int tir = tid - rg * 32;
    float ww[40];
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
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)(tir * 8)))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_1_0);
        ww[0 + 1] = __uint_as_float(_ldv8_1_1);
        ww[0 + 2] = __uint_as_float(_ldv8_1_2);
        ww[0 + 3] = __uint_as_float(_ldv8_1_3);
        ww[0 + 4] = __uint_as_float(_ldv8_1_4);
        ww[0 + 5] = __uint_as_float(_ldv8_1_5);
        ww[0 + 6] = __uint_as_float(_ldv8_1_6);
        ww[0 + 7] = __uint_as_float(_ldv8_1_7);
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
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + ((unsigned long long)((32 + tir) * 8)))) : "memory");
        ww[8 + 0] = __uint_as_float(_ldv8_2_0);
        ww[8 + 1] = __uint_as_float(_ldv8_2_1);
        ww[8 + 2] = __uint_as_float(_ldv8_2_2);
        ww[8 + 3] = __uint_as_float(_ldv8_2_3);
        ww[8 + 4] = __uint_as_float(_ldv8_2_4);
        ww[8 + 5] = __uint_as_float(_ldv8_2_5);
        ww[8 + 6] = __uint_as_float(_ldv8_2_6);
        ww[8 + 7] = __uint_as_float(_ldv8_2_7);
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
            : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(w + ((unsigned long long)((64 + tir) * 8)))) : "memory");
        ww[16 + 0] = __uint_as_float(_ldv8_3_0);
        ww[16 + 1] = __uint_as_float(_ldv8_3_1);
        ww[16 + 2] = __uint_as_float(_ldv8_3_2);
        ww[16 + 3] = __uint_as_float(_ldv8_3_3);
        ww[16 + 4] = __uint_as_float(_ldv8_3_4);
        ww[16 + 5] = __uint_as_float(_ldv8_3_5);
        ww[16 + 6] = __uint_as_float(_ldv8_3_6);
        ww[16 + 7] = __uint_as_float(_ldv8_3_7);
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
            : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(w + ((unsigned long long)((96 + tir) * 8)))) : "memory");
        ww[24 + 0] = __uint_as_float(_ldv8_4_0);
        ww[24 + 1] = __uint_as_float(_ldv8_4_1);
        ww[24 + 2] = __uint_as_float(_ldv8_4_2);
        ww[24 + 3] = __uint_as_float(_ldv8_4_3);
        ww[24 + 4] = __uint_as_float(_ldv8_4_4);
        ww[24 + 5] = __uint_as_float(_ldv8_4_5);
        ww[24 + 6] = __uint_as_float(_ldv8_4_6);
        ww[24 + 7] = __uint_as_float(_ldv8_4_7);
    }
    {
        unsigned _ldv8_5_0;
        unsigned _ldv8_5_1;
        unsigned _ldv8_5_2;
        unsigned _ldv8_5_3;
        unsigned _ldv8_5_4;
        unsigned _ldv8_5_5;
        unsigned _ldv8_5_6;
        unsigned _ldv8_5_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(w + ((unsigned long long)((128 + tir) * 8)))) : "memory");
        ww[32 + 0] = __uint_as_float(_ldv8_5_0);
        ww[32 + 1] = __uint_as_float(_ldv8_5_1);
        ww[32 + 2] = __uint_as_float(_ldv8_5_2);
        ww[32 + 3] = __uint_as_float(_ldv8_5_3);
        ww[32 + 4] = __uint_as_float(_ldv8_5_4);
        ww[32 + 5] = __uint_as_float(_ldv8_5_5);
        ww[32 + 6] = __uint_as_float(_ldv8_5_6);
        ww[32 + 7] = __uint_as_float(_ldv8_5_7);
    }
    float acc[40];
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
    int c = 0;
    int first_c = 0;
    sums[rg * 1280 + tir] = 0.0f;
    sums[rg * 1280 + 32 + tir] = 0.0f;
    sums[rg * 1280 + 64 + tir] = 0.0f;
    sums[rg * 1280 + 96 + tir] = 0.0f;
    sums[rg * 1280 + 128 + tir] = 0.0f;
    sums[rg * 1280 + 160 + tir] = 0.0f;
    sums[rg * 1280 + 192 + tir] = 0.0f;
    sums[rg * 1280 + 224 + tir] = 0.0f;
    sums[rg * 1280 + 256 + tir] = 0.0f;
    sums[rg * 1280 + 288 + tir] = 0.0f;
    sums[rg * 1280 + 320 + tir] = 0.0f;
    sums[rg * 1280 + 352 + tir] = 0.0f;
    sums[rg * 1280 + 384 + tir] = 0.0f;
    sums[rg * 1280 + 416 + tir] = 0.0f;
    sums[rg * 1280 + 448 + tir] = 0.0f;
    sums[rg * 1280 + 480 + tir] = 0.0f;
    sums[rg * 1280 + 512 + tir] = 0.0f;
    sums[rg * 1280 + 544 + tir] = 0.0f;
    sums[rg * 1280 + 576 + tir] = 0.0f;
    sums[rg * 1280 + 608 + tir] = 0.0f;
    sums[rg * 1280 + 640 + tir] = 0.0f;
    sums[rg * 1280 + 672 + tir] = 0.0f;
    sums[rg * 1280 + 704 + tir] = 0.0f;
    sums[rg * 1280 + 736 + tir] = 0.0f;
    sums[rg * 1280 + 768 + tir] = 0.0f;
    sums[rg * 1280 + 800 + tir] = 0.0f;
    sums[rg * 1280 + 832 + tir] = 0.0f;
    sums[rg * 1280 + 864 + tir] = 0.0f;
    sums[rg * 1280 + 896 + tir] = 0.0f;
    sums[rg * 1280 + 928 + tir] = 0.0f;
    sums[rg * 1280 + 960 + tir] = 0.0f;
    sums[rg * 1280 + 992 + tir] = 0.0f;
    sums[rg * 1280 + 1024 + tir] = 0.0f;
    sums[rg * 1280 + 1056 + tir] = 0.0f;
    sums[rg * 1280 + 1088 + tir] = 0.0f;
    sums[rg * 1280 + 1120 + tir] = 0.0f;
    sums[rg * 1280 + 1152 + tir] = 0.0f;
    sums[rg * 1280 + 1184 + tir] = 0.0f;
    sums[rg * 1280 + 1216 + tir] = 0.0f;
    sums[rg * 1280 + 1248 + tir] = 0.0f;
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[20];
    unsigned int xw[20];
    unsigned int hw[1];
    float rr[2];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        pc = (int)nxt;
        first_c = pc;
        if (pc < n_chunks) {
            int stages = (T + 3) / 4;
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
            int _min_6 = ((begin_it * 4) < (T) ? (begin_it * 4) : (T));
            prow = _min_6;
            int _min_7 = ((begin_it * 4 + span * 4) < (T) ? (begin_it * 4 + span * 4) : (T));
            pend = _min_7;
            unsigned int _atomic_fetch_add_8 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_8;
        }
        int slot = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_1 = (T + 3) / 4;
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
                    int _min_9 = ((begin_it_1 * 4) < (T) ? (begin_it_1 * 4) : (T));
                    prow = _min_9;
                    int _min_10 = ((begin_it_1 * 4 + span_1 * 4) < (T) ? (begin_it_1 * 4 + span_1 * 4) : (T));
                    pend = _min_10;
                    unsigned int _atomic_fetch_add_11 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_11;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_12 = ((pend - prow) < (4) ? (pend - prow) : (4));
            nval = _min_12;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 20480);
            int sbase = ring_at + slot * 20480;
            int _min_13 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_13);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_14 = ((nval - 1) < (1) ? (nval - 1) : (1));
            long long row_i_1 = (long long)(prow + _min_14);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 2560), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12800), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_15 = ((nval - 1) < (2) ? (nval - 1) : (2));
            long long row_i_2 = (long long)(prow + _min_15);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 5120), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 15360), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_16 = ((nval - 1) < (3) ? (nval - 1) : (3));
            long long row_i_3 = (long long)(prow + _min_16);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 7680), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 17920), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr);
        }
        prow = prow + nval;
        int slot_0 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_2 = (T + 3) / 4;
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
                    int _min_17 = ((begin_it_2 * 4) < (T) ? (begin_it_2 * 4) : (T));
                    prow = _min_17;
                    int _min_18 = ((begin_it_2 * 4 + span_2 * 4) < (T) ? (begin_it_2 * 4 + span_2 * 4) : (T));
                    pend = _min_18;
                    unsigned int _atomic_fetch_add_19 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_19;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_20 = ((pend - prow) < (4) ? (pend - prow) : (4));
            nval_0 = _min_20;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 20480);
            int sbase_1 = ring_at + slot_0 * 20480;
            int _min_21 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_4 = (long long)(prow + _min_21);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_4 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_4 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_22 = ((nval_0 - 1) < (1) ? (nval_0 - 1) : (1));
            long long row_i_5 = (long long)(prow + _min_22);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 2560), "l"(reinterpret_cast<const uint8_t*>(g + row_i_5 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12800), "l"(reinterpret_cast<const uint8_t*>(x + row_i_5 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_23 = ((nval_0 - 1) < (2) ? (nval_0 - 1) : (2));
            long long row_i_6 = (long long)(prow + _min_23);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 5120), "l"(reinterpret_cast<const uint8_t*>(g + row_i_6 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 15360), "l"(reinterpret_cast<const uint8_t*>(x + row_i_6 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_24 = ((nval_0 - 1) < (3) ? (nval_0 - 1) : (3));
            long long row_i_7 = (long long)(prow + _min_24);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 7680), "l"(reinterpret_cast<const uint8_t*>(g + row_i_7 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 17920), "l"(reinterpret_cast<const uint8_t*>(x + row_i_7 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr + 8);
        }
        prow = prow + nval_0;
        int slot_1 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_3 = (T + 3) / 4;
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
                    int _min_25 = ((begin_it_3 * 4) < (T) ? (begin_it_3 * 4) : (T));
                    prow = _min_25;
                    int _min_26 = ((begin_it_3 * 4 + span_3 * 4) < (T) ? (begin_it_3 * 4 + span_3 * 4) : (T));
                    pend = _min_26;
                    unsigned int _atomic_fetch_add_27 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_27;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_28 = ((pend - prow) < (4) ? (pend - prow) : (4));
            nval_1 = _min_28;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 20480);
            int sbase_2 = ring_at + slot_1 * 20480;
            int _min_29 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_8 = (long long)(prow + _min_29);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_8 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_8 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_30 = ((nval_1 - 1) < (1) ? (nval_1 - 1) : (1));
            long long row_i_9 = (long long)(prow + _min_30);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 2560), "l"(reinterpret_cast<const uint8_t*>(g + row_i_9 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12800), "l"(reinterpret_cast<const uint8_t*>(x + row_i_9 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_31 = ((nval_1 - 1) < (2) ? (nval_1 - 1) : (2));
            long long row_i_10 = (long long)(prow + _min_31);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 5120), "l"(reinterpret_cast<const uint8_t*>(g + row_i_10 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 15360), "l"(reinterpret_cast<const uint8_t*>(x + row_i_10 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_32 = ((nval_1 - 1) < (3) ? (nval_1 - 1) : (3));
            long long row_i_11 = (long long)(prow + _min_32);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 7680), "l"(reinterpret_cast<const uint8_t*>(g + row_i_11 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 17920), "l"(reinterpret_cast<const uint8_t*>(x + row_i_11 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
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
        int rbase = ring_at + k_4 % 3 * 20480 + rg * 2560;
        __int128_t _smem_raw_33;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_33) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[3];
        __int128_t _smem_raw_34;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_34) : "r"(rbase + 10240 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[3];
        __int128_t _smem_raw_35;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_35) : "r"(rbase + (32 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[3];
        __int128_t _smem_raw_36;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_36) : "r"(rbase + 10240 + (32 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[3];
        __int128_t _smem_raw_37;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_37) : "r"(rbase + (64 + tir) * 8 * 2));
        gw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[0];
        gw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[1];
        gw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[2];
        gw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[3];
        __int128_t _smem_raw_38;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_38) : "r"(rbase + 10240 + (64 + tir) * 8 * 2));
        xw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[0];
        xw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[1];
        xw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[2];
        xw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[3];
        __int128_t _smem_raw_39;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_39) : "r"(rbase + (96 + tir) * 8 * 2));
        gw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[0];
        gw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[1];
        gw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[2];
        gw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[3];
        __int128_t _smem_raw_40;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_40) : "r"(rbase + 10240 + (96 + tir) * 8 * 2));
        xw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[0];
        xw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[1];
        xw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[2];
        xw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[3];
        __int128_t _smem_raw_41;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_41) : "r"(rbase + (128 + tir) * 8 * 2));
        gw[16] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[0];
        gw[17] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[1];
        gw[18] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[2];
        gw[19] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[3];
        __int128_t _smem_raw_42;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_42) : "r"(rbase + 10240 + (128 + tir) * 8 * 2));
        xw[16] = reinterpret_cast<const uint32_t*>(&_smem_raw_42)[0];
        xw[17] = reinterpret_cast<const uint32_t*>(&_smem_raw_42)[1];
        xw[18] = reinterpret_cast<const uint32_t*>(&_smem_raw_42)[2];
        xw[19] = reinterpret_cast<const uint32_t*>(&_smem_raw_42)[3];
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
        dq = __fmaf_rn(__uint_as_float(gw[8] << 16) * ww[16], __uint_as_float(xw[8] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[8] & 4294901760u) * ww[17], __uint_as_float(xw[8] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[9] << 16) * ww[18], __uint_as_float(xw[9] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[9] & 4294901760u) * ww[19], __uint_as_float(xw[9] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[10] << 16) * ww[20], __uint_as_float(xw[10] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[10] & 4294901760u) * ww[21], __uint_as_float(xw[10] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[11] << 16) * ww[22], __uint_as_float(xw[11] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[11] & 4294901760u) * ww[23], __uint_as_float(xw[11] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[12] << 16) * ww[24], __uint_as_float(xw[12] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[12] & 4294901760u) * ww[25], __uint_as_float(xw[12] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[13] << 16) * ww[26], __uint_as_float(xw[13] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[13] & 4294901760u) * ww[27], __uint_as_float(xw[13] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[14] << 16) * ww[28], __uint_as_float(xw[14] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[14] & 4294901760u) * ww[29], __uint_as_float(xw[14] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[15] << 16) * ww[30], __uint_as_float(xw[15] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[15] & 4294901760u) * ww[31], __uint_as_float(xw[15] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[16] << 16) * ww[32], __uint_as_float(xw[16] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[16] & 4294901760u) * ww[33], __uint_as_float(xw[16] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[17] << 16) * ww[34], __uint_as_float(xw[17] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[17] & 4294901760u) * ww[35], __uint_as_float(xw[17] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[18] << 16) * ww[36], __uint_as_float(xw[18] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[18] & 4294901760u) * ww[37], __uint_as_float(xw[18] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[19] << 16) * ww[38], __uint_as_float(xw[19] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[19] & 4294901760u) * ww[39], __uint_as_float(xw[19] & 4294901760u), dq);
        float _warp_reduce_43 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_43 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_43, offset);
        dot[0] = _warp_reduce_43;
        __syncthreads();
        if (tid == 0) {
            int slot_2 = (k_4 + 3) % 3;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt;
                    if (pc < n_chunks) {
                        int stages_4 = (T + 3) / 4;
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
                        int _min_44 = ((begin_it_4 * 4) < (T) ? (begin_it_4 * 4) : (T));
                        prow = _min_44;
                        int _min_45 = ((begin_it_4 * 4 + span_4 * 4) < (T) ? (begin_it_4 * 4 + span_4 * 4) : (T));
                        pend = _min_45;
                        unsigned int _atomic_fetch_add_46 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_46;
                    }
                }
            }
            int nval_2_1 = 0;
            if (prow < pend) {
                int _min_47 = ((pend - prow) < (4) ? (pend - prow) : (4));
                nval_2_1 = _min_47;
            }
            meta[2 * slot_2] = prow;
            meta[2 * slot_2 + 1] = nval_2_1;
            if (nval_2_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8, 20480);
                int sbase_3 = ring_at + slot_2 * 20480;
                int _min_48 = ((nval_2_1 - 1) < (0) ? (nval_2_1 - 1) : (0));
                long long row_i_12 = (long long)(prow + _min_48);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_12 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_12 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                int _min_49 = ((nval_2_1 - 1) < (1) ? (nval_2_1 - 1) : (1));
                long long row_i_13 = (long long)(prow + _min_49);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 2560), "l"(reinterpret_cast<const uint8_t*>(g + row_i_13 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 12800), "l"(reinterpret_cast<const uint8_t*>(x + row_i_13 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                int _min_50 = ((nval_2_1 - 1) < (2) ? (nval_2_1 - 1) : (2));
                long long row_i_14 = (long long)(prow + _min_50);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 5120), "l"(reinterpret_cast<const uint8_t*>(g + row_i_14 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 15360), "l"(reinterpret_cast<const uint8_t*>(x + row_i_14 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                int _min_51 = ((nval_2_1 - 1) < (3) ? (nval_2_1 - 1) : (3));
                long long row_i_15 = (long long)(prow + _min_51);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 7680), "l"(reinterpret_cast<const uint8_t*>(g + row_i_15 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 17920), "l"(reinterpret_cast<const uint8_t*>(x + row_i_15 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
            } else {
                mbarrier_arrive(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8);
            }
            prow = prow + nval_2_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 1280.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row0 + rg < row0 + nval_2) {
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1280ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1280ull + (unsigned long long)((32 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            float out_2[8];
            float g0_8 = __uint_as_float(gw[8] << 16);
            float g1_8 = __uint_as_float(gw[8] & 4294901760u);
            float x0_8 = __uint_as_float(xw[8] << 16);
            float x1_8 = __uint_as_float(xw[8] & 4294901760u);
            acc[16] = __fmaf_rn(g0_8 * x0_8, rq, acc[16]);
            acc[17] = __fmaf_rn(g1_8 * x1_8, rq, acc[17]);
            float y0_8 = __fmaf_rn(nkx, x0_8, rq * (g0_8 * ww[16]));
            float y1_8 = __fmaf_rn(nkx, x1_8, rq * (g1_8 * ww[17]));
            out_2[0] = y0_8;
            out_2[1] = y1_8;
            float g0_9 = __uint_as_float(gw[9] << 16);
            float g1_9 = __uint_as_float(gw[9] & 4294901760u);
            float x0_9 = __uint_as_float(xw[9] << 16);
            float x1_9 = __uint_as_float(xw[9] & 4294901760u);
            acc[18] = __fmaf_rn(g0_9 * x0_9, rq, acc[18]);
            acc[19] = __fmaf_rn(g1_9 * x1_9, rq, acc[19]);
            float y0_9 = __fmaf_rn(nkx, x0_9, rq * (g0_9 * ww[18]));
            float y1_9 = __fmaf_rn(nkx, x1_9, rq * (g1_9 * ww[19]));
            out_2[2] = y0_9;
            out_2[3] = y1_9;
            float g0_10 = __uint_as_float(gw[10] << 16);
            float g1_10 = __uint_as_float(gw[10] & 4294901760u);
            float x0_10 = __uint_as_float(xw[10] << 16);
            float x1_10 = __uint_as_float(xw[10] & 4294901760u);
            acc[20] = __fmaf_rn(g0_10 * x0_10, rq, acc[20]);
            acc[21] = __fmaf_rn(g1_10 * x1_10, rq, acc[21]);
            float y0_10 = __fmaf_rn(nkx, x0_10, rq * (g0_10 * ww[20]));
            float y1_10 = __fmaf_rn(nkx, x1_10, rq * (g1_10 * ww[21]));
            out_2[4] = y0_10;
            out_2[5] = y1_10;
            float g0_11 = __uint_as_float(gw[11] << 16);
            float g1_11 = __uint_as_float(gw[11] & 4294901760u);
            float x0_11 = __uint_as_float(xw[11] << 16);
            float x1_11 = __uint_as_float(xw[11] & 4294901760u);
            acc[22] = __fmaf_rn(g0_11 * x0_11, rq, acc[22]);
            acc[23] = __fmaf_rn(g1_11 * x1_11, rq, acc[23]);
            float y0_11 = __fmaf_rn(nkx, x0_11, rq * (g0_11 * ww[22]));
            float y1_11 = __fmaf_rn(nkx, x1_11, rq * (g1_11 * ww[23]));
            out_2[6] = y0_11;
            out_2[7] = y1_11;
            unsigned int packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1280ull + (unsigned long long)((64 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
            float out_3[8];
            float g0_12 = __uint_as_float(gw[12] << 16);
            float g1_12 = __uint_as_float(gw[12] & 4294901760u);
            float x0_12 = __uint_as_float(xw[12] << 16);
            float x1_12 = __uint_as_float(xw[12] & 4294901760u);
            acc[24] = __fmaf_rn(g0_12 * x0_12, rq, acc[24]);
            acc[25] = __fmaf_rn(g1_12 * x1_12, rq, acc[25]);
            float y0_12 = __fmaf_rn(nkx, x0_12, rq * (g0_12 * ww[24]));
            float y1_12 = __fmaf_rn(nkx, x1_12, rq * (g1_12 * ww[25]));
            out_3[0] = y0_12;
            out_3[1] = y1_12;
            float g0_13 = __uint_as_float(gw[13] << 16);
            float g1_13 = __uint_as_float(gw[13] & 4294901760u);
            float x0_13 = __uint_as_float(xw[13] << 16);
            float x1_13 = __uint_as_float(xw[13] & 4294901760u);
            acc[26] = __fmaf_rn(g0_13 * x0_13, rq, acc[26]);
            acc[27] = __fmaf_rn(g1_13 * x1_13, rq, acc[27]);
            float y0_13 = __fmaf_rn(nkx, x0_13, rq * (g0_13 * ww[26]));
            float y1_13 = __fmaf_rn(nkx, x1_13, rq * (g1_13 * ww[27]));
            out_3[2] = y0_13;
            out_3[3] = y1_13;
            float g0_14 = __uint_as_float(gw[14] << 16);
            float g1_14 = __uint_as_float(gw[14] & 4294901760u);
            float x0_14 = __uint_as_float(xw[14] << 16);
            float x1_14 = __uint_as_float(xw[14] & 4294901760u);
            acc[28] = __fmaf_rn(g0_14 * x0_14, rq, acc[28]);
            acc[29] = __fmaf_rn(g1_14 * x1_14, rq, acc[29]);
            float y0_14 = __fmaf_rn(nkx, x0_14, rq * (g0_14 * ww[28]));
            float y1_14 = __fmaf_rn(nkx, x1_14, rq * (g1_14 * ww[29]));
            out_3[4] = y0_14;
            out_3[5] = y1_14;
            float g0_15 = __uint_as_float(gw[15] << 16);
            float g1_15 = __uint_as_float(gw[15] & 4294901760u);
            float x0_15 = __uint_as_float(xw[15] << 16);
            float x1_15 = __uint_as_float(xw[15] & 4294901760u);
            acc[30] = __fmaf_rn(g0_15 * x0_15, rq, acc[30]);
            acc[31] = __fmaf_rn(g1_15 * x1_15, rq, acc[31]);
            float y0_15 = __fmaf_rn(nkx, x0_15, rq * (g0_15 * ww[30]));
            float y1_15 = __fmaf_rn(nkx, x1_15, rq * (g1_15 * ww[31]));
            out_3[6] = y0_15;
            out_3[7] = y1_15;
            unsigned int packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1280ull + (unsigned long long)((96 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
            float out_4[8];
            float g0_16 = __uint_as_float(gw[16] << 16);
            float g1_16 = __uint_as_float(gw[16] & 4294901760u);
            float x0_16 = __uint_as_float(xw[16] << 16);
            float x1_16 = __uint_as_float(xw[16] & 4294901760u);
            acc[32] = __fmaf_rn(g0_16 * x0_16, rq, acc[32]);
            acc[33] = __fmaf_rn(g1_16 * x1_16, rq, acc[33]);
            float y0_16 = __fmaf_rn(nkx, x0_16, rq * (g0_16 * ww[32]));
            float y1_16 = __fmaf_rn(nkx, x1_16, rq * (g1_16 * ww[33]));
            out_4[0] = y0_16;
            out_4[1] = y1_16;
            float g0_17 = __uint_as_float(gw[17] << 16);
            float g1_17 = __uint_as_float(gw[17] & 4294901760u);
            float x0_17 = __uint_as_float(xw[17] << 16);
            float x1_17 = __uint_as_float(xw[17] & 4294901760u);
            acc[34] = __fmaf_rn(g0_17 * x0_17, rq, acc[34]);
            acc[35] = __fmaf_rn(g1_17 * x1_17, rq, acc[35]);
            float y0_17 = __fmaf_rn(nkx, x0_17, rq * (g0_17 * ww[34]));
            float y1_17 = __fmaf_rn(nkx, x1_17, rq * (g1_17 * ww[35]));
            out_4[2] = y0_17;
            out_4[3] = y1_17;
            float g0_18 = __uint_as_float(gw[18] << 16);
            float g1_18 = __uint_as_float(gw[18] & 4294901760u);
            float x0_18 = __uint_as_float(xw[18] << 16);
            float x1_18 = __uint_as_float(xw[18] & 4294901760u);
            acc[36] = __fmaf_rn(g0_18 * x0_18, rq, acc[36]);
            acc[37] = __fmaf_rn(g1_18 * x1_18, rq, acc[37]);
            float y0_18 = __fmaf_rn(nkx, x0_18, rq * (g0_18 * ww[36]));
            float y1_18 = __fmaf_rn(nkx, x1_18, rq * (g1_18 * ww[37]));
            out_4[4] = y0_18;
            out_4[5] = y1_18;
            float g0_19 = __uint_as_float(gw[19] << 16);
            float g1_19 = __uint_as_float(gw[19] & 4294901760u);
            float x0_19 = __uint_as_float(xw[19] << 16);
            float x1_19 = __uint_as_float(xw[19] & 4294901760u);
            acc[38] = __fmaf_rn(g0_19 * x0_19, rq, acc[38]);
            acc[39] = __fmaf_rn(g1_19 * x1_19, rq, acc[39]);
            float y0_19 = __fmaf_rn(nkx, x0_19, rq * (g0_19 * ww[38]));
            float y1_19 = __fmaf_rn(nkx, x1_19, rq * (g1_19 * ww[39]));
            out_4[6] = y0_19;
            out_4[7] = y1_19;
            unsigned int packed_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 0], out_4[_lp*2+1 + 0]));
                packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1280ull + (unsigned long long)((128 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_4)[0];
        }
        row0 = row0_n;
        nval_2 = nval_n;
        rr[0] = rr[1];
        if ((k_4 + 1 & 15) == 0) {
            sums[rg * 1280 + tir] = sums[rg * 1280 + tir] + acc[0];
            acc[0] = 0.0f;
            sums[rg * 1280 + 32 + tir] = sums[rg * 1280 + 32 + tir] + acc[1];
            acc[1] = 0.0f;
            sums[rg * 1280 + 64 + tir] = sums[rg * 1280 + 64 + tir] + acc[2];
            acc[2] = 0.0f;
            sums[rg * 1280 + 96 + tir] = sums[rg * 1280 + 96 + tir] + acc[3];
            acc[3] = 0.0f;
            sums[rg * 1280 + 128 + tir] = sums[rg * 1280 + 128 + tir] + acc[4];
            acc[4] = 0.0f;
            sums[rg * 1280 + 160 + tir] = sums[rg * 1280 + 160 + tir] + acc[5];
            acc[5] = 0.0f;
            sums[rg * 1280 + 192 + tir] = sums[rg * 1280 + 192 + tir] + acc[6];
            acc[6] = 0.0f;
            sums[rg * 1280 + 224 + tir] = sums[rg * 1280 + 224 + tir] + acc[7];
            acc[7] = 0.0f;
            sums[rg * 1280 + 256 + tir] = sums[rg * 1280 + 256 + tir] + acc[8];
            acc[8] = 0.0f;
            sums[rg * 1280 + 288 + tir] = sums[rg * 1280 + 288 + tir] + acc[9];
            acc[9] = 0.0f;
            sums[rg * 1280 + 320 + tir] = sums[rg * 1280 + 320 + tir] + acc[10];
            acc[10] = 0.0f;
            sums[rg * 1280 + 352 + tir] = sums[rg * 1280 + 352 + tir] + acc[11];
            acc[11] = 0.0f;
            sums[rg * 1280 + 384 + tir] = sums[rg * 1280 + 384 + tir] + acc[12];
            acc[12] = 0.0f;
            sums[rg * 1280 + 416 + tir] = sums[rg * 1280 + 416 + tir] + acc[13];
            acc[13] = 0.0f;
            sums[rg * 1280 + 448 + tir] = sums[rg * 1280 + 448 + tir] + acc[14];
            acc[14] = 0.0f;
            sums[rg * 1280 + 480 + tir] = sums[rg * 1280 + 480 + tir] + acc[15];
            acc[15] = 0.0f;
            sums[rg * 1280 + 512 + tir] = sums[rg * 1280 + 512 + tir] + acc[16];
            acc[16] = 0.0f;
            sums[rg * 1280 + 544 + tir] = sums[rg * 1280 + 544 + tir] + acc[17];
            acc[17] = 0.0f;
            sums[rg * 1280 + 576 + tir] = sums[rg * 1280 + 576 + tir] + acc[18];
            acc[18] = 0.0f;
            sums[rg * 1280 + 608 + tir] = sums[rg * 1280 + 608 + tir] + acc[19];
            acc[19] = 0.0f;
            sums[rg * 1280 + 640 + tir] = sums[rg * 1280 + 640 + tir] + acc[20];
            acc[20] = 0.0f;
            sums[rg * 1280 + 672 + tir] = sums[rg * 1280 + 672 + tir] + acc[21];
            acc[21] = 0.0f;
            sums[rg * 1280 + 704 + tir] = sums[rg * 1280 + 704 + tir] + acc[22];
            acc[22] = 0.0f;
            sums[rg * 1280 + 736 + tir] = sums[rg * 1280 + 736 + tir] + acc[23];
            acc[23] = 0.0f;
            sums[rg * 1280 + 768 + tir] = sums[rg * 1280 + 768 + tir] + acc[24];
            acc[24] = 0.0f;
            sums[rg * 1280 + 800 + tir] = sums[rg * 1280 + 800 + tir] + acc[25];
            acc[25] = 0.0f;
            sums[rg * 1280 + 832 + tir] = sums[rg * 1280 + 832 + tir] + acc[26];
            acc[26] = 0.0f;
            sums[rg * 1280 + 864 + tir] = sums[rg * 1280 + 864 + tir] + acc[27];
            acc[27] = 0.0f;
            sums[rg * 1280 + 896 + tir] = sums[rg * 1280 + 896 + tir] + acc[28];
            acc[28] = 0.0f;
            sums[rg * 1280 + 928 + tir] = sums[rg * 1280 + 928 + tir] + acc[29];
            acc[29] = 0.0f;
            sums[rg * 1280 + 960 + tir] = sums[rg * 1280 + 960 + tir] + acc[30];
            acc[30] = 0.0f;
            sums[rg * 1280 + 992 + tir] = sums[rg * 1280 + 992 + tir] + acc[31];
            acc[31] = 0.0f;
            sums[rg * 1280 + 1024 + tir] = sums[rg * 1280 + 1024 + tir] + acc[32];
            acc[32] = 0.0f;
            sums[rg * 1280 + 1056 + tir] = sums[rg * 1280 + 1056 + tir] + acc[33];
            acc[33] = 0.0f;
            sums[rg * 1280 + 1088 + tir] = sums[rg * 1280 + 1088 + tir] + acc[34];
            acc[34] = 0.0f;
            sums[rg * 1280 + 1120 + tir] = sums[rg * 1280 + 1120 + tir] + acc[35];
            acc[35] = 0.0f;
            sums[rg * 1280 + 1152 + tir] = sums[rg * 1280 + 1152 + tir] + acc[36];
            acc[36] = 0.0f;
            sums[rg * 1280 + 1184 + tir] = sums[rg * 1280 + 1184 + tir] + acc[37];
            acc[37] = 0.0f;
            sums[rg * 1280 + 1216 + tir] = sums[rg * 1280 + 1216 + tir] + acc[38];
            acc[38] = 0.0f;
            sums[rg * 1280 + 1248 + tir] = sums[rg * 1280 + 1248 + tir] + acc[39];
            acc[39] = 0.0f;
        }
    }
    sums[rg * 1280 + tir] = sums[rg * 1280 + tir] + acc[0];
    acc[0] = 0.0f;
    sums[rg * 1280 + 32 + tir] = sums[rg * 1280 + 32 + tir] + acc[1];
    acc[1] = 0.0f;
    sums[rg * 1280 + 64 + tir] = sums[rg * 1280 + 64 + tir] + acc[2];
    acc[2] = 0.0f;
    sums[rg * 1280 + 96 + tir] = sums[rg * 1280 + 96 + tir] + acc[3];
    acc[3] = 0.0f;
    sums[rg * 1280 + 128 + tir] = sums[rg * 1280 + 128 + tir] + acc[4];
    acc[4] = 0.0f;
    sums[rg * 1280 + 160 + tir] = sums[rg * 1280 + 160 + tir] + acc[5];
    acc[5] = 0.0f;
    sums[rg * 1280 + 192 + tir] = sums[rg * 1280 + 192 + tir] + acc[6];
    acc[6] = 0.0f;
    sums[rg * 1280 + 224 + tir] = sums[rg * 1280 + 224 + tir] + acc[7];
    acc[7] = 0.0f;
    sums[rg * 1280 + 256 + tir] = sums[rg * 1280 + 256 + tir] + acc[8];
    acc[8] = 0.0f;
    sums[rg * 1280 + 288 + tir] = sums[rg * 1280 + 288 + tir] + acc[9];
    acc[9] = 0.0f;
    sums[rg * 1280 + 320 + tir] = sums[rg * 1280 + 320 + tir] + acc[10];
    acc[10] = 0.0f;
    sums[rg * 1280 + 352 + tir] = sums[rg * 1280 + 352 + tir] + acc[11];
    acc[11] = 0.0f;
    sums[rg * 1280 + 384 + tir] = sums[rg * 1280 + 384 + tir] + acc[12];
    acc[12] = 0.0f;
    sums[rg * 1280 + 416 + tir] = sums[rg * 1280 + 416 + tir] + acc[13];
    acc[13] = 0.0f;
    sums[rg * 1280 + 448 + tir] = sums[rg * 1280 + 448 + tir] + acc[14];
    acc[14] = 0.0f;
    sums[rg * 1280 + 480 + tir] = sums[rg * 1280 + 480 + tir] + acc[15];
    acc[15] = 0.0f;
    sums[rg * 1280 + 512 + tir] = sums[rg * 1280 + 512 + tir] + acc[16];
    acc[16] = 0.0f;
    sums[rg * 1280 + 544 + tir] = sums[rg * 1280 + 544 + tir] + acc[17];
    acc[17] = 0.0f;
    sums[rg * 1280 + 576 + tir] = sums[rg * 1280 + 576 + tir] + acc[18];
    acc[18] = 0.0f;
    sums[rg * 1280 + 608 + tir] = sums[rg * 1280 + 608 + tir] + acc[19];
    acc[19] = 0.0f;
    sums[rg * 1280 + 640 + tir] = sums[rg * 1280 + 640 + tir] + acc[20];
    acc[20] = 0.0f;
    sums[rg * 1280 + 672 + tir] = sums[rg * 1280 + 672 + tir] + acc[21];
    acc[21] = 0.0f;
    sums[rg * 1280 + 704 + tir] = sums[rg * 1280 + 704 + tir] + acc[22];
    acc[22] = 0.0f;
    sums[rg * 1280 + 736 + tir] = sums[rg * 1280 + 736 + tir] + acc[23];
    acc[23] = 0.0f;
    sums[rg * 1280 + 768 + tir] = sums[rg * 1280 + 768 + tir] + acc[24];
    acc[24] = 0.0f;
    sums[rg * 1280 + 800 + tir] = sums[rg * 1280 + 800 + tir] + acc[25];
    acc[25] = 0.0f;
    sums[rg * 1280 + 832 + tir] = sums[rg * 1280 + 832 + tir] + acc[26];
    acc[26] = 0.0f;
    sums[rg * 1280 + 864 + tir] = sums[rg * 1280 + 864 + tir] + acc[27];
    acc[27] = 0.0f;
    sums[rg * 1280 + 896 + tir] = sums[rg * 1280 + 896 + tir] + acc[28];
    acc[28] = 0.0f;
    sums[rg * 1280 + 928 + tir] = sums[rg * 1280 + 928 + tir] + acc[29];
    acc[29] = 0.0f;
    sums[rg * 1280 + 960 + tir] = sums[rg * 1280 + 960 + tir] + acc[30];
    acc[30] = 0.0f;
    sums[rg * 1280 + 992 + tir] = sums[rg * 1280 + 992 + tir] + acc[31];
    acc[31] = 0.0f;
    sums[rg * 1280 + 1024 + tir] = sums[rg * 1280 + 1024 + tir] + acc[32];
    acc[32] = 0.0f;
    sums[rg * 1280 + 1056 + tir] = sums[rg * 1280 + 1056 + tir] + acc[33];
    acc[33] = 0.0f;
    sums[rg * 1280 + 1088 + tir] = sums[rg * 1280 + 1088 + tir] + acc[34];
    acc[34] = 0.0f;
    sums[rg * 1280 + 1120 + tir] = sums[rg * 1280 + 1120 + tir] + acc[35];
    acc[35] = 0.0f;
    sums[rg * 1280 + 1152 + tir] = sums[rg * 1280 + 1152 + tir] + acc[36];
    acc[36] = 0.0f;
    sums[rg * 1280 + 1184 + tir] = sums[rg * 1280 + 1184 + tir] + acc[37];
    acc[37] = 0.0f;
    sums[rg * 1280 + 1216 + tir] = sums[rg * 1280 + 1216 + tir] + acc[38];
    acc[38] = 0.0f;
    sums[rg * 1280 + 1248 + tir] = sums[rg * 1280 + 1248 + tir] + acc[39];
    acc[39] = 0.0f;
    c = pc;
    int _min_52 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
#if __CUDA_ARCH__ == 1070
    int nrep = ((_min_52) < (25) ? (_min_52) : (25));
#else
    int nrep = ((_min_52) < (2) ? (_min_52) : (2));
#endif
    __syncthreads();
    int at_0 = tid;
    float rv_0[4];
    rv_0[0] = sums[at_0];
    rv_0[1] = sums[1280 + at_0];
    rv_0[2] = sums[2560 + at_0];
    rv_0[3] = sums[3840 + at_0];
    rv_0[0] = rv_0[0] + rv_0[1];
    rv_0[2] = rv_0[2] + rv_0[3];
    rv_0[0] = rv_0[0] + rv_0[2];
    sums[at_0] = rv_0[0];
    int at_1 = 128 + tid;
    float rv_1[4];
    rv_1[0] = sums[at_1];
    rv_1[1] = sums[1280 + at_1];
    rv_1[2] = sums[2560 + at_1];
    rv_1[3] = sums[3840 + at_1];
    rv_1[0] = rv_1[0] + rv_1[1];
    rv_1[2] = rv_1[2] + rv_1[3];
    rv_1[0] = rv_1[0] + rv_1[2];
    sums[at_1] = rv_1[0];
    int at_2 = 256 + tid;
    float rv_2[4];
    rv_2[0] = sums[at_2];
    rv_2[1] = sums[1280 + at_2];
    rv_2[2] = sums[2560 + at_2];
    rv_2[3] = sums[3840 + at_2];
    rv_2[0] = rv_2[0] + rv_2[1];
    rv_2[2] = rv_2[2] + rv_2[3];
    rv_2[0] = rv_2[0] + rv_2[2];
    sums[at_2] = rv_2[0];
    int at_3 = 384 + tid;
    float rv_3[4];
    rv_3[0] = sums[at_3];
    rv_3[1] = sums[1280 + at_3];
    rv_3[2] = sums[2560 + at_3];
    rv_3[3] = sums[3840 + at_3];
    rv_3[0] = rv_3[0] + rv_3[1];
    rv_3[2] = rv_3[2] + rv_3[3];
    rv_3[0] = rv_3[0] + rv_3[2];
    sums[at_3] = rv_3[0];
    int at_4 = 512 + tid;
    float rv_4[4];
    rv_4[0] = sums[at_4];
    rv_4[1] = sums[1280 + at_4];
    rv_4[2] = sums[2560 + at_4];
    rv_4[3] = sums[3840 + at_4];
    rv_4[0] = rv_4[0] + rv_4[1];
    rv_4[2] = rv_4[2] + rv_4[3];
    rv_4[0] = rv_4[0] + rv_4[2];
    sums[at_4] = rv_4[0];
    int at_5 = 640 + tid;
    float rv_5[4];
    rv_5[0] = sums[at_5];
    rv_5[1] = sums[1280 + at_5];
    rv_5[2] = sums[2560 + at_5];
    rv_5[3] = sums[3840 + at_5];
    rv_5[0] = rv_5[0] + rv_5[1];
    rv_5[2] = rv_5[2] + rv_5[3];
    rv_5[0] = rv_5[0] + rv_5[2];
    sums[at_5] = rv_5[0];
    int at_6 = 768 + tid;
    float rv_6[4];
    rv_6[0] = sums[at_6];
    rv_6[1] = sums[1280 + at_6];
    rv_6[2] = sums[2560 + at_6];
    rv_6[3] = sums[3840 + at_6];
    rv_6[0] = rv_6[0] + rv_6[1];
    rv_6[2] = rv_6[2] + rv_6[3];
    rv_6[0] = rv_6[0] + rv_6[2];
    sums[at_6] = rv_6[0];
    int at_7 = 896 + tid;
    float rv_7[4];
    rv_7[0] = sums[at_7];
    rv_7[1] = sums[1280 + at_7];
    rv_7[2] = sums[2560 + at_7];
    rv_7[3] = sums[3840 + at_7];
    rv_7[0] = rv_7[0] + rv_7[1];
    rv_7[2] = rv_7[2] + rv_7[3];
    rv_7[0] = rv_7[0] + rv_7[2];
    sums[at_7] = rv_7[0];
    int at_8 = 1024 + tid;
    float rv_8[4];
    rv_8[0] = sums[at_8];
    rv_8[1] = sums[1280 + at_8];
    rv_8[2] = sums[2560 + at_8];
    rv_8[3] = sums[3840 + at_8];
    rv_8[0] = rv_8[0] + rv_8[1];
    rv_8[2] = rv_8[2] + rv_8[3];
    rv_8[0] = rv_8[0] + rv_8[2];
    sums[at_8] = rv_8[0];
    int at_9 = 1152 + tid;
    float rv_9[4];
    rv_9[0] = sums[at_9];
    rv_9[1] = sums[1280 + at_9];
    rv_9[2] = sums[2560 + at_9];
    rv_9[3] = sums[3840 + at_9];
    rv_9[0] = rv_9[0] + rv_9[1];
    rv_9[2] = rv_9[2] + rv_9[3];
    rv_9[0] = rv_9[0] + rv_9[2];
    sums[at_9] = rv_9[0];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_53 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_53), "r"(sums_addr + 0), "r"((uint32_t)(1280))
                    : "memory");
            }
            {
                void* _cpred_dst_54 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280 + 320));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_54), "r"(sums_addr + 1280), "r"((uint32_t)(1280))
                    : "memory");
            }
            {
                void* _cpred_dst_55 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280 + 640));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_55), "r"(sums_addr + 2560), "r"((uint32_t)(1280))
                    : "memory");
            }
            {
                void* _cpred_dst_56 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280 + 960));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_56), "r"(sums_addr + 3840), "r"((uint32_t)(1280))
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
        float buf[25];
#else
        float buf[40];
#endif
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_57;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_57);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_58;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (1280 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_58);
            }
        }
        buf[2] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_59;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (2560 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (128 + tid))) : "memory");
#endif
                buf[2] = __uint_as_float(_scalar_bits_59);
            }
        }
        buf[3] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (3 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_60;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (3840 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (1408 + tid))) : "memory");
#endif
                buf[3] = __uint_as_float(_scalar_bits_60);
            }
        }
        buf[4] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (4 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_61;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (5120 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (256 + tid))) : "memory");
#endif
                buf[4] = __uint_as_float(_scalar_bits_61);
            }
        }
        buf[5] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (5 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_62;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (6400 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (1536 + tid))) : "memory");
#endif
                buf[5] = __uint_as_float(_scalar_bits_62);
            }
        }
        buf[6] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (6 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_63;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (7680 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (384 + tid))) : "memory");
#endif
                buf[6] = __uint_as_float(_scalar_bits_63);
            }
        }
        buf[7] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (7 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_64;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (8960 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (1664 + tid))) : "memory");
#endif
                buf[7] = __uint_as_float(_scalar_bits_64);
            }
        }
        buf[8] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (8 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_65;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (10240 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (512 + tid))) : "memory");
#endif
                buf[8] = __uint_as_float(_scalar_bits_65);
            }
        }
        buf[9] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (9 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_66;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (11520 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (1792 + tid))) : "memory");
#endif
                buf[9] = __uint_as_float(_scalar_bits_66);
            }
        }
        buf[10] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (10 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_67;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (12800 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (640 + tid))) : "memory");
#endif
                buf[10] = __uint_as_float(_scalar_bits_67);
            }
        }
        buf[11] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (11 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_68;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (14080 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (1920 + tid))) : "memory");
#endif
                buf[11] = __uint_as_float(_scalar_bits_68);
            }
        }
        buf[12] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (12 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_69;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (15360 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (768 + tid))) : "memory");
#endif
                buf[12] = __uint_as_float(_scalar_bits_69);
            }
        }
        buf[13] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (13 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_70;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (16640 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (2048 + tid))) : "memory");
#endif
                buf[13] = __uint_as_float(_scalar_bits_70);
            }
        }
        buf[14] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (14 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_71;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (17920 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (896 + tid))) : "memory");
#endif
                buf[14] = __uint_as_float(_scalar_bits_71);
            }
        }
        buf[15] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (15 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_72;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (19200 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (2176 + tid))) : "memory");
#endif
                buf[15] = __uint_as_float(_scalar_bits_72);
            }
        }
        buf[16] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (16 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_73;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (20480 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (1024 + tid))) : "memory");
#endif
                buf[16] = __uint_as_float(_scalar_bits_73);
            }
        }
        buf[17] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (17 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_74;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (21760 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (2304 + tid))) : "memory");
#endif
                buf[17] = __uint_as_float(_scalar_bits_74);
            }
        }
        buf[18] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (18 < nrep) {
#else
        if (0 < nrep) {
#endif
            {
                uint32_t _scalar_bits_75;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (23040 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (1152 + tid))) : "memory");
#endif
                buf[18] = __uint_as_float(_scalar_bits_75);
            }
        }
        buf[19] = 0.0f;
#if __CUDA_ARCH__ == 1070
        if (19 < nrep) {
#else
        if (1 < nrep) {
#endif
            {
                uint32_t _scalar_bits_76;
                asm volatile("ld.global.cg.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1070
                    : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (24320 + tid))) : "memory");
#else
                    : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (2432 + tid))) : "memory");
#endif
                buf[19] = __uint_as_float(_scalar_bits_76);
            }
        }
#if __CUDA_ARCH__ == 1070
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_77;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (25600 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_77);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_78;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_78) : "l"((const void*)(partial + (26880 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_78);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_79;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_79) : "l"((const void*)(partial + (28160 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_79);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_80;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_80) : "l"((const void*)(partial + (29440 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_80);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_81;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_81) : "l"((const void*)(partial + (30720 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_81);
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
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_0 = tid;
        dwbuf[(at_w_0 / 32 / 8 * 32 + at_w_0 % 32) * 8 + at_w_0 / 32 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_0)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_0)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_0)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_0)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_0)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_0)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_0)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_0)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_0)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_0)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_0)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_0)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_0)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_0)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_0)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_0)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_0)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_0)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_0)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_0)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_0)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_0)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_0)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_0)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_82;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_82) : "l"((const void*)(partial + (128 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_82);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_83;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_83) : "l"((const void*)(partial + (1408 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_83);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_84;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_84) : "l"((const void*)(partial + (2688 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_84);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_85;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_85) : "l"((const void*)(partial + (3968 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_85);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_86;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_86) : "l"((const void*)(partial + (5248 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_86);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_87;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_87) : "l"((const void*)(partial + (6528 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_87);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_88;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_88) : "l"((const void*)(partial + (7808 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_88);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_89;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_89) : "l"((const void*)(partial + (9088 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_89);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_90;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_90) : "l"((const void*)(partial + (10368 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_90);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_91;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_91) : "l"((const void*)(partial + (11648 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_91);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_92;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_92) : "l"((const void*)(partial + (12928 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_92);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_93;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_93) : "l"((const void*)(partial + (14208 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_93);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_94;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_94) : "l"((const void*)(partial + (15488 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_94);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_95;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_95) : "l"((const void*)(partial + (16768 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_95);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_96;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_96) : "l"((const void*)(partial + (18048 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_96);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_97;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_97) : "l"((const void*)(partial + (19328 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_97);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_98;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_98) : "l"((const void*)(partial + (20608 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_98);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_99;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_99) : "l"((const void*)(partial + (21888 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_99);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_100;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_100) : "l"((const void*)(partial + (23168 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_100);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_101;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_101) : "l"((const void*)(partial + (24448 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_101);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_102;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_102) : "l"((const void*)(partial + (25728 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_102);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_103;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_103) : "l"((const void*)(partial + (27008 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_103);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_104;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_104) : "l"((const void*)(partial + (28288 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_104);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_105;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_105) : "l"((const void*)(partial + (29568 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_105);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_106;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_106) : "l"((const void*)(partial + (30848 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_106);
            }
        }
        buf[0] = buf[0] + buf[1];
#endif
        buf[2] = buf[2] + buf[3];
#if __CUDA_ARCH__ == 1070
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
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_1 = 128 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_1 / 32 / 8 * 32 + at_w_1 % 32) * 8 + at_w_1 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_1 / 32 / 8 * 32 + at_w_1 % 32) * 8 + at_w_1 / 32 % 8] = buf[2];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_1)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_1)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_1)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_1)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_1)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_1)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_1)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_1)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_1)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_1)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_1)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_1)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_1)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_1)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_1)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_1)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_1)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_1)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_1)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_1)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_1)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_1)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_1)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_1)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_1)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_107;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_107) : "l"((const void*)(partial + (256 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_107);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_108;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_108) : "l"((const void*)(partial + (1536 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_108);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_109;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_109) : "l"((const void*)(partial + (2816 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_109);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_110;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_110) : "l"((const void*)(partial + (4096 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_110);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_111;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_111) : "l"((const void*)(partial + (5376 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_111);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_112;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_112) : "l"((const void*)(partial + (6656 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_112);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_113;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_113) : "l"((const void*)(partial + (7936 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_113);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_114;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_114) : "l"((const void*)(partial + (9216 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_114);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_115;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_115) : "l"((const void*)(partial + (10496 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_115);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_116;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_116) : "l"((const void*)(partial + (11776 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_116);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_117;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_117) : "l"((const void*)(partial + (13056 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_117);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_118;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_118) : "l"((const void*)(partial + (14336 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_118);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_119;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_119) : "l"((const void*)(partial + (15616 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_119);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_120;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_120) : "l"((const void*)(partial + (16896 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_120);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_121;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_121) : "l"((const void*)(partial + (18176 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_121);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_122;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_122) : "l"((const void*)(partial + (19456 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_122);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_123;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_123) : "l"((const void*)(partial + (20736 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_123);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_124;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_124) : "l"((const void*)(partial + (22016 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_124);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_125;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_125) : "l"((const void*)(partial + (23296 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_125);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_126;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_126) : "l"((const void*)(partial + (24576 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_126);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_127;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_127) : "l"((const void*)(partial + (25856 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_127);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_128;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_128) : "l"((const void*)(partial + (27136 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_128);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_129;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_129) : "l"((const void*)(partial + (28416 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_129);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_130;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_130) : "l"((const void*)(partial + (29696 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_130);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_131;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_131) : "l"((const void*)(partial + (30976 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_131);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
#endif
        buf[4] = buf[4] + buf[5];
#if __CUDA_ARCH__ == 1070
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_2 = 256 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_2 / 32 / 8 * 32 + at_w_2 % 32) * 8 + at_w_2 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_2 / 32 / 8 * 32 + at_w_2 % 32) * 8 + at_w_2 / 32 % 8] = buf[4];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_2)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_2)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_2)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_2)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_2)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_2)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_2)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_2)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_2)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_2)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_2)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_2)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_2)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_2)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_2)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_2)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_2)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_2)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_2)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_2)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_2)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_2)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_2)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_2)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_2)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_132;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_132) : "l"((const void*)(partial + (384 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_132);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_133;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_133) : "l"((const void*)(partial + (1664 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_133);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_134;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_134) : "l"((const void*)(partial + (2944 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_134);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_135;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_135) : "l"((const void*)(partial + (4224 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_135);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_136;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_136) : "l"((const void*)(partial + (5504 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_136);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_137;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_137) : "l"((const void*)(partial + (6784 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_137);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_138;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_138) : "l"((const void*)(partial + (8064 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_138);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_139;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_139) : "l"((const void*)(partial + (9344 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_139);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_140;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_140) : "l"((const void*)(partial + (10624 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_140);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_141;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_141) : "l"((const void*)(partial + (11904 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_141);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_142;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_142) : "l"((const void*)(partial + (13184 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_142);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_143;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_143) : "l"((const void*)(partial + (14464 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_143);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_144;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_144) : "l"((const void*)(partial + (15744 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_144);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_145;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_145) : "l"((const void*)(partial + (17024 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_145);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_146;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_146) : "l"((const void*)(partial + (18304 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_146);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_147;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_147) : "l"((const void*)(partial + (19584 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_147);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_148;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_148) : "l"((const void*)(partial + (20864 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_148);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_149;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_149) : "l"((const void*)(partial + (22144 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_149);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_150;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_150) : "l"((const void*)(partial + (23424 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_150);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_151;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_151) : "l"((const void*)(partial + (24704 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_151);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_152;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_152) : "l"((const void*)(partial + (25984 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_152);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_153;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_153) : "l"((const void*)(partial + (27264 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_153);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_154;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_154) : "l"((const void*)(partial + (28544 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_154);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_155;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_155) : "l"((const void*)(partial + (29824 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_155);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_156;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_156) : "l"((const void*)(partial + (31104 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_156);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
#endif
        buf[6] = buf[6] + buf[7];
#if __CUDA_ARCH__ == 1070
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_3 = 384 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_3 / 32 / 8 * 32 + at_w_3 % 32) * 8 + at_w_3 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_3 / 32 / 8 * 32 + at_w_3 % 32) * 8 + at_w_3 / 32 % 8] = buf[6];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_3)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_3)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_3)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_3)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_3)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_3)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_3)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_3)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_3)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_3)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_3)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_3)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_3)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_3)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_3)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_3)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_3)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_3)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_3)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_3)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_3)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_3)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_3)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_3)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_3)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_157;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_157) : "l"((const void*)(partial + (512 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_157);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_158;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_158) : "l"((const void*)(partial + (1792 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_158);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_159;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_159) : "l"((const void*)(partial + (3072 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_159);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_160;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_160) : "l"((const void*)(partial + (4352 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_160);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_161;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_161) : "l"((const void*)(partial + (5632 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_161);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_162;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_162) : "l"((const void*)(partial + (6912 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_162);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_163;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_163) : "l"((const void*)(partial + (8192 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_163);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_164;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_164) : "l"((const void*)(partial + (9472 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_164);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_165;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_165) : "l"((const void*)(partial + (10752 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_165);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_166;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_166) : "l"((const void*)(partial + (12032 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_166);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_167;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_167) : "l"((const void*)(partial + (13312 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_167);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_168;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_168) : "l"((const void*)(partial + (14592 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_168);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_169;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_169) : "l"((const void*)(partial + (15872 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_169);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_170;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_170) : "l"((const void*)(partial + (17152 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_170);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_171;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_171) : "l"((const void*)(partial + (18432 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_171);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_172;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_172) : "l"((const void*)(partial + (19712 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_172);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_173;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_173) : "l"((const void*)(partial + (20992 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_173);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_174;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_174) : "l"((const void*)(partial + (22272 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_174);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_175;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_175) : "l"((const void*)(partial + (23552 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_175);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_176;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_176) : "l"((const void*)(partial + (24832 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_176);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_177;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_177) : "l"((const void*)(partial + (26112 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_177);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_178;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_178) : "l"((const void*)(partial + (27392 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_178);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_179;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_179) : "l"((const void*)(partial + (28672 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_179);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_180;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_180) : "l"((const void*)(partial + (29952 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_180);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_181;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_181) : "l"((const void*)(partial + (31232 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_181);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
#endif
        buf[8] = buf[8] + buf[9];
#if __CUDA_ARCH__ == 1070
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_4 = 512 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_4 / 32 / 8 * 32 + at_w_4 % 32) * 8 + at_w_4 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_4 / 32 / 8 * 32 + at_w_4 % 32) * 8 + at_w_4 / 32 % 8] = buf[8];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_4)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_4)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_4)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_4)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_4)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_4)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_4)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_4)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_4)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_4)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_4)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_4)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_4)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_4)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_4)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_4)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_4)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_4)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_4)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_4)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_4)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_4)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_4)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_4)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_4)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_182;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_182) : "l"((const void*)(partial + (640 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_182);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_183;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_183) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_183);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_184;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_184) : "l"((const void*)(partial + (3200 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_184);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_185;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_185) : "l"((const void*)(partial + (4480 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_185);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_186;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_186) : "l"((const void*)(partial + (5760 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_186);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_187;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_187) : "l"((const void*)(partial + (7040 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_187);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_188;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_188) : "l"((const void*)(partial + (8320 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_188);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_189;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_189) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_189);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_190;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_190) : "l"((const void*)(partial + (10880 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_190);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_191;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_191) : "l"((const void*)(partial + (12160 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_191);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_192;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_192) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_192);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_193;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_193) : "l"((const void*)(partial + (14720 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_193);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_194;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_194) : "l"((const void*)(partial + (16000 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_194);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_195;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_195) : "l"((const void*)(partial + (17280 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_195);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_196;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_196) : "l"((const void*)(partial + (18560 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_196);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_197;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_197) : "l"((const void*)(partial + (19840 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_197);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_198;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_198) : "l"((const void*)(partial + (21120 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_198);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_199;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_199) : "l"((const void*)(partial + (22400 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_199);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_200;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_200) : "l"((const void*)(partial + (23680 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_200);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_201;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_201) : "l"((const void*)(partial + (24960 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_201);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_202;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_202) : "l"((const void*)(partial + (26240 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_202);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_203;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_203) : "l"((const void*)(partial + (27520 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_203);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_204;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_204) : "l"((const void*)(partial + (28800 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_204);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_205;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_205) : "l"((const void*)(partial + (30080 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_205);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_206;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_206) : "l"((const void*)(partial + (31360 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_206);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
#endif
        buf[10] = buf[10] + buf[11];
#if __CUDA_ARCH__ == 1070
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_5 = 640 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_5 / 32 / 8 * 32 + at_w_5 % 32) * 8 + at_w_5 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_5 / 32 / 8 * 32 + at_w_5 % 32) * 8 + at_w_5 / 32 % 8] = buf[10];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_5)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_5)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_5)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_5)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_5)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_5)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_5)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_5)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_5)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_5)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_5)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_5)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_5)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_5)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_5)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_5)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_5)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_5)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_5)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_5)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_5)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_5)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_5)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_5)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_5)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_207;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_207) : "l"((const void*)(partial + (768 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_207);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_208;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_208) : "l"((const void*)(partial + (2048 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_208);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_209;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_209) : "l"((const void*)(partial + (3328 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_209);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_210;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_210) : "l"((const void*)(partial + (4608 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_210);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_211;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_211) : "l"((const void*)(partial + (5888 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_211);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_212;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_212) : "l"((const void*)(partial + (7168 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_212);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_213;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_213) : "l"((const void*)(partial + (8448 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_213);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_214;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_214) : "l"((const void*)(partial + (9728 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_214);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_215;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_215) : "l"((const void*)(partial + (11008 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_215);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_216;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_216) : "l"((const void*)(partial + (12288 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_216);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_217;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_217) : "l"((const void*)(partial + (13568 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_217);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_218;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_218) : "l"((const void*)(partial + (14848 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_218);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_219;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_219) : "l"((const void*)(partial + (16128 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_219);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_220;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_220) : "l"((const void*)(partial + (17408 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_220);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_221;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_221) : "l"((const void*)(partial + (18688 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_221);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_222;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_222) : "l"((const void*)(partial + (19968 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_222);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_223;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_223) : "l"((const void*)(partial + (21248 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_223);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_224;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_224) : "l"((const void*)(partial + (22528 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_224);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_225;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_225) : "l"((const void*)(partial + (23808 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_225);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_226;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_226) : "l"((const void*)(partial + (25088 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_226);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_227;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_227) : "l"((const void*)(partial + (26368 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_227);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_228;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_228) : "l"((const void*)(partial + (27648 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_228);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_229;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_229) : "l"((const void*)(partial + (28928 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_229);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_230;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_230) : "l"((const void*)(partial + (30208 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_230);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_231;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_231) : "l"((const void*)(partial + (31488 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_231);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
#endif
        buf[12] = buf[12] + buf[13];
#if __CUDA_ARCH__ == 1070
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_6 = 768 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_6 / 32 / 8 * 32 + at_w_6 % 32) * 8 + at_w_6 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_6 / 32 / 8 * 32 + at_w_6 % 32) * 8 + at_w_6 / 32 % 8] = buf[12];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_6)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_6)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_6)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_6)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_6)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_6)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_6)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_6)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_6)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_6)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_6)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_6)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_6)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_6)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_6)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_6)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_6)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_6)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_6)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_6)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_6)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_6)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_6)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_6)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_6)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_232;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_232) : "l"((const void*)(partial + (896 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_232);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_233;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_233) : "l"((const void*)(partial + (2176 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_233);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_234;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_234) : "l"((const void*)(partial + (3456 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_234);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_235;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_235) : "l"((const void*)(partial + (4736 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_235);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_236;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_236) : "l"((const void*)(partial + (6016 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_236);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_237;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_237) : "l"((const void*)(partial + (7296 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_237);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_238;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_238) : "l"((const void*)(partial + (8576 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_238);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_239;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_239) : "l"((const void*)(partial + (9856 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_239);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_240;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_240) : "l"((const void*)(partial + (11136 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_240);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_241;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_241) : "l"((const void*)(partial + (12416 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_241);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_242;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_242) : "l"((const void*)(partial + (13696 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_242);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_243;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_243) : "l"((const void*)(partial + (14976 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_243);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_244;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_244) : "l"((const void*)(partial + (16256 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_244);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_245;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_245) : "l"((const void*)(partial + (17536 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_245);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_246;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_246) : "l"((const void*)(partial + (18816 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_246);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_247;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_247) : "l"((const void*)(partial + (20096 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_247);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_248;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_248) : "l"((const void*)(partial + (21376 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_248);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_249;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_249) : "l"((const void*)(partial + (22656 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_249);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_250;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_250) : "l"((const void*)(partial + (23936 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_250);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_251;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_251) : "l"((const void*)(partial + (25216 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_251);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_252;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_252) : "l"((const void*)(partial + (26496 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_252);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_253;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_253) : "l"((const void*)(partial + (27776 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_253);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_254;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_254) : "l"((const void*)(partial + (29056 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_254);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_255;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_255) : "l"((const void*)(partial + (30336 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_255);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_256;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_256) : "l"((const void*)(partial + (31616 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_256);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
#endif
        buf[14] = buf[14] + buf[15];
#if __CUDA_ARCH__ == 1070
        buf[16] = buf[16] + buf[17];
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_7 = 896 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_7 / 32 / 8 * 32 + at_w_7 % 32) * 8 + at_w_7 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_7 / 32 / 8 * 32 + at_w_7 % 32) * 8 + at_w_7 / 32 % 8] = buf[14];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_7)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_7)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_7)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_7)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_7)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_7)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_7)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_7)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_7)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_7)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_7)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_7)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_7)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_7)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_7)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_7)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_7)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_7)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_7)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_7)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_7)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_7)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_7)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_7)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_7)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_257;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_257) : "l"((const void*)(partial + (1024 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_257);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_258;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_258) : "l"((const void*)(partial + (2304 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_258);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_259;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_259) : "l"((const void*)(partial + (3584 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_259);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_260;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_260) : "l"((const void*)(partial + (4864 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_260);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_261;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_261) : "l"((const void*)(partial + (6144 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_261);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_262;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_262) : "l"((const void*)(partial + (7424 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_262);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_263;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_263) : "l"((const void*)(partial + (8704 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_263);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_264;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_264) : "l"((const void*)(partial + (9984 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_264);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_265;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_265) : "l"((const void*)(partial + (11264 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_265);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_266;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_266) : "l"((const void*)(partial + (12544 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_266);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_267;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_267) : "l"((const void*)(partial + (13824 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_267);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_268;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_268) : "l"((const void*)(partial + (15104 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_268);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_269;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_269) : "l"((const void*)(partial + (16384 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_269);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_270;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_270) : "l"((const void*)(partial + (17664 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_270);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_271;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_271) : "l"((const void*)(partial + (18944 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_271);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_272;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_272) : "l"((const void*)(partial + (20224 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_272);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_273;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_273) : "l"((const void*)(partial + (21504 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_273);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_274;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_274) : "l"((const void*)(partial + (22784 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_274);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_275;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_275) : "l"((const void*)(partial + (24064 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_275);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_276;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_276) : "l"((const void*)(partial + (25344 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_276);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_277;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_277) : "l"((const void*)(partial + (26624 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_277);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_278;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_278) : "l"((const void*)(partial + (27904 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_278);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_279;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_279) : "l"((const void*)(partial + (29184 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_279);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_280;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_280) : "l"((const void*)(partial + (30464 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_280);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_281;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_281) : "l"((const void*)(partial + (31744 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_281);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
#endif
        buf[16] = buf[16] + buf[17];
#if __CUDA_ARCH__ == 1070
        buf[18] = buf[18] + buf[19];
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_8 = 1024 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_8 / 32 / 8 * 32 + at_w_8 % 32) * 8 + at_w_8 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_8 / 32 / 8 * 32 + at_w_8 % 32) * 8 + at_w_8 / 32 % 8] = buf[16];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_8)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_8)) = 0.0f;
        }
#if __CUDA_ARCH__ == 1070
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_8)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_8)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_8)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_8)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_8)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_8)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_8)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_8)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_8)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_8)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_8)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_8)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_8)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_8)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_8)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_8)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_8)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_8)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_8)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_8)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_8)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_8)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_8)) = 0.0f;
        }
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_282;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_282) : "l"((const void*)(partial + (1152 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_282);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_283;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_283) : "l"((const void*)(partial + (2432 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_283);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_284;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_284) : "l"((const void*)(partial + (3712 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_284);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_285;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_285) : "l"((const void*)(partial + (4992 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_285);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_286;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_286) : "l"((const void*)(partial + (6272 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_286);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_287;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_287) : "l"((const void*)(partial + (7552 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_287);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_288;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_288) : "l"((const void*)(partial + (8832 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_288);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_289;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_289) : "l"((const void*)(partial + (10112 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_289);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_290;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_290) : "l"((const void*)(partial + (11392 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_290);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_291;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_291) : "l"((const void*)(partial + (12672 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_291);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_292;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_292) : "l"((const void*)(partial + (13952 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_292);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_293;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_293) : "l"((const void*)(partial + (15232 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_293);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_294;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_294) : "l"((const void*)(partial + (16512 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_294);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_295;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_295) : "l"((const void*)(partial + (17792 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_295);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_296;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_296) : "l"((const void*)(partial + (19072 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_296);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_297;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_297) : "l"((const void*)(partial + (20352 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_297);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_298;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_298) : "l"((const void*)(partial + (21632 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_298);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_299;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_299) : "l"((const void*)(partial + (22912 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_299);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_300;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_300) : "l"((const void*)(partial + (24192 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_300);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_301;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_301) : "l"((const void*)(partial + (25472 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_301);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_302;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_302) : "l"((const void*)(partial + (26752 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_302);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_303;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_303) : "l"((const void*)(partial + (28032 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_303);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_304;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_304) : "l"((const void*)(partial + (29312 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_304);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_305;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_305) : "l"((const void*)(partial + (30592 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_305);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_306;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_306) : "l"((const void*)(partial + (31872 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_306);
            }
        }
        buf[0] = buf[0] + buf[1];
        buf[2] = buf[2] + buf[3];
        buf[4] = buf[4] + buf[5];
        buf[6] = buf[6] + buf[7];
        buf[8] = buf[8] + buf[9];
        buf[10] = buf[10] + buf[11];
        buf[12] = buf[12] + buf[13];
        buf[14] = buf[14] + buf[15];
        buf[16] = buf[16] + buf[17];
#endif
        buf[18] = buf[18] + buf[19];
#if __CUDA_ARCH__ == 1070
        buf[20] = buf[20] + buf[21];
        buf[22] = buf[22] + buf[23];
        buf[0] = buf[0] + buf[2];
        buf[4] = buf[4] + buf[6];
        buf[8] = buf[8] + buf[10];
        buf[12] = buf[12] + buf[14];
        buf[16] = buf[16] + buf[18];
        buf[20] = buf[20] + buf[22];
        buf[0] = buf[0] + buf[4];
        buf[8] = buf[8] + buf[12];
        buf[16] = buf[16] + buf[20];
        buf[0] = buf[0] + buf[8];
        buf[16] = buf[16] + buf[24];
        buf[0] = buf[0] + buf[16];
#endif
        int at_w_9 = 1152 + tid;
#if __CUDA_ARCH__ == 1070
        dwbuf[(at_w_9 / 32 / 8 * 32 + at_w_9 % 32) * 8 + at_w_9 / 32 % 8] = buf[0];
#else
        dwbuf[(at_w_9 / 32 / 8 * 32 + at_w_9 % 32) * 8 + at_w_9 / 32 % 8] = buf[18];
#endif
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_9)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_9)) = 0.0f;
#if __CUDA_ARCH__ == 1070
        }
        if (2 < nrep) {
            *(reinterpret_cast<float*>(partial) + (2560 + at_w_9)) = 0.0f;
        }
        if (3 < nrep) {
            *(reinterpret_cast<float*>(partial) + (3840 + at_w_9)) = 0.0f;
        }
        if (4 < nrep) {
            *(reinterpret_cast<float*>(partial) + (5120 + at_w_9)) = 0.0f;
        }
        if (5 < nrep) {
            *(reinterpret_cast<float*>(partial) + (6400 + at_w_9)) = 0.0f;
        }
        if (6 < nrep) {
            *(reinterpret_cast<float*>(partial) + (7680 + at_w_9)) = 0.0f;
        }
        if (7 < nrep) {
            *(reinterpret_cast<float*>(partial) + (8960 + at_w_9)) = 0.0f;
        }
        if (8 < nrep) {
            *(reinterpret_cast<float*>(partial) + (10240 + at_w_9)) = 0.0f;
        }
        if (9 < nrep) {
            *(reinterpret_cast<float*>(partial) + (11520 + at_w_9)) = 0.0f;
        }
        if (10 < nrep) {
            *(reinterpret_cast<float*>(partial) + (12800 + at_w_9)) = 0.0f;
        }
        if (11 < nrep) {
            *(reinterpret_cast<float*>(partial) + (14080 + at_w_9)) = 0.0f;
        }
        if (12 < nrep) {
            *(reinterpret_cast<float*>(partial) + (15360 + at_w_9)) = 0.0f;
        }
        if (13 < nrep) {
            *(reinterpret_cast<float*>(partial) + (16640 + at_w_9)) = 0.0f;
        }
        if (14 < nrep) {
            *(reinterpret_cast<float*>(partial) + (17920 + at_w_9)) = 0.0f;
        }
        if (15 < nrep) {
            *(reinterpret_cast<float*>(partial) + (19200 + at_w_9)) = 0.0f;
        }
        if (16 < nrep) {
            *(reinterpret_cast<float*>(partial) + (20480 + at_w_9)) = 0.0f;
        }
        if (17 < nrep) {
            *(reinterpret_cast<float*>(partial) + (21760 + at_w_9)) = 0.0f;
        }
        if (18 < nrep) {
            *(reinterpret_cast<float*>(partial) + (23040 + at_w_9)) = 0.0f;
        }
        if (19 < nrep) {
            *(reinterpret_cast<float*>(partial) + (24320 + at_w_9)) = 0.0f;
        }
        if (20 < nrep) {
            *(reinterpret_cast<float*>(partial) + (25600 + at_w_9)) = 0.0f;
        }
        if (21 < nrep) {
            *(reinterpret_cast<float*>(partial) + (26880 + at_w_9)) = 0.0f;
        }
        if (22 < nrep) {
            *(reinterpret_cast<float*>(partial) + (28160 + at_w_9)) = 0.0f;
        }
        if (23 < nrep) {
            *(reinterpret_cast<float*>(partial) + (29440 + at_w_9)) = 0.0f;
        }
        if (24 < nrep) {
            *(reinterpret_cast<float*>(partial) + (30720 + at_w_9)) = 0.0f;
#endif
        }
        __syncthreads();
        int col_0 = tid * 4;
        if (col_0 < 1280) {
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
        int col_1 = (128 + tid) * 4;
        if (col_1 < 1280) {
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
        int col_2 = (256 + tid) * 4;
        if (col_2 < 1280) {
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
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
