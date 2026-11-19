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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_f0aebb74650c8dc3f128(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    float* sums = reinterpret_cast<float*>(smem_raw + 1152);
    const int sums_addr = smem + 1152;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1152);
    const int dwbuf_addr = smem + 1152;
    int* meta = reinterpret_cast<int*>(smem_raw + 1120);
    const int meta_addr = smem + 1120;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 17536);
    const int ring_addr = smem + 17536;

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
    int rg = warp / 8;
    int tir = tid - rg * 256;
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
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((256 + tir) * 8));
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
    int rows_mine = 0;
    int slot = 0;
    sums[rg * 4096 + tir] = 0.0f;
    sums[rg * 4096 + 256 + tir] = 0.0f;
    sums[rg * 4096 + 512 + tir] = 0.0f;
    sums[rg * 4096 + 768 + tir] = 0.0f;
    sums[rg * 4096 + 1024 + tir] = 0.0f;
    sums[rg * 4096 + 1280 + tir] = 0.0f;
    sums[rg * 4096 + 1536 + tir] = 0.0f;
    sums[rg * 4096 + 1792 + tir] = 0.0f;
    sums[rg * 4096 + 2048 + tir] = 0.0f;
    sums[rg * 4096 + 2304 + tir] = 0.0f;
    sums[rg * 4096 + 2560 + tir] = 0.0f;
    sums[rg * 4096 + 2816 + tir] = 0.0f;
    sums[rg * 4096 + 3072 + tir] = 0.0f;
    sums[rg * 4096 + 3328 + tir] = 0.0f;
    sums[rg * 4096 + 3584 + tir] = 0.0f;
    sums[rg * 4096 + 3840 + tir] = 0.0f;
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[1];
    float rr[1];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        pc = (int)nxt;
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
            int _min_3 = ((begin_it) < (T) ? (begin_it) : (T));
            prow = _min_3;
            int _min_4 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
            pend = _min_4;
            unsigned int _atomic_fetch_add_5 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_5;
            unsigned int _atomic_fetch_add_6 = atomicAdd(&counters[3], 1);
            slot = (int)_atomic_fetch_add_6;
        }
        int slot_0 = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
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
                    int _min_7 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                    prow = _min_7;
                    int _min_8 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                    pend = _min_8;
                    unsigned int _atomic_fetch_add_9 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_9;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_10 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval = _min_10;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 16384);
            int sbase = ring_at + slot_0 * 16384;
            int _min_11 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_11);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr);
        }
        prow = prow + nval;
        int slot_1 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
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
                    int _min_12 = ((begin_it_2) < (T) ? (begin_it_2) : (T));
                    prow = _min_12;
                    int _min_13 = ((begin_it_2 + span_2) < (T) ? (begin_it_2 + span_2) : (T));
                    pend = _min_13;
                    unsigned int _atomic_fetch_add_14 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_14;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_15 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_0 = _min_15;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 16384);
            int sbase_1 = ring_at + slot_1 * 16384;
            int _min_16 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_1 = (long long)(prow + _min_16);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr + 8);
        }
        prow = prow + nval_0;
        int slot_2 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
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
                    int _min_17 = ((begin_it_3) < (T) ? (begin_it_3) : (T));
                    prow = _min_17;
                    int _min_18 = ((begin_it_3 + span_3) < (T) ? (begin_it_3 + span_3) : (T));
                    pend = _min_18;
                    unsigned int _atomic_fetch_add_19 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_19;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_20 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_1 = _min_20;
        }
        meta[2 * slot_2] = prow;
        meta[2 * slot_2 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 16384);
            int sbase_2 = ring_at + slot_2 * 16384;
            int _min_21 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_21);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr + 16),
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
        rows_mine = rows_mine + nval_2;
        int _stage_full = (int)((unsigned int)k_4 % 3);
        int _parity_full = (int)((unsigned int)k_4 / 3 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_4 % 3 * 16384 + rg * 8192;
        __int128_t _smem_raw_22;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_22) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[3];
        __int128_t _smem_raw_23;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_23) : "r"(rbase + 8192 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[3];
        __int128_t _smem_raw_24;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_24) : "r"(rbase + (256 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[3];
        __int128_t _smem_raw_25;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_25) : "r"(rbase + 8192 + (256 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[3];
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
        float _warp_reduce_26 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_26 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_26, offset);
        dot[0] = _warp_reduce_26;
        if (lane == 0) {
            dots[(k_4 & 1) * 8 + warp] = dot[0];
        }
        __syncthreads();
        float tot = 0.0f;
        tot += dots[(k_4 & 1) * 8 + rg * 8];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 1];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 2];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 3];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 4];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 5];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 6];
        tot += dots[(k_4 & 1) * 8 + rg * 8 + 7];
        dot[0] = tot;
        if (tid == 0) {
            int slot_3 = (k_4 + 3) % 3;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt;
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
                        int _min_27 = ((begin_it_4) < (T) ? (begin_it_4) : (T));
                        prow = _min_27;
                        int _min_28 = ((begin_it_4 + span_4) < (T) ? (begin_it_4 + span_4) : (T));
                        pend = _min_28;
                        unsigned int _atomic_fetch_add_29 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_29;
                    }
                }
            }
            int nval_2_1 = 0;
            if (prow < pend) {
                int _min_30 = ((pend - prow) < (1) ? (pend - prow) : (1));
                nval_2_1 = _min_30;
            }
            meta[2 * slot_3] = prow;
            meta[2 * slot_3 + 1] = nval_2_1;
            if (nval_2_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8, 16384);
                int sbase_3 = ring_at + slot_3 * 16384;
                int _min_31 = ((nval_2_1 - 1) < (0) ? (nval_2_1 - 1) : (0));
                long long row_i_3 = (long long)(prow + _min_31);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
            } else {
                mbarrier_arrive(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8);
            }
            prow = prow + nval_2_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 4096.0f);
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
            float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
            float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
            out_0[0] = y0_0;
            out_0[1] = y1_0;
            float g0_1 = __uint_as_float(gw[1] << 16);
            float g1_1 = __uint_as_float(gw[1] & 4294901760u);
            float x0_1 = __uint_as_float(xw[1] << 16);
            float x1_1 = __uint_as_float(xw[1] & 4294901760u);
            acc[2] = __fmaf_rn(g0_1 * x0_1, rq, acc[2]);
            acc[3] = __fmaf_rn(g1_1 * x1_1, rq, acc[3]);
            float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
            float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
            out_0[2] = y0_1;
            out_0[3] = y1_1;
            float g0_2 = __uint_as_float(gw[2] << 16);
            float g1_2 = __uint_as_float(gw[2] & 4294901760u);
            float x0_2 = __uint_as_float(xw[2] << 16);
            float x1_2 = __uint_as_float(xw[2] & 4294901760u);
            acc[4] = __fmaf_rn(g0_2 * x0_2, rq, acc[4]);
            acc[5] = __fmaf_rn(g1_2 * x1_2, rq, acc[5]);
            float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
            float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
            out_0[4] = y0_2;
            out_0[5] = y1_2;
            float g0_3 = __uint_as_float(gw[3] << 16);
            float g1_3 = __uint_as_float(gw[3] & 4294901760u);
            float x0_3 = __uint_as_float(xw[3] << 16);
            float x1_3 = __uint_as_float(xw[3] & 4294901760u);
            acc[6] = __fmaf_rn(g0_3 * x0_3, rq, acc[6]);
            acc[7] = __fmaf_rn(g1_3 * x1_3, rq, acc[7]);
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 4096ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
            float out_1[8];
            float g0_4 = __uint_as_float(gw[4] << 16);
            float g1_4 = __uint_as_float(gw[4] & 4294901760u);
            float x0_4 = __uint_as_float(xw[4] << 16);
            float x1_4 = __uint_as_float(xw[4] & 4294901760u);
            acc[8] = __fmaf_rn(g0_4 * x0_4, rq, acc[8]);
            acc[9] = __fmaf_rn(g1_4 * x1_4, rq, acc[9]);
            float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(ww[4] << 16)));
            float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(ww[4] & 4294901760u)));
            out_1[0] = y0_4;
            out_1[1] = y1_4;
            float g0_5 = __uint_as_float(gw[5] << 16);
            float g1_5 = __uint_as_float(gw[5] & 4294901760u);
            float x0_5 = __uint_as_float(xw[5] << 16);
            float x1_5 = __uint_as_float(xw[5] & 4294901760u);
            acc[10] = __fmaf_rn(g0_5 * x0_5, rq, acc[10]);
            acc[11] = __fmaf_rn(g1_5 * x1_5, rq, acc[11]);
            float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(ww[5] << 16)));
            float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(ww[5] & 4294901760u)));
            out_1[2] = y0_5;
            out_1[3] = y1_5;
            float g0_6 = __uint_as_float(gw[6] << 16);
            float g1_6 = __uint_as_float(gw[6] & 4294901760u);
            float x0_6 = __uint_as_float(xw[6] << 16);
            float x1_6 = __uint_as_float(xw[6] & 4294901760u);
            acc[12] = __fmaf_rn(g0_6 * x0_6, rq, acc[12]);
            acc[13] = __fmaf_rn(g1_6 * x1_6, rq, acc[13]);
            float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(ww[6] << 16)));
            float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(ww[6] & 4294901760u)));
            out_1[4] = y0_6;
            out_1[5] = y1_6;
            float g0_7 = __uint_as_float(gw[7] << 16);
            float g1_7 = __uint_as_float(gw[7] & 4294901760u);
            float x0_7 = __uint_as_float(xw[7] << 16);
            float x1_7 = __uint_as_float(xw[7] & 4294901760u);
            acc[14] = __fmaf_rn(g0_7 * x0_7, rq, acc[14]);
            acc[15] = __fmaf_rn(g1_7 * x1_7, rq, acc[15]);
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 4096ull + (unsigned long long)((256 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        row0 = meta[2 * ((k_4 + 1) % 3)];
        nval_2 = meta[2 * ((k_4 + 1) % 3) + 1];
        if (nval_2 > rg) {
            rr[0] = r[row0 + rg];
        }
        if ((k_4 + 1 & 31) == 0) {
            sums[rg * 4096 + tir] = sums[rg * 4096 + tir] + acc[0];
            acc[0] = 0.0f;
            sums[rg * 4096 + 256 + tir] = sums[rg * 4096 + 256 + tir] + acc[1];
            acc[1] = 0.0f;
            sums[rg * 4096 + 512 + tir] = sums[rg * 4096 + 512 + tir] + acc[2];
            acc[2] = 0.0f;
            sums[rg * 4096 + 768 + tir] = sums[rg * 4096 + 768 + tir] + acc[3];
            acc[3] = 0.0f;
            sums[rg * 4096 + 1024 + tir] = sums[rg * 4096 + 1024 + tir] + acc[4];
            acc[4] = 0.0f;
            sums[rg * 4096 + 1280 + tir] = sums[rg * 4096 + 1280 + tir] + acc[5];
            acc[5] = 0.0f;
            sums[rg * 4096 + 1536 + tir] = sums[rg * 4096 + 1536 + tir] + acc[6];
            acc[6] = 0.0f;
            sums[rg * 4096 + 1792 + tir] = sums[rg * 4096 + 1792 + tir] + acc[7];
            acc[7] = 0.0f;
            sums[rg * 4096 + 2048 + tir] = sums[rg * 4096 + 2048 + tir] + acc[8];
            acc[8] = 0.0f;
            sums[rg * 4096 + 2304 + tir] = sums[rg * 4096 + 2304 + tir] + acc[9];
            acc[9] = 0.0f;
            sums[rg * 4096 + 2560 + tir] = sums[rg * 4096 + 2560 + tir] + acc[10];
            acc[10] = 0.0f;
            sums[rg * 4096 + 2816 + tir] = sums[rg * 4096 + 2816 + tir] + acc[11];
            acc[11] = 0.0f;
            sums[rg * 4096 + 3072 + tir] = sums[rg * 4096 + 3072 + tir] + acc[12];
            acc[12] = 0.0f;
            sums[rg * 4096 + 3328 + tir] = sums[rg * 4096 + 3328 + tir] + acc[13];
            acc[13] = 0.0f;
            sums[rg * 4096 + 3584 + tir] = sums[rg * 4096 + 3584 + tir] + acc[14];
            acc[14] = 0.0f;
            sums[rg * 4096 + 3840 + tir] = sums[rg * 4096 + 3840 + tir] + acc[15];
            acc[15] = 0.0f;
        }
    }
    sums[rg * 4096 + tir] = sums[rg * 4096 + tir] + acc[0];
    acc[0] = 0.0f;
    sums[rg * 4096 + 256 + tir] = sums[rg * 4096 + 256 + tir] + acc[1];
    acc[1] = 0.0f;
    sums[rg * 4096 + 512 + tir] = sums[rg * 4096 + 512 + tir] + acc[2];
    acc[2] = 0.0f;
    sums[rg * 4096 + 768 + tir] = sums[rg * 4096 + 768 + tir] + acc[3];
    acc[3] = 0.0f;
    sums[rg * 4096 + 1024 + tir] = sums[rg * 4096 + 1024 + tir] + acc[4];
    acc[4] = 0.0f;
    sums[rg * 4096 + 1280 + tir] = sums[rg * 4096 + 1280 + tir] + acc[5];
    acc[5] = 0.0f;
    sums[rg * 4096 + 1536 + tir] = sums[rg * 4096 + 1536 + tir] + acc[6];
    acc[6] = 0.0f;
    sums[rg * 4096 + 1792 + tir] = sums[rg * 4096 + 1792 + tir] + acc[7];
    acc[7] = 0.0f;
    sums[rg * 4096 + 2048 + tir] = sums[rg * 4096 + 2048 + tir] + acc[8];
    acc[8] = 0.0f;
    sums[rg * 4096 + 2304 + tir] = sums[rg * 4096 + 2304 + tir] + acc[9];
    acc[9] = 0.0f;
    sums[rg * 4096 + 2560 + tir] = sums[rg * 4096 + 2560 + tir] + acc[10];
    acc[10] = 0.0f;
    sums[rg * 4096 + 2816 + tir] = sums[rg * 4096 + 2816 + tir] + acc[11];
    acc[11] = 0.0f;
    sums[rg * 4096 + 3072 + tir] = sums[rg * 4096 + 3072 + tir] + acc[12];
    acc[12] = 0.0f;
    sums[rg * 4096 + 3328 + tir] = sums[rg * 4096 + 3328 + tir] + acc[13];
    acc[13] = 0.0f;
    sums[rg * 4096 + 3584 + tir] = sums[rg * 4096 + 3584 + tir] + acc[14];
    acc[14] = 0.0f;
    sums[rg * 4096 + 3840 + tir] = sums[rg * 4096 + 3840 + tir] + acc[15];
    acc[15] = 0.0f;
    c = pc;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpbulk_dst_32 = reinterpret_cast<void*>(partial + (long long)(slot * 4096));
                asm volatile(
                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                    :: "l"(_cpbulk_dst_32), "r"(sums_addr), "r"((uint32_t)(16384))
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
    __syncthreads();
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
        __syncthreads();
        asm volatile("fence.acquire.gpu;" ::: "memory");
        unsigned int nw[1];
        {
            uint32_t _scalar_bits_33;
            asm volatile("ld.global.cg.b32 %0, [%1];"
                : "=r"(_scalar_bits_33) : "l"((const void*)(counters + (3))) : "memory");
            nw[0] = (unsigned int)_scalar_bits_33;
        }
        int grid = (int)num_bids;
        int helpers = ((grid) < (128) ? (grid) : (128));
        int cl = tid % 8;
        int rl = tid / 8;
        #pragma unroll 1
        for (int kk = 0; kk < 128; kk++) {
            int sl_5 = hs_c + kk * helpers;
            if (sl_5 >= 128) {
                break;
            }
            int at0 = sl_5 * 32 + 4 * cl;
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
                        unsigned _v4_34_0;
                        unsigned _v4_34_1;
                        unsigned _v4_34_2;
                        unsigned _v4_34_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_34_0), "=r"(_v4_34_1), "=r"(_v4_34_2), "=r"(_v4_34_3) : "l"((const void*)(partial + (s0 * 4096 + at0))) : "memory");
                        buf[0 + 0] = __uint_as_float(_v4_34_0);
                        buf[0 + 1] = __uint_as_float(_v4_34_1);
                        buf[0 + 2] = __uint_as_float(_v4_34_2);
                        buf[0 + 3] = __uint_as_float(_v4_34_3);
                    }
                }
                if (s0 + 32 < grid) {
                    {
                        unsigned _v4_35_0;
                        unsigned _v4_35_1;
                        unsigned _v4_35_2;
                        unsigned _v4_35_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_35_0), "=r"(_v4_35_1), "=r"(_v4_35_2), "=r"(_v4_35_3) : "l"((const void*)(partial + ((s0 + 32) * 4096 + at0))) : "memory");
                        buf[4 + 0] = __uint_as_float(_v4_35_0);
                        buf[4 + 1] = __uint_as_float(_v4_35_1);
                        buf[4 + 2] = __uint_as_float(_v4_35_2);
                        buf[4 + 3] = __uint_as_float(_v4_35_3);
                    }
                }
                if (s0 + 64 < grid) {
                    {
                        unsigned _v4_36_0;
                        unsigned _v4_36_1;
                        unsigned _v4_36_2;
                        unsigned _v4_36_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_36_0), "=r"(_v4_36_1), "=r"(_v4_36_2), "=r"(_v4_36_3) : "l"((const void*)(partial + ((s0 + 64) * 4096 + at0))) : "memory");
                        buf[8 + 0] = __uint_as_float(_v4_36_0);
                        buf[8 + 1] = __uint_as_float(_v4_36_1);
                        buf[8 + 2] = __uint_as_float(_v4_36_2);
                        buf[8 + 3] = __uint_as_float(_v4_36_3);
                    }
                }
                if (s0 + 96 < grid) {
                    {
                        unsigned _v4_37_0;
                        unsigned _v4_37_1;
                        unsigned _v4_37_2;
                        unsigned _v4_37_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_37_0), "=r"(_v4_37_1), "=r"(_v4_37_2), "=r"(_v4_37_3) : "l"((const void*)(partial + ((s0 + 96) * 4096 + at0))) : "memory");
                        buf[12 + 0] = __uint_as_float(_v4_37_0);
                        buf[12 + 1] = __uint_as_float(_v4_37_1);
                        buf[12 + 2] = __uint_as_float(_v4_37_2);
                        buf[12 + 3] = __uint_as_float(_v4_37_3);
                    }
                }
                if (s0 + 128 < grid) {
                    {
                        unsigned _v4_38_0;
                        unsigned _v4_38_1;
                        unsigned _v4_38_2;
                        unsigned _v4_38_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_38_0), "=r"(_v4_38_1), "=r"(_v4_38_2), "=r"(_v4_38_3) : "l"((const void*)(partial + ((s0 + 128) * 4096 + at0))) : "memory");
                        buf[16 + 0] = __uint_as_float(_v4_38_0);
                        buf[16 + 1] = __uint_as_float(_v4_38_1);
                        buf[16 + 2] = __uint_as_float(_v4_38_2);
                        buf[16 + 3] = __uint_as_float(_v4_38_3);
                    }
                }
                if (s0 + 160 < grid) {
                    {
                        unsigned _v4_39_0;
                        unsigned _v4_39_1;
                        unsigned _v4_39_2;
                        unsigned _v4_39_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_39_0), "=r"(_v4_39_1), "=r"(_v4_39_2), "=r"(_v4_39_3) : "l"((const void*)(partial + ((s0 + 160) * 4096 + at0))) : "memory");
                        buf[20 + 0] = __uint_as_float(_v4_39_0);
                        buf[20 + 1] = __uint_as_float(_v4_39_1);
                        buf[20 + 2] = __uint_as_float(_v4_39_2);
                        buf[20 + 3] = __uint_as_float(_v4_39_3);
                    }
                }
                if (s0 + 192 < grid) {
                    {
                        unsigned _v4_40_0;
                        unsigned _v4_40_1;
                        unsigned _v4_40_2;
                        unsigned _v4_40_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_40_0), "=r"(_v4_40_1), "=r"(_v4_40_2), "=r"(_v4_40_3) : "l"((const void*)(partial + ((s0 + 192) * 4096 + at0))) : "memory");
                        buf[24 + 0] = __uint_as_float(_v4_40_0);
                        buf[24 + 1] = __uint_as_float(_v4_40_1);
                        buf[24 + 2] = __uint_as_float(_v4_40_2);
                        buf[24 + 3] = __uint_as_float(_v4_40_3);
                    }
                }
                if (s0 + 224 < grid) {
                    {
                        unsigned _v4_41_0;
                        unsigned _v4_41_1;
                        unsigned _v4_41_2;
                        unsigned _v4_41_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_41_0), "=r"(_v4_41_1), "=r"(_v4_41_2), "=r"(_v4_41_3) : "l"((const void*)(partial + ((s0 + 224) * 4096 + at0))) : "memory");
                        buf[28 + 0] = __uint_as_float(_v4_41_0);
                        buf[28 + 1] = __uint_as_float(_v4_41_1);
                        buf[28 + 2] = __uint_as_float(_v4_41_2);
                        buf[28 + 3] = __uint_as_float(_v4_41_3);
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
            __syncthreads();
            if (tid < 8) {
                p[0] = p[0] + sums[4 * (8 + cl)];
                p[1] = p[1] + sums[4 * (8 + cl) + 1];
                p[2] = p[2] + sums[4 * (8 + cl) + 2];
                p[3] = p[3] + sums[4 * (8 + cl) + 3];
                p[0] = p[0] + sums[4 * (16 + cl)];
                p[1] = p[1] + sums[4 * (16 + cl) + 1];
                p[2] = p[2] + sums[4 * (16 + cl) + 2];
                p[3] = p[3] + sums[4 * (16 + cl) + 3];
                p[0] = p[0] + sums[4 * (24 + cl)];
                p[1] = p[1] + sums[4 * (24 + cl) + 1];
                p[2] = p[2] + sums[4 * (24 + cl) + 2];
                p[3] = p[3] + sums[4 * (24 + cl) + 3];
                p[0] = p[0] + sums[4 * (32 + cl)];
                p[1] = p[1] + sums[4 * (32 + cl) + 1];
                p[2] = p[2] + sums[4 * (32 + cl) + 2];
                p[3] = p[3] + sums[4 * (32 + cl) + 3];
                p[0] = p[0] + sums[4 * (40 + cl)];
                p[1] = p[1] + sums[4 * (40 + cl) + 1];
                p[2] = p[2] + sums[4 * (40 + cl) + 2];
                p[3] = p[3] + sums[4 * (40 + cl) + 3];
                p[0] = p[0] + sums[4 * (48 + cl)];
                p[1] = p[1] + sums[4 * (48 + cl) + 1];
                p[2] = p[2] + sums[4 * (48 + cl) + 2];
                p[3] = p[3] + sums[4 * (48 + cl) + 3];
                p[0] = p[0] + sums[4 * (56 + cl)];
                p[1] = p[1] + sums[4 * (56 + cl) + 1];
                p[2] = p[2] + sums[4 * (56 + cl) + 2];
                p[3] = p[3] + sums[4 * (56 + cl) + 3];
                p[0] = p[0] + sums[4 * (64 + cl)];
                p[1] = p[1] + sums[4 * (64 + cl) + 1];
                p[2] = p[2] + sums[4 * (64 + cl) + 2];
                p[3] = p[3] + sums[4 * (64 + cl) + 3];
                p[0] = p[0] + sums[4 * (72 + cl)];
                p[1] = p[1] + sums[4 * (72 + cl) + 1];
                p[2] = p[2] + sums[4 * (72 + cl) + 2];
                p[3] = p[3] + sums[4 * (72 + cl) + 3];
                p[0] = p[0] + sums[4 * (80 + cl)];
                p[1] = p[1] + sums[4 * (80 + cl) + 1];
                p[2] = p[2] + sums[4 * (80 + cl) + 2];
                p[3] = p[3] + sums[4 * (80 + cl) + 3];
                p[0] = p[0] + sums[4 * (88 + cl)];
                p[1] = p[1] + sums[4 * (88 + cl) + 1];
                p[2] = p[2] + sums[4 * (88 + cl) + 2];
                p[3] = p[3] + sums[4 * (88 + cl) + 3];
                p[0] = p[0] + sums[4 * (96 + cl)];
                p[1] = p[1] + sums[4 * (96 + cl) + 1];
                p[2] = p[2] + sums[4 * (96 + cl) + 2];
                p[3] = p[3] + sums[4 * (96 + cl) + 3];
                p[0] = p[0] + sums[4 * (104 + cl)];
                p[1] = p[1] + sums[4 * (104 + cl) + 1];
                p[2] = p[2] + sums[4 * (104 + cl) + 2];
                p[3] = p[3] + sums[4 * (104 + cl) + 3];
                p[0] = p[0] + sums[4 * (112 + cl)];
                p[1] = p[1] + sums[4 * (112 + cl) + 1];
                p[2] = p[2] + sums[4 * (112 + cl) + 2];
                p[3] = p[3] + sums[4 * (112 + cl) + 3];
                p[0] = p[0] + sums[4 * (120 + cl)];
                p[1] = p[1] + sums[4 * (120 + cl) + 1];
                p[2] = p[2] + sums[4 * (120 + cl) + 2];
                p[3] = p[3] + sums[4 * (120 + cl) + 3];
                p[0] = p[0] + sums[4 * (128 + cl)];
                p[1] = p[1] + sums[4 * (128 + cl) + 1];
                p[2] = p[2] + sums[4 * (128 + cl) + 2];
                p[3] = p[3] + sums[4 * (128 + cl) + 3];
                p[0] = p[0] + sums[4 * (136 + cl)];
                p[1] = p[1] + sums[4 * (136 + cl) + 1];
                p[2] = p[2] + sums[4 * (136 + cl) + 2];
                p[3] = p[3] + sums[4 * (136 + cl) + 3];
                p[0] = p[0] + sums[4 * (144 + cl)];
                p[1] = p[1] + sums[4 * (144 + cl) + 1];
                p[2] = p[2] + sums[4 * (144 + cl) + 2];
                p[3] = p[3] + sums[4 * (144 + cl) + 3];
                p[0] = p[0] + sums[4 * (152 + cl)];
                p[1] = p[1] + sums[4 * (152 + cl) + 1];
                p[2] = p[2] + sums[4 * (152 + cl) + 2];
                p[3] = p[3] + sums[4 * (152 + cl) + 3];
                p[0] = p[0] + sums[4 * (160 + cl)];
                p[1] = p[1] + sums[4 * (160 + cl) + 1];
                p[2] = p[2] + sums[4 * (160 + cl) + 2];
                p[3] = p[3] + sums[4 * (160 + cl) + 3];
                p[0] = p[0] + sums[4 * (168 + cl)];
                p[1] = p[1] + sums[4 * (168 + cl) + 1];
                p[2] = p[2] + sums[4 * (168 + cl) + 2];
                p[3] = p[3] + sums[4 * (168 + cl) + 3];
                p[0] = p[0] + sums[4 * (176 + cl)];
                p[1] = p[1] + sums[4 * (176 + cl) + 1];
                p[2] = p[2] + sums[4 * (176 + cl) + 2];
                p[3] = p[3] + sums[4 * (176 + cl) + 3];
                p[0] = p[0] + sums[4 * (184 + cl)];
                p[1] = p[1] + sums[4 * (184 + cl) + 1];
                p[2] = p[2] + sums[4 * (184 + cl) + 2];
                p[3] = p[3] + sums[4 * (184 + cl) + 3];
                p[0] = p[0] + sums[4 * (192 + cl)];
                p[1] = p[1] + sums[4 * (192 + cl) + 1];
                p[2] = p[2] + sums[4 * (192 + cl) + 2];
                p[3] = p[3] + sums[4 * (192 + cl) + 3];
                p[0] = p[0] + sums[4 * (200 + cl)];
                p[1] = p[1] + sums[4 * (200 + cl) + 1];
                p[2] = p[2] + sums[4 * (200 + cl) + 2];
                p[3] = p[3] + sums[4 * (200 + cl) + 3];
                p[0] = p[0] + sums[4 * (208 + cl)];
                p[1] = p[1] + sums[4 * (208 + cl) + 1];
                p[2] = p[2] + sums[4 * (208 + cl) + 2];
                p[3] = p[3] + sums[4 * (208 + cl) + 3];
                p[0] = p[0] + sums[4 * (216 + cl)];
                p[1] = p[1] + sums[4 * (216 + cl) + 1];
                p[2] = p[2] + sums[4 * (216 + cl) + 2];
                p[3] = p[3] + sums[4 * (216 + cl) + 3];
                p[0] = p[0] + sums[4 * (224 + cl)];
                p[1] = p[1] + sums[4 * (224 + cl) + 1];
                p[2] = p[2] + sums[4 * (224 + cl) + 2];
                p[3] = p[3] + sums[4 * (224 + cl) + 3];
                p[0] = p[0] + sums[4 * (232 + cl)];
                p[1] = p[1] + sums[4 * (232 + cl) + 1];
                p[2] = p[2] + sums[4 * (232 + cl) + 2];
                p[3] = p[3] + sums[4 * (232 + cl) + 3];
                p[0] = p[0] + sums[4 * (240 + cl)];
                p[1] = p[1] + sums[4 * (240 + cl) + 1];
                p[2] = p[2] + sums[4 * (240 + cl) + 2];
                p[3] = p[3] + sums[4 * (240 + cl) + 3];
                p[0] = p[0] + sums[4 * (248 + cl)];
                p[1] = p[1] + sums[4 * (248 + cl) + 1];
                p[2] = p[2] + sums[4 * (248 + cl) + 2];
                p[3] = p[3] + sums[4 * (248 + cl) + 3];
                *(reinterpret_cast<float*>(dw) + ((at0 / 256 / 8 * 256 + at0 % 256) * 8 + at0 / 256 % 8)) = p[0];
                *(reinterpret_cast<float*>(dw) + (((at0 + 1) / 256 / 8 * 256 + (at0 + 1) % 256) * 8 + (at0 + 1) / 256 % 8)) = p[1];
                *(reinterpret_cast<float*>(dw) + (((at0 + 2) / 256 / 8 * 256 + (at0 + 2) % 256) * 8 + (at0 + 2) / 256 % 8)) = p[2];
                *(reinterpret_cast<float*>(dw) + (((at0 + 3) / 256 / 8 * 256 + (at0 + 3) % 256) * 8 + (at0 + 3) / 256 % 8)) = p[3];
            }
            __syncthreads();
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

    // Cleanup
    __syncthreads();
}

} // extern "C"
