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

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_d874fb9fe6cdc5b5d292(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    float* sums = reinterpret_cast<float*>(smem_raw + 1136);
    const int sums_addr = smem + 1136;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1136);
    const int dwbuf_addr = smem + 1136;
    int* meta = reinterpret_cast<int*>(smem_raw + 1104);
    const int meta_addr = smem + 1104;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 6272);
    const int ring_addr = smem + 6272;

    // Mbarrier init (4 barriers at smem_raw[0..32))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
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
    int rg = warp / 5;
    int tir = tid - rg * 160;
    unsigned int ww[4];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    float acc[8];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    int c = 0;
    int first_c = 0;
    sums[rg * 1280 + tir] = 0.0f;
    sums[rg * 1280 + 160 + tir] = 0.0f;
    sums[rg * 1280 + 320 + tir] = 0.0f;
    sums[rg * 1280 + 480 + tir] = 0.0f;
    sums[rg * 1280 + 640 + tir] = 0.0f;
    sums[rg * 1280 + 800 + tir] = 0.0f;
    sums[rg * 1280 + 960 + tir] = 0.0f;
    sums[rg * 1280 + 1120 + tir] = 0.0f;
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[4];
    unsigned int xw[4];
    unsigned int hw[1];
    float rr[2];
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
            int _min_2 = ((begin_it) < (T) ? (begin_it) : (T));
            prow = _min_2;
            int _min_3 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
            pend = _min_3;
            unsigned int _atomic_fetch_add_4 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_4;
        }
        int slot = 0;
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
                    int _min_5 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                    prow = _min_5;
                    int _min_6 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                    pend = _min_6;
                    unsigned int _atomic_fetch_add_7 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_7;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_8 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval = _min_8;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 5120);
            int sbase = ring_at + slot * 5120;
            int _min_9 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_9);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 2560), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr),
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
                    int _min_10 = ((begin_it_2) < (T) ? (begin_it_2) : (T));
                    prow = _min_10;
                    int _min_11 = ((begin_it_2 + span_2) < (T) ? (begin_it_2 + span_2) : (T));
                    pend = _min_11;
                    unsigned int _atomic_fetch_add_12 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_12;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_13 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_0 = _min_13;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 5120);
            int sbase_1 = ring_at + slot_0 * 5120;
            int _min_14 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_1 = (long long)(prow + _min_14);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 2560), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 8),
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
                    int _min_15 = ((begin_it_3) < (T) ? (begin_it_3) : (T));
                    prow = _min_15;
                    int _min_16 = ((begin_it_3 + span_3) < (T) ? (begin_it_3 + span_3) : (T));
                    pend = _min_16;
                    unsigned int _atomic_fetch_add_17 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_17;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_18 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_1 = _min_18;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 5120);
            int sbase_2 = ring_at + slot_1 * 5120;
            int _min_19 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_19);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 2560), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr + 16);
        }
        prow = prow + nval_1;
        int slot_2 = 3;
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
                    for (int k_4 = 0; k_4 < 5; k_4++) {
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
                    int _min_20 = ((begin_it_4) < (T) ? (begin_it_4) : (T));
                    prow = _min_20;
                    int _min_21 = ((begin_it_4 + span_4) < (T) ? (begin_it_4 + span_4) : (T));
                    pend = _min_21;
                    unsigned int _atomic_fetch_add_22 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_22;
                }
            }
        }
        int nval_2 = 0;
        if (prow < pend) {
            int _min_23 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_2 = _min_23;
        }
        meta[2 * slot_2] = prow;
        meta[2 * slot_2 + 1] = nval_2;
        if (nval_2 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 24, 5120);
            int sbase_3 = ring_at + slot_2 * 5120;
            int _min_24 = ((nval_2 - 1) < (0) ? (nval_2 - 1) : (0));
            long long row_i_3 = (long long)(prow + _min_24);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + 24),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 2560), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + 24),
                   "l"(0x12F0000000000000ULL) : "memory");
        } else {
            mbarrier_arrive(full_addr + 24);
        }
        prow = prow + nval_2;
    }
    __syncthreads();
    int row0 = meta[0];
    int nval_3 = meta[1];
    rr[0] = 0.0f;
    if (nval_3 > rg) {
        rr[0] = r[row0 + rg];
    }
    float dot[1];
    #pragma unroll 1
    for (int k_5 = 0; k_5 < T + n_chunks; k_5++) {
        if (nval_3 < 1) {
            break;
        }
        row0_n = meta[2 * ((k_5 + 1) % 4)];
        nval_n = meta[2 * ((k_5 + 1) % 4) + 1];
        if (nval_n > rg) {
            rr[1] = r[row0_n + rg];
        }
        int _stage_full = (int)((unsigned int)k_5 % 4);
        int _parity_full = (int)((unsigned int)k_5 / 4 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_5 % 4 * 5120 + rg * 2560;
        __int128_t _smem_raw_25;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_25) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[3];
        __int128_t _smem_raw_26;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_26) : "r"(rbase + 2560 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[3];
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
        float _warp_reduce_27 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_27 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_27, offset);
        dot[0] = _warp_reduce_27;
        if (lane == 0) {
            dots[(k_5 & 1) * 5 + warp] = dot[0];
        }
        __syncthreads();
        float tot = 0.0f;
        tot += dots[(k_5 & 1) * 5 + rg * 5];
        tot += dots[(k_5 & 1) * 5 + rg * 5 + 1];
        tot += dots[(k_5 & 1) * 5 + rg * 5 + 2];
        tot += dots[(k_5 & 1) * 5 + rg * 5 + 3];
        tot += dots[(k_5 & 1) * 5 + rg * 5 + 4];
        dot[0] = tot;
        if (tid == 0) {
            int slot_3 = (k_5 + 4) % 4;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt;
                    if (pc < n_chunks) {
                        int stages_5 = T;
                        int sl_5 = rows_per_chunk;
                        if (sl_5 > stages_5) {
                            sl_5 = stages_5;
                        }
                        if (sl_5 < 1) {
                            sl_5 = 1;
                        }
                        int share_5 = (stages_5 + sl_5 - 1) / sl_5;
                        int tail_5 = (share_5 + 11) / 12;
                        if (tail_5 < 1) {
                            tail_5 = 1;
                        }
                        int rem_5 = stages_5;
                        int size_5 = (share_5 + 1) / 2;
                        int first_5 = 0;
                        int base_5 = 0;
                        int span_5 = 0;
                        int halving_5 = 1;
                        int begin_it_5 = 0;
                        #pragma unroll 1
                        for (int k_0 = 0; k_0 < 5; k_0++) {
                            if (halving_5 != 0) {
                                if (size_5 > tail_5) {
                                    if (rem_5 >= sl_5 * size_5) {
                                        if (pc < first_5 + sl_5) {
                                            span_5 = size_5;
                                            begin_it_5 = base_5 + (pc - first_5) * size_5;
                                            halving_5 = 0;
                                        } else {
                                            first_5 = first_5 + sl_5;
                                            base_5 = base_5 + sl_5 * size_5;
                                            rem_5 = rem_5 - sl_5 * size_5;
                                            size_5 = (size_5 + 1) / 2;
                                        }
                                    } else {
                                        halving_5 = 0;
                                    }
                                } else {
                                    halving_5 = 0;
                                }
                            }
                        }
                        if (span_5 == 0) {
                            int quot_5 = rem_5 / sl_5;
                            int extra_5 = rem_5 - quot_5 * sl_5;
                            if (pc < first_5 + extra_5) {
                                span_5 = quot_5 + 1;
                                begin_it_5 = base_5 + (pc - first_5) * (quot_5 + 1);
                            } else if (pc < first_5 + sl_5) {
                                span_5 = quot_5;
                                begin_it_5 = base_5 + extra_5 * (quot_5 + 1) + (pc - first_5 - extra_5) * quot_5;
                            } else {
                                begin_it_5 = stages_5;
                            }
                        }
                        int _min_28 = ((begin_it_5) < (T) ? (begin_it_5) : (T));
                        prow = _min_28;
                        int _min_29 = ((begin_it_5 + span_5) < (T) ? (begin_it_5 + span_5) : (T));
                        pend = _min_29;
                        unsigned int _atomic_fetch_add_30 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_30;
                    }
                }
            }
            int nval_3_1 = 0;
            if (prow < pend) {
                int _min_31 = ((pend - prow) < (1) ? (pend - prow) : (1));
                nval_3_1 = _min_31;
            }
            meta[2 * slot_3] = prow;
            meta[2 * slot_3 + 1] = nval_3_1;
            if (nval_3_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8, 5120);
                int sbase_4 = ring_at + slot_3 * 5120;
                int _min_32 = ((nval_3_1 - 1) < (0) ? (nval_3_1 - 1) : (0));
                long long row_i_4 = (long long)(prow + _min_32);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4), "l"(reinterpret_cast<const uint8_t*>(g + row_i_4 * g_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 2560), "l"(reinterpret_cast<const uint8_t*>(x + row_i_4 * x_stride)), "r"((unsigned)(2560)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
            } else {
                mbarrier_arrive(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8);
            }
            prow = prow + nval_3_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 1280.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row0 + rg < row0 + nval_3) {
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1280ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        row0 = row0_n;
        nval_3 = nval_n;
        rr[0] = rr[1];
        if ((k_5 + 1 & 15) == 0) {
            sums[rg * 1280 + tir] = sums[rg * 1280 + tir] + acc[0];
            acc[0] = 0.0f;
            sums[rg * 1280 + 160 + tir] = sums[rg * 1280 + 160 + tir] + acc[1];
            acc[1] = 0.0f;
            sums[rg * 1280 + 320 + tir] = sums[rg * 1280 + 320 + tir] + acc[2];
            acc[2] = 0.0f;
            sums[rg * 1280 + 480 + tir] = sums[rg * 1280 + 480 + tir] + acc[3];
            acc[3] = 0.0f;
            sums[rg * 1280 + 640 + tir] = sums[rg * 1280 + 640 + tir] + acc[4];
            acc[4] = 0.0f;
            sums[rg * 1280 + 800 + tir] = sums[rg * 1280 + 800 + tir] + acc[5];
            acc[5] = 0.0f;
            sums[rg * 1280 + 960 + tir] = sums[rg * 1280 + 960 + tir] + acc[6];
            acc[6] = 0.0f;
            sums[rg * 1280 + 1120 + tir] = sums[rg * 1280 + 1120 + tir] + acc[7];
            acc[7] = 0.0f;
        }
    }
    sums[rg * 1280 + tir] = sums[rg * 1280 + tir] + acc[0];
    acc[0] = 0.0f;
    sums[rg * 1280 + 160 + tir] = sums[rg * 1280 + 160 + tir] + acc[1];
    acc[1] = 0.0f;
    sums[rg * 1280 + 320 + tir] = sums[rg * 1280 + 320 + tir] + acc[2];
    acc[2] = 0.0f;
    sums[rg * 1280 + 480 + tir] = sums[rg * 1280 + 480 + tir] + acc[3];
    acc[3] = 0.0f;
    sums[rg * 1280 + 640 + tir] = sums[rg * 1280 + 640 + tir] + acc[4];
    acc[4] = 0.0f;
    sums[rg * 1280 + 800 + tir] = sums[rg * 1280 + 800 + tir] + acc[5];
    acc[5] = 0.0f;
    sums[rg * 1280 + 960 + tir] = sums[rg * 1280 + 960 + tir] + acc[6];
    acc[6] = 0.0f;
    sums[rg * 1280 + 1120 + tir] = sums[rg * 1280 + 1120 + tir] + acc[7];
    acc[7] = 0.0f;
    c = pc;
    int _min_33 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
    int nrep = ((_min_33) < (25) ? (_min_33) : (25));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_34 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_34), "r"(sums_addr + 0), "r"((uint32_t)(1280))
                    : "memory");
            }
            {
                void* _cpred_dst_35 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280 + 320));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_35), "r"(sums_addr + 1280), "r"((uint32_t)(1280))
                    : "memory");
            }
            {
                void* _cpred_dst_36 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280 + 640));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_36), "r"(sums_addr + 2560), "r"((uint32_t)(1280))
                    : "memory");
            }
            {
                void* _cpred_dst_37 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1280 + 960));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_37), "r"(sums_addr + 3840), "r"((uint32_t)(1280))
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
        float buf[25];
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_38;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_38) : "l"((const void*)(partial + (tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_38);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_39;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_39) : "l"((const void*)(partial + (1280 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_39);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_40;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_40) : "l"((const void*)(partial + (2560 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_40);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_41;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_41) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_41);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_42;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_42) : "l"((const void*)(partial + (5120 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_42);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_43;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_43) : "l"((const void*)(partial + (6400 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_43);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_44;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_44) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_44);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_45;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_45) : "l"((const void*)(partial + (8960 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_45);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_46;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_46) : "l"((const void*)(partial + (10240 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_46);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_47;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_47) : "l"((const void*)(partial + (11520 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_47);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_48;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_48) : "l"((const void*)(partial + (12800 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_48);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_49;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_49) : "l"((const void*)(partial + (14080 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_49);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_50;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_50) : "l"((const void*)(partial + (15360 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_50);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_51;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_51) : "l"((const void*)(partial + (16640 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_51);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_52;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_52) : "l"((const void*)(partial + (17920 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_52);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_53;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_53) : "l"((const void*)(partial + (19200 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_53);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_54;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_54) : "l"((const void*)(partial + (20480 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_54);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_55;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_55) : "l"((const void*)(partial + (21760 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_55);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_56;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_56) : "l"((const void*)(partial + (23040 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_56);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_57;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (24320 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_57);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_58;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (25600 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_58);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_59;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (26880 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_59);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_60;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (28160 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_60);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_61;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (29440 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_61);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_62;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (30720 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_62);
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
        int at_w_0 = tid;
        dwbuf[(at_w_0 / 160 / 8 * 160 + at_w_0 % 160) * 8 + at_w_0 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_0)) = 0.0f;
        }
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
                uint32_t _scalar_bits_63;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (160 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_63);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_64;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (1440 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_64);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_65;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (2720 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_65);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_66;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (4000 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_66);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_67;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (5280 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_67);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_68;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (6560 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_68);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_69;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (7840 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_69);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_70;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_70) : "l"((const void*)(partial + (9120 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_70);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_71;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_71) : "l"((const void*)(partial + (10400 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_71);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_72;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_72) : "l"((const void*)(partial + (11680 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_72);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_73;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_73) : "l"((const void*)(partial + (12960 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_73);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_74;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_74) : "l"((const void*)(partial + (14240 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_74);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_75;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_75) : "l"((const void*)(partial + (15520 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_75);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_76;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_76) : "l"((const void*)(partial + (16800 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_76);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_77;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_77) : "l"((const void*)(partial + (18080 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_77);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_78;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_78) : "l"((const void*)(partial + (19360 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_78);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_79;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_79) : "l"((const void*)(partial + (20640 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_79);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_80;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_80) : "l"((const void*)(partial + (21920 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_80);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_81;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_81) : "l"((const void*)(partial + (23200 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_81);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_82;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_82) : "l"((const void*)(partial + (24480 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_82);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_83;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_83) : "l"((const void*)(partial + (25760 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_83);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_84;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_84) : "l"((const void*)(partial + (27040 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_84);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_85;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_85) : "l"((const void*)(partial + (28320 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_85);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_86;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_86) : "l"((const void*)(partial + (29600 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_86);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_87;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_87) : "l"((const void*)(partial + (30880 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_87);
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
        int at_w_1 = 160 + tid;
        dwbuf[(at_w_1 / 160 / 8 * 160 + at_w_1 % 160) * 8 + at_w_1 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_1)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_1)) = 0.0f;
        }
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
                uint32_t _scalar_bits_88;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_88) : "l"((const void*)(partial + (320 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_88);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_89;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_89) : "l"((const void*)(partial + (1600 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_89);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_90;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_90) : "l"((const void*)(partial + (2880 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_90);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_91;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_91) : "l"((const void*)(partial + (4160 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_91);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_92;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_92) : "l"((const void*)(partial + (5440 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_92);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_93;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_93) : "l"((const void*)(partial + (6720 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_93);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_94;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_94) : "l"((const void*)(partial + (8000 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_94);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_95;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_95) : "l"((const void*)(partial + (9280 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_95);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_96;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_96) : "l"((const void*)(partial + (10560 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_96);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_97;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_97) : "l"((const void*)(partial + (11840 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_97);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_98;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_98) : "l"((const void*)(partial + (13120 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_98);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_99;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_99) : "l"((const void*)(partial + (14400 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_99);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_100;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_100) : "l"((const void*)(partial + (15680 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_100);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_101;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_101) : "l"((const void*)(partial + (16960 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_101);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_102;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_102) : "l"((const void*)(partial + (18240 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_102);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_103;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_103) : "l"((const void*)(partial + (19520 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_103);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_104;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_104) : "l"((const void*)(partial + (20800 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_104);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_105;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_105) : "l"((const void*)(partial + (22080 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_105);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_106;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_106) : "l"((const void*)(partial + (23360 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_106);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_107;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_107) : "l"((const void*)(partial + (24640 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_107);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_108;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_108) : "l"((const void*)(partial + (25920 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_108);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_109;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_109) : "l"((const void*)(partial + (27200 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_109);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_110;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_110) : "l"((const void*)(partial + (28480 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_110);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_111;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_111) : "l"((const void*)(partial + (29760 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_111);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_112;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_112) : "l"((const void*)(partial + (31040 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_112);
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
        int at_w_2 = 320 + tid;
        dwbuf[(at_w_2 / 160 / 8 * 160 + at_w_2 % 160) * 8 + at_w_2 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_2)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_2)) = 0.0f;
        }
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
                uint32_t _scalar_bits_113;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_113) : "l"((const void*)(partial + (480 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_113);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_114;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_114) : "l"((const void*)(partial + (1760 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_114);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_115;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_115) : "l"((const void*)(partial + (3040 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_115);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_116;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_116) : "l"((const void*)(partial + (4320 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_116);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_117;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_117) : "l"((const void*)(partial + (5600 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_117);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_118;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_118) : "l"((const void*)(partial + (6880 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_118);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_119;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_119) : "l"((const void*)(partial + (8160 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_119);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_120;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_120) : "l"((const void*)(partial + (9440 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_120);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_121;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_121) : "l"((const void*)(partial + (10720 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_121);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_122;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_122) : "l"((const void*)(partial + (12000 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_122);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_123;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_123) : "l"((const void*)(partial + (13280 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_123);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_124;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_124) : "l"((const void*)(partial + (14560 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_124);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_125;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_125) : "l"((const void*)(partial + (15840 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_125);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_126;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_126) : "l"((const void*)(partial + (17120 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_126);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_127;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_127) : "l"((const void*)(partial + (18400 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_127);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_128;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_128) : "l"((const void*)(partial + (19680 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_128);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_129;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_129) : "l"((const void*)(partial + (20960 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_129);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_130;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_130) : "l"((const void*)(partial + (22240 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_130);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_131;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_131) : "l"((const void*)(partial + (23520 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_131);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_132;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_132) : "l"((const void*)(partial + (24800 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_132);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_133;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_133) : "l"((const void*)(partial + (26080 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_133);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_134;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_134) : "l"((const void*)(partial + (27360 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_134);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_135;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_135) : "l"((const void*)(partial + (28640 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_135);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_136;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_136) : "l"((const void*)(partial + (29920 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_136);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_137;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_137) : "l"((const void*)(partial + (31200 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_137);
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
        int at_w_3 = 480 + tid;
        dwbuf[(at_w_3 / 160 / 8 * 160 + at_w_3 % 160) * 8 + at_w_3 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_3)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_3)) = 0.0f;
        }
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
                uint32_t _scalar_bits_138;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_138) : "l"((const void*)(partial + (640 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_138);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_139;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_139) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_139);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_140;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_140) : "l"((const void*)(partial + (3200 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_140);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_141;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_141) : "l"((const void*)(partial + (4480 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_141);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_142;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_142) : "l"((const void*)(partial + (5760 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_142);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_143;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_143) : "l"((const void*)(partial + (7040 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_143);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_144;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_144) : "l"((const void*)(partial + (8320 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_144);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_145;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_145) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_145);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_146;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_146) : "l"((const void*)(partial + (10880 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_146);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_147;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_147) : "l"((const void*)(partial + (12160 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_147);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_148;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_148) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_148);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_149;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_149) : "l"((const void*)(partial + (14720 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_149);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_150;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_150) : "l"((const void*)(partial + (16000 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_150);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_151;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_151) : "l"((const void*)(partial + (17280 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_151);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_152;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_152) : "l"((const void*)(partial + (18560 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_152);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_153;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_153) : "l"((const void*)(partial + (19840 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_153);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_154;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_154) : "l"((const void*)(partial + (21120 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_154);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_155;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_155) : "l"((const void*)(partial + (22400 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_155);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_156;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_156) : "l"((const void*)(partial + (23680 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_156);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_157;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_157) : "l"((const void*)(partial + (24960 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_157);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_158;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_158) : "l"((const void*)(partial + (26240 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_158);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_159;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_159) : "l"((const void*)(partial + (27520 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_159);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_160;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_160) : "l"((const void*)(partial + (28800 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_160);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_161;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_161) : "l"((const void*)(partial + (30080 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_161);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_162;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_162) : "l"((const void*)(partial + (31360 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_162);
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
        int at_w_4 = 640 + tid;
        dwbuf[(at_w_4 / 160 / 8 * 160 + at_w_4 % 160) * 8 + at_w_4 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_4)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_4)) = 0.0f;
        }
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
                uint32_t _scalar_bits_163;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_163) : "l"((const void*)(partial + (800 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_163);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_164;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_164) : "l"((const void*)(partial + (2080 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_164);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_165;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_165) : "l"((const void*)(partial + (3360 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_165);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_166;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_166) : "l"((const void*)(partial + (4640 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_166);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_167;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_167) : "l"((const void*)(partial + (5920 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_167);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_168;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_168) : "l"((const void*)(partial + (7200 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_168);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_169;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_169) : "l"((const void*)(partial + (8480 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_169);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_170;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_170) : "l"((const void*)(partial + (9760 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_170);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_171;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_171) : "l"((const void*)(partial + (11040 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_171);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_172;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_172) : "l"((const void*)(partial + (12320 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_172);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_173;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_173) : "l"((const void*)(partial + (13600 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_173);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_174;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_174) : "l"((const void*)(partial + (14880 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_174);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_175;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_175) : "l"((const void*)(partial + (16160 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_175);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_176;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_176) : "l"((const void*)(partial + (17440 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_176);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_177;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_177) : "l"((const void*)(partial + (18720 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_177);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_178;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_178) : "l"((const void*)(partial + (20000 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_178);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_179;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_179) : "l"((const void*)(partial + (21280 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_179);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_180;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_180) : "l"((const void*)(partial + (22560 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_180);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_181;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_181) : "l"((const void*)(partial + (23840 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_181);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_182;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_182) : "l"((const void*)(partial + (25120 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_182);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_183;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_183) : "l"((const void*)(partial + (26400 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_183);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_184;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_184) : "l"((const void*)(partial + (27680 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_184);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_185;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_185) : "l"((const void*)(partial + (28960 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_185);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_186;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_186) : "l"((const void*)(partial + (30240 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_186);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_187;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_187) : "l"((const void*)(partial + (31520 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_187);
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
        int at_w_5 = 800 + tid;
        dwbuf[(at_w_5 / 160 / 8 * 160 + at_w_5 % 160) * 8 + at_w_5 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_5)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_5)) = 0.0f;
        }
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
                uint32_t _scalar_bits_188;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_188) : "l"((const void*)(partial + (960 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_188);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_189;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_189) : "l"((const void*)(partial + (2240 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_189);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_190;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_190) : "l"((const void*)(partial + (3520 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_190);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_191;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_191) : "l"((const void*)(partial + (4800 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_191);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_192;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_192) : "l"((const void*)(partial + (6080 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_192);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_193;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_193) : "l"((const void*)(partial + (7360 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_193);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_194;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_194) : "l"((const void*)(partial + (8640 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_194);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_195;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_195) : "l"((const void*)(partial + (9920 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_195);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_196;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_196) : "l"((const void*)(partial + (11200 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_196);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_197;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_197) : "l"((const void*)(partial + (12480 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_197);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_198;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_198) : "l"((const void*)(partial + (13760 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_198);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_199;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_199) : "l"((const void*)(partial + (15040 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_199);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_200;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_200) : "l"((const void*)(partial + (16320 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_200);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_201;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_201) : "l"((const void*)(partial + (17600 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_201);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_202;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_202) : "l"((const void*)(partial + (18880 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_202);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_203;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_203) : "l"((const void*)(partial + (20160 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_203);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_204;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_204) : "l"((const void*)(partial + (21440 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_204);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_205;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_205) : "l"((const void*)(partial + (22720 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_205);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_206;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_206) : "l"((const void*)(partial + (24000 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_206);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_207;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_207) : "l"((const void*)(partial + (25280 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_207);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_208;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_208) : "l"((const void*)(partial + (26560 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_208);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_209;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_209) : "l"((const void*)(partial + (27840 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_209);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_210;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_210) : "l"((const void*)(partial + (29120 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_210);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_211;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_211) : "l"((const void*)(partial + (30400 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_211);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_212;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_212) : "l"((const void*)(partial + (31680 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_212);
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
        int at_w_6 = 960 + tid;
        dwbuf[(at_w_6 / 160 / 8 * 160 + at_w_6 % 160) * 8 + at_w_6 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_6)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_6)) = 0.0f;
        }
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
                uint32_t _scalar_bits_213;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_213) : "l"((const void*)(partial + (1120 + tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_213);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_214;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_214) : "l"((const void*)(partial + (2400 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_214);
            }
        }
        buf[2] = 0.0f;
        if (2 < nrep) {
            {
                uint32_t _scalar_bits_215;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_215) : "l"((const void*)(partial + (3680 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_215);
            }
        }
        buf[3] = 0.0f;
        if (3 < nrep) {
            {
                uint32_t _scalar_bits_216;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_216) : "l"((const void*)(partial + (4960 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_216);
            }
        }
        buf[4] = 0.0f;
        if (4 < nrep) {
            {
                uint32_t _scalar_bits_217;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_217) : "l"((const void*)(partial + (6240 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_217);
            }
        }
        buf[5] = 0.0f;
        if (5 < nrep) {
            {
                uint32_t _scalar_bits_218;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_218) : "l"((const void*)(partial + (7520 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_218);
            }
        }
        buf[6] = 0.0f;
        if (6 < nrep) {
            {
                uint32_t _scalar_bits_219;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_219) : "l"((const void*)(partial + (8800 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_219);
            }
        }
        buf[7] = 0.0f;
        if (7 < nrep) {
            {
                uint32_t _scalar_bits_220;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_220) : "l"((const void*)(partial + (10080 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_220);
            }
        }
        buf[8] = 0.0f;
        if (8 < nrep) {
            {
                uint32_t _scalar_bits_221;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_221) : "l"((const void*)(partial + (11360 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_221);
            }
        }
        buf[9] = 0.0f;
        if (9 < nrep) {
            {
                uint32_t _scalar_bits_222;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_222) : "l"((const void*)(partial + (12640 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_222);
            }
        }
        buf[10] = 0.0f;
        if (10 < nrep) {
            {
                uint32_t _scalar_bits_223;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_223) : "l"((const void*)(partial + (13920 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_223);
            }
        }
        buf[11] = 0.0f;
        if (11 < nrep) {
            {
                uint32_t _scalar_bits_224;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_224) : "l"((const void*)(partial + (15200 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_224);
            }
        }
        buf[12] = 0.0f;
        if (12 < nrep) {
            {
                uint32_t _scalar_bits_225;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_225) : "l"((const void*)(partial + (16480 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_225);
            }
        }
        buf[13] = 0.0f;
        if (13 < nrep) {
            {
                uint32_t _scalar_bits_226;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_226) : "l"((const void*)(partial + (17760 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_226);
            }
        }
        buf[14] = 0.0f;
        if (14 < nrep) {
            {
                uint32_t _scalar_bits_227;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_227) : "l"((const void*)(partial + (19040 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_227);
            }
        }
        buf[15] = 0.0f;
        if (15 < nrep) {
            {
                uint32_t _scalar_bits_228;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_228) : "l"((const void*)(partial + (20320 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_228);
            }
        }
        buf[16] = 0.0f;
        if (16 < nrep) {
            {
                uint32_t _scalar_bits_229;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_229) : "l"((const void*)(partial + (21600 + tid))) : "memory");
                buf[16] = __uint_as_float(_scalar_bits_229);
            }
        }
        buf[17] = 0.0f;
        if (17 < nrep) {
            {
                uint32_t _scalar_bits_230;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_230) : "l"((const void*)(partial + (22880 + tid))) : "memory");
                buf[17] = __uint_as_float(_scalar_bits_230);
            }
        }
        buf[18] = 0.0f;
        if (18 < nrep) {
            {
                uint32_t _scalar_bits_231;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_231) : "l"((const void*)(partial + (24160 + tid))) : "memory");
                buf[18] = __uint_as_float(_scalar_bits_231);
            }
        }
        buf[19] = 0.0f;
        if (19 < nrep) {
            {
                uint32_t _scalar_bits_232;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_232) : "l"((const void*)(partial + (25440 + tid))) : "memory");
                buf[19] = __uint_as_float(_scalar_bits_232);
            }
        }
        buf[20] = 0.0f;
        if (20 < nrep) {
            {
                uint32_t _scalar_bits_233;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_233) : "l"((const void*)(partial + (26720 + tid))) : "memory");
                buf[20] = __uint_as_float(_scalar_bits_233);
            }
        }
        buf[21] = 0.0f;
        if (21 < nrep) {
            {
                uint32_t _scalar_bits_234;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_234) : "l"((const void*)(partial + (28000 + tid))) : "memory");
                buf[21] = __uint_as_float(_scalar_bits_234);
            }
        }
        buf[22] = 0.0f;
        if (22 < nrep) {
            {
                uint32_t _scalar_bits_235;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_235) : "l"((const void*)(partial + (29280 + tid))) : "memory");
                buf[22] = __uint_as_float(_scalar_bits_235);
            }
        }
        buf[23] = 0.0f;
        if (23 < nrep) {
            {
                uint32_t _scalar_bits_236;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_236) : "l"((const void*)(partial + (30560 + tid))) : "memory");
                buf[23] = __uint_as_float(_scalar_bits_236);
            }
        }
        buf[24] = 0.0f;
        if (24 < nrep) {
            {
                uint32_t _scalar_bits_237;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_237) : "l"((const void*)(partial + (31840 + tid))) : "memory");
                buf[24] = __uint_as_float(_scalar_bits_237);
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
        int at_w_7 = 1120 + tid;
        dwbuf[(at_w_7 / 160 / 8 * 160 + at_w_7 % 160) * 8 + at_w_7 / 160 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_7)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1280 + at_w_7)) = 0.0f;
        }
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
        int col_1 = (160 + tid) * 4;
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
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
