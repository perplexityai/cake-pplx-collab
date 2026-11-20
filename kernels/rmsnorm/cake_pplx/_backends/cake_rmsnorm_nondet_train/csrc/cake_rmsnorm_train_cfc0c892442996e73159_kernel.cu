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
kernel_cake_rmsnorm_train_cfc0c892442996e73159(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    int rg = warp;
    int tir = tid - rg * 32;
    unsigned int ww[16];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((32 + tir) * 8));
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(w + (unsigned long long)((64 + tir) * 8));
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&ww[8]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_3[_blk] = _vptr_3[_blk];
        }
    }
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(w + (unsigned long long)((96 + tir) * 8));
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&ww[12]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_4[_blk] = _vptr_4[_blk];
        }
    }
    float acc[32];
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
    int c = 0;
    int first_c = 0;
    sums[rg * 1024 + tir] = 0.0f;
    sums[rg * 1024 + 32 + tir] = 0.0f;
    sums[rg * 1024 + 64 + tir] = 0.0f;
    sums[rg * 1024 + 96 + tir] = 0.0f;
    sums[rg * 1024 + 128 + tir] = 0.0f;
    sums[rg * 1024 + 160 + tir] = 0.0f;
    sums[rg * 1024 + 192 + tir] = 0.0f;
    sums[rg * 1024 + 224 + tir] = 0.0f;
    sums[rg * 1024 + 256 + tir] = 0.0f;
    sums[rg * 1024 + 288 + tir] = 0.0f;
    sums[rg * 1024 + 320 + tir] = 0.0f;
    sums[rg * 1024 + 352 + tir] = 0.0f;
    sums[rg * 1024 + 384 + tir] = 0.0f;
    sums[rg * 1024 + 416 + tir] = 0.0f;
    sums[rg * 1024 + 448 + tir] = 0.0f;
    sums[rg * 1024 + 480 + tir] = 0.0f;
    sums[rg * 1024 + 512 + tir] = 0.0f;
    sums[rg * 1024 + 544 + tir] = 0.0f;
    sums[rg * 1024 + 576 + tir] = 0.0f;
    sums[rg * 1024 + 608 + tir] = 0.0f;
    sums[rg * 1024 + 640 + tir] = 0.0f;
    sums[rg * 1024 + 672 + tir] = 0.0f;
    sums[rg * 1024 + 704 + tir] = 0.0f;
    sums[rg * 1024 + 736 + tir] = 0.0f;
    sums[rg * 1024 + 768 + tir] = 0.0f;
    sums[rg * 1024 + 800 + tir] = 0.0f;
    sums[rg * 1024 + 832 + tir] = 0.0f;
    sums[rg * 1024 + 864 + tir] = 0.0f;
    sums[rg * 1024 + 896 + tir] = 0.0f;
    sums[rg * 1024 + 928 + tir] = 0.0f;
    sums[rg * 1024 + 960 + tir] = 0.0f;
    sums[rg * 1024 + 992 + tir] = 0.0f;
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[16];
    unsigned int xw[16];
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
            int _min_5 = ((begin_it * 4) < (T) ? (begin_it * 4) : (T));
            prow = _min_5;
            int _min_6 = ((begin_it * 4 + span * 4) < (T) ? (begin_it * 4 + span * 4) : (T));
            pend = _min_6;
            unsigned int _atomic_fetch_add_7 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_7;
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
                    int _min_8 = ((begin_it_1 * 4) < (T) ? (begin_it_1 * 4) : (T));
                    prow = _min_8;
                    int _min_9 = ((begin_it_1 * 4 + span_1 * 4) < (T) ? (begin_it_1 * 4 + span_1 * 4) : (T));
                    pend = _min_9;
                    unsigned int _atomic_fetch_add_10 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_10;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_11 = ((pend - prow) < (4) ? (pend - prow) : (4));
            nval = _min_11;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 16384);
            int sbase = ring_at + slot * 16384;
            int _min_12 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_12);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_13 = ((nval - 1) < (1) ? (nval - 1) : (1));
            long long row_i_1 = (long long)(prow + _min_13);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 2048), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_14 = ((nval - 1) < (2) ? (nval - 1) : (2));
            long long row_i_2 = (long long)(prow + _min_14);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_15 = ((nval - 1) < (3) ? (nval - 1) : (3));
            long long row_i_3 = (long long)(prow + _min_15);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 6144), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 14336), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr),
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
                    int _min_16 = ((begin_it_2 * 4) < (T) ? (begin_it_2 * 4) : (T));
                    prow = _min_16;
                    int _min_17 = ((begin_it_2 * 4 + span_2 * 4) < (T) ? (begin_it_2 * 4 + span_2 * 4) : (T));
                    pend = _min_17;
                    unsigned int _atomic_fetch_add_18 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_18;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_19 = ((pend - prow) < (4) ? (pend - prow) : (4));
            nval_0 = _min_19;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 16384);
            int sbase_1 = ring_at + slot_0 * 16384;
            int _min_20 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_4 = (long long)(prow + _min_20);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_4 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_4 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_21 = ((nval_0 - 1) < (1) ? (nval_0 - 1) : (1));
            long long row_i_5 = (long long)(prow + _min_21);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 2048), "l"(reinterpret_cast<const uint8_t*>(g + row_i_5 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_5 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_22 = ((nval_0 - 1) < (2) ? (nval_0 - 1) : (2));
            long long row_i_6 = (long long)(prow + _min_22);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_6 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_6 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_23 = ((nval_0 - 1) < (3) ? (nval_0 - 1) : (3));
            long long row_i_7 = (long long)(prow + _min_23);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 6144), "l"(reinterpret_cast<const uint8_t*>(g + row_i_7 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 14336), "l"(reinterpret_cast<const uint8_t*>(x + row_i_7 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 8),
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
                    int _min_24 = ((begin_it_3 * 4) < (T) ? (begin_it_3 * 4) : (T));
                    prow = _min_24;
                    int _min_25 = ((begin_it_3 * 4 + span_3 * 4) < (T) ? (begin_it_3 * 4 + span_3 * 4) : (T));
                    pend = _min_25;
                    unsigned int _atomic_fetch_add_26 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_26;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_27 = ((pend - prow) < (4) ? (pend - prow) : (4));
            nval_1 = _min_27;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 16384);
            int sbase_2 = ring_at + slot_1 * 16384;
            int _min_28 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_8 = (long long)(prow + _min_28);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_8 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_8 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_29 = ((nval_1 - 1) < (1) ? (nval_1 - 1) : (1));
            long long row_i_9 = (long long)(prow + _min_29);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 2048), "l"(reinterpret_cast<const uint8_t*>(g + row_i_9 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_9 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_30 = ((nval_1 - 1) < (2) ? (nval_1 - 1) : (2));
            long long row_i_10 = (long long)(prow + _min_30);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_10 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_10 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            int _min_31 = ((nval_1 - 1) < (3) ? (nval_1 - 1) : (3));
            long long row_i_11 = (long long)(prow + _min_31);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 6144), "l"(reinterpret_cast<const uint8_t*>(g + row_i_11 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 14336), "l"(reinterpret_cast<const uint8_t*>(x + row_i_11 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + 16),
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
        int rbase = ring_at + k_4 % 3 * 16384 + rg * 2048;
        __int128_t _smem_raw_32;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_32) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_32)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_32)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_32)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_32)[3];
        __int128_t _smem_raw_33;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_33) : "r"(rbase + 8192 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_33)[3];
        __int128_t _smem_raw_34;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_34) : "r"(rbase + (32 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[3];
        __int128_t _smem_raw_35;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_35) : "r"(rbase + 8192 + (32 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[3];
        __int128_t _smem_raw_36;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_36) : "r"(rbase + (64 + tir) * 8 * 2));
        gw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[0];
        gw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[1];
        gw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[2];
        gw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[3];
        __int128_t _smem_raw_37;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_37) : "r"(rbase + 8192 + (64 + tir) * 8 * 2));
        xw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[0];
        xw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[1];
        xw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[2];
        xw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[3];
        __int128_t _smem_raw_38;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_38) : "r"(rbase + (96 + tir) * 8 * 2));
        gw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[0];
        gw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[1];
        gw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[2];
        gw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[3];
        __int128_t _smem_raw_39;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_39) : "r"(rbase + 8192 + (96 + tir) * 8 * 2));
        xw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[0];
        xw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[1];
        xw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[2];
        xw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[3];
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
        dq = __fmaf_rn(__uint_as_float(gw[8] << 16) * __uint_as_float(ww[8] << 16), __uint_as_float(xw[8] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[8] & 4294901760u) * __uint_as_float(ww[8] & 4294901760u), __uint_as_float(xw[8] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[9] << 16) * __uint_as_float(ww[9] << 16), __uint_as_float(xw[9] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[9] & 4294901760u) * __uint_as_float(ww[9] & 4294901760u), __uint_as_float(xw[9] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[10] << 16) * __uint_as_float(ww[10] << 16), __uint_as_float(xw[10] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[10] & 4294901760u) * __uint_as_float(ww[10] & 4294901760u), __uint_as_float(xw[10] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[11] << 16) * __uint_as_float(ww[11] << 16), __uint_as_float(xw[11] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[11] & 4294901760u) * __uint_as_float(ww[11] & 4294901760u), __uint_as_float(xw[11] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[12] << 16) * __uint_as_float(ww[12] << 16), __uint_as_float(xw[12] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[12] & 4294901760u) * __uint_as_float(ww[12] & 4294901760u), __uint_as_float(xw[12] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[13] << 16) * __uint_as_float(ww[13] << 16), __uint_as_float(xw[13] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[13] & 4294901760u) * __uint_as_float(ww[13] & 4294901760u), __uint_as_float(xw[13] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[14] << 16) * __uint_as_float(ww[14] << 16), __uint_as_float(xw[14] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[14] & 4294901760u) * __uint_as_float(ww[14] & 4294901760u), __uint_as_float(xw[14] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[15] << 16) * __uint_as_float(ww[15] << 16), __uint_as_float(xw[15] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[15] & 4294901760u) * __uint_as_float(ww[15] & 4294901760u), __uint_as_float(xw[15] & 4294901760u), dq);
        float _warp_reduce_40 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_40 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_40, offset);
        dot[0] = _warp_reduce_40;
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
                        int _min_41 = ((begin_it_4 * 4) < (T) ? (begin_it_4 * 4) : (T));
                        prow = _min_41;
                        int _min_42 = ((begin_it_4 * 4 + span_4 * 4) < (T) ? (begin_it_4 * 4 + span_4 * 4) : (T));
                        pend = _min_42;
                        unsigned int _atomic_fetch_add_43 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_43;
                    }
                }
            }
            int nval_2_1 = 0;
            if (prow < pend) {
                int _min_44 = ((pend - prow) < (4) ? (pend - prow) : (4));
                nval_2_1 = _min_44;
            }
            meta[2 * slot_2] = prow;
            meta[2 * slot_2 + 1] = nval_2_1;
            if (nval_2_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8, 16384);
                int sbase_3 = ring_at + slot_2 * 16384;
                int _min_45 = ((nval_2_1 - 1) < (0) ? (nval_2_1 - 1) : (0));
                long long row_i_12 = (long long)(prow + _min_45);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_12 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_12 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                int _min_46 = ((nval_2_1 - 1) < (1) ? (nval_2_1 - 1) : (1));
                long long row_i_13 = (long long)(prow + _min_46);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 2048), "l"(reinterpret_cast<const uint8_t*>(g + row_i_13 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 10240), "l"(reinterpret_cast<const uint8_t*>(x + row_i_13 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                int _min_47 = ((nval_2_1 - 1) < (2) ? (nval_2_1 - 1) : (2));
                long long row_i_14 = (long long)(prow + _min_47);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_14 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_14 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                int _min_48 = ((nval_2_1 - 1) < (3) ? (nval_2_1 - 1) : (3));
                long long row_i_15 = (long long)(prow + _min_48);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 6144), "l"(reinterpret_cast<const uint8_t*>(g + row_i_15 * g_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 14336), "l"(reinterpret_cast<const uint8_t*>(x + row_i_15 * x_stride)), "r"((unsigned)(2048)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
            } else {
                mbarrier_arrive(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8);
            }
            prow = prow + nval_2_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 1024.0f);
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1024ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1024ull + (unsigned long long)((32 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            float out_2[8];
            float g0_8 = __uint_as_float(gw[8] << 16);
            float g1_8 = __uint_as_float(gw[8] & 4294901760u);
            float x0_8 = __uint_as_float(xw[8] << 16);
            float x1_8 = __uint_as_float(xw[8] & 4294901760u);
            acc[16] = __fmaf_rn(g0_8 * x0_8, rq, acc[16]);
            acc[17] = __fmaf_rn(g1_8 * x1_8, rq, acc[17]);
            float y0_8 = __fmaf_rn(nkx, x0_8, rq * (g0_8 * __uint_as_float(ww[8] << 16)));
            float y1_8 = __fmaf_rn(nkx, x1_8, rq * (g1_8 * __uint_as_float(ww[8] & 4294901760u)));
            out_2[0] = y0_8;
            out_2[1] = y1_8;
            float g0_9 = __uint_as_float(gw[9] << 16);
            float g1_9 = __uint_as_float(gw[9] & 4294901760u);
            float x0_9 = __uint_as_float(xw[9] << 16);
            float x1_9 = __uint_as_float(xw[9] & 4294901760u);
            acc[18] = __fmaf_rn(g0_9 * x0_9, rq, acc[18]);
            acc[19] = __fmaf_rn(g1_9 * x1_9, rq, acc[19]);
            float y0_9 = __fmaf_rn(nkx, x0_9, rq * (g0_9 * __uint_as_float(ww[9] << 16)));
            float y1_9 = __fmaf_rn(nkx, x1_9, rq * (g1_9 * __uint_as_float(ww[9] & 4294901760u)));
            out_2[2] = y0_9;
            out_2[3] = y1_9;
            float g0_10 = __uint_as_float(gw[10] << 16);
            float g1_10 = __uint_as_float(gw[10] & 4294901760u);
            float x0_10 = __uint_as_float(xw[10] << 16);
            float x1_10 = __uint_as_float(xw[10] & 4294901760u);
            acc[20] = __fmaf_rn(g0_10 * x0_10, rq, acc[20]);
            acc[21] = __fmaf_rn(g1_10 * x1_10, rq, acc[21]);
            float y0_10 = __fmaf_rn(nkx, x0_10, rq * (g0_10 * __uint_as_float(ww[10] << 16)));
            float y1_10 = __fmaf_rn(nkx, x1_10, rq * (g1_10 * __uint_as_float(ww[10] & 4294901760u)));
            out_2[4] = y0_10;
            out_2[5] = y1_10;
            float g0_11 = __uint_as_float(gw[11] << 16);
            float g1_11 = __uint_as_float(gw[11] & 4294901760u);
            float x0_11 = __uint_as_float(xw[11] << 16);
            float x1_11 = __uint_as_float(xw[11] & 4294901760u);
            acc[22] = __fmaf_rn(g0_11 * x0_11, rq, acc[22]);
            acc[23] = __fmaf_rn(g1_11 * x1_11, rq, acc[23]);
            float y0_11 = __fmaf_rn(nkx, x0_11, rq * (g0_11 * __uint_as_float(ww[11] << 16)));
            float y1_11 = __fmaf_rn(nkx, x1_11, rq * (g1_11 * __uint_as_float(ww[11] & 4294901760u)));
            out_2[6] = y0_11;
            out_2[7] = y1_11;
            unsigned int packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1024ull + (unsigned long long)((64 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
            float out_3[8];
            float g0_12 = __uint_as_float(gw[12] << 16);
            float g1_12 = __uint_as_float(gw[12] & 4294901760u);
            float x0_12 = __uint_as_float(xw[12] << 16);
            float x1_12 = __uint_as_float(xw[12] & 4294901760u);
            acc[24] = __fmaf_rn(g0_12 * x0_12, rq, acc[24]);
            acc[25] = __fmaf_rn(g1_12 * x1_12, rq, acc[25]);
            float y0_12 = __fmaf_rn(nkx, x0_12, rq * (g0_12 * __uint_as_float(ww[12] << 16)));
            float y1_12 = __fmaf_rn(nkx, x1_12, rq * (g1_12 * __uint_as_float(ww[12] & 4294901760u)));
            out_3[0] = y0_12;
            out_3[1] = y1_12;
            float g0_13 = __uint_as_float(gw[13] << 16);
            float g1_13 = __uint_as_float(gw[13] & 4294901760u);
            float x0_13 = __uint_as_float(xw[13] << 16);
            float x1_13 = __uint_as_float(xw[13] & 4294901760u);
            acc[26] = __fmaf_rn(g0_13 * x0_13, rq, acc[26]);
            acc[27] = __fmaf_rn(g1_13 * x1_13, rq, acc[27]);
            float y0_13 = __fmaf_rn(nkx, x0_13, rq * (g0_13 * __uint_as_float(ww[13] << 16)));
            float y1_13 = __fmaf_rn(nkx, x1_13, rq * (g1_13 * __uint_as_float(ww[13] & 4294901760u)));
            out_3[2] = y0_13;
            out_3[3] = y1_13;
            float g0_14 = __uint_as_float(gw[14] << 16);
            float g1_14 = __uint_as_float(gw[14] & 4294901760u);
            float x0_14 = __uint_as_float(xw[14] << 16);
            float x1_14 = __uint_as_float(xw[14] & 4294901760u);
            acc[28] = __fmaf_rn(g0_14 * x0_14, rq, acc[28]);
            acc[29] = __fmaf_rn(g1_14 * x1_14, rq, acc[29]);
            float y0_14 = __fmaf_rn(nkx, x0_14, rq * (g0_14 * __uint_as_float(ww[14] << 16)));
            float y1_14 = __fmaf_rn(nkx, x1_14, rq * (g1_14 * __uint_as_float(ww[14] & 4294901760u)));
            out_3[4] = y0_14;
            out_3[5] = y1_14;
            float g0_15 = __uint_as_float(gw[15] << 16);
            float g1_15 = __uint_as_float(gw[15] & 4294901760u);
            float x0_15 = __uint_as_float(xw[15] << 16);
            float x1_15 = __uint_as_float(xw[15] & 4294901760u);
            acc[30] = __fmaf_rn(g0_15 * x0_15, rq, acc[30]);
            acc[31] = __fmaf_rn(g1_15 * x1_15, rq, acc[31]);
            float y0_15 = __fmaf_rn(nkx, x0_15, rq * (g0_15 * __uint_as_float(ww[15] << 16)));
            float y1_15 = __fmaf_rn(nkx, x1_15, rq * (g1_15 * __uint_as_float(ww[15] & 4294901760u)));
            out_3[6] = y0_15;
            out_3[7] = y1_15;
            unsigned int packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 1024ull + (unsigned long long)((96 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        row0 = row0_n;
        nval_2 = nval_n;
        rr[0] = rr[1];
        if ((k_4 + 1 & 15) == 0) {
            sums[rg * 1024 + tir] = sums[rg * 1024 + tir] + acc[0];
            acc[0] = 0.0f;
            sums[rg * 1024 + 32 + tir] = sums[rg * 1024 + 32 + tir] + acc[1];
            acc[1] = 0.0f;
            sums[rg * 1024 + 64 + tir] = sums[rg * 1024 + 64 + tir] + acc[2];
            acc[2] = 0.0f;
            sums[rg * 1024 + 96 + tir] = sums[rg * 1024 + 96 + tir] + acc[3];
            acc[3] = 0.0f;
            sums[rg * 1024 + 128 + tir] = sums[rg * 1024 + 128 + tir] + acc[4];
            acc[4] = 0.0f;
            sums[rg * 1024 + 160 + tir] = sums[rg * 1024 + 160 + tir] + acc[5];
            acc[5] = 0.0f;
            sums[rg * 1024 + 192 + tir] = sums[rg * 1024 + 192 + tir] + acc[6];
            acc[6] = 0.0f;
            sums[rg * 1024 + 224 + tir] = sums[rg * 1024 + 224 + tir] + acc[7];
            acc[7] = 0.0f;
            sums[rg * 1024 + 256 + tir] = sums[rg * 1024 + 256 + tir] + acc[8];
            acc[8] = 0.0f;
            sums[rg * 1024 + 288 + tir] = sums[rg * 1024 + 288 + tir] + acc[9];
            acc[9] = 0.0f;
            sums[rg * 1024 + 320 + tir] = sums[rg * 1024 + 320 + tir] + acc[10];
            acc[10] = 0.0f;
            sums[rg * 1024 + 352 + tir] = sums[rg * 1024 + 352 + tir] + acc[11];
            acc[11] = 0.0f;
            sums[rg * 1024 + 384 + tir] = sums[rg * 1024 + 384 + tir] + acc[12];
            acc[12] = 0.0f;
            sums[rg * 1024 + 416 + tir] = sums[rg * 1024 + 416 + tir] + acc[13];
            acc[13] = 0.0f;
            sums[rg * 1024 + 448 + tir] = sums[rg * 1024 + 448 + tir] + acc[14];
            acc[14] = 0.0f;
            sums[rg * 1024 + 480 + tir] = sums[rg * 1024 + 480 + tir] + acc[15];
            acc[15] = 0.0f;
            sums[rg * 1024 + 512 + tir] = sums[rg * 1024 + 512 + tir] + acc[16];
            acc[16] = 0.0f;
            sums[rg * 1024 + 544 + tir] = sums[rg * 1024 + 544 + tir] + acc[17];
            acc[17] = 0.0f;
            sums[rg * 1024 + 576 + tir] = sums[rg * 1024 + 576 + tir] + acc[18];
            acc[18] = 0.0f;
            sums[rg * 1024 + 608 + tir] = sums[rg * 1024 + 608 + tir] + acc[19];
            acc[19] = 0.0f;
            sums[rg * 1024 + 640 + tir] = sums[rg * 1024 + 640 + tir] + acc[20];
            acc[20] = 0.0f;
            sums[rg * 1024 + 672 + tir] = sums[rg * 1024 + 672 + tir] + acc[21];
            acc[21] = 0.0f;
            sums[rg * 1024 + 704 + tir] = sums[rg * 1024 + 704 + tir] + acc[22];
            acc[22] = 0.0f;
            sums[rg * 1024 + 736 + tir] = sums[rg * 1024 + 736 + tir] + acc[23];
            acc[23] = 0.0f;
            sums[rg * 1024 + 768 + tir] = sums[rg * 1024 + 768 + tir] + acc[24];
            acc[24] = 0.0f;
            sums[rg * 1024 + 800 + tir] = sums[rg * 1024 + 800 + tir] + acc[25];
            acc[25] = 0.0f;
            sums[rg * 1024 + 832 + tir] = sums[rg * 1024 + 832 + tir] + acc[26];
            acc[26] = 0.0f;
            sums[rg * 1024 + 864 + tir] = sums[rg * 1024 + 864 + tir] + acc[27];
            acc[27] = 0.0f;
            sums[rg * 1024 + 896 + tir] = sums[rg * 1024 + 896 + tir] + acc[28];
            acc[28] = 0.0f;
            sums[rg * 1024 + 928 + tir] = sums[rg * 1024 + 928 + tir] + acc[29];
            acc[29] = 0.0f;
            sums[rg * 1024 + 960 + tir] = sums[rg * 1024 + 960 + tir] + acc[30];
            acc[30] = 0.0f;
            sums[rg * 1024 + 992 + tir] = sums[rg * 1024 + 992 + tir] + acc[31];
            acc[31] = 0.0f;
        }
    }
    sums[rg * 1024 + tir] = sums[rg * 1024 + tir] + acc[0];
    acc[0] = 0.0f;
    sums[rg * 1024 + 32 + tir] = sums[rg * 1024 + 32 + tir] + acc[1];
    acc[1] = 0.0f;
    sums[rg * 1024 + 64 + tir] = sums[rg * 1024 + 64 + tir] + acc[2];
    acc[2] = 0.0f;
    sums[rg * 1024 + 96 + tir] = sums[rg * 1024 + 96 + tir] + acc[3];
    acc[3] = 0.0f;
    sums[rg * 1024 + 128 + tir] = sums[rg * 1024 + 128 + tir] + acc[4];
    acc[4] = 0.0f;
    sums[rg * 1024 + 160 + tir] = sums[rg * 1024 + 160 + tir] + acc[5];
    acc[5] = 0.0f;
    sums[rg * 1024 + 192 + tir] = sums[rg * 1024 + 192 + tir] + acc[6];
    acc[6] = 0.0f;
    sums[rg * 1024 + 224 + tir] = sums[rg * 1024 + 224 + tir] + acc[7];
    acc[7] = 0.0f;
    sums[rg * 1024 + 256 + tir] = sums[rg * 1024 + 256 + tir] + acc[8];
    acc[8] = 0.0f;
    sums[rg * 1024 + 288 + tir] = sums[rg * 1024 + 288 + tir] + acc[9];
    acc[9] = 0.0f;
    sums[rg * 1024 + 320 + tir] = sums[rg * 1024 + 320 + tir] + acc[10];
    acc[10] = 0.0f;
    sums[rg * 1024 + 352 + tir] = sums[rg * 1024 + 352 + tir] + acc[11];
    acc[11] = 0.0f;
    sums[rg * 1024 + 384 + tir] = sums[rg * 1024 + 384 + tir] + acc[12];
    acc[12] = 0.0f;
    sums[rg * 1024 + 416 + tir] = sums[rg * 1024 + 416 + tir] + acc[13];
    acc[13] = 0.0f;
    sums[rg * 1024 + 448 + tir] = sums[rg * 1024 + 448 + tir] + acc[14];
    acc[14] = 0.0f;
    sums[rg * 1024 + 480 + tir] = sums[rg * 1024 + 480 + tir] + acc[15];
    acc[15] = 0.0f;
    sums[rg * 1024 + 512 + tir] = sums[rg * 1024 + 512 + tir] + acc[16];
    acc[16] = 0.0f;
    sums[rg * 1024 + 544 + tir] = sums[rg * 1024 + 544 + tir] + acc[17];
    acc[17] = 0.0f;
    sums[rg * 1024 + 576 + tir] = sums[rg * 1024 + 576 + tir] + acc[18];
    acc[18] = 0.0f;
    sums[rg * 1024 + 608 + tir] = sums[rg * 1024 + 608 + tir] + acc[19];
    acc[19] = 0.0f;
    sums[rg * 1024 + 640 + tir] = sums[rg * 1024 + 640 + tir] + acc[20];
    acc[20] = 0.0f;
    sums[rg * 1024 + 672 + tir] = sums[rg * 1024 + 672 + tir] + acc[21];
    acc[21] = 0.0f;
    sums[rg * 1024 + 704 + tir] = sums[rg * 1024 + 704 + tir] + acc[22];
    acc[22] = 0.0f;
    sums[rg * 1024 + 736 + tir] = sums[rg * 1024 + 736 + tir] + acc[23];
    acc[23] = 0.0f;
    sums[rg * 1024 + 768 + tir] = sums[rg * 1024 + 768 + tir] + acc[24];
    acc[24] = 0.0f;
    sums[rg * 1024 + 800 + tir] = sums[rg * 1024 + 800 + tir] + acc[25];
    acc[25] = 0.0f;
    sums[rg * 1024 + 832 + tir] = sums[rg * 1024 + 832 + tir] + acc[26];
    acc[26] = 0.0f;
    sums[rg * 1024 + 864 + tir] = sums[rg * 1024 + 864 + tir] + acc[27];
    acc[27] = 0.0f;
    sums[rg * 1024 + 896 + tir] = sums[rg * 1024 + 896 + tir] + acc[28];
    acc[28] = 0.0f;
    sums[rg * 1024 + 928 + tir] = sums[rg * 1024 + 928 + tir] + acc[29];
    acc[29] = 0.0f;
    sums[rg * 1024 + 960 + tir] = sums[rg * 1024 + 960 + tir] + acc[30];
    acc[30] = 0.0f;
    sums[rg * 1024 + 992 + tir] = sums[rg * 1024 + 992 + tir] + acc[31];
    acc[31] = 0.0f;
    c = pc;
    int _min_49 = (((int)num_bids) < (n_chunks) ? ((int)num_bids) : (n_chunks));
    int nrep = ((_min_49) < (2) ? (_min_49) : (2));
    __syncthreads();
    int at_0 = tid;
    float rv_0[4];
    rv_0[0] = sums[at_0];
    rv_0[1] = sums[1024 + at_0];
    rv_0[2] = sums[2048 + at_0];
    rv_0[3] = sums[3072 + at_0];
    rv_0[0] = rv_0[0] + rv_0[1];
    rv_0[2] = rv_0[2] + rv_0[3];
    rv_0[0] = rv_0[0] + rv_0[2];
    sums[at_0] = rv_0[0];
    int at_1 = 128 + tid;
    float rv_1[4];
    rv_1[0] = sums[at_1];
    rv_1[1] = sums[1024 + at_1];
    rv_1[2] = sums[2048 + at_1];
    rv_1[3] = sums[3072 + at_1];
    rv_1[0] = rv_1[0] + rv_1[1];
    rv_1[2] = rv_1[2] + rv_1[3];
    rv_1[0] = rv_1[0] + rv_1[2];
    sums[at_1] = rv_1[0];
    int at_2 = 256 + tid;
    float rv_2[4];
    rv_2[0] = sums[at_2];
    rv_2[1] = sums[1024 + at_2];
    rv_2[2] = sums[2048 + at_2];
    rv_2[3] = sums[3072 + at_2];
    rv_2[0] = rv_2[0] + rv_2[1];
    rv_2[2] = rv_2[2] + rv_2[3];
    rv_2[0] = rv_2[0] + rv_2[2];
    sums[at_2] = rv_2[0];
    int at_3 = 384 + tid;
    float rv_3[4];
    rv_3[0] = sums[at_3];
    rv_3[1] = sums[1024 + at_3];
    rv_3[2] = sums[2048 + at_3];
    rv_3[3] = sums[3072 + at_3];
    rv_3[0] = rv_3[0] + rv_3[1];
    rv_3[2] = rv_3[2] + rv_3[3];
    rv_3[0] = rv_3[0] + rv_3[2];
    sums[at_3] = rv_3[0];
    int at_4 = 512 + tid;
    float rv_4[4];
    rv_4[0] = sums[at_4];
    rv_4[1] = sums[1024 + at_4];
    rv_4[2] = sums[2048 + at_4];
    rv_4[3] = sums[3072 + at_4];
    rv_4[0] = rv_4[0] + rv_4[1];
    rv_4[2] = rv_4[2] + rv_4[3];
    rv_4[0] = rv_4[0] + rv_4[2];
    sums[at_4] = rv_4[0];
    int at_5 = 640 + tid;
    float rv_5[4];
    rv_5[0] = sums[at_5];
    rv_5[1] = sums[1024 + at_5];
    rv_5[2] = sums[2048 + at_5];
    rv_5[3] = sums[3072 + at_5];
    rv_5[0] = rv_5[0] + rv_5[1];
    rv_5[2] = rv_5[2] + rv_5[3];
    rv_5[0] = rv_5[0] + rv_5[2];
    sums[at_5] = rv_5[0];
    int at_6 = 768 + tid;
    float rv_6[4];
    rv_6[0] = sums[at_6];
    rv_6[1] = sums[1024 + at_6];
    rv_6[2] = sums[2048 + at_6];
    rv_6[3] = sums[3072 + at_6];
    rv_6[0] = rv_6[0] + rv_6[1];
    rv_6[2] = rv_6[2] + rv_6[3];
    rv_6[0] = rv_6[0] + rv_6[2];
    sums[at_6] = rv_6[0];
    int at_7 = 896 + tid;
    float rv_7[4];
    rv_7[0] = sums[at_7];
    rv_7[1] = sums[1024 + at_7];
    rv_7[2] = sums[2048 + at_7];
    rv_7[3] = sums[3072 + at_7];
    rv_7[0] = rv_7[0] + rv_7[1];
    rv_7[2] = rv_7[2] + rv_7[3];
    rv_7[0] = rv_7[0] + rv_7[2];
    sums[at_7] = rv_7[0];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_50 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1024));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_50), "r"(sums_addr + 0), "r"((uint32_t)(1024))
                    : "memory");
            }
            {
                void* _cpred_dst_51 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1024 + 256));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_51), "r"(sums_addr + 1024), "r"((uint32_t)(1024))
                    : "memory");
            }
            {
                void* _cpred_dst_52 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1024 + 512));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_52), "r"(sums_addr + 2048), "r"((uint32_t)(1024))
                    : "memory");
            }
            {
                void* _cpred_dst_53 = reinterpret_cast<void*>(partial + (long long)((int)bid % nrep * 1024 + 768));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_53), "r"(sums_addr + 3072), "r"((uint32_t)(1024))
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
        float buf[32];
        buf[0] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_54;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_54) : "l"((const void*)(partial + (tid))) : "memory");
                buf[0] = __uint_as_float(_scalar_bits_54);
            }
        }
        buf[1] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_55;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_55) : "l"((const void*)(partial + (1024 + tid))) : "memory");
                buf[1] = __uint_as_float(_scalar_bits_55);
            }
        }
        buf[2] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_56;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_56) : "l"((const void*)(partial + (128 + tid))) : "memory");
                buf[2] = __uint_as_float(_scalar_bits_56);
            }
        }
        buf[3] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_57;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_57) : "l"((const void*)(partial + (1152 + tid))) : "memory");
                buf[3] = __uint_as_float(_scalar_bits_57);
            }
        }
        buf[4] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_58;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_58) : "l"((const void*)(partial + (256 + tid))) : "memory");
                buf[4] = __uint_as_float(_scalar_bits_58);
            }
        }
        buf[5] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_59;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_59) : "l"((const void*)(partial + (1280 + tid))) : "memory");
                buf[5] = __uint_as_float(_scalar_bits_59);
            }
        }
        buf[6] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_60;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_60) : "l"((const void*)(partial + (384 + tid))) : "memory");
                buf[6] = __uint_as_float(_scalar_bits_60);
            }
        }
        buf[7] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_61;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_61) : "l"((const void*)(partial + (1408 + tid))) : "memory");
                buf[7] = __uint_as_float(_scalar_bits_61);
            }
        }
        buf[8] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_62;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_62) : "l"((const void*)(partial + (512 + tid))) : "memory");
                buf[8] = __uint_as_float(_scalar_bits_62);
            }
        }
        buf[9] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_63;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_63) : "l"((const void*)(partial + (1536 + tid))) : "memory");
                buf[9] = __uint_as_float(_scalar_bits_63);
            }
        }
        buf[10] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_64;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_64) : "l"((const void*)(partial + (640 + tid))) : "memory");
                buf[10] = __uint_as_float(_scalar_bits_64);
            }
        }
        buf[11] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_65;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_65) : "l"((const void*)(partial + (1664 + tid))) : "memory");
                buf[11] = __uint_as_float(_scalar_bits_65);
            }
        }
        buf[12] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_66;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_66) : "l"((const void*)(partial + (768 + tid))) : "memory");
                buf[12] = __uint_as_float(_scalar_bits_66);
            }
        }
        buf[13] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_67;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_67) : "l"((const void*)(partial + (1792 + tid))) : "memory");
                buf[13] = __uint_as_float(_scalar_bits_67);
            }
        }
        buf[14] = 0.0f;
        if (0 < nrep) {
            {
                uint32_t _scalar_bits_68;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_68) : "l"((const void*)(partial + (896 + tid))) : "memory");
                buf[14] = __uint_as_float(_scalar_bits_68);
            }
        }
        buf[15] = 0.0f;
        if (1 < nrep) {
            {
                uint32_t _scalar_bits_69;
                asm volatile("ld.global.cg.b32 %0, [%1];"
                    : "=r"(_scalar_bits_69) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                buf[15] = __uint_as_float(_scalar_bits_69);
            }
        }
        buf[0] = buf[0] + buf[1];
        int at_w_0 = tid;
        dwbuf[(at_w_0 / 32 / 8 * 32 + at_w_0 % 32) * 8 + at_w_0 / 32 % 8] = buf[0];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_0)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_0)) = 0.0f;
        }
        buf[2] = buf[2] + buf[3];
        int at_w_1 = 128 + tid;
        dwbuf[(at_w_1 / 32 / 8 * 32 + at_w_1 % 32) * 8 + at_w_1 / 32 % 8] = buf[2];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_1)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_1)) = 0.0f;
        }
        buf[4] = buf[4] + buf[5];
        int at_w_2 = 256 + tid;
        dwbuf[(at_w_2 / 32 / 8 * 32 + at_w_2 % 32) * 8 + at_w_2 / 32 % 8] = buf[4];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_2)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_2)) = 0.0f;
        }
        buf[6] = buf[6] + buf[7];
        int at_w_3 = 384 + tid;
        dwbuf[(at_w_3 / 32 / 8 * 32 + at_w_3 % 32) * 8 + at_w_3 / 32 % 8] = buf[6];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_3)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_3)) = 0.0f;
        }
        buf[8] = buf[8] + buf[9];
        int at_w_4 = 512 + tid;
        dwbuf[(at_w_4 / 32 / 8 * 32 + at_w_4 % 32) * 8 + at_w_4 / 32 % 8] = buf[8];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_4)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_4)) = 0.0f;
        }
        buf[10] = buf[10] + buf[11];
        int at_w_5 = 640 + tid;
        dwbuf[(at_w_5 / 32 / 8 * 32 + at_w_5 % 32) * 8 + at_w_5 / 32 % 8] = buf[10];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_5)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_5)) = 0.0f;
        }
        buf[12] = buf[12] + buf[13];
        int at_w_6 = 768 + tid;
        dwbuf[(at_w_6 / 32 / 8 * 32 + at_w_6 % 32) * 8 + at_w_6 / 32 % 8] = buf[12];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_6)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_6)) = 0.0f;
        }
        buf[14] = buf[14] + buf[15];
        int at_w_7 = 896 + tid;
        dwbuf[(at_w_7 / 32 / 8 * 32 + at_w_7 % 32) * 8 + at_w_7 / 32 % 8] = buf[14];
        if (0 < nrep) {
            *(reinterpret_cast<float*>(partial) + (at_w_7)) = 0.0f;
        }
        if (1 < nrep) {
            *(reinterpret_cast<float*>(partial) + (1024 + at_w_7)) = 0.0f;
        }
        __syncthreads();
        int col_0 = tid * 4;
        if (col_0 < 1024) {
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
        if (col_1 < 1024) {
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
