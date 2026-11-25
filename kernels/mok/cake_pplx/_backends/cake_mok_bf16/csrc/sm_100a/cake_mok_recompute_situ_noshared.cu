/*
 * Copyright 2026 Cursor Research
 * Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from the Apache-2.0 Mixture of Kittens BF16 training kernels.
 * Modified: generated CUDA implementation and standalone TVM-FFI bindings.
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};




__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_exp2_noftz(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)






__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mok_recompute_situ_noshared(const __grid_constant__ CUtensorMap x_routed, const __grid_constant__ CUtensorMap wg_routed, const __grid_constant__ CUtensorMap wu_routed, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap hidden_routed_out, __nv_bfloat16* __restrict__ x_routed_ptr, unsigned long long* __restrict__ x_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float situ_beta, float situ_lb, CakeFastDivmod mini_tasks_div, CakeFastDivmod fused_col_blocks_div)
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
    #define gemm_arrived_addr (mbar_base + 0)
    #define gemm_finished_addr (mbar_base + 32)
    #define output_arrived_addr (mbar_base + 64)
    #define output_finished_addr (mbar_base + 80)
    #define schedule_arrived_addr (mbar_base + 96)
    #define schedule_finished_addr (mbar_base + 104)
    #define drain_arrived_0_addr (mbar_base + 112)
    #define drain_arrived_1_addr (mbar_base + 120)
    #define drain_arrived_2_addr (mbar_base + 128)
    #define drain_arrived_3_addr (mbar_base + 136)
    #define drain_arrived_4_addr (mbar_base + 144)
    #define drain_arrived_5_addr (mbar_base + 152)
    #define drain_arrived_6_addr (mbar_base + 160)
    #define drain_arrived_7_addr (mbar_base + 168)
    #define drain_finished_0_addr (mbar_base + 176)
    #define drain_finished_1_addr (mbar_base + 184)
    #define drain_finished_2_addr (mbar_base + 192)
    #define drain_finished_3_addr (mbar_base + 200)
    #define drain_finished_4_addr (mbar_base + 208)
    #define drain_finished_5_addr (mbar_base + 216)
    #define drain_finished_6_addr (mbar_base + 224)
    #define drain_finished_7_addr (mbar_base + 232)
    #define range_arrived_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* a_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int a_smem_addr = smem + 1024;
    __nv_bfloat16* b_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int b_smem_addr = smem + 66560;
    __nv_bfloat16* b_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int b_hi_addr = smem + 132096;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
    __nv_bfloat16* flat_view = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr = smem + 1024;
    unsigned int* flat_view_0 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_0 = smem + 1024;
    float* flat_view_1 = reinterpret_cast<float*>(smem_raw + 197632);
    const int flat_view_addr_1 = smem + 197632;
    int* flat_view_2 = reinterpret_cast<int*>(smem_raw + 199168);
    const int flat_view_addr_2 = smem + 199168;

    // Kernel setup ops
    int tokens = num_tokens[0];
    int routed_tokens = ((tokens) < (macro_size) ? (tokens) : (macro_size));
    int shared_rows = 0;
    int shared_tasks = shared_rows * (intermediate / 256);
    int mini_tasks = mini_size / 256 * (intermediate / 256);
    int comm_clusters = comm_sms / 2;
    int routed_minis = (routed_tokens + mini_size - 1) / mini_size;
    int true_clusters = comm_clusters + shared_tasks + routed_minis * mini_tasks;
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (33 barriers at smem_raw[0..264))

    if (threadIdx.x == 0) {
        // gemm_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        mbarrier_init(smem + 24, 1);
        // gemm_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 32, 1);
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        mbarrier_init(smem + 56, 1);
        // output_arrived: 2 barriers, init_count=1
        mbarrier_init(smem + 64, 1);
        mbarrier_init(smem + 72, 1);
        // output_finished: 2 barriers, init_count=2
        mbarrier_init(smem + 80, 2);
        mbarrier_init(smem + 88, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 96, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 104, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 112, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 120, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 128, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 144, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 152, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 160, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 176, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 184, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 192, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 200, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 208, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 216, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 224, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 232, 2);
        // range_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 240, 1);
        mbarrier_init(smem + 248, 1);
        mbarrier_init(smem + 256, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 0) {
        int _tmem_hold = smem + 264;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_accumulator = taddr;

    // Kernel post-init ops
    unsigned int taddr_1 = taddr;
    int cluster = bid / 2;
    int cta_rank_0 = cta_rank;
    int gemm_deep = 0;
    int gemm_wide = 0;
    int gemm_layout = 0;
    int gemm_out0 = 0;
    int gemm_out1 = 0;
    int gemm_acc_tiles = 0;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int col_blocks = (hidden + 511) / 512;
        int unit_rows = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
        int block_units = (128 + unit_rows - 1) / unit_rows;
        int macro_offset = 0;
        int _min_0 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
        int macro_blocks = _min_0 / 128;
        int blocks = 0;
        if (macro_blocks > cluster * 2 + cta_rank_0) {
            blocks = (macro_blocks - (cluster * 2 + cta_rank_0) + comm_sms - 1) / comm_sms;
        }
        int units = blocks * block_units;
        int _min_0_1 = ((units) < (2) ? (units) : (2));
        #pragma unroll 1
        for (int unit = 0; unit < _min_0_1; unit++) {
            int buffer = unit % 3;
            int block = cluster * 2 + cta_rank_0 + unit / block_units * comm_sms;
            int first_row = block * 128 + unit % block_units * unit_rows;
            int rows = ((unit_rows) < (block * 128 + 128 - first_row) ? (unit_rows) : (block * 128 + 128 - first_row));
            int peer = -1;
            int peer_token = -1;
            if (rows > tid) {
                peer = schedule_rank[macro_offset + first_row + tid];
                peer_token = schedule_token[macro_offset + first_row + tid];
                flat_view_2[tid] = peer;
            }
            uint32_t cta_count = __syncthreads_count(peer >= 0);
            if (tid == 0) {
                mbarrier_arrive_expect_tx(range_arrived_addr + (buffer) * 8, (unsigned int)(cta_count * (unsigned int)hidden * 2));
            }
            if (cta_count < (unsigned int)rows) {
                #pragma unroll 1
                for (int pad = 0; pad < rows; pad++) {
                    int pad_peer = flat_view_2[pad];
                    if (pad_peer < 0) {
                        #pragma unroll 1
                        for (int vec = tid; vec < hidden / 8; vec += 256) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_0) + (buffer * 65536 + pad * hidden * 2 + vec * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
            }
            __syncthreads();
            if (peer >= 0) {
                cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)((buffer * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + (unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden) * (unsigned long long)2), hidden * 2, range_arrived_addr + (buffer) * 8);
            }
        }
        #pragma unroll 1
        for (int unit_1 = 0; unit_1 < units; unit_1++) {
            int buffer_1 = unit_1 % 3;
            int block_1 = cluster * 2 + cta_rank_0 + unit_1 / block_units * comm_sms;
            int first_row_1 = block_1 * 128 + unit_1 % block_units * unit_rows;
            int rows_1 = ((unit_rows) < (block_1 * 128 + 128 - first_row_1) ? (unit_rows) : (block_1 * 128 + 128 - first_row_1));
            int _parity_range_arrived = (int)((unsigned int)unit_1 / 3 % 2);
            mbarrier_wait(range_arrived_addr + (buffer_1) * 8, _parity_range_arrived);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid == 0) {
                {
                    void* _cpbulk_dst_1 = reinterpret_cast<void*>(x_routed_ptr + (unsigned long long)first_row_1 * (unsigned long long)hidden);
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                        :: "l"(_cpbulk_dst_1), "r"(flat_view_addr + (unsigned int)(buffer_1 * 65536)), "r"((uint32_t)((unsigned int)(rows_1 * hidden * 2)))
                        : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;");
                if (units > unit_1 + 2) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                if (unit_1 >= 2 && (unit_1 - 2) % block_units == block_units - 1) {
                    asm volatile("cp.async.bulk.wait_group 2;");
                    int done_block = cluster * 2 + cta_rank_0 + (unit_1 - 2) / block_units * comm_sms;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + done_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                }
            }
            if (units > unit_1 + 2) {
                int buffer_0 = (unit_1 + 2) % 3;
                int block_0 = cluster * 2 + cta_rank_0 + (unit_1 + 2) / block_units * comm_sms;
                int first_row_0 = block_0 * 128 + (unit_1 + 2) % block_units * unit_rows;
                int rows_0 = ((unit_rows) < (block_0 * 128 + 128 - first_row_0) ? (unit_rows) : (block_0 * 128 + 128 - first_row_0));
                int peer_1 = -1;
                int peer_token_1 = -1;
                if (rows_0 > tid) {
                    peer_1 = schedule_rank[macro_offset + first_row_0 + tid];
                    peer_token_1 = schedule_token[macro_offset + first_row_0 + tid];
                    flat_view_2[tid] = peer_1;
                }
                uint32_t cta_count_1 = __syncthreads_count(peer_1 >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(range_arrived_addr + (buffer_0) * 8, (unsigned int)(cta_count_1 * (unsigned int)hidden * 2));
                }
                if (cta_count_1 < (unsigned int)rows_0) {
                    #pragma unroll 1
                    for (int pad_1 = 0; pad_1 < rows_0; pad_1++) {
                        int pad_peer_1 = flat_view_2[pad_1];
                        if (pad_peer_1 < 0) {
                            #pragma unroll 1
                            for (int vec_1 = tid; vec_1 < hidden / 8; vec_1 += 256) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_0) + (buffer_0 * 65536 + pad_1 * hidden * 2 + vec_1 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                }
                __syncthreads();
                if (peer_1 >= 0) {
                    cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)((buffer_0 * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + (unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden) * (unsigned long long)2), hidden * 2, range_arrived_addr + (buffer_0) * 8);
                }
            }
        }
        if (tid == 0) {
            asm volatile("cp.async.bulk.wait_group 0;");
            int _max_0 = ((0) > (units - 2) ? (0) : (units - 2));
            #pragma unroll 1
            for (int tail = _max_0; tail < units; tail++) {
                if (tail % block_units == block_units - 1) {
                    int tail_block = cluster * 2 + cta_rank_0 + tail / block_units * comm_sms;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + tail_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                }
            }
        }
    } else {
        int iteration = 0;
        float k1 = __fdiv_rn(-2.88539f, situ_beta);
        float k2 = __fdiv_rn(-2.88539f, situ_lb);
        float bl;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(bl) : "f"(situ_beta), "f"(situ_lb));
        while (cluster >= 0 && cluster < true_clusters) {
            if (tid / 32 == 5) {
                if (warp == 5) {
                    if (elect_sync()) {
                        if (cta_rank_0 == 0) {
                            int _parity_schedule_finished = (int)((unsigned int)iteration % 2) ^ 1;
                            mbarrier_wait(schedule_finished_addr, _parity_schedule_finished);
                            asm volatile(
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                    " [%0], [%1];"
                                :: "r"(smem + 512 + 0 * 16 + 0 * 16), "r"(schedule_arrived_addr + 0 * 8)
                                : "memory");
                        }
                        mbarrier_arrive_expect_tx(schedule_arrived_addr, 16);
                    }
                }
            }
            int compute = cluster - comm_clusters;
            if (compute < shared_tasks) {
            } else {
                int mini = 0;
                int task = 0;
                int swiglu = 0;
                if (compute >= shared_tasks) {
                    uint32_t _fast_div_q_0 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(compute - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(compute - shared_tasks));
                    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)(compute - shared_tasks)) - _fast_div_q_0 * (uint32_t)(mini_tasks_div.divisor);
                    mini = (int)_fast_div_q_0;
                    task = (int)_fast_div_r_0;
                }
                int col_blocks_1 = (intermediate + 256 - 1) / 256;
                int x = -1;
                int y = -1;
                int expert = -1;
                int k_start = 0;
                int k_end = 0;
                int first = 0;
                int first_block = mini * (mini_size / 256);
                int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                int block_2 = first_block;
                uint32_t fast_div_q = (fused_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task), fused_col_blocks_div.multiplier) >> fused_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task);
                uint32_t fast_div_r = (uint32_t)((unsigned int)task) - fast_div_q * (uint32_t)(fused_col_blocks_div.divisor);
                block_2 = first_block + (int)fast_div_q;
                if (block_2 < end_block) {
                    int index = counts[3 * experts + block_2];
                    int offset = counts[experts + index] / 256;
                    int first_row_2 = ((first_block) > (offset) ? (first_block) : (offset));
                    int _min_2 = ((end_block) < (offset + counts[index] / 256) ? (end_block) : (offset + counts[index] / 256));
                    int rows_2 = _min_2 - first_row_2;
                    int supergroup = (task - (first_row_2 - first_block) * col_blocks_1) / (rows_2 * 8);
                    int full_cols = col_blocks_1 / 8 * 8;
                    int row = 0;
                    int col = 0;
                    if (task - (first_row_2 - first_block) * col_blocks_1 < rows_2 * full_cols) {
                        row = (task - (first_row_2 - first_block) * col_blocks_1) % (rows_2 * 8) / 8;
                        col = supergroup * 8 + (task - (first_row_2 - first_block) * col_blocks_1) % 8;
                    } else {
                        row = (task - (first_row_2 - first_block) * col_blocks_1 - rows_2 * full_cols) / (col_blocks_1 - full_cols);
                        col = full_cols + (task - (first_row_2 - first_block) * col_blocks_1 - rows_2 * full_cols) % (col_blocks_1 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row = rows_2 - row - 1;
                    }
                    x = first_row_2 + row;
                    y = col;
                    expert = index;
                }
                int has_hi = 0;
                has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                int global_mini = mini;
                int macro_rows = 0;
                int iterations = hidden / 64;
                if (expert < 0) {
                    if (tid == 0) {
                    }
                } else {
                    if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_3 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int mini_rows = ((0) > (_min_3) ? (0) : (_min_3));
                                    int required = (mini_rows + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value = 1;
                                    if (enabled_value != 0) {
                                        int poll_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_first) : "l"(x_ready + global_mini) : "memory");
                                        int poll = poll_first;
                                        while (poll < required) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_next) : "l"(x_ready + global_mini) : "memory");
                                            poll = poll_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                #pragma unroll 1
                                for (int idx = 0; idx < iterations; idx++) {
                                    int _stage_gemm_finished = (int)((unsigned int)(gemm_wide + idx) % 4);
                                    int _parity_gemm_finished = (int)((unsigned int)(gemm_wide + idx) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished) * 8, _parity_gemm_finished);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&x_routed)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    if (cta_rank_0 == 0) {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y * 256), "r"(idx), "r"(expert), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    } else {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y * 256), "r"(idx), "r"(expert), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                    if (cta_rank_0 == 0) {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y * 256 + 128), "r"(idx), "r"(expert), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    } else {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y * 256 + 128), "r"(idx), "r"(expert), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        }
                    } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int head = ((3) < (iterations) ? (3) : (iterations));
                                int tail_1 = 0;
                                int _min_4 = ((3) < (iterations - head) ? (3) : (iterations - head));
                                tail_1 = _min_4;
                                int _stage_output_finished = (int)((unsigned int)gemm_acc_tiles % 2);
                                int _parity_output_finished = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished) * 8, _parity_output_finished);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_1 = 0; idx_1 < head; idx_1++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_1) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived = (int)((unsigned int)(gemm_wide + idx_1) % 4);
                                    int _parity_gemm_arrived = (int)((unsigned int)(gemm_wide + idx_1) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived) * 8, _parity_gemm_arrived);
                                    int _smem_desc_lo_5 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_6 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_5), "r"(_smem_desc_lo_6), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                }
                                int _stage_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                int _parity_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished_1) * 8, _parity_output_finished_1);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_2 = 0; idx_2 < head; idx_2++) {
                                    int _smem_desc_lo_7 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
                                    int _smem_desc_lo_8 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
                                    asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_7), "r"(_smem_desc_lo_8), "r"((tmem_accumulator + (256))), "r"(((idx_2 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_3 = head; idx_3 < iterations - tail_1; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) % 4);
                                    int _parity_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_1) * 8, _parity_gemm_arrived_1);
                                    int _smem_desc_lo_9 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_10 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_9), "r"(_smem_desc_lo_10), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_11 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_12 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_11), "r"(_smem_desc_lo_12), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_4 = iterations - tail_1; idx_4 < iterations; idx_4++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) % 4);
                                    int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
                                    int _smem_desc_lo_13 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
                                    int _smem_desc_lo_14 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
                                    asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_13), "r"(_smem_desc_lo_14), "r"(tmem_accumulator), "r"(((idx_4 == 0) ? 0 : 1)) : "memory");
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                #pragma unroll 1
                                for (int idx_5 = iterations - tail_1; idx_5 < iterations; idx_5++) {
                                    int _smem_desc_lo_15 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
                                    int _smem_desc_lo_16 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
                                    asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_smem_desc_lo_15), "r"(_smem_desc_lo_16), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_5) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                            }
                        }
                    } else {
                        if (tid < 128) {
                            int _parity_output_arrived = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived);
                            unsigned int gate_packed_0[32];
                            unsigned int up_packed_0[32];
                            unsigned int address_3_0 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16);
                            float _tmem_load_0[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                : "r"(address_3_0));
                            float _tmem_load_32_0[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[15]))
                                : "r"(address_3_0 + 128));
                            gate_packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1])));
                            up_packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[0], _tmem_load_32_0[1])));
                            gate_packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3])));
                            up_packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[2], _tmem_load_32_0[3])));
                            gate_packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[4], _tmem_load_0[5])));
                            up_packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[4], _tmem_load_32_0[5])));
                            gate_packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[6], _tmem_load_0[7])));
                            up_packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[6], _tmem_load_32_0[7])));
                            gate_packed_0[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[8], _tmem_load_0[9])));
                            up_packed_0[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[8], _tmem_load_32_0[9])));
                            gate_packed_0[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[10], _tmem_load_0[11])));
                            up_packed_0[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[10], _tmem_load_32_0[11])));
                            gate_packed_0[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[12], _tmem_load_0[13])));
                            up_packed_0[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[12], _tmem_load_32_0[13])));
                            gate_packed_0[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[14], _tmem_load_0[15])));
                            up_packed_0[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[14], _tmem_load_32_0[15])));
                            unsigned int address_3_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16);
                            float _tmem_load_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                : "r"(address_3_1));
                            float _tmem_load_32_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[15]))
                                : "r"(address_3_1 + 128));
                            gate_packed_0[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1])));
                            up_packed_0[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[0], _tmem_load_32_1[1])));
                            gate_packed_0[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3])));
                            up_packed_0[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[2], _tmem_load_32_1[3])));
                            gate_packed_0[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[4], _tmem_load_1[5])));
                            up_packed_0[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[4], _tmem_load_32_1[5])));
                            gate_packed_0[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[6], _tmem_load_1[7])));
                            up_packed_0[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[6], _tmem_load_32_1[7])));
                            gate_packed_0[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[8], _tmem_load_1[9])));
                            up_packed_0[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[8], _tmem_load_32_1[9])));
                            gate_packed_0[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[10], _tmem_load_1[11])));
                            up_packed_0[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[10], _tmem_load_32_1[11])));
                            gate_packed_0[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[12], _tmem_load_1[13])));
                            up_packed_0[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[12], _tmem_load_32_1[13])));
                            gate_packed_0[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[14], _tmem_load_1[15])));
                            up_packed_0[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[14], _tmem_load_32_1[15])));
                            unsigned int address_3_2 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 32;
                            float _tmem_load_2[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                : "r"(address_3_2));
                            float _tmem_load_32_2[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[15]))
                                : "r"(address_3_2 + 128));
                            gate_packed_0[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[0], _tmem_load_2[1])));
                            up_packed_0[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[0], _tmem_load_32_2[1])));
                            gate_packed_0[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[2], _tmem_load_2[3])));
                            up_packed_0[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[2], _tmem_load_32_2[3])));
                            gate_packed_0[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[4], _tmem_load_2[5])));
                            up_packed_0[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[4], _tmem_load_32_2[5])));
                            gate_packed_0[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[6], _tmem_load_2[7])));
                            up_packed_0[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[6], _tmem_load_32_2[7])));
                            gate_packed_0[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[8], _tmem_load_2[9])));
                            up_packed_0[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[8], _tmem_load_32_2[9])));
                            gate_packed_0[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[10], _tmem_load_2[11])));
                            up_packed_0[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[10], _tmem_load_32_2[11])));
                            gate_packed_0[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[12], _tmem_load_2[13])));
                            up_packed_0[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[12], _tmem_load_32_2[13])));
                            gate_packed_0[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[14], _tmem_load_2[15])));
                            up_packed_0[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[14], _tmem_load_32_2[15])));
                            unsigned int address_3_3 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 32;
                            float _tmem_load_3[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                : "r"(address_3_3));
                            float _tmem_load_32_3[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[15]))
                                : "r"(address_3_3 + 128));
                            gate_packed_0[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[0], _tmem_load_3[1])));
                            up_packed_0[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[0], _tmem_load_32_3[1])));
                            gate_packed_0[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[2], _tmem_load_3[3])));
                            up_packed_0[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[2], _tmem_load_32_3[3])));
                            gate_packed_0[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[4], _tmem_load_3[5])));
                            up_packed_0[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[4], _tmem_load_32_3[5])));
                            gate_packed_0[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[6], _tmem_load_3[7])));
                            up_packed_0[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[6], _tmem_load_32_3[7])));
                            gate_packed_0[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[8], _tmem_load_3[9])));
                            up_packed_0[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[8], _tmem_load_32_3[9])));
                            gate_packed_0[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[10], _tmem_load_3[11])));
                            up_packed_0[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[10], _tmem_load_32_3[11])));
                            gate_packed_0[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[12], _tmem_load_3[13])));
                            up_packed_0[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[12], _tmem_load_32_3[13])));
                            gate_packed_0[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[14], _tmem_load_3[15])));
                            up_packed_0[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[14], _tmem_load_32_3[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1 = tid / 32;
                            int lane_1 = tid % 32;
                            #pragma unroll
                            for (int half = 0; half < 2; half++) {
                                #pragma unroll
                                for (int col_tile = 0; col_tile < 2; col_tile++) {
                                    int row_1 = warp_1 * 32 + half * 16 + lane_1 % 16;
                                    int col_1 = col_tile * 16 + lane_1 / 16 * 8;
                                    unsigned int address = d_smem_addr + (unsigned int)((row_1 * 32 + col_1) * 2);
                                    address = address ^ (address & 511) >> 7 << 4;
                                    int offset_1 = half * 8 + col_tile * 4;
                                    uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(address);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_0 = tid / 32;
                            int lane_0 = tid % 32;
                            #pragma unroll
                            for (int half_1 = 0; half_1 < 2; half_1++) {
                                #pragma unroll
                                for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                    int row_2 = warp_0 * 32 + half_1 * 16 + lane_0 % 16;
                                    int col_2 = col_tile_1 * 16 + lane_0 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + 8192 + (unsigned int)((row_2 * 32 + col_2) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset_2 = half_1 * 8 + col_tile_1 * 4;
                                    uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1_1 = tid / 32;
                            int lane_1_1 = tid % 32;
                            #pragma unroll
                            for (int half_2 = 0; half_2 < 2; half_2++) {
                                #pragma unroll
                                for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                    int row_3 = warp_1_1 * 32 + half_2 * 16 + lane_1_1 % 16;
                                    int col_3 = col_tile_2 * 16 + lane_1_1 / 16 * 8;
                                    unsigned int address_2 = d_smem_addr + 16384 + (unsigned int)((row_3 * 32 + col_3) * 2);
                                    address_2 = address_2 ^ (address_2 & 511) >> 7 << 4;
                                    int offset_3 = 16 + half_2 * 8 + col_tile_2 * 4;
                                    uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(address_2);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_2 = tid / 32;
                            int lane_2 = tid % 32;
                            #pragma unroll
                            for (int half_3 = 0; half_3 < 2; half_3++) {
                                #pragma unroll
                                for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                    int row_4 = warp_2 * 32 + half_3 * 16 + lane_2 % 16;
                                    int col_4 = col_tile_3 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_3 = d_smem_addr + (unsigned int)((row_4 * 32 + col_4) * 2);
                                    address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                    int offset_4 = 16 + half_3 * 8 + col_tile_3 * 4;
                                    uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(address_3);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[0]));
                            float2 v_0 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[0]));
                            unsigned int sign_0 = (gate_packed_0[0] ^ up_packed_0[0]) & 2147516416u;
                            float _abs_21;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(x_0.x));
                            float _abs_22;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(x_0.y));
                            float2 ax = make_float2(_abs_21, _abs_22);
                            float _abs_23;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(v_0.x));
                            float _abs_24;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(v_0.y));
                            float2 av = make_float2(_abs_23, _abs_24);
                            float2 _f32x2_mul_b_25 = make_float2(k1, k1);
                            float2 te;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
                            float2 _f32x2_mul_b_26 = make_float2(k2, k2);
                            float2 tw;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw) : "l"(*(const unsigned long long*)&av), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
                            float _exp2_27 = approx_exp2(te.x);
                            float _exp2_28 = approx_exp2(te.y);
                            float2 e = make_float2(_exp2_27, _exp2_28);
                            float _exp2_29 = approx_exp2(tw.x);
                            float _exp2_30 = approx_exp2(tw.y);
                            float2 w = make_float2(_exp2_29, _exp2_30);
                            float2 _f32x2_mul_b_31 = make_float2(-1.442695f, -1.442695f);
                            float2 tp;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                            float _exp2_32 = approx_exp2_noftz(tp.x);
                            float _exp2_33 = approx_exp2_noftz(tp.y);
                            float2 p = make_float2(_exp2_32, _exp2_33);
                            float2 d1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                            float2 d2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p);
                            float2 dw = add_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                            float2 ng = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                            float2 nu = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                            float _rcp_34 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1, d2), dw).x);
                            float _rcp_35 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1, d2), dw).y);
                            float2 r_0 = make_float2(_rcp_34, _rcp_35);
                            float2 r = r_0;
                            float2 _f32x2_mul_a_36 = mul_f32x2_noftz(ng, nu);
                            float2 _f32x2_mul_b_37 = mul_f32x2_noftz(r, make_float2(bl, bl));
                            float2 m;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                            float2 ns = make_float2(((x_0.x < 0.0f) ? p.x : 1.0f), ((x_0.y < 0.0f) ? p.y : 1.0f));
                            gate_packed_0[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m, ns))) | sign_0;
                            float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[1]));
                            float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[1]));
                            unsigned int sign_1 = (gate_packed_0[1] ^ up_packed_0[1]) & 2147516416u;
                            float _abs_38;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(x_1.x));
                            float _abs_39;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(x_1.y));
                            float2 ax_0 = make_float2(_abs_38, _abs_39);
                            float _abs_40;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(v_1.x));
                            float _abs_41;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_41) : "f"(v_1.y));
                            float2 av_0 = make_float2(_abs_40, _abs_41);
                            float2 _f32x2_mul_b_42 = make_float2(k1, k1);
                            float2 te_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_0) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
                            float2 _f32x2_mul_b_43 = make_float2(k2, k2);
                            float2 tw_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_0) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                            float _exp2_44 = approx_exp2(te_0.x);
                            float _exp2_45 = approx_exp2(te_0.y);
                            float2 e_0 = make_float2(_exp2_44, _exp2_45);
                            float _exp2_46 = approx_exp2(tw_0.x);
                            float _exp2_47 = approx_exp2(tw_0.y);
                            float2 w_0 = make_float2(_exp2_46, _exp2_47);
                            float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_0) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
                            float _exp2_49 = approx_exp2_noftz(tp_0.x);
                            float _exp2_50 = approx_exp2_noftz(tp_0.y);
                            float2 p_0 = make_float2(_exp2_49, _exp2_50);
                            float2 d1_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                            float2 d2_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_0);
                            float2 dw_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                            float2 ng_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                            float2 nu_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                            float _rcp_51 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_0, d2_0), dw_0).x);
                            float _rcp_52 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_0, d2_0), dw_0).y);
                            float2 r_2 = make_float2(_rcp_51, _rcp_52);
                            float2 r_1 = r_2;
                            float2 _f32x2_mul_a_53 = mul_f32x2_noftz(ng_0, nu_0);
                            float2 _f32x2_mul_b_54 = mul_f32x2_noftz(r_1, make_float2(bl, bl));
                            float2 m_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
                            float2 ns_0 = make_float2(((x_1.x < 0.0f) ? p_0.x : 1.0f), ((x_1.y < 0.0f) ? p_0.y : 1.0f));
                            gate_packed_0[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_0, ns_0))) | sign_1;
                            float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[2]));
                            float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[2]));
                            unsigned int sign_2 = (gate_packed_0[2] ^ up_packed_0[2]) & 2147516416u;
                            float _abs_55;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(x_2.x));
                            float _abs_56;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(x_2.y));
                            float2 ax_1 = make_float2(_abs_55, _abs_56);
                            float _abs_57;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(v_2.x));
                            float _abs_58;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(v_2.y));
                            float2 av_1 = make_float2(_abs_57, _abs_58);
                            float2 _f32x2_mul_b_59 = make_float2(k1, k1);
                            float2 te_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_1) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
                            float2 _f32x2_mul_b_60 = make_float2(k2, k2);
                            float2 tw_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_1) : "l"(*(const unsigned long long*)&av_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
                            float _exp2_61 = approx_exp2(te_1.x);
                            float _exp2_62 = approx_exp2(te_1.y);
                            float2 e_1 = make_float2(_exp2_61, _exp2_62);
                            float _exp2_63 = approx_exp2(tw_1.x);
                            float _exp2_64 = approx_exp2(tw_1.y);
                            float2 w_1 = make_float2(_exp2_63, _exp2_64);
                            float2 _f32x2_mul_b_65 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_1) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                            float _exp2_66 = approx_exp2_noftz(tp_1.x);
                            float _exp2_67 = approx_exp2_noftz(tp_1.y);
                            float2 p_1 = make_float2(_exp2_66, _exp2_67);
                            float2 d1_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                            float2 d2_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_1);
                            float2 dw_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                            float2 ng_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                            float2 nu_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                            float _rcp_68 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_1, d2_1), dw_1).x);
                            float _rcp_69 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_1, d2_1), dw_1).y);
                            float2 r_4 = make_float2(_rcp_68, _rcp_69);
                            float2 r_3 = r_4;
                            float2 _f32x2_mul_a_70 = mul_f32x2_noftz(ng_1, nu_1);
                            float2 _f32x2_mul_b_71 = mul_f32x2_noftz(r_3, make_float2(bl, bl));
                            float2 m_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                            float2 ns_1 = make_float2(((x_2.x < 0.0f) ? p_1.x : 1.0f), ((x_2.y < 0.0f) ? p_1.y : 1.0f));
                            gate_packed_0[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_1, ns_1))) | sign_2;
                            float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[3]));
                            float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[3]));
                            unsigned int sign_3 = (gate_packed_0[3] ^ up_packed_0[3]) & 2147516416u;
                            float _abs_72;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_72) : "f"(x_3.x));
                            float _abs_73;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(x_3.y));
                            float2 ax_2 = make_float2(_abs_72, _abs_73);
                            float _abs_74;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(v_3.x));
                            float _abs_75;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(v_3.y));
                            float2 av_2 = make_float2(_abs_74, _abs_75);
                            float2 _f32x2_mul_b_76 = make_float2(k1, k1);
                            float2 te_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_2) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
                            float2 _f32x2_mul_b_77 = make_float2(k2, k2);
                            float2 tw_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_2) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                            float _exp2_78 = approx_exp2(te_2.x);
                            float _exp2_79 = approx_exp2(te_2.y);
                            float2 e_2 = make_float2(_exp2_78, _exp2_79);
                            float _exp2_80 = approx_exp2(tw_2.x);
                            float _exp2_81 = approx_exp2(tw_2.y);
                            float2 w_2 = make_float2(_exp2_80, _exp2_81);
                            float2 _f32x2_mul_b_82 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_2) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
                            float _exp2_83 = approx_exp2_noftz(tp_2.x);
                            float _exp2_84 = approx_exp2_noftz(tp_2.y);
                            float2 p_2 = make_float2(_exp2_83, _exp2_84);
                            float2 d1_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                            float2 d2_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_2);
                            float2 dw_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                            float2 ng_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                            float2 nu_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                            float _rcp_85 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_2, d2_2), dw_2).x);
                            float _rcp_86 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_2, d2_2), dw_2).y);
                            float2 r_6 = make_float2(_rcp_85, _rcp_86);
                            float2 r_5 = r_6;
                            float2 _f32x2_mul_a_87 = mul_f32x2_noftz(ng_2, nu_2);
                            float2 _f32x2_mul_b_88 = mul_f32x2_noftz(r_5, make_float2(bl, bl));
                            float2 m_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
                            float2 ns_2 = make_float2(((x_3.x < 0.0f) ? p_2.x : 1.0f), ((x_3.y < 0.0f) ? p_2.y : 1.0f));
                            gate_packed_0[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_2, ns_2))) | sign_3;
                            float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[4]));
                            float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[4]));
                            unsigned int sign_4 = (gate_packed_0[4] ^ up_packed_0[4]) & 2147516416u;
                            float _abs_89;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(x_4.x));
                            float _abs_90;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(x_4.y));
                            float2 ax_3 = make_float2(_abs_89, _abs_90);
                            float _abs_91;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(v_4.x));
                            float _abs_92;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(v_4.y));
                            float2 av_3 = make_float2(_abs_91, _abs_92);
                            float2 _f32x2_mul_b_93 = make_float2(k1, k1);
                            float2 te_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
                            float2 _f32x2_mul_b_94 = make_float2(k2, k2);
                            float2 tw_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3) : "l"(*(const unsigned long long*)&av_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
                            float _exp2_95 = approx_exp2(te_3.x);
                            float _exp2_96 = approx_exp2(te_3.y);
                            float2 e_3 = make_float2(_exp2_95, _exp2_96);
                            float _exp2_97 = approx_exp2(tw_3.x);
                            float _exp2_98 = approx_exp2(tw_3.y);
                            float2 w_3 = make_float2(_exp2_97, _exp2_98);
                            float2 _f32x2_mul_b_99 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_3) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
                            float _exp2_100 = approx_exp2_noftz(tp_3.x);
                            float _exp2_101 = approx_exp2_noftz(tp_3.y);
                            float2 p_3 = make_float2(_exp2_100, _exp2_101);
                            float2 d1_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                            float2 d2_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3);
                            float2 dw_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                            float2 ng_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                            float2 nu_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                            float _rcp_102 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_3, d2_3), dw_3).x);
                            float _rcp_103 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_3, d2_3), dw_3).y);
                            float2 r_8 = make_float2(_rcp_102, _rcp_103);
                            float2 r_7 = r_8;
                            float2 _f32x2_mul_a_104 = mul_f32x2_noftz(ng_3, nu_3);
                            float2 _f32x2_mul_b_105 = mul_f32x2_noftz(r_7, make_float2(bl, bl));
                            float2 m_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
                            float2 ns_3 = make_float2(((x_4.x < 0.0f) ? p_3.x : 1.0f), ((x_4.y < 0.0f) ? p_3.y : 1.0f));
                            gate_packed_0[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_3, ns_3))) | sign_4;
                            float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[5]));
                            float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[5]));
                            unsigned int sign_5 = (gate_packed_0[5] ^ up_packed_0[5]) & 2147516416u;
                            float _abs_106;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_106) : "f"(x_5.x));
                            float _abs_107;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_107) : "f"(x_5.y));
                            float2 ax_4 = make_float2(_abs_106, _abs_107);
                            float _abs_108;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_108) : "f"(v_5.x));
                            float _abs_109;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_109) : "f"(v_5.y));
                            float2 av_4 = make_float2(_abs_108, _abs_109);
                            float2 _f32x2_mul_b_110 = make_float2(k1, k1);
                            float2 te_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_4) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
                            float2 _f32x2_mul_b_111 = make_float2(k2, k2);
                            float2 tw_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_4) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_111));
                            float _exp2_112 = approx_exp2(te_4.x);
                            float _exp2_113 = approx_exp2(te_4.y);
                            float2 e_4 = make_float2(_exp2_112, _exp2_113);
                            float _exp2_114 = approx_exp2(tw_4.x);
                            float _exp2_115 = approx_exp2(tw_4.y);
                            float2 w_4 = make_float2(_exp2_114, _exp2_115);
                            float2 _f32x2_mul_b_116 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_4) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
                            float _exp2_117 = approx_exp2_noftz(tp_4.x);
                            float _exp2_118 = approx_exp2_noftz(tp_4.y);
                            float2 p_4 = make_float2(_exp2_117, _exp2_118);
                            float2 d1_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                            float2 d2_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_4);
                            float2 dw_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                            float2 ng_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                            float2 nu_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                            float _rcp_119 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_4, d2_4), dw_4).x);
                            float _rcp_120 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_4, d2_4), dw_4).y);
                            float2 r_10 = make_float2(_rcp_119, _rcp_120);
                            float2 r_9 = r_10;
                            float2 _f32x2_mul_a_121 = mul_f32x2_noftz(ng_4, nu_4);
                            float2 _f32x2_mul_b_122 = mul_f32x2_noftz(r_9, make_float2(bl, bl));
                            float2 m_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
                            float2 ns_4 = make_float2(((x_5.x < 0.0f) ? p_4.x : 1.0f), ((x_5.y < 0.0f) ? p_4.y : 1.0f));
                            gate_packed_0[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_4, ns_4))) | sign_5;
                            float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[6]));
                            float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[6]));
                            unsigned int sign_6 = (gate_packed_0[6] ^ up_packed_0[6]) & 2147516416u;
                            float _abs_123;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(x_6.x));
                            float _abs_124;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_124) : "f"(x_6.y));
                            float2 ax_5 = make_float2(_abs_123, _abs_124);
                            float _abs_125;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_125) : "f"(v_6.x));
                            float _abs_126;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(v_6.y));
                            float2 av_5 = make_float2(_abs_125, _abs_126);
                            float2 _f32x2_mul_b_127 = make_float2(k1, k1);
                            float2 te_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_5) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
                            float2 _f32x2_mul_b_128 = make_float2(k2, k2);
                            float2 tw_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_5) : "l"(*(const unsigned long long*)&av_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_128));
                            float _exp2_129 = approx_exp2(te_5.x);
                            float _exp2_130 = approx_exp2(te_5.y);
                            float2 e_5 = make_float2(_exp2_129, _exp2_130);
                            float _exp2_131 = approx_exp2(tw_5.x);
                            float _exp2_132 = approx_exp2(tw_5.y);
                            float2 w_5 = make_float2(_exp2_131, _exp2_132);
                            float2 _f32x2_mul_b_133 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_5) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
                            float _exp2_134 = approx_exp2_noftz(tp_5.x);
                            float _exp2_135 = approx_exp2_noftz(tp_5.y);
                            float2 p_5 = make_float2(_exp2_134, _exp2_135);
                            float2 d1_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                            float2 d2_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_5);
                            float2 dw_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                            float2 ng_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                            float2 nu_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                            float _rcp_136 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_5, d2_5), dw_5).x);
                            float _rcp_137 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_5, d2_5), dw_5).y);
                            float2 r_12 = make_float2(_rcp_136, _rcp_137);
                            float2 r_11 = r_12;
                            float2 _f32x2_mul_a_138 = mul_f32x2_noftz(ng_5, nu_5);
                            float2 _f32x2_mul_b_139 = mul_f32x2_noftz(r_11, make_float2(bl, bl));
                            float2 m_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
                            float2 ns_5 = make_float2(((x_6.x < 0.0f) ? p_5.x : 1.0f), ((x_6.y < 0.0f) ? p_5.y : 1.0f));
                            gate_packed_0[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_5, ns_5))) | sign_6;
                            float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[7]));
                            float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[7]));
                            unsigned int sign_7 = (gate_packed_0[7] ^ up_packed_0[7]) & 2147516416u;
                            float _abs_140;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_140) : "f"(x_7.x));
                            float _abs_141;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_141) : "f"(x_7.y));
                            float2 ax_6 = make_float2(_abs_140, _abs_141);
                            float _abs_142;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_142) : "f"(v_7.x));
                            float _abs_143;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_143) : "f"(v_7.y));
                            float2 av_6 = make_float2(_abs_142, _abs_143);
                            float2 _f32x2_mul_b_144 = make_float2(k1, k1);
                            float2 te_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_6) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_144));
                            float2 _f32x2_mul_b_145 = make_float2(k2, k2);
                            float2 tw_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_6) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
                            float _exp2_146 = approx_exp2(te_6.x);
                            float _exp2_147 = approx_exp2(te_6.y);
                            float2 e_6 = make_float2(_exp2_146, _exp2_147);
                            float _exp2_148 = approx_exp2(tw_6.x);
                            float _exp2_149 = approx_exp2(tw_6.y);
                            float2 w_6 = make_float2(_exp2_148, _exp2_149);
                            float2 _f32x2_mul_b_150 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_6) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_150));
                            float _exp2_151 = approx_exp2_noftz(tp_6.x);
                            float _exp2_152 = approx_exp2_noftz(tp_6.y);
                            float2 p_6 = make_float2(_exp2_151, _exp2_152);
                            float2 d1_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_6);
                            float2 d2_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_6);
                            float2 dw_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_6);
                            float2 ng_6 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_6);
                            float2 nu_6 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_6);
                            float _rcp_153 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_6, d2_6), dw_6).x);
                            float _rcp_154 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_6, d2_6), dw_6).y);
                            float2 r_14 = make_float2(_rcp_153, _rcp_154);
                            float2 r_13 = r_14;
                            float2 _f32x2_mul_a_155 = mul_f32x2_noftz(ng_6, nu_6);
                            float2 _f32x2_mul_b_156 = mul_f32x2_noftz(r_13, make_float2(bl, bl));
                            float2 m_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&_f32x2_mul_b_156));
                            float2 ns_6 = make_float2(((x_7.x < 0.0f) ? p_6.x : 1.0f), ((x_7.y < 0.0f) ? p_6.y : 1.0f));
                            gate_packed_0[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_6, ns_6))) | sign_7;
                            float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[8]));
                            float2 v_8 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[8]));
                            unsigned int sign_8 = (gate_packed_0[8] ^ up_packed_0[8]) & 2147516416u;
                            float _abs_157;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_157) : "f"(x_8.x));
                            float _abs_158;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_158) : "f"(x_8.y));
                            float2 ax_7 = make_float2(_abs_157, _abs_158);
                            float _abs_159;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_159) : "f"(v_8.x));
                            float _abs_160;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(v_8.y));
                            float2 av_7 = make_float2(_abs_159, _abs_160);
                            float2 _f32x2_mul_b_161 = make_float2(k1, k1);
                            float2 te_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_7) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
                            float2 _f32x2_mul_b_162 = make_float2(k2, k2);
                            float2 tw_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_7) : "l"(*(const unsigned long long*)&av_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_162));
                            float _exp2_163 = approx_exp2(te_7.x);
                            float _exp2_164 = approx_exp2(te_7.y);
                            float2 e_7 = make_float2(_exp2_163, _exp2_164);
                            float _exp2_165 = approx_exp2(tw_7.x);
                            float _exp2_166 = approx_exp2(tw_7.y);
                            float2 w_7 = make_float2(_exp2_165, _exp2_166);
                            float2 _f32x2_mul_b_167 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_7) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
                            float _exp2_168 = approx_exp2_noftz(tp_7.x);
                            float _exp2_169 = approx_exp2_noftz(tp_7.y);
                            float2 p_7 = make_float2(_exp2_168, _exp2_169);
                            float2 d1_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_7);
                            float2 d2_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_7);
                            float2 dw_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_7);
                            float2 ng_7 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_7);
                            float2 nu_7 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_7);
                            float _rcp_170 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_7, d2_7), dw_7).x);
                            float _rcp_171 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_7, d2_7), dw_7).y);
                            float2 r_16 = make_float2(_rcp_170, _rcp_171);
                            float2 r_15 = r_16;
                            float2 _f32x2_mul_a_172 = mul_f32x2_noftz(ng_7, nu_7);
                            float2 _f32x2_mul_b_173 = mul_f32x2_noftz(r_15, make_float2(bl, bl));
                            float2 m_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
                            float2 ns_7 = make_float2(((x_8.x < 0.0f) ? p_7.x : 1.0f), ((x_8.y < 0.0f) ? p_7.y : 1.0f));
                            gate_packed_0[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_7, ns_7))) | sign_8;
                            float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[9]));
                            float2 v_9 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[9]));
                            unsigned int sign_9 = (gate_packed_0[9] ^ up_packed_0[9]) & 2147516416u;
                            float _abs_174;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_174) : "f"(x_9.x));
                            float _abs_175;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_175) : "f"(x_9.y));
                            float2 ax_8 = make_float2(_abs_174, _abs_175);
                            float _abs_176;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(v_9.x));
                            float _abs_177;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_177) : "f"(v_9.y));
                            float2 av_8 = make_float2(_abs_176, _abs_177);
                            float2 _f32x2_mul_b_178 = make_float2(k1, k1);
                            float2 te_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_8) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
                            float2 _f32x2_mul_b_179 = make_float2(k2, k2);
                            float2 tw_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_8) : "l"(*(const unsigned long long*)&av_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_179));
                            float _exp2_180 = approx_exp2(te_8.x);
                            float _exp2_181 = approx_exp2(te_8.y);
                            float2 e_8 = make_float2(_exp2_180, _exp2_181);
                            float _exp2_182 = approx_exp2(tw_8.x);
                            float _exp2_183 = approx_exp2(tw_8.y);
                            float2 w_8 = make_float2(_exp2_182, _exp2_183);
                            float2 _f32x2_mul_b_184 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_8) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_184));
                            float _exp2_185 = approx_exp2_noftz(tp_8.x);
                            float _exp2_186 = approx_exp2_noftz(tp_8.y);
                            float2 p_8 = make_float2(_exp2_185, _exp2_186);
                            float2 d1_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_8);
                            float2 d2_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_8);
                            float2 dw_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_8);
                            float2 ng_8 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_8);
                            float2 nu_8 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_8);
                            float _rcp_187 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_8, d2_8), dw_8).x);
                            float _rcp_188 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_8, d2_8), dw_8).y);
                            float2 r_18 = make_float2(_rcp_187, _rcp_188);
                            float2 r_17 = r_18;
                            float2 _f32x2_mul_a_189 = mul_f32x2_noftz(ng_8, nu_8);
                            float2 _f32x2_mul_b_190 = mul_f32x2_noftz(r_17, make_float2(bl, bl));
                            float2 m_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_189), "l"(*(const unsigned long long*)&_f32x2_mul_b_190));
                            float2 ns_8 = make_float2(((x_9.x < 0.0f) ? p_8.x : 1.0f), ((x_9.y < 0.0f) ? p_8.y : 1.0f));
                            gate_packed_0[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_8, ns_8))) | sign_9;
                            float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[10]));
                            float2 v_10 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[10]));
                            unsigned int sign_10 = (gate_packed_0[10] ^ up_packed_0[10]) & 2147516416u;
                            float _abs_191;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(x_10.x));
                            float _abs_192;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_192) : "f"(x_10.y));
                            float2 ax_9 = make_float2(_abs_191, _abs_192);
                            float _abs_193;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(v_10.x));
                            float _abs_194;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_194) : "f"(v_10.y));
                            float2 av_9 = make_float2(_abs_193, _abs_194);
                            float2 _f32x2_mul_b_195 = make_float2(k1, k1);
                            float2 te_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_9) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_195));
                            float2 _f32x2_mul_b_196 = make_float2(k2, k2);
                            float2 tw_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_9) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_196));
                            float _exp2_197 = approx_exp2(te_9.x);
                            float _exp2_198 = approx_exp2(te_9.y);
                            float2 e_9 = make_float2(_exp2_197, _exp2_198);
                            float _exp2_199 = approx_exp2(tw_9.x);
                            float _exp2_200 = approx_exp2(tw_9.y);
                            float2 w_9 = make_float2(_exp2_199, _exp2_200);
                            float2 _f32x2_mul_b_201 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_9) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
                            float _exp2_202 = approx_exp2_noftz(tp_9.x);
                            float _exp2_203 = approx_exp2_noftz(tp_9.y);
                            float2 p_9 = make_float2(_exp2_202, _exp2_203);
                            float2 d1_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_9);
                            float2 d2_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_9);
                            float2 dw_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_9);
                            float2 ng_9 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_9);
                            float2 nu_9 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_9);
                            float _rcp_204 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_9, d2_9), dw_9).x);
                            float _rcp_205 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_9, d2_9), dw_9).y);
                            float2 r_20 = make_float2(_rcp_204, _rcp_205);
                            float2 r_19 = r_20;
                            float2 _f32x2_mul_a_206 = mul_f32x2_noftz(ng_9, nu_9);
                            float2 _f32x2_mul_b_207 = mul_f32x2_noftz(r_19, make_float2(bl, bl));
                            float2 m_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
                            float2 ns_9 = make_float2(((x_10.x < 0.0f) ? p_9.x : 1.0f), ((x_10.y < 0.0f) ? p_9.y : 1.0f));
                            gate_packed_0[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_9, ns_9))) | sign_10;
                            float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[11]));
                            float2 v_11 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[11]));
                            unsigned int sign_11 = (gate_packed_0[11] ^ up_packed_0[11]) & 2147516416u;
                            float _abs_208;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_208) : "f"(x_11.x));
                            float _abs_209;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(x_11.y));
                            float2 ax_10 = make_float2(_abs_208, _abs_209);
                            float _abs_210;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_210) : "f"(v_11.x));
                            float _abs_211;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_211) : "f"(v_11.y));
                            float2 av_10 = make_float2(_abs_210, _abs_211);
                            float2 _f32x2_mul_b_212 = make_float2(k1, k1);
                            float2 te_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_10) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_212));
                            float2 _f32x2_mul_b_213 = make_float2(k2, k2);
                            float2 tw_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_10) : "l"(*(const unsigned long long*)&av_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
                            float _exp2_214 = approx_exp2(te_10.x);
                            float _exp2_215 = approx_exp2(te_10.y);
                            float2 e_10 = make_float2(_exp2_214, _exp2_215);
                            float _exp2_216 = approx_exp2(tw_10.x);
                            float _exp2_217 = approx_exp2(tw_10.y);
                            float2 w_10 = make_float2(_exp2_216, _exp2_217);
                            float2 _f32x2_mul_b_218 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_10) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_218));
                            float _exp2_219 = approx_exp2_noftz(tp_10.x);
                            float _exp2_220 = approx_exp2_noftz(tp_10.y);
                            float2 p_10 = make_float2(_exp2_219, _exp2_220);
                            float2 d1_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_10);
                            float2 d2_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_10);
                            float2 dw_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_10);
                            float2 ng_10 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_10);
                            float2 nu_10 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_10);
                            float _rcp_221 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_10, d2_10), dw_10).x);
                            float _rcp_222 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_10, d2_10), dw_10).y);
                            float2 r_22 = make_float2(_rcp_221, _rcp_222);
                            float2 r_21 = r_22;
                            float2 _f32x2_mul_a_223 = mul_f32x2_noftz(ng_10, nu_10);
                            float2 _f32x2_mul_b_224 = mul_f32x2_noftz(r_21, make_float2(bl, bl));
                            float2 m_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&_f32x2_mul_b_224));
                            float2 ns_10 = make_float2(((x_11.x < 0.0f) ? p_10.x : 1.0f), ((x_11.y < 0.0f) ? p_10.y : 1.0f));
                            gate_packed_0[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_10, ns_10))) | sign_11;
                            float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[12]));
                            float2 v_12 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[12]));
                            unsigned int sign_12 = (gate_packed_0[12] ^ up_packed_0[12]) & 2147516416u;
                            float _abs_225;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_225) : "f"(x_12.x));
                            float _abs_226;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_226) : "f"(x_12.y));
                            float2 ax_11 = make_float2(_abs_225, _abs_226);
                            float _abs_227;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_227) : "f"(v_12.x));
                            float _abs_228;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_228) : "f"(v_12.y));
                            float2 av_11 = make_float2(_abs_227, _abs_228);
                            float2 _f32x2_mul_b_229 = make_float2(k1, k1);
                            float2 te_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_11) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
                            float2 _f32x2_mul_b_230 = make_float2(k2, k2);
                            float2 tw_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_11) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
                            float _exp2_231 = approx_exp2(te_11.x);
                            float _exp2_232 = approx_exp2(te_11.y);
                            float2 e_11 = make_float2(_exp2_231, _exp2_232);
                            float _exp2_233 = approx_exp2(tw_11.x);
                            float _exp2_234 = approx_exp2(tw_11.y);
                            float2 w_11 = make_float2(_exp2_233, _exp2_234);
                            float2 _f32x2_mul_b_235 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_11) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_235));
                            float _exp2_236 = approx_exp2_noftz(tp_11.x);
                            float _exp2_237 = approx_exp2_noftz(tp_11.y);
                            float2 p_11 = make_float2(_exp2_236, _exp2_237);
                            float2 d1_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_11);
                            float2 d2_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_11);
                            float2 dw_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_11);
                            float2 ng_11 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_11);
                            float2 nu_11 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_11);
                            float _rcp_238 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_11, d2_11), dw_11).x);
                            float _rcp_239 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_11, d2_11), dw_11).y);
                            float2 r_24 = make_float2(_rcp_238, _rcp_239);
                            float2 r_23 = r_24;
                            float2 _f32x2_mul_a_240 = mul_f32x2_noftz(ng_11, nu_11);
                            float2 _f32x2_mul_b_241 = mul_f32x2_noftz(r_23, make_float2(bl, bl));
                            float2 m_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_240), "l"(*(const unsigned long long*)&_f32x2_mul_b_241));
                            float2 ns_11 = make_float2(((x_12.x < 0.0f) ? p_11.x : 1.0f), ((x_12.y < 0.0f) ? p_11.y : 1.0f));
                            gate_packed_0[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_11, ns_11))) | sign_12;
                            float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[13]));
                            float2 v_13 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[13]));
                            unsigned int sign_13 = (gate_packed_0[13] ^ up_packed_0[13]) & 2147516416u;
                            float _abs_242;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_242) : "f"(x_13.x));
                            float _abs_243;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_243) : "f"(x_13.y));
                            float2 ax_12 = make_float2(_abs_242, _abs_243);
                            float _abs_244;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_244) : "f"(v_13.x));
                            float _abs_245;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_245) : "f"(v_13.y));
                            float2 av_12 = make_float2(_abs_244, _abs_245);
                            float2 _f32x2_mul_b_246 = make_float2(k1, k1);
                            float2 te_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_12) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_246));
                            float2 _f32x2_mul_b_247 = make_float2(k2, k2);
                            float2 tw_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_12) : "l"(*(const unsigned long long*)&av_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_247));
                            float _exp2_248 = approx_exp2(te_12.x);
                            float _exp2_249 = approx_exp2(te_12.y);
                            float2 e_12 = make_float2(_exp2_248, _exp2_249);
                            float _exp2_250 = approx_exp2(tw_12.x);
                            float _exp2_251 = approx_exp2(tw_12.y);
                            float2 w_12 = make_float2(_exp2_250, _exp2_251);
                            float2 _f32x2_mul_b_252 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_12) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_252));
                            float _exp2_253 = approx_exp2_noftz(tp_12.x);
                            float _exp2_254 = approx_exp2_noftz(tp_12.y);
                            float2 p_12 = make_float2(_exp2_253, _exp2_254);
                            float2 d1_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_12);
                            float2 d2_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_12);
                            float2 dw_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_12);
                            float2 ng_12 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_12);
                            float2 nu_12 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_12);
                            float _rcp_255 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_12, d2_12), dw_12).x);
                            float _rcp_256 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_12, d2_12), dw_12).y);
                            float2 r_26 = make_float2(_rcp_255, _rcp_256);
                            float2 r_25 = r_26;
                            float2 _f32x2_mul_a_257 = mul_f32x2_noftz(ng_12, nu_12);
                            float2 _f32x2_mul_b_258 = mul_f32x2_noftz(r_25, make_float2(bl, bl));
                            float2 m_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_257), "l"(*(const unsigned long long*)&_f32x2_mul_b_258));
                            float2 ns_12 = make_float2(((x_13.x < 0.0f) ? p_12.x : 1.0f), ((x_13.y < 0.0f) ? p_12.y : 1.0f));
                            gate_packed_0[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_12, ns_12))) | sign_13;
                            float2 x_14 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[14]));
                            float2 v_14 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[14]));
                            unsigned int sign_14 = (gate_packed_0[14] ^ up_packed_0[14]) & 2147516416u;
                            float _abs_259;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_259) : "f"(x_14.x));
                            float _abs_260;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_260) : "f"(x_14.y));
                            float2 ax_13 = make_float2(_abs_259, _abs_260);
                            float _abs_261;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(v_14.x));
                            float _abs_262;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_262) : "f"(v_14.y));
                            float2 av_13 = make_float2(_abs_261, _abs_262);
                            float2 _f32x2_mul_b_263 = make_float2(k1, k1);
                            float2 te_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_13) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_263));
                            float2 _f32x2_mul_b_264 = make_float2(k2, k2);
                            float2 tw_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_13) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_264));
                            float _exp2_265 = approx_exp2(te_13.x);
                            float _exp2_266 = approx_exp2(te_13.y);
                            float2 e_13 = make_float2(_exp2_265, _exp2_266);
                            float _exp2_267 = approx_exp2(tw_13.x);
                            float _exp2_268 = approx_exp2(tw_13.y);
                            float2 w_13 = make_float2(_exp2_267, _exp2_268);
                            float2 _f32x2_mul_b_269 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_13) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_269));
                            float _exp2_270 = approx_exp2_noftz(tp_13.x);
                            float _exp2_271 = approx_exp2_noftz(tp_13.y);
                            float2 p_13 = make_float2(_exp2_270, _exp2_271);
                            float2 d1_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_13);
                            float2 d2_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_13);
                            float2 dw_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_13);
                            float2 ng_13 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_13);
                            float2 nu_13 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_13);
                            float _rcp_272 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_13, d2_13), dw_13).x);
                            float _rcp_273 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_13, d2_13), dw_13).y);
                            float2 r_28 = make_float2(_rcp_272, _rcp_273);
                            float2 r_27 = r_28;
                            float2 _f32x2_mul_a_274 = mul_f32x2_noftz(ng_13, nu_13);
                            float2 _f32x2_mul_b_275 = mul_f32x2_noftz(r_27, make_float2(bl, bl));
                            float2 m_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_274), "l"(*(const unsigned long long*)&_f32x2_mul_b_275));
                            float2 ns_13 = make_float2(((x_14.x < 0.0f) ? p_13.x : 1.0f), ((x_14.y < 0.0f) ? p_13.y : 1.0f));
                            gate_packed_0[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_13, ns_13))) | sign_14;
                            float2 x_15 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[15]));
                            float2 v_15 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[15]));
                            unsigned int sign_15 = (gate_packed_0[15] ^ up_packed_0[15]) & 2147516416u;
                            float _abs_276;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_276) : "f"(x_15.x));
                            float _abs_277;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_277) : "f"(x_15.y));
                            float2 ax_14 = make_float2(_abs_276, _abs_277);
                            float _abs_278;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_278) : "f"(v_15.x));
                            float _abs_279;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_279) : "f"(v_15.y));
                            float2 av_14 = make_float2(_abs_278, _abs_279);
                            float2 _f32x2_mul_b_280 = make_float2(k1, k1);
                            float2 te_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_14) : "l"(*(const unsigned long long*)&ax_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_280));
                            float2 _f32x2_mul_b_281 = make_float2(k2, k2);
                            float2 tw_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_14) : "l"(*(const unsigned long long*)&av_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_281));
                            float _exp2_282 = approx_exp2(te_14.x);
                            float _exp2_283 = approx_exp2(te_14.y);
                            float2 e_14 = make_float2(_exp2_282, _exp2_283);
                            float _exp2_284 = approx_exp2(tw_14.x);
                            float _exp2_285 = approx_exp2(tw_14.y);
                            float2 w_14 = make_float2(_exp2_284, _exp2_285);
                            float2 _f32x2_mul_b_286 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_14) : "l"(*(const unsigned long long*)&ax_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_286));
                            float _exp2_287 = approx_exp2_noftz(tp_14.x);
                            float _exp2_288 = approx_exp2_noftz(tp_14.y);
                            float2 p_14 = make_float2(_exp2_287, _exp2_288);
                            float2 d1_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_14);
                            float2 d2_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_14);
                            float2 dw_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_14);
                            float2 ng_14 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_14);
                            float2 nu_14 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_14);
                            float _rcp_289 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_14, d2_14), dw_14).x);
                            float _rcp_290 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_14, d2_14), dw_14).y);
                            float2 r_30 = make_float2(_rcp_289, _rcp_290);
                            float2 r_29 = r_30;
                            float2 _f32x2_mul_a_291 = mul_f32x2_noftz(ng_14, nu_14);
                            float2 _f32x2_mul_b_292 = mul_f32x2_noftz(r_29, make_float2(bl, bl));
                            float2 m_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_291), "l"(*(const unsigned long long*)&_f32x2_mul_b_292));
                            float2 ns_14 = make_float2(((x_15.x < 0.0f) ? p_14.x : 1.0f), ((x_15.y < 0.0f) ? p_14.y : 1.0f));
                            gate_packed_0[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_14, ns_14))) | sign_15;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_3 = tid / 32;
                            int lane_3 = tid % 32;
                            #pragma unroll
                            for (int half_4 = 0; half_4 < 2; half_4++) {
                                #pragma unroll
                                for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                    int row_5 = warp_3 * 32 + half_4 * 16 + lane_3 % 16;
                                    int col_5 = col_tile_4 * 16 + lane_3 / 16 * 8;
                                    unsigned int address_4 = d_smem_addr + 8192 + (unsigned int)((row_5 * 32 + col_5) * 2);
                                    address_4 = address_4 ^ (address_4 & 511) >> 7 << 4;
                                    int offset_5 = half_4 * 8 + col_tile_4 * 4;
                                    uint32_t _stmatrix_addr_293 = static_cast<uint32_t>(address_4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_293), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_16 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[16]));
                            float2 v_16 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[16]));
                            unsigned int sign_16 = (gate_packed_0[16] ^ up_packed_0[16]) & 2147516416u;
                            float _abs_294;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_294) : "f"(x_16.x));
                            float _abs_295;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_295) : "f"(x_16.y));
                            float2 ax_15 = make_float2(_abs_294, _abs_295);
                            float _abs_296;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_296) : "f"(v_16.x));
                            float _abs_297;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_297) : "f"(v_16.y));
                            float2 av_15 = make_float2(_abs_296, _abs_297);
                            float2 _f32x2_mul_b_298 = make_float2(k1, k1);
                            float2 te_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_15) : "l"(*(const unsigned long long*)&ax_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_298));
                            float2 _f32x2_mul_b_299 = make_float2(k2, k2);
                            float2 tw_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_15) : "l"(*(const unsigned long long*)&av_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_299));
                            float _exp2_300 = approx_exp2(te_15.x);
                            float _exp2_301 = approx_exp2(te_15.y);
                            float2 e_15 = make_float2(_exp2_300, _exp2_301);
                            float _exp2_302 = approx_exp2(tw_15.x);
                            float _exp2_303 = approx_exp2(tw_15.y);
                            float2 w_15 = make_float2(_exp2_302, _exp2_303);
                            float2 _f32x2_mul_b_304 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_15) : "l"(*(const unsigned long long*)&ax_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_304));
                            float _exp2_305 = approx_exp2_noftz(tp_15.x);
                            float _exp2_306 = approx_exp2_noftz(tp_15.y);
                            float2 p_15 = make_float2(_exp2_305, _exp2_306);
                            float2 d1_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_15);
                            float2 d2_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_15);
                            float2 dw_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_15);
                            float2 ng_15 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_15);
                            float2 nu_15 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_15);
                            float _rcp_307 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_15, d2_15), dw_15).x);
                            float _rcp_308 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_15, d2_15), dw_15).y);
                            float2 r_32 = make_float2(_rcp_307, _rcp_308);
                            float2 r_31 = r_32;
                            float2 _f32x2_mul_a_309 = mul_f32x2_noftz(ng_15, nu_15);
                            float2 _f32x2_mul_b_310 = mul_f32x2_noftz(r_31, make_float2(bl, bl));
                            float2 m_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_309), "l"(*(const unsigned long long*)&_f32x2_mul_b_310));
                            float2 ns_15 = make_float2(((x_16.x < 0.0f) ? p_15.x : 1.0f), ((x_16.y < 0.0f) ? p_15.y : 1.0f));
                            gate_packed_0[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_15, ns_15))) | sign_16;
                            float2 x_17 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[17]));
                            float2 v_17 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[17]));
                            unsigned int sign_17 = (gate_packed_0[17] ^ up_packed_0[17]) & 2147516416u;
                            float _abs_311;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_311) : "f"(x_17.x));
                            float _abs_312;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_312) : "f"(x_17.y));
                            float2 ax_16 = make_float2(_abs_311, _abs_312);
                            float _abs_313;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_313) : "f"(v_17.x));
                            float _abs_314;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_314) : "f"(v_17.y));
                            float2 av_16 = make_float2(_abs_313, _abs_314);
                            float2 _f32x2_mul_b_315 = make_float2(k1, k1);
                            float2 te_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_16) : "l"(*(const unsigned long long*)&ax_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_315));
                            float2 _f32x2_mul_b_316 = make_float2(k2, k2);
                            float2 tw_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_16) : "l"(*(const unsigned long long*)&av_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_316));
                            float _exp2_317 = approx_exp2(te_16.x);
                            float _exp2_318 = approx_exp2(te_16.y);
                            float2 e_16 = make_float2(_exp2_317, _exp2_318);
                            float _exp2_319 = approx_exp2(tw_16.x);
                            float _exp2_320 = approx_exp2(tw_16.y);
                            float2 w_16 = make_float2(_exp2_319, _exp2_320);
                            float2 _f32x2_mul_b_321 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_16) : "l"(*(const unsigned long long*)&ax_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_321));
                            float _exp2_322 = approx_exp2_noftz(tp_16.x);
                            float _exp2_323 = approx_exp2_noftz(tp_16.y);
                            float2 p_16 = make_float2(_exp2_322, _exp2_323);
                            float2 d1_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_16);
                            float2 d2_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_16);
                            float2 dw_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_16);
                            float2 ng_16 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_16);
                            float2 nu_16 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_16);
                            float _rcp_324 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_16, d2_16), dw_16).x);
                            float _rcp_325 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_16, d2_16), dw_16).y);
                            float2 r_34 = make_float2(_rcp_324, _rcp_325);
                            float2 r_33 = r_34;
                            float2 _f32x2_mul_a_326 = mul_f32x2_noftz(ng_16, nu_16);
                            float2 _f32x2_mul_b_327 = mul_f32x2_noftz(r_33, make_float2(bl, bl));
                            float2 m_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_326), "l"(*(const unsigned long long*)&_f32x2_mul_b_327));
                            float2 ns_16 = make_float2(((x_17.x < 0.0f) ? p_16.x : 1.0f), ((x_17.y < 0.0f) ? p_16.y : 1.0f));
                            gate_packed_0[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_16, ns_16))) | sign_17;
                            float2 x_18 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[18]));
                            float2 v_18 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[18]));
                            unsigned int sign_18 = (gate_packed_0[18] ^ up_packed_0[18]) & 2147516416u;
                            float _abs_328;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_328) : "f"(x_18.x));
                            float _abs_329;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_329) : "f"(x_18.y));
                            float2 ax_17 = make_float2(_abs_328, _abs_329);
                            float _abs_330;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_330) : "f"(v_18.x));
                            float _abs_331;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_331) : "f"(v_18.y));
                            float2 av_17 = make_float2(_abs_330, _abs_331);
                            float2 _f32x2_mul_b_332 = make_float2(k1, k1);
                            float2 te_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_17) : "l"(*(const unsigned long long*)&ax_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_332));
                            float2 _f32x2_mul_b_333 = make_float2(k2, k2);
                            float2 tw_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_17) : "l"(*(const unsigned long long*)&av_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_333));
                            float _exp2_334 = approx_exp2(te_17.x);
                            float _exp2_335 = approx_exp2(te_17.y);
                            float2 e_17 = make_float2(_exp2_334, _exp2_335);
                            float _exp2_336 = approx_exp2(tw_17.x);
                            float _exp2_337 = approx_exp2(tw_17.y);
                            float2 w_17 = make_float2(_exp2_336, _exp2_337);
                            float2 _f32x2_mul_b_338 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_17) : "l"(*(const unsigned long long*)&ax_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_338));
                            float _exp2_339 = approx_exp2_noftz(tp_17.x);
                            float _exp2_340 = approx_exp2_noftz(tp_17.y);
                            float2 p_17 = make_float2(_exp2_339, _exp2_340);
                            float2 d1_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_17);
                            float2 d2_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_17);
                            float2 dw_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_17);
                            float2 ng_17 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_17);
                            float2 nu_17 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_17);
                            float _rcp_341 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_17, d2_17), dw_17).x);
                            float _rcp_342 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_17, d2_17), dw_17).y);
                            float2 r_36 = make_float2(_rcp_341, _rcp_342);
                            float2 r_35 = r_36;
                            float2 _f32x2_mul_a_343 = mul_f32x2_noftz(ng_17, nu_17);
                            float2 _f32x2_mul_b_344 = mul_f32x2_noftz(r_35, make_float2(bl, bl));
                            float2 m_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_343), "l"(*(const unsigned long long*)&_f32x2_mul_b_344));
                            float2 ns_17 = make_float2(((x_18.x < 0.0f) ? p_17.x : 1.0f), ((x_18.y < 0.0f) ? p_17.y : 1.0f));
                            gate_packed_0[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_17, ns_17))) | sign_18;
                            float2 x_19 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[19]));
                            float2 v_19 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[19]));
                            unsigned int sign_19 = (gate_packed_0[19] ^ up_packed_0[19]) & 2147516416u;
                            float _abs_345;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_345) : "f"(x_19.x));
                            float _abs_346;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_346) : "f"(x_19.y));
                            float2 ax_18 = make_float2(_abs_345, _abs_346);
                            float _abs_347;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_347) : "f"(v_19.x));
                            float _abs_348;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_348) : "f"(v_19.y));
                            float2 av_18 = make_float2(_abs_347, _abs_348);
                            float2 _f32x2_mul_b_349 = make_float2(k1, k1);
                            float2 te_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_18) : "l"(*(const unsigned long long*)&ax_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_349));
                            float2 _f32x2_mul_b_350 = make_float2(k2, k2);
                            float2 tw_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_18) : "l"(*(const unsigned long long*)&av_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_350));
                            float _exp2_351 = approx_exp2(te_18.x);
                            float _exp2_352 = approx_exp2(te_18.y);
                            float2 e_18 = make_float2(_exp2_351, _exp2_352);
                            float _exp2_353 = approx_exp2(tw_18.x);
                            float _exp2_354 = approx_exp2(tw_18.y);
                            float2 w_18 = make_float2(_exp2_353, _exp2_354);
                            float2 _f32x2_mul_b_355 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_18) : "l"(*(const unsigned long long*)&ax_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_355));
                            float _exp2_356 = approx_exp2_noftz(tp_18.x);
                            float _exp2_357 = approx_exp2_noftz(tp_18.y);
                            float2 p_18 = make_float2(_exp2_356, _exp2_357);
                            float2 d1_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_18);
                            float2 d2_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_18);
                            float2 dw_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_18);
                            float2 ng_18 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_18);
                            float2 nu_18 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_18);
                            float _rcp_358 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_18, d2_18), dw_18).x);
                            float _rcp_359 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_18, d2_18), dw_18).y);
                            float2 r_38 = make_float2(_rcp_358, _rcp_359);
                            float2 r_37 = r_38;
                            float2 _f32x2_mul_a_360 = mul_f32x2_noftz(ng_18, nu_18);
                            float2 _f32x2_mul_b_361 = mul_f32x2_noftz(r_37, make_float2(bl, bl));
                            float2 m_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_360), "l"(*(const unsigned long long*)&_f32x2_mul_b_361));
                            float2 ns_18 = make_float2(((x_19.x < 0.0f) ? p_18.x : 1.0f), ((x_19.y < 0.0f) ? p_18.y : 1.0f));
                            gate_packed_0[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_18, ns_18))) | sign_19;
                            float2 x_20 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[20]));
                            float2 v_20 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[20]));
                            unsigned int sign_20 = (gate_packed_0[20] ^ up_packed_0[20]) & 2147516416u;
                            float _abs_362;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_362) : "f"(x_20.x));
                            float _abs_363;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_363) : "f"(x_20.y));
                            float2 ax_19 = make_float2(_abs_362, _abs_363);
                            float _abs_364;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_364) : "f"(v_20.x));
                            float _abs_365;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_365) : "f"(v_20.y));
                            float2 av_19 = make_float2(_abs_364, _abs_365);
                            float2 _f32x2_mul_b_366 = make_float2(k1, k1);
                            float2 te_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_19) : "l"(*(const unsigned long long*)&ax_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_366));
                            float2 _f32x2_mul_b_367 = make_float2(k2, k2);
                            float2 tw_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_19) : "l"(*(const unsigned long long*)&av_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_367));
                            float _exp2_368 = approx_exp2(te_19.x);
                            float _exp2_369 = approx_exp2(te_19.y);
                            float2 e_19 = make_float2(_exp2_368, _exp2_369);
                            float _exp2_370 = approx_exp2(tw_19.x);
                            float _exp2_371 = approx_exp2(tw_19.y);
                            float2 w_19 = make_float2(_exp2_370, _exp2_371);
                            float2 _f32x2_mul_b_372 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_19) : "l"(*(const unsigned long long*)&ax_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_372));
                            float _exp2_373 = approx_exp2_noftz(tp_19.x);
                            float _exp2_374 = approx_exp2_noftz(tp_19.y);
                            float2 p_19 = make_float2(_exp2_373, _exp2_374);
                            float2 d1_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_19);
                            float2 d2_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_19);
                            float2 dw_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_19);
                            float2 ng_19 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_19);
                            float2 nu_19 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_19);
                            float _rcp_375 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_19, d2_19), dw_19).x);
                            float _rcp_376 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_19, d2_19), dw_19).y);
                            float2 r_40 = make_float2(_rcp_375, _rcp_376);
                            float2 r_39 = r_40;
                            float2 _f32x2_mul_a_377 = mul_f32x2_noftz(ng_19, nu_19);
                            float2 _f32x2_mul_b_378 = mul_f32x2_noftz(r_39, make_float2(bl, bl));
                            float2 m_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_377), "l"(*(const unsigned long long*)&_f32x2_mul_b_378));
                            float2 ns_19 = make_float2(((x_20.x < 0.0f) ? p_19.x : 1.0f), ((x_20.y < 0.0f) ? p_19.y : 1.0f));
                            gate_packed_0[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_19, ns_19))) | sign_20;
                            float2 x_21 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[21]));
                            float2 v_21 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[21]));
                            unsigned int sign_21 = (gate_packed_0[21] ^ up_packed_0[21]) & 2147516416u;
                            float _abs_379;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_379) : "f"(x_21.x));
                            float _abs_380;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_380) : "f"(x_21.y));
                            float2 ax_20 = make_float2(_abs_379, _abs_380);
                            float _abs_381;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_381) : "f"(v_21.x));
                            float _abs_382;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_382) : "f"(v_21.y));
                            float2 av_20 = make_float2(_abs_381, _abs_382);
                            float2 _f32x2_mul_b_383 = make_float2(k1, k1);
                            float2 te_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_20) : "l"(*(const unsigned long long*)&ax_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_383));
                            float2 _f32x2_mul_b_384 = make_float2(k2, k2);
                            float2 tw_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_20) : "l"(*(const unsigned long long*)&av_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_384));
                            float _exp2_385 = approx_exp2(te_20.x);
                            float _exp2_386 = approx_exp2(te_20.y);
                            float2 e_20 = make_float2(_exp2_385, _exp2_386);
                            float _exp2_387 = approx_exp2(tw_20.x);
                            float _exp2_388 = approx_exp2(tw_20.y);
                            float2 w_20 = make_float2(_exp2_387, _exp2_388);
                            float2 _f32x2_mul_b_389 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_20) : "l"(*(const unsigned long long*)&ax_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_389));
                            float _exp2_390 = approx_exp2_noftz(tp_20.x);
                            float _exp2_391 = approx_exp2_noftz(tp_20.y);
                            float2 p_20 = make_float2(_exp2_390, _exp2_391);
                            float2 d1_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_20);
                            float2 d2_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_20);
                            float2 dw_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_20);
                            float2 ng_20 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_20);
                            float2 nu_20 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_20);
                            float _rcp_392 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_20, d2_20), dw_20).x);
                            float _rcp_393 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_20, d2_20), dw_20).y);
                            float2 r_42 = make_float2(_rcp_392, _rcp_393);
                            float2 r_41 = r_42;
                            float2 _f32x2_mul_a_394 = mul_f32x2_noftz(ng_20, nu_20);
                            float2 _f32x2_mul_b_395 = mul_f32x2_noftz(r_41, make_float2(bl, bl));
                            float2 m_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_394), "l"(*(const unsigned long long*)&_f32x2_mul_b_395));
                            float2 ns_20 = make_float2(((x_21.x < 0.0f) ? p_20.x : 1.0f), ((x_21.y < 0.0f) ? p_20.y : 1.0f));
                            gate_packed_0[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_20, ns_20))) | sign_21;
                            float2 x_22 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[22]));
                            float2 v_22 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[22]));
                            unsigned int sign_22 = (gate_packed_0[22] ^ up_packed_0[22]) & 2147516416u;
                            float _abs_396;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_396) : "f"(x_22.x));
                            float _abs_397;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_397) : "f"(x_22.y));
                            float2 ax_21 = make_float2(_abs_396, _abs_397);
                            float _abs_398;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_398) : "f"(v_22.x));
                            float _abs_399;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_399) : "f"(v_22.y));
                            float2 av_21 = make_float2(_abs_398, _abs_399);
                            float2 _f32x2_mul_b_400 = make_float2(k1, k1);
                            float2 te_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_21) : "l"(*(const unsigned long long*)&ax_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_400));
                            float2 _f32x2_mul_b_401 = make_float2(k2, k2);
                            float2 tw_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_21) : "l"(*(const unsigned long long*)&av_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_401));
                            float _exp2_402 = approx_exp2(te_21.x);
                            float _exp2_403 = approx_exp2(te_21.y);
                            float2 e_21 = make_float2(_exp2_402, _exp2_403);
                            float _exp2_404 = approx_exp2(tw_21.x);
                            float _exp2_405 = approx_exp2(tw_21.y);
                            float2 w_21 = make_float2(_exp2_404, _exp2_405);
                            float2 _f32x2_mul_b_406 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_21) : "l"(*(const unsigned long long*)&ax_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_406));
                            float _exp2_407 = approx_exp2_noftz(tp_21.x);
                            float _exp2_408 = approx_exp2_noftz(tp_21.y);
                            float2 p_21 = make_float2(_exp2_407, _exp2_408);
                            float2 d1_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_21);
                            float2 d2_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_21);
                            float2 dw_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_21);
                            float2 ng_21 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_21);
                            float2 nu_21 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_21);
                            float _rcp_409 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_21, d2_21), dw_21).x);
                            float _rcp_410 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_21, d2_21), dw_21).y);
                            float2 r_44 = make_float2(_rcp_409, _rcp_410);
                            float2 r_43 = r_44;
                            float2 _f32x2_mul_a_411 = mul_f32x2_noftz(ng_21, nu_21);
                            float2 _f32x2_mul_b_412 = mul_f32x2_noftz(r_43, make_float2(bl, bl));
                            float2 m_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_411), "l"(*(const unsigned long long*)&_f32x2_mul_b_412));
                            float2 ns_21 = make_float2(((x_22.x < 0.0f) ? p_21.x : 1.0f), ((x_22.y < 0.0f) ? p_21.y : 1.0f));
                            gate_packed_0[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_21, ns_21))) | sign_22;
                            float2 x_23 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[23]));
                            float2 v_23 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[23]));
                            unsigned int sign_23 = (gate_packed_0[23] ^ up_packed_0[23]) & 2147516416u;
                            float _abs_413;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_413) : "f"(x_23.x));
                            float _abs_414;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_414) : "f"(x_23.y));
                            float2 ax_22 = make_float2(_abs_413, _abs_414);
                            float _abs_415;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_415) : "f"(v_23.x));
                            float _abs_416;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_416) : "f"(v_23.y));
                            float2 av_22 = make_float2(_abs_415, _abs_416);
                            float2 _f32x2_mul_b_417 = make_float2(k1, k1);
                            float2 te_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_22) : "l"(*(const unsigned long long*)&ax_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_417));
                            float2 _f32x2_mul_b_418 = make_float2(k2, k2);
                            float2 tw_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_22) : "l"(*(const unsigned long long*)&av_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_418));
                            float _exp2_419 = approx_exp2(te_22.x);
                            float _exp2_420 = approx_exp2(te_22.y);
                            float2 e_22 = make_float2(_exp2_419, _exp2_420);
                            float _exp2_421 = approx_exp2(tw_22.x);
                            float _exp2_422 = approx_exp2(tw_22.y);
                            float2 w_22 = make_float2(_exp2_421, _exp2_422);
                            float2 _f32x2_mul_b_423 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_22) : "l"(*(const unsigned long long*)&ax_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_423));
                            float _exp2_424 = approx_exp2_noftz(tp_22.x);
                            float _exp2_425 = approx_exp2_noftz(tp_22.y);
                            float2 p_22 = make_float2(_exp2_424, _exp2_425);
                            float2 d1_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_22);
                            float2 d2_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_22);
                            float2 dw_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_22);
                            float2 ng_22 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_22);
                            float2 nu_22 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_22);
                            float _rcp_426 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_22, d2_22), dw_22).x);
                            float _rcp_427 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_22, d2_22), dw_22).y);
                            float2 r_46 = make_float2(_rcp_426, _rcp_427);
                            float2 r_45 = r_46;
                            float2 _f32x2_mul_a_428 = mul_f32x2_noftz(ng_22, nu_22);
                            float2 _f32x2_mul_b_429 = mul_f32x2_noftz(r_45, make_float2(bl, bl));
                            float2 m_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_428), "l"(*(const unsigned long long*)&_f32x2_mul_b_429));
                            float2 ns_22 = make_float2(((x_23.x < 0.0f) ? p_22.x : 1.0f), ((x_23.y < 0.0f) ? p_22.y : 1.0f));
                            gate_packed_0[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_22, ns_22))) | sign_23;
                            float2 x_24 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[24]));
                            float2 v_24 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[24]));
                            unsigned int sign_24 = (gate_packed_0[24] ^ up_packed_0[24]) & 2147516416u;
                            float _abs_430;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_430) : "f"(x_24.x));
                            float _abs_431;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_431) : "f"(x_24.y));
                            float2 ax_23 = make_float2(_abs_430, _abs_431);
                            float _abs_432;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_432) : "f"(v_24.x));
                            float _abs_433;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_433) : "f"(v_24.y));
                            float2 av_23 = make_float2(_abs_432, _abs_433);
                            float2 _f32x2_mul_b_434 = make_float2(k1, k1);
                            float2 te_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_23) : "l"(*(const unsigned long long*)&ax_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_434));
                            float2 _f32x2_mul_b_435 = make_float2(k2, k2);
                            float2 tw_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_23) : "l"(*(const unsigned long long*)&av_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_435));
                            float _exp2_436 = approx_exp2(te_23.x);
                            float _exp2_437 = approx_exp2(te_23.y);
                            float2 e_23 = make_float2(_exp2_436, _exp2_437);
                            float _exp2_438 = approx_exp2(tw_23.x);
                            float _exp2_439 = approx_exp2(tw_23.y);
                            float2 w_23 = make_float2(_exp2_438, _exp2_439);
                            float2 _f32x2_mul_b_440 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_23) : "l"(*(const unsigned long long*)&ax_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_440));
                            float _exp2_441 = approx_exp2_noftz(tp_23.x);
                            float _exp2_442 = approx_exp2_noftz(tp_23.y);
                            float2 p_23 = make_float2(_exp2_441, _exp2_442);
                            float2 d1_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_23);
                            float2 d2_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_23);
                            float2 dw_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_23);
                            float2 ng_23 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_23);
                            float2 nu_23 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_23);
                            float _rcp_443 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_23, d2_23), dw_23).x);
                            float _rcp_444 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_23, d2_23), dw_23).y);
                            float2 r_48 = make_float2(_rcp_443, _rcp_444);
                            float2 r_47 = r_48;
                            float2 _f32x2_mul_a_445 = mul_f32x2_noftz(ng_23, nu_23);
                            float2 _f32x2_mul_b_446 = mul_f32x2_noftz(r_47, make_float2(bl, bl));
                            float2 m_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_445), "l"(*(const unsigned long long*)&_f32x2_mul_b_446));
                            float2 ns_23 = make_float2(((x_24.x < 0.0f) ? p_23.x : 1.0f), ((x_24.y < 0.0f) ? p_23.y : 1.0f));
                            gate_packed_0[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_23, ns_23))) | sign_24;
                            float2 x_25 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[25]));
                            float2 v_25 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[25]));
                            unsigned int sign_25 = (gate_packed_0[25] ^ up_packed_0[25]) & 2147516416u;
                            float _abs_447;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_447) : "f"(x_25.x));
                            float _abs_448;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_448) : "f"(x_25.y));
                            float2 ax_24 = make_float2(_abs_447, _abs_448);
                            float _abs_449;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_449) : "f"(v_25.x));
                            float _abs_450;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_450) : "f"(v_25.y));
                            float2 av_24 = make_float2(_abs_449, _abs_450);
                            float2 _f32x2_mul_b_451 = make_float2(k1, k1);
                            float2 te_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_24) : "l"(*(const unsigned long long*)&ax_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_451));
                            float2 _f32x2_mul_b_452 = make_float2(k2, k2);
                            float2 tw_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_24) : "l"(*(const unsigned long long*)&av_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_452));
                            float _exp2_453 = approx_exp2(te_24.x);
                            float _exp2_454 = approx_exp2(te_24.y);
                            float2 e_24 = make_float2(_exp2_453, _exp2_454);
                            float _exp2_455 = approx_exp2(tw_24.x);
                            float _exp2_456 = approx_exp2(tw_24.y);
                            float2 w_24 = make_float2(_exp2_455, _exp2_456);
                            float2 _f32x2_mul_b_457 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_24) : "l"(*(const unsigned long long*)&ax_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_457));
                            float _exp2_458 = approx_exp2_noftz(tp_24.x);
                            float _exp2_459 = approx_exp2_noftz(tp_24.y);
                            float2 p_24 = make_float2(_exp2_458, _exp2_459);
                            float2 d1_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_24);
                            float2 d2_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_24);
                            float2 dw_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_24);
                            float2 ng_24 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_24);
                            float2 nu_24 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_24);
                            float _rcp_460 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_24, d2_24), dw_24).x);
                            float _rcp_461 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_24, d2_24), dw_24).y);
                            float2 r_50 = make_float2(_rcp_460, _rcp_461);
                            float2 r_49 = r_50;
                            float2 _f32x2_mul_a_462 = mul_f32x2_noftz(ng_24, nu_24);
                            float2 _f32x2_mul_b_463 = mul_f32x2_noftz(r_49, make_float2(bl, bl));
                            float2 m_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_462), "l"(*(const unsigned long long*)&_f32x2_mul_b_463));
                            float2 ns_24 = make_float2(((x_25.x < 0.0f) ? p_24.x : 1.0f), ((x_25.y < 0.0f) ? p_24.y : 1.0f));
                            gate_packed_0[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_24, ns_24))) | sign_25;
                            float2 x_26 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[26]));
                            float2 v_26 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[26]));
                            unsigned int sign_26 = (gate_packed_0[26] ^ up_packed_0[26]) & 2147516416u;
                            float _abs_464;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_464) : "f"(x_26.x));
                            float _abs_465;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_465) : "f"(x_26.y));
                            float2 ax_25 = make_float2(_abs_464, _abs_465);
                            float _abs_466;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_466) : "f"(v_26.x));
                            float _abs_467;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_467) : "f"(v_26.y));
                            float2 av_25 = make_float2(_abs_466, _abs_467);
                            float2 _f32x2_mul_b_468 = make_float2(k1, k1);
                            float2 te_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_25) : "l"(*(const unsigned long long*)&ax_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_468));
                            float2 _f32x2_mul_b_469 = make_float2(k2, k2);
                            float2 tw_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_25) : "l"(*(const unsigned long long*)&av_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_469));
                            float _exp2_470 = approx_exp2(te_25.x);
                            float _exp2_471 = approx_exp2(te_25.y);
                            float2 e_25 = make_float2(_exp2_470, _exp2_471);
                            float _exp2_472 = approx_exp2(tw_25.x);
                            float _exp2_473 = approx_exp2(tw_25.y);
                            float2 w_25 = make_float2(_exp2_472, _exp2_473);
                            float2 _f32x2_mul_b_474 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_25) : "l"(*(const unsigned long long*)&ax_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_474));
                            float _exp2_475 = approx_exp2_noftz(tp_25.x);
                            float _exp2_476 = approx_exp2_noftz(tp_25.y);
                            float2 p_25 = make_float2(_exp2_475, _exp2_476);
                            float2 d1_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_25);
                            float2 d2_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_25);
                            float2 dw_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_25);
                            float2 ng_25 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_25);
                            float2 nu_25 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_25);
                            float _rcp_477 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_25, d2_25), dw_25).x);
                            float _rcp_478 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_25, d2_25), dw_25).y);
                            float2 r_52 = make_float2(_rcp_477, _rcp_478);
                            float2 r_51 = r_52;
                            float2 _f32x2_mul_a_479 = mul_f32x2_noftz(ng_25, nu_25);
                            float2 _f32x2_mul_b_480 = mul_f32x2_noftz(r_51, make_float2(bl, bl));
                            float2 m_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_479), "l"(*(const unsigned long long*)&_f32x2_mul_b_480));
                            float2 ns_25 = make_float2(((x_26.x < 0.0f) ? p_25.x : 1.0f), ((x_26.y < 0.0f) ? p_25.y : 1.0f));
                            gate_packed_0[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_25, ns_25))) | sign_26;
                            float2 x_27 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[27]));
                            float2 v_27 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[27]));
                            unsigned int sign_27 = (gate_packed_0[27] ^ up_packed_0[27]) & 2147516416u;
                            float _abs_481;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_481) : "f"(x_27.x));
                            float _abs_482;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_482) : "f"(x_27.y));
                            float2 ax_26 = make_float2(_abs_481, _abs_482);
                            float _abs_483;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_483) : "f"(v_27.x));
                            float _abs_484;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_484) : "f"(v_27.y));
                            float2 av_26 = make_float2(_abs_483, _abs_484);
                            float2 _f32x2_mul_b_485 = make_float2(k1, k1);
                            float2 te_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_26) : "l"(*(const unsigned long long*)&ax_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_485));
                            float2 _f32x2_mul_b_486 = make_float2(k2, k2);
                            float2 tw_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_26) : "l"(*(const unsigned long long*)&av_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_486));
                            float _exp2_487 = approx_exp2(te_26.x);
                            float _exp2_488 = approx_exp2(te_26.y);
                            float2 e_26 = make_float2(_exp2_487, _exp2_488);
                            float _exp2_489 = approx_exp2(tw_26.x);
                            float _exp2_490 = approx_exp2(tw_26.y);
                            float2 w_26 = make_float2(_exp2_489, _exp2_490);
                            float2 _f32x2_mul_b_491 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_26) : "l"(*(const unsigned long long*)&ax_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_491));
                            float _exp2_492 = approx_exp2_noftz(tp_26.x);
                            float _exp2_493 = approx_exp2_noftz(tp_26.y);
                            float2 p_26 = make_float2(_exp2_492, _exp2_493);
                            float2 d1_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_26);
                            float2 d2_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_26);
                            float2 dw_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_26);
                            float2 ng_26 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_26);
                            float2 nu_26 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_26);
                            float _rcp_494 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_26, d2_26), dw_26).x);
                            float _rcp_495 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_26, d2_26), dw_26).y);
                            float2 r_54 = make_float2(_rcp_494, _rcp_495);
                            float2 r_53 = r_54;
                            float2 _f32x2_mul_a_496 = mul_f32x2_noftz(ng_26, nu_26);
                            float2 _f32x2_mul_b_497 = mul_f32x2_noftz(r_53, make_float2(bl, bl));
                            float2 m_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_496), "l"(*(const unsigned long long*)&_f32x2_mul_b_497));
                            float2 ns_26 = make_float2(((x_27.x < 0.0f) ? p_26.x : 1.0f), ((x_27.y < 0.0f) ? p_26.y : 1.0f));
                            gate_packed_0[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_26, ns_26))) | sign_27;
                            float2 x_28 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[28]));
                            float2 v_28 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[28]));
                            unsigned int sign_28 = (gate_packed_0[28] ^ up_packed_0[28]) & 2147516416u;
                            float _abs_498;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_498) : "f"(x_28.x));
                            float _abs_499;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_499) : "f"(x_28.y));
                            float2 ax_27 = make_float2(_abs_498, _abs_499);
                            float _abs_500;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_500) : "f"(v_28.x));
                            float _abs_501;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_501) : "f"(v_28.y));
                            float2 av_27 = make_float2(_abs_500, _abs_501);
                            float2 _f32x2_mul_b_502 = make_float2(k1, k1);
                            float2 te_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_27) : "l"(*(const unsigned long long*)&ax_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_502));
                            float2 _f32x2_mul_b_503 = make_float2(k2, k2);
                            float2 tw_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_27) : "l"(*(const unsigned long long*)&av_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_503));
                            float _exp2_504 = approx_exp2(te_27.x);
                            float _exp2_505 = approx_exp2(te_27.y);
                            float2 e_27 = make_float2(_exp2_504, _exp2_505);
                            float _exp2_506 = approx_exp2(tw_27.x);
                            float _exp2_507 = approx_exp2(tw_27.y);
                            float2 w_27 = make_float2(_exp2_506, _exp2_507);
                            float2 _f32x2_mul_b_508 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_27) : "l"(*(const unsigned long long*)&ax_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_508));
                            float _exp2_509 = approx_exp2_noftz(tp_27.x);
                            float _exp2_510 = approx_exp2_noftz(tp_27.y);
                            float2 p_27 = make_float2(_exp2_509, _exp2_510);
                            float2 d1_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_27);
                            float2 d2_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_27);
                            float2 dw_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_27);
                            float2 ng_27 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_27);
                            float2 nu_27 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_27);
                            float _rcp_511 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_27, d2_27), dw_27).x);
                            float _rcp_512 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_27, d2_27), dw_27).y);
                            float2 r_56 = make_float2(_rcp_511, _rcp_512);
                            float2 r_55 = r_56;
                            float2 _f32x2_mul_a_513 = mul_f32x2_noftz(ng_27, nu_27);
                            float2 _f32x2_mul_b_514 = mul_f32x2_noftz(r_55, make_float2(bl, bl));
                            float2 m_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_513), "l"(*(const unsigned long long*)&_f32x2_mul_b_514));
                            float2 ns_27 = make_float2(((x_28.x < 0.0f) ? p_27.x : 1.0f), ((x_28.y < 0.0f) ? p_27.y : 1.0f));
                            gate_packed_0[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_27, ns_27))) | sign_28;
                            float2 x_29 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[29]));
                            float2 v_29 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[29]));
                            unsigned int sign_29 = (gate_packed_0[29] ^ up_packed_0[29]) & 2147516416u;
                            float _abs_515;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_515) : "f"(x_29.x));
                            float _abs_516;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_516) : "f"(x_29.y));
                            float2 ax_28 = make_float2(_abs_515, _abs_516);
                            float _abs_517;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_517) : "f"(v_29.x));
                            float _abs_518;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_518) : "f"(v_29.y));
                            float2 av_28 = make_float2(_abs_517, _abs_518);
                            float2 _f32x2_mul_b_519 = make_float2(k1, k1);
                            float2 te_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_28) : "l"(*(const unsigned long long*)&ax_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_519));
                            float2 _f32x2_mul_b_520 = make_float2(k2, k2);
                            float2 tw_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_28) : "l"(*(const unsigned long long*)&av_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_520));
                            float _exp2_521 = approx_exp2(te_28.x);
                            float _exp2_522 = approx_exp2(te_28.y);
                            float2 e_28 = make_float2(_exp2_521, _exp2_522);
                            float _exp2_523 = approx_exp2(tw_28.x);
                            float _exp2_524 = approx_exp2(tw_28.y);
                            float2 w_28 = make_float2(_exp2_523, _exp2_524);
                            float2 _f32x2_mul_b_525 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_28) : "l"(*(const unsigned long long*)&ax_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_525));
                            float _exp2_526 = approx_exp2_noftz(tp_28.x);
                            float _exp2_527 = approx_exp2_noftz(tp_28.y);
                            float2 p_28 = make_float2(_exp2_526, _exp2_527);
                            float2 d1_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_28);
                            float2 d2_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_28);
                            float2 dw_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_28);
                            float2 ng_28 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_28);
                            float2 nu_28 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_28);
                            float _rcp_528 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_28, d2_28), dw_28).x);
                            float _rcp_529 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_28, d2_28), dw_28).y);
                            float2 r_58 = make_float2(_rcp_528, _rcp_529);
                            float2 r_57 = r_58;
                            float2 _f32x2_mul_a_530 = mul_f32x2_noftz(ng_28, nu_28);
                            float2 _f32x2_mul_b_531 = mul_f32x2_noftz(r_57, make_float2(bl, bl));
                            float2 m_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_530), "l"(*(const unsigned long long*)&_f32x2_mul_b_531));
                            float2 ns_28 = make_float2(((x_29.x < 0.0f) ? p_28.x : 1.0f), ((x_29.y < 0.0f) ? p_28.y : 1.0f));
                            gate_packed_0[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_28, ns_28))) | sign_29;
                            float2 x_30 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[30]));
                            float2 v_30 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[30]));
                            unsigned int sign_30 = (gate_packed_0[30] ^ up_packed_0[30]) & 2147516416u;
                            float _abs_532;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_532) : "f"(x_30.x));
                            float _abs_533;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_533) : "f"(x_30.y));
                            float2 ax_29 = make_float2(_abs_532, _abs_533);
                            float _abs_534;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_534) : "f"(v_30.x));
                            float _abs_535;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_535) : "f"(v_30.y));
                            float2 av_29 = make_float2(_abs_534, _abs_535);
                            float2 _f32x2_mul_b_536 = make_float2(k1, k1);
                            float2 te_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_29) : "l"(*(const unsigned long long*)&ax_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_536));
                            float2 _f32x2_mul_b_537 = make_float2(k2, k2);
                            float2 tw_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_29) : "l"(*(const unsigned long long*)&av_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_537));
                            float _exp2_538 = approx_exp2(te_29.x);
                            float _exp2_539 = approx_exp2(te_29.y);
                            float2 e_29 = make_float2(_exp2_538, _exp2_539);
                            float _exp2_540 = approx_exp2(tw_29.x);
                            float _exp2_541 = approx_exp2(tw_29.y);
                            float2 w_29 = make_float2(_exp2_540, _exp2_541);
                            float2 _f32x2_mul_b_542 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_29) : "l"(*(const unsigned long long*)&ax_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_542));
                            float _exp2_543 = approx_exp2_noftz(tp_29.x);
                            float _exp2_544 = approx_exp2_noftz(tp_29.y);
                            float2 p_29 = make_float2(_exp2_543, _exp2_544);
                            float2 d1_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_29);
                            float2 d2_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_29);
                            float2 dw_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_29);
                            float2 ng_29 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_29);
                            float2 nu_29 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_29);
                            float _rcp_545 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_29, d2_29), dw_29).x);
                            float _rcp_546 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_29, d2_29), dw_29).y);
                            float2 r_60 = make_float2(_rcp_545, _rcp_546);
                            float2 r_59 = r_60;
                            float2 _f32x2_mul_a_547 = mul_f32x2_noftz(ng_29, nu_29);
                            float2 _f32x2_mul_b_548 = mul_f32x2_noftz(r_59, make_float2(bl, bl));
                            float2 m_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_547), "l"(*(const unsigned long long*)&_f32x2_mul_b_548));
                            float2 ns_29 = make_float2(((x_30.x < 0.0f) ? p_29.x : 1.0f), ((x_30.y < 0.0f) ? p_29.y : 1.0f));
                            gate_packed_0[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_29, ns_29))) | sign_30;
                            float2 x_31 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[31]));
                            float2 v_31 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[31]));
                            unsigned int sign_31 = (gate_packed_0[31] ^ up_packed_0[31]) & 2147516416u;
                            float _abs_549;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_549) : "f"(x_31.x));
                            float _abs_550;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_550) : "f"(x_31.y));
                            float2 ax_30 = make_float2(_abs_549, _abs_550);
                            float _abs_551;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_551) : "f"(v_31.x));
                            float _abs_552;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_552) : "f"(v_31.y));
                            float2 av_30 = make_float2(_abs_551, _abs_552);
                            float2 _f32x2_mul_b_553 = make_float2(k1, k1);
                            float2 te_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_30) : "l"(*(const unsigned long long*)&ax_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_553));
                            float2 _f32x2_mul_b_554 = make_float2(k2, k2);
                            float2 tw_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_30) : "l"(*(const unsigned long long*)&av_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_554));
                            float _exp2_555 = approx_exp2(te_30.x);
                            float _exp2_556 = approx_exp2(te_30.y);
                            float2 e_30 = make_float2(_exp2_555, _exp2_556);
                            float _exp2_557 = approx_exp2(tw_30.x);
                            float _exp2_558 = approx_exp2(tw_30.y);
                            float2 w_30 = make_float2(_exp2_557, _exp2_558);
                            float2 _f32x2_mul_b_559 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_30) : "l"(*(const unsigned long long*)&ax_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_559));
                            float _exp2_560 = approx_exp2_noftz(tp_30.x);
                            float _exp2_561 = approx_exp2_noftz(tp_30.y);
                            float2 p_30 = make_float2(_exp2_560, _exp2_561);
                            float2 d1_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_30);
                            float2 d2_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_30);
                            float2 dw_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_30);
                            float2 ng_30 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_30);
                            float2 nu_30 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_30);
                            float _rcp_562 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_30, d2_30), dw_30).x);
                            float _rcp_563 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_30, d2_30), dw_30).y);
                            float2 r_62 = make_float2(_rcp_562, _rcp_563);
                            float2 r_61 = r_62;
                            float2 _f32x2_mul_a_564 = mul_f32x2_noftz(ng_30, nu_30);
                            float2 _f32x2_mul_b_565 = mul_f32x2_noftz(r_61, make_float2(bl, bl));
                            float2 m_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_564), "l"(*(const unsigned long long*)&_f32x2_mul_b_565));
                            float2 ns_30 = make_float2(((x_31.x < 0.0f) ? p_30.x : 1.0f), ((x_31.y < 0.0f) ? p_30.y : 1.0f));
                            gate_packed_0[31] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_30, ns_30))) | sign_31;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_4 = tid / 32;
                            int lane_4 = tid % 32;
                            #pragma unroll
                            for (int half_5 = 0; half_5 < 2; half_5++) {
                                #pragma unroll
                                for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                    int row_6 = warp_4 * 32 + half_5 * 16 + lane_4 % 16;
                                    int col_6 = col_tile_5 * 16 + lane_4 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + 16384 + (unsigned int)((row_6 * 32 + col_6) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_6 = 16 + half_5 * 8 + col_tile_5 * 4;
                                    uint32_t _stmatrix_addr_566 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_566), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int gate_packed_1[32];
                            unsigned int up_packed_1[32];
                            unsigned int address_3_4 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 64;
                            float _tmem_load_4[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                                : "r"(address_3_4));
                            float _tmem_load_32_4[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[15]))
                                : "r"(address_3_4 + 128));
                            gate_packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[0], _tmem_load_4[1])));
                            up_packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[0], _tmem_load_32_4[1])));
                            gate_packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[2], _tmem_load_4[3])));
                            up_packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[2], _tmem_load_32_4[3])));
                            gate_packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[4], _tmem_load_4[5])));
                            up_packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[4], _tmem_load_32_4[5])));
                            gate_packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[6], _tmem_load_4[7])));
                            up_packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[6], _tmem_load_32_4[7])));
                            gate_packed_1[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[8], _tmem_load_4[9])));
                            up_packed_1[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[8], _tmem_load_32_4[9])));
                            gate_packed_1[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[10], _tmem_load_4[11])));
                            up_packed_1[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[10], _tmem_load_32_4[11])));
                            gate_packed_1[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[12], _tmem_load_4[13])));
                            up_packed_1[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[12], _tmem_load_32_4[13])));
                            gate_packed_1[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[14], _tmem_load_4[15])));
                            up_packed_1[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[14], _tmem_load_32_4[15])));
                            unsigned int address_3_5 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 64;
                            float _tmem_load_5[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                                : "r"(address_3_5));
                            float _tmem_load_32_5[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[15]))
                                : "r"(address_3_5 + 128));
                            gate_packed_1[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[0], _tmem_load_5[1])));
                            up_packed_1[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[0], _tmem_load_32_5[1])));
                            gate_packed_1[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[2], _tmem_load_5[3])));
                            up_packed_1[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[2], _tmem_load_32_5[3])));
                            gate_packed_1[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[4], _tmem_load_5[5])));
                            up_packed_1[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[4], _tmem_load_32_5[5])));
                            gate_packed_1[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[6], _tmem_load_5[7])));
                            up_packed_1[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[6], _tmem_load_32_5[7])));
                            gate_packed_1[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[8], _tmem_load_5[9])));
                            up_packed_1[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[8], _tmem_load_32_5[9])));
                            gate_packed_1[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[10], _tmem_load_5[11])));
                            up_packed_1[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[10], _tmem_load_32_5[11])));
                            gate_packed_1[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[12], _tmem_load_5[13])));
                            up_packed_1[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[12], _tmem_load_32_5[13])));
                            gate_packed_1[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[14], _tmem_load_5[15])));
                            up_packed_1[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[14], _tmem_load_32_5[15])));
                            unsigned int address_3_6 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 96;
                            float _tmem_load_6[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                                : "r"(address_3_6));
                            float _tmem_load_32_6[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[15]))
                                : "r"(address_3_6 + 128));
                            gate_packed_1[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[0], _tmem_load_6[1])));
                            up_packed_1[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[0], _tmem_load_32_6[1])));
                            gate_packed_1[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[2], _tmem_load_6[3])));
                            up_packed_1[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[2], _tmem_load_32_6[3])));
                            gate_packed_1[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[4], _tmem_load_6[5])));
                            up_packed_1[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[4], _tmem_load_32_6[5])));
                            gate_packed_1[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[6], _tmem_load_6[7])));
                            up_packed_1[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[6], _tmem_load_32_6[7])));
                            gate_packed_1[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[8], _tmem_load_6[9])));
                            up_packed_1[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[8], _tmem_load_32_6[9])));
                            gate_packed_1[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[10], _tmem_load_6[11])));
                            up_packed_1[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[10], _tmem_load_32_6[11])));
                            gate_packed_1[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[12], _tmem_load_6[13])));
                            up_packed_1[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[12], _tmem_load_32_6[13])));
                            gate_packed_1[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[14], _tmem_load_6[15])));
                            up_packed_1[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[14], _tmem_load_32_6[15])));
                            unsigned int address_3_7 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 96;
                            float _tmem_load_7[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                                : "r"(address_3_7));
                            float _tmem_load_32_7[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[15]))
                                : "r"(address_3_7 + 128));
                            gate_packed_1[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[0], _tmem_load_7[1])));
                            up_packed_1[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[0], _tmem_load_32_7[1])));
                            gate_packed_1[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[2], _tmem_load_7[3])));
                            up_packed_1[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[2], _tmem_load_32_7[3])));
                            gate_packed_1[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[4], _tmem_load_7[5])));
                            up_packed_1[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[4], _tmem_load_32_7[5])));
                            gate_packed_1[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[6], _tmem_load_7[7])));
                            up_packed_1[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[6], _tmem_load_32_7[7])));
                            gate_packed_1[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[8], _tmem_load_7[9])));
                            up_packed_1[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[8], _tmem_load_32_7[9])));
                            gate_packed_1[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[10], _tmem_load_7[11])));
                            up_packed_1[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[10], _tmem_load_32_7[11])));
                            gate_packed_1[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[12], _tmem_load_7[13])));
                            up_packed_1[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[12], _tmem_load_32_7[13])));
                            gate_packed_1[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[14], _tmem_load_7[15])));
                            up_packed_1[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[14], _tmem_load_32_7[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_5 = tid / 32;
                            int lane_5 = tid % 32;
                            #pragma unroll
                            for (int half_6 = 0; half_6 < 2; half_6++) {
                                #pragma unroll
                                for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                    int row_7 = warp_5 * 32 + half_6 * 16 + lane_5 % 16;
                                    int col_7 = col_tile_6 * 16 + lane_5 / 16 * 8;
                                    unsigned int address_6 = d_smem_addr + (unsigned int)((row_7 * 32 + col_7) * 2);
                                    address_6 = address_6 ^ (address_6 & 511) >> 7 << 4;
                                    int offset_7 = half_6 * 8 + col_tile_6 * 4;
                                    uint32_t _stmatrix_addr_567 = static_cast<uint32_t>(address_6);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_567), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_6 = tid / 32;
                            int lane_6 = tid % 32;
                            #pragma unroll
                            for (int half_7 = 0; half_7 < 2; half_7++) {
                                #pragma unroll
                                for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                    int row_8 = warp_6 * 32 + half_7 * 16 + lane_6 % 16;
                                    int col_8 = col_tile_7 * 16 + lane_6 / 16 * 8;
                                    unsigned int address_7 = d_smem_addr + 8192 + (unsigned int)((row_8 * 32 + col_8) * 2);
                                    address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                    int offset_8 = half_7 * 8 + col_tile_7 * 4;
                                    uint32_t _stmatrix_addr_568 = static_cast<uint32_t>(address_7);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_568), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_7 = tid / 32;
                            int lane_7 = tid % 32;
                            #pragma unroll
                            for (int half_8 = 0; half_8 < 2; half_8++) {
                                #pragma unroll
                                for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                    int row_9 = warp_7 * 32 + half_8 * 16 + lane_7 % 16;
                                    int col_9 = col_tile_8 * 16 + lane_7 / 16 * 8;
                                    unsigned int address_8 = d_smem_addr + 16384 + (unsigned int)((row_9 * 32 + col_9) * 2);
                                    address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                    int offset_9 = 16 + half_8 * 8 + col_tile_8 * 4;
                                    uint32_t _stmatrix_addr_569 = static_cast<uint32_t>(address_8);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_569), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_8 = tid / 32;
                            int lane_8 = tid % 32;
                            #pragma unroll
                            for (int half_9 = 0; half_9 < 2; half_9++) {
                                #pragma unroll
                                for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                    int row_10 = warp_8 * 32 + half_9 * 16 + lane_8 % 16;
                                    int col_10 = col_tile_9 * 16 + lane_8 / 16 * 8;
                                    unsigned int address_9 = d_smem_addr + (unsigned int)((row_10 * 32 + col_10) * 2);
                                    address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                    int offset_10 = 16 + half_9 * 8 + col_tile_9 * 4;
                                    uint32_t _stmatrix_addr_570 = static_cast<uint32_t>(address_9);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_570), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_32 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[0]));
                            float2 v_32 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[0]));
                            unsigned int sign_32 = (gate_packed_1[0] ^ up_packed_1[0]) & 2147516416u;
                            float _abs_571;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_571) : "f"(x_32.x));
                            float _abs_572;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_572) : "f"(x_32.y));
                            float2 ax_31 = make_float2(_abs_571, _abs_572);
                            float _abs_573;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_573) : "f"(v_32.x));
                            float _abs_574;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_574) : "f"(v_32.y));
                            float2 av_31 = make_float2(_abs_573, _abs_574);
                            float2 _f32x2_mul_b_575 = make_float2(k1, k1);
                            float2 te_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_31) : "l"(*(const unsigned long long*)&ax_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_575));
                            float2 _f32x2_mul_b_576 = make_float2(k2, k2);
                            float2 tw_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_31) : "l"(*(const unsigned long long*)&av_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_576));
                            float _exp2_577 = approx_exp2(te_31.x);
                            float _exp2_578 = approx_exp2(te_31.y);
                            float2 e_31 = make_float2(_exp2_577, _exp2_578);
                            float _exp2_579 = approx_exp2(tw_31.x);
                            float _exp2_580 = approx_exp2(tw_31.y);
                            float2 w_31 = make_float2(_exp2_579, _exp2_580);
                            float2 _f32x2_mul_b_581 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_31) : "l"(*(const unsigned long long*)&ax_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_581));
                            float _exp2_582 = approx_exp2_noftz(tp_31.x);
                            float _exp2_583 = approx_exp2_noftz(tp_31.y);
                            float2 p_31 = make_float2(_exp2_582, _exp2_583);
                            float2 d1_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_31);
                            float2 d2_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_31);
                            float2 dw_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_31);
                            float2 ng_31 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_31);
                            float2 nu_31 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_31);
                            float _rcp_584 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_31, d2_31), dw_31).x);
                            float _rcp_585 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_31, d2_31), dw_31).y);
                            float2 r_64 = make_float2(_rcp_584, _rcp_585);
                            float2 r_63 = r_64;
                            float2 _f32x2_mul_a_586 = mul_f32x2_noftz(ng_31, nu_31);
                            float2 _f32x2_mul_b_587 = mul_f32x2_noftz(r_63, make_float2(bl, bl));
                            float2 m_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_586), "l"(*(const unsigned long long*)&_f32x2_mul_b_587));
                            float2 ns_31 = make_float2(((x_32.x < 0.0f) ? p_31.x : 1.0f), ((x_32.y < 0.0f) ? p_31.y : 1.0f));
                            gate_packed_1[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_31, ns_31))) | sign_32;
                            float2 x_33 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[1]));
                            float2 v_33 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[1]));
                            unsigned int sign_33 = (gate_packed_1[1] ^ up_packed_1[1]) & 2147516416u;
                            float _abs_588;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_588) : "f"(x_33.x));
                            float _abs_589;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_589) : "f"(x_33.y));
                            float2 ax_32 = make_float2(_abs_588, _abs_589);
                            float _abs_590;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_590) : "f"(v_33.x));
                            float _abs_591;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_591) : "f"(v_33.y));
                            float2 av_32 = make_float2(_abs_590, _abs_591);
                            float2 _f32x2_mul_b_592 = make_float2(k1, k1);
                            float2 te_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_32) : "l"(*(const unsigned long long*)&ax_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_592));
                            float2 _f32x2_mul_b_593 = make_float2(k2, k2);
                            float2 tw_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_32) : "l"(*(const unsigned long long*)&av_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_593));
                            float _exp2_594 = approx_exp2(te_32.x);
                            float _exp2_595 = approx_exp2(te_32.y);
                            float2 e_32 = make_float2(_exp2_594, _exp2_595);
                            float _exp2_596 = approx_exp2(tw_32.x);
                            float _exp2_597 = approx_exp2(tw_32.y);
                            float2 w_32 = make_float2(_exp2_596, _exp2_597);
                            float2 _f32x2_mul_b_598 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_32) : "l"(*(const unsigned long long*)&ax_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_598));
                            float _exp2_599 = approx_exp2_noftz(tp_32.x);
                            float _exp2_600 = approx_exp2_noftz(tp_32.y);
                            float2 p_32 = make_float2(_exp2_599, _exp2_600);
                            float2 d1_32 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_32);
                            float2 d2_32 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_32);
                            float2 dw_32 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_32);
                            float2 ng_32 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_32);
                            float2 nu_32 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_32);
                            float _rcp_601 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_32, d2_32), dw_32).x);
                            float _rcp_602 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_32, d2_32), dw_32).y);
                            float2 r_66 = make_float2(_rcp_601, _rcp_602);
                            float2 r_65 = r_66;
                            float2 _f32x2_mul_a_603 = mul_f32x2_noftz(ng_32, nu_32);
                            float2 _f32x2_mul_b_604 = mul_f32x2_noftz(r_65, make_float2(bl, bl));
                            float2 m_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_603), "l"(*(const unsigned long long*)&_f32x2_mul_b_604));
                            float2 ns_32 = make_float2(((x_33.x < 0.0f) ? p_32.x : 1.0f), ((x_33.y < 0.0f) ? p_32.y : 1.0f));
                            gate_packed_1[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_32, ns_32))) | sign_33;
                            float2 x_34 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[2]));
                            float2 v_34 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[2]));
                            unsigned int sign_34 = (gate_packed_1[2] ^ up_packed_1[2]) & 2147516416u;
                            float _abs_605;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_605) : "f"(x_34.x));
                            float _abs_606;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_606) : "f"(x_34.y));
                            float2 ax_33 = make_float2(_abs_605, _abs_606);
                            float _abs_607;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_607) : "f"(v_34.x));
                            float _abs_608;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_608) : "f"(v_34.y));
                            float2 av_33 = make_float2(_abs_607, _abs_608);
                            float2 _f32x2_mul_b_609 = make_float2(k1, k1);
                            float2 te_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_33) : "l"(*(const unsigned long long*)&ax_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_609));
                            float2 _f32x2_mul_b_610 = make_float2(k2, k2);
                            float2 tw_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_33) : "l"(*(const unsigned long long*)&av_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_610));
                            float _exp2_611 = approx_exp2(te_33.x);
                            float _exp2_612 = approx_exp2(te_33.y);
                            float2 e_33 = make_float2(_exp2_611, _exp2_612);
                            float _exp2_613 = approx_exp2(tw_33.x);
                            float _exp2_614 = approx_exp2(tw_33.y);
                            float2 w_33 = make_float2(_exp2_613, _exp2_614);
                            float2 _f32x2_mul_b_615 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_33) : "l"(*(const unsigned long long*)&ax_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_615));
                            float _exp2_616 = approx_exp2_noftz(tp_33.x);
                            float _exp2_617 = approx_exp2_noftz(tp_33.y);
                            float2 p_33 = make_float2(_exp2_616, _exp2_617);
                            float2 d1_33 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_33);
                            float2 d2_33 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_33);
                            float2 dw_33 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_33);
                            float2 ng_33 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_33);
                            float2 nu_33 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_33);
                            float _rcp_618 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_33, d2_33), dw_33).x);
                            float _rcp_619 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_33, d2_33), dw_33).y);
                            float2 r_68 = make_float2(_rcp_618, _rcp_619);
                            float2 r_67 = r_68;
                            float2 _f32x2_mul_a_620 = mul_f32x2_noftz(ng_33, nu_33);
                            float2 _f32x2_mul_b_621 = mul_f32x2_noftz(r_67, make_float2(bl, bl));
                            float2 m_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_620), "l"(*(const unsigned long long*)&_f32x2_mul_b_621));
                            float2 ns_33 = make_float2(((x_34.x < 0.0f) ? p_33.x : 1.0f), ((x_34.y < 0.0f) ? p_33.y : 1.0f));
                            gate_packed_1[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_33, ns_33))) | sign_34;
                            float2 x_35 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[3]));
                            float2 v_35 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[3]));
                            unsigned int sign_35 = (gate_packed_1[3] ^ up_packed_1[3]) & 2147516416u;
                            float _abs_622;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_622) : "f"(x_35.x));
                            float _abs_623;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_623) : "f"(x_35.y));
                            float2 ax_34 = make_float2(_abs_622, _abs_623);
                            float _abs_624;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_624) : "f"(v_35.x));
                            float _abs_625;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_625) : "f"(v_35.y));
                            float2 av_34 = make_float2(_abs_624, _abs_625);
                            float2 _f32x2_mul_b_626 = make_float2(k1, k1);
                            float2 te_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_34) : "l"(*(const unsigned long long*)&ax_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_626));
                            float2 _f32x2_mul_b_627 = make_float2(k2, k2);
                            float2 tw_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_34) : "l"(*(const unsigned long long*)&av_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_627));
                            float _exp2_628 = approx_exp2(te_34.x);
                            float _exp2_629 = approx_exp2(te_34.y);
                            float2 e_34 = make_float2(_exp2_628, _exp2_629);
                            float _exp2_630 = approx_exp2(tw_34.x);
                            float _exp2_631 = approx_exp2(tw_34.y);
                            float2 w_34 = make_float2(_exp2_630, _exp2_631);
                            float2 _f32x2_mul_b_632 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_34) : "l"(*(const unsigned long long*)&ax_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_632));
                            float _exp2_633 = approx_exp2_noftz(tp_34.x);
                            float _exp2_634 = approx_exp2_noftz(tp_34.y);
                            float2 p_34 = make_float2(_exp2_633, _exp2_634);
                            float2 d1_34 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_34);
                            float2 d2_34 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_34);
                            float2 dw_34 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_34);
                            float2 ng_34 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_34);
                            float2 nu_34 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_34);
                            float _rcp_635 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_34, d2_34), dw_34).x);
                            float _rcp_636 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_34, d2_34), dw_34).y);
                            float2 r_70 = make_float2(_rcp_635, _rcp_636);
                            float2 r_69 = r_70;
                            float2 _f32x2_mul_a_637 = mul_f32x2_noftz(ng_34, nu_34);
                            float2 _f32x2_mul_b_638 = mul_f32x2_noftz(r_69, make_float2(bl, bl));
                            float2 m_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_637), "l"(*(const unsigned long long*)&_f32x2_mul_b_638));
                            float2 ns_34 = make_float2(((x_35.x < 0.0f) ? p_34.x : 1.0f), ((x_35.y < 0.0f) ? p_34.y : 1.0f));
                            gate_packed_1[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_34, ns_34))) | sign_35;
                            float2 x_36 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[4]));
                            float2 v_36 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[4]));
                            unsigned int sign_36 = (gate_packed_1[4] ^ up_packed_1[4]) & 2147516416u;
                            float _abs_639;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_639) : "f"(x_36.x));
                            float _abs_640;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_640) : "f"(x_36.y));
                            float2 ax_35 = make_float2(_abs_639, _abs_640);
                            float _abs_641;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_641) : "f"(v_36.x));
                            float _abs_642;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_642) : "f"(v_36.y));
                            float2 av_35 = make_float2(_abs_641, _abs_642);
                            float2 _f32x2_mul_b_643 = make_float2(k1, k1);
                            float2 te_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_35) : "l"(*(const unsigned long long*)&ax_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_643));
                            float2 _f32x2_mul_b_644 = make_float2(k2, k2);
                            float2 tw_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_35) : "l"(*(const unsigned long long*)&av_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_644));
                            float _exp2_645 = approx_exp2(te_35.x);
                            float _exp2_646 = approx_exp2(te_35.y);
                            float2 e_35 = make_float2(_exp2_645, _exp2_646);
                            float _exp2_647 = approx_exp2(tw_35.x);
                            float _exp2_648 = approx_exp2(tw_35.y);
                            float2 w_35 = make_float2(_exp2_647, _exp2_648);
                            float2 _f32x2_mul_b_649 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_35) : "l"(*(const unsigned long long*)&ax_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_649));
                            float _exp2_650 = approx_exp2_noftz(tp_35.x);
                            float _exp2_651 = approx_exp2_noftz(tp_35.y);
                            float2 p_35 = make_float2(_exp2_650, _exp2_651);
                            float2 d1_35 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_35);
                            float2 d2_35 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_35);
                            float2 dw_35 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_35);
                            float2 ng_35 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_35);
                            float2 nu_35 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_35);
                            float _rcp_652 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_35, d2_35), dw_35).x);
                            float _rcp_653 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_35, d2_35), dw_35).y);
                            float2 r_72 = make_float2(_rcp_652, _rcp_653);
                            float2 r_71 = r_72;
                            float2 _f32x2_mul_a_654 = mul_f32x2_noftz(ng_35, nu_35);
                            float2 _f32x2_mul_b_655 = mul_f32x2_noftz(r_71, make_float2(bl, bl));
                            float2 m_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_654), "l"(*(const unsigned long long*)&_f32x2_mul_b_655));
                            float2 ns_35 = make_float2(((x_36.x < 0.0f) ? p_35.x : 1.0f), ((x_36.y < 0.0f) ? p_35.y : 1.0f));
                            gate_packed_1[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_35, ns_35))) | sign_36;
                            float2 x_37 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[5]));
                            float2 v_37 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[5]));
                            unsigned int sign_37 = (gate_packed_1[5] ^ up_packed_1[5]) & 2147516416u;
                            float _abs_656;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_656) : "f"(x_37.x));
                            float _abs_657;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_657) : "f"(x_37.y));
                            float2 ax_36 = make_float2(_abs_656, _abs_657);
                            float _abs_658;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_658) : "f"(v_37.x));
                            float _abs_659;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_659) : "f"(v_37.y));
                            float2 av_36 = make_float2(_abs_658, _abs_659);
                            float2 _f32x2_mul_b_660 = make_float2(k1, k1);
                            float2 te_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_36) : "l"(*(const unsigned long long*)&ax_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_660));
                            float2 _f32x2_mul_b_661 = make_float2(k2, k2);
                            float2 tw_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_36) : "l"(*(const unsigned long long*)&av_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_661));
                            float _exp2_662 = approx_exp2(te_36.x);
                            float _exp2_663 = approx_exp2(te_36.y);
                            float2 e_36 = make_float2(_exp2_662, _exp2_663);
                            float _exp2_664 = approx_exp2(tw_36.x);
                            float _exp2_665 = approx_exp2(tw_36.y);
                            float2 w_36 = make_float2(_exp2_664, _exp2_665);
                            float2 _f32x2_mul_b_666 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_36) : "l"(*(const unsigned long long*)&ax_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_666));
                            float _exp2_667 = approx_exp2_noftz(tp_36.x);
                            float _exp2_668 = approx_exp2_noftz(tp_36.y);
                            float2 p_36 = make_float2(_exp2_667, _exp2_668);
                            float2 d1_36 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_36);
                            float2 d2_36 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_36);
                            float2 dw_36 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_36);
                            float2 ng_36 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_36);
                            float2 nu_36 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_36);
                            float _rcp_669 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_36, d2_36), dw_36).x);
                            float _rcp_670 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_36, d2_36), dw_36).y);
                            float2 r_74 = make_float2(_rcp_669, _rcp_670);
                            float2 r_73 = r_74;
                            float2 _f32x2_mul_a_671 = mul_f32x2_noftz(ng_36, nu_36);
                            float2 _f32x2_mul_b_672 = mul_f32x2_noftz(r_73, make_float2(bl, bl));
                            float2 m_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_671), "l"(*(const unsigned long long*)&_f32x2_mul_b_672));
                            float2 ns_36 = make_float2(((x_37.x < 0.0f) ? p_36.x : 1.0f), ((x_37.y < 0.0f) ? p_36.y : 1.0f));
                            gate_packed_1[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_36, ns_36))) | sign_37;
                            float2 x_38 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[6]));
                            float2 v_38 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[6]));
                            unsigned int sign_38 = (gate_packed_1[6] ^ up_packed_1[6]) & 2147516416u;
                            float _abs_673;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_673) : "f"(x_38.x));
                            float _abs_674;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_674) : "f"(x_38.y));
                            float2 ax_37 = make_float2(_abs_673, _abs_674);
                            float _abs_675;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_675) : "f"(v_38.x));
                            float _abs_676;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_676) : "f"(v_38.y));
                            float2 av_37 = make_float2(_abs_675, _abs_676);
                            float2 _f32x2_mul_b_677 = make_float2(k1, k1);
                            float2 te_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_37) : "l"(*(const unsigned long long*)&ax_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_677));
                            float2 _f32x2_mul_b_678 = make_float2(k2, k2);
                            float2 tw_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_37) : "l"(*(const unsigned long long*)&av_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_678));
                            float _exp2_679 = approx_exp2(te_37.x);
                            float _exp2_680 = approx_exp2(te_37.y);
                            float2 e_37 = make_float2(_exp2_679, _exp2_680);
                            float _exp2_681 = approx_exp2(tw_37.x);
                            float _exp2_682 = approx_exp2(tw_37.y);
                            float2 w_37 = make_float2(_exp2_681, _exp2_682);
                            float2 _f32x2_mul_b_683 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_37) : "l"(*(const unsigned long long*)&ax_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_683));
                            float _exp2_684 = approx_exp2_noftz(tp_37.x);
                            float _exp2_685 = approx_exp2_noftz(tp_37.y);
                            float2 p_37 = make_float2(_exp2_684, _exp2_685);
                            float2 d1_37 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_37);
                            float2 d2_37 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_37);
                            float2 dw_37 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_37);
                            float2 ng_37 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_37);
                            float2 nu_37 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_37);
                            float _rcp_686 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_37, d2_37), dw_37).x);
                            float _rcp_687 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_37, d2_37), dw_37).y);
                            float2 r_76 = make_float2(_rcp_686, _rcp_687);
                            float2 r_75 = r_76;
                            float2 _f32x2_mul_a_688 = mul_f32x2_noftz(ng_37, nu_37);
                            float2 _f32x2_mul_b_689 = mul_f32x2_noftz(r_75, make_float2(bl, bl));
                            float2 m_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_688), "l"(*(const unsigned long long*)&_f32x2_mul_b_689));
                            float2 ns_37 = make_float2(((x_38.x < 0.0f) ? p_37.x : 1.0f), ((x_38.y < 0.0f) ? p_37.y : 1.0f));
                            gate_packed_1[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_37, ns_37))) | sign_38;
                            float2 x_39 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[7]));
                            float2 v_39 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[7]));
                            unsigned int sign_39 = (gate_packed_1[7] ^ up_packed_1[7]) & 2147516416u;
                            float _abs_690;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_690) : "f"(x_39.x));
                            float _abs_691;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_691) : "f"(x_39.y));
                            float2 ax_38 = make_float2(_abs_690, _abs_691);
                            float _abs_692;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_692) : "f"(v_39.x));
                            float _abs_693;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_693) : "f"(v_39.y));
                            float2 av_38 = make_float2(_abs_692, _abs_693);
                            float2 _f32x2_mul_b_694 = make_float2(k1, k1);
                            float2 te_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_38) : "l"(*(const unsigned long long*)&ax_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_694));
                            float2 _f32x2_mul_b_695 = make_float2(k2, k2);
                            float2 tw_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_38) : "l"(*(const unsigned long long*)&av_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_695));
                            float _exp2_696 = approx_exp2(te_38.x);
                            float _exp2_697 = approx_exp2(te_38.y);
                            float2 e_38 = make_float2(_exp2_696, _exp2_697);
                            float _exp2_698 = approx_exp2(tw_38.x);
                            float _exp2_699 = approx_exp2(tw_38.y);
                            float2 w_38 = make_float2(_exp2_698, _exp2_699);
                            float2 _f32x2_mul_b_700 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_38) : "l"(*(const unsigned long long*)&ax_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_700));
                            float _exp2_701 = approx_exp2_noftz(tp_38.x);
                            float _exp2_702 = approx_exp2_noftz(tp_38.y);
                            float2 p_38 = make_float2(_exp2_701, _exp2_702);
                            float2 d1_38 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_38);
                            float2 d2_38 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_38);
                            float2 dw_38 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_38);
                            float2 ng_38 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_38);
                            float2 nu_38 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_38);
                            float _rcp_703 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_38, d2_38), dw_38).x);
                            float _rcp_704 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_38, d2_38), dw_38).y);
                            float2 r_78 = make_float2(_rcp_703, _rcp_704);
                            float2 r_77 = r_78;
                            float2 _f32x2_mul_a_705 = mul_f32x2_noftz(ng_38, nu_38);
                            float2 _f32x2_mul_b_706 = mul_f32x2_noftz(r_77, make_float2(bl, bl));
                            float2 m_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_705), "l"(*(const unsigned long long*)&_f32x2_mul_b_706));
                            float2 ns_38 = make_float2(((x_39.x < 0.0f) ? p_38.x : 1.0f), ((x_39.y < 0.0f) ? p_38.y : 1.0f));
                            gate_packed_1[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_38, ns_38))) | sign_39;
                            float2 x_40 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[8]));
                            float2 v_40 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[8]));
                            unsigned int sign_40 = (gate_packed_1[8] ^ up_packed_1[8]) & 2147516416u;
                            float _abs_707;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_707) : "f"(x_40.x));
                            float _abs_708;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_708) : "f"(x_40.y));
                            float2 ax_39 = make_float2(_abs_707, _abs_708);
                            float _abs_709;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_709) : "f"(v_40.x));
                            float _abs_710;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_710) : "f"(v_40.y));
                            float2 av_39 = make_float2(_abs_709, _abs_710);
                            float2 _f32x2_mul_b_711 = make_float2(k1, k1);
                            float2 te_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_39) : "l"(*(const unsigned long long*)&ax_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_711));
                            float2 _f32x2_mul_b_712 = make_float2(k2, k2);
                            float2 tw_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_39) : "l"(*(const unsigned long long*)&av_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_712));
                            float _exp2_713 = approx_exp2(te_39.x);
                            float _exp2_714 = approx_exp2(te_39.y);
                            float2 e_39 = make_float2(_exp2_713, _exp2_714);
                            float _exp2_715 = approx_exp2(tw_39.x);
                            float _exp2_716 = approx_exp2(tw_39.y);
                            float2 w_39 = make_float2(_exp2_715, _exp2_716);
                            float2 _f32x2_mul_b_717 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_39) : "l"(*(const unsigned long long*)&ax_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_717));
                            float _exp2_718 = approx_exp2_noftz(tp_39.x);
                            float _exp2_719 = approx_exp2_noftz(tp_39.y);
                            float2 p_39 = make_float2(_exp2_718, _exp2_719);
                            float2 d1_39 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_39);
                            float2 d2_39 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_39);
                            float2 dw_39 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_39);
                            float2 ng_39 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_39);
                            float2 nu_39 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_39);
                            float _rcp_720 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_39, d2_39), dw_39).x);
                            float _rcp_721 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_39, d2_39), dw_39).y);
                            float2 r_80 = make_float2(_rcp_720, _rcp_721);
                            float2 r_79 = r_80;
                            float2 _f32x2_mul_a_722 = mul_f32x2_noftz(ng_39, nu_39);
                            float2 _f32x2_mul_b_723 = mul_f32x2_noftz(r_79, make_float2(bl, bl));
                            float2 m_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_722), "l"(*(const unsigned long long*)&_f32x2_mul_b_723));
                            float2 ns_39 = make_float2(((x_40.x < 0.0f) ? p_39.x : 1.0f), ((x_40.y < 0.0f) ? p_39.y : 1.0f));
                            gate_packed_1[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_39, ns_39))) | sign_40;
                            float2 x_41 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[9]));
                            float2 v_41 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[9]));
                            unsigned int sign_41 = (gate_packed_1[9] ^ up_packed_1[9]) & 2147516416u;
                            float _abs_724;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_724) : "f"(x_41.x));
                            float _abs_725;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_725) : "f"(x_41.y));
                            float2 ax_40 = make_float2(_abs_724, _abs_725);
                            float _abs_726;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_726) : "f"(v_41.x));
                            float _abs_727;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_727) : "f"(v_41.y));
                            float2 av_40 = make_float2(_abs_726, _abs_727);
                            float2 _f32x2_mul_b_728 = make_float2(k1, k1);
                            float2 te_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_40) : "l"(*(const unsigned long long*)&ax_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_728));
                            float2 _f32x2_mul_b_729 = make_float2(k2, k2);
                            float2 tw_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_40) : "l"(*(const unsigned long long*)&av_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_729));
                            float _exp2_730 = approx_exp2(te_40.x);
                            float _exp2_731 = approx_exp2(te_40.y);
                            float2 e_40 = make_float2(_exp2_730, _exp2_731);
                            float _exp2_732 = approx_exp2(tw_40.x);
                            float _exp2_733 = approx_exp2(tw_40.y);
                            float2 w_40 = make_float2(_exp2_732, _exp2_733);
                            float2 _f32x2_mul_b_734 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_40) : "l"(*(const unsigned long long*)&ax_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_734));
                            float _exp2_735 = approx_exp2_noftz(tp_40.x);
                            float _exp2_736 = approx_exp2_noftz(tp_40.y);
                            float2 p_40 = make_float2(_exp2_735, _exp2_736);
                            float2 d1_40 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_40);
                            float2 d2_40 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_40);
                            float2 dw_40 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_40);
                            float2 ng_40 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_40);
                            float2 nu_40 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_40);
                            float _rcp_737 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_40, d2_40), dw_40).x);
                            float _rcp_738 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_40, d2_40), dw_40).y);
                            float2 r_82 = make_float2(_rcp_737, _rcp_738);
                            float2 r_81 = r_82;
                            float2 _f32x2_mul_a_739 = mul_f32x2_noftz(ng_40, nu_40);
                            float2 _f32x2_mul_b_740 = mul_f32x2_noftz(r_81, make_float2(bl, bl));
                            float2 m_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_739), "l"(*(const unsigned long long*)&_f32x2_mul_b_740));
                            float2 ns_40 = make_float2(((x_41.x < 0.0f) ? p_40.x : 1.0f), ((x_41.y < 0.0f) ? p_40.y : 1.0f));
                            gate_packed_1[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_40, ns_40))) | sign_41;
                            float2 x_42 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[10]));
                            float2 v_42 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[10]));
                            unsigned int sign_42 = (gate_packed_1[10] ^ up_packed_1[10]) & 2147516416u;
                            float _abs_741;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_741) : "f"(x_42.x));
                            float _abs_742;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_742) : "f"(x_42.y));
                            float2 ax_41 = make_float2(_abs_741, _abs_742);
                            float _abs_743;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_743) : "f"(v_42.x));
                            float _abs_744;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_744) : "f"(v_42.y));
                            float2 av_41 = make_float2(_abs_743, _abs_744);
                            float2 _f32x2_mul_b_745 = make_float2(k1, k1);
                            float2 te_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_41) : "l"(*(const unsigned long long*)&ax_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_745));
                            float2 _f32x2_mul_b_746 = make_float2(k2, k2);
                            float2 tw_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_41) : "l"(*(const unsigned long long*)&av_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_746));
                            float _exp2_747 = approx_exp2(te_41.x);
                            float _exp2_748 = approx_exp2(te_41.y);
                            float2 e_41 = make_float2(_exp2_747, _exp2_748);
                            float _exp2_749 = approx_exp2(tw_41.x);
                            float _exp2_750 = approx_exp2(tw_41.y);
                            float2 w_41 = make_float2(_exp2_749, _exp2_750);
                            float2 _f32x2_mul_b_751 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_41) : "l"(*(const unsigned long long*)&ax_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_751));
                            float _exp2_752 = approx_exp2_noftz(tp_41.x);
                            float _exp2_753 = approx_exp2_noftz(tp_41.y);
                            float2 p_41 = make_float2(_exp2_752, _exp2_753);
                            float2 d1_41 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_41);
                            float2 d2_41 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_41);
                            float2 dw_41 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_41);
                            float2 ng_41 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_41);
                            float2 nu_41 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_41);
                            float _rcp_754 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_41, d2_41), dw_41).x);
                            float _rcp_755 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_41, d2_41), dw_41).y);
                            float2 r_84 = make_float2(_rcp_754, _rcp_755);
                            float2 r_83 = r_84;
                            float2 _f32x2_mul_a_756 = mul_f32x2_noftz(ng_41, nu_41);
                            float2 _f32x2_mul_b_757 = mul_f32x2_noftz(r_83, make_float2(bl, bl));
                            float2 m_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_756), "l"(*(const unsigned long long*)&_f32x2_mul_b_757));
                            float2 ns_41 = make_float2(((x_42.x < 0.0f) ? p_41.x : 1.0f), ((x_42.y < 0.0f) ? p_41.y : 1.0f));
                            gate_packed_1[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_41, ns_41))) | sign_42;
                            float2 x_43 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[11]));
                            float2 v_43 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[11]));
                            unsigned int sign_43 = (gate_packed_1[11] ^ up_packed_1[11]) & 2147516416u;
                            float _abs_758;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_758) : "f"(x_43.x));
                            float _abs_759;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_759) : "f"(x_43.y));
                            float2 ax_42 = make_float2(_abs_758, _abs_759);
                            float _abs_760;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_760) : "f"(v_43.x));
                            float _abs_761;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_761) : "f"(v_43.y));
                            float2 av_42 = make_float2(_abs_760, _abs_761);
                            float2 _f32x2_mul_b_762 = make_float2(k1, k1);
                            float2 te_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_42) : "l"(*(const unsigned long long*)&ax_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_762));
                            float2 _f32x2_mul_b_763 = make_float2(k2, k2);
                            float2 tw_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_42) : "l"(*(const unsigned long long*)&av_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_763));
                            float _exp2_764 = approx_exp2(te_42.x);
                            float _exp2_765 = approx_exp2(te_42.y);
                            float2 e_42 = make_float2(_exp2_764, _exp2_765);
                            float _exp2_766 = approx_exp2(tw_42.x);
                            float _exp2_767 = approx_exp2(tw_42.y);
                            float2 w_42 = make_float2(_exp2_766, _exp2_767);
                            float2 _f32x2_mul_b_768 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_42) : "l"(*(const unsigned long long*)&ax_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_768));
                            float _exp2_769 = approx_exp2_noftz(tp_42.x);
                            float _exp2_770 = approx_exp2_noftz(tp_42.y);
                            float2 p_42 = make_float2(_exp2_769, _exp2_770);
                            float2 d1_42 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_42);
                            float2 d2_42 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_42);
                            float2 dw_42 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_42);
                            float2 ng_42 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_42);
                            float2 nu_42 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_42);
                            float _rcp_771 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_42, d2_42), dw_42).x);
                            float _rcp_772 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_42, d2_42), dw_42).y);
                            float2 r_86 = make_float2(_rcp_771, _rcp_772);
                            float2 r_85 = r_86;
                            float2 _f32x2_mul_a_773 = mul_f32x2_noftz(ng_42, nu_42);
                            float2 _f32x2_mul_b_774 = mul_f32x2_noftz(r_85, make_float2(bl, bl));
                            float2 m_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_773), "l"(*(const unsigned long long*)&_f32x2_mul_b_774));
                            float2 ns_42 = make_float2(((x_43.x < 0.0f) ? p_42.x : 1.0f), ((x_43.y < 0.0f) ? p_42.y : 1.0f));
                            gate_packed_1[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_42, ns_42))) | sign_43;
                            float2 x_44 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[12]));
                            float2 v_44 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[12]));
                            unsigned int sign_44 = (gate_packed_1[12] ^ up_packed_1[12]) & 2147516416u;
                            float _abs_775;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_775) : "f"(x_44.x));
                            float _abs_776;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_776) : "f"(x_44.y));
                            float2 ax_43 = make_float2(_abs_775, _abs_776);
                            float _abs_777;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_777) : "f"(v_44.x));
                            float _abs_778;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_778) : "f"(v_44.y));
                            float2 av_43 = make_float2(_abs_777, _abs_778);
                            float2 _f32x2_mul_b_779 = make_float2(k1, k1);
                            float2 te_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_43) : "l"(*(const unsigned long long*)&ax_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_779));
                            float2 _f32x2_mul_b_780 = make_float2(k2, k2);
                            float2 tw_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_43) : "l"(*(const unsigned long long*)&av_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_780));
                            float _exp2_781 = approx_exp2(te_43.x);
                            float _exp2_782 = approx_exp2(te_43.y);
                            float2 e_43 = make_float2(_exp2_781, _exp2_782);
                            float _exp2_783 = approx_exp2(tw_43.x);
                            float _exp2_784 = approx_exp2(tw_43.y);
                            float2 w_43 = make_float2(_exp2_783, _exp2_784);
                            float2 _f32x2_mul_b_785 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_43) : "l"(*(const unsigned long long*)&ax_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_785));
                            float _exp2_786 = approx_exp2_noftz(tp_43.x);
                            float _exp2_787 = approx_exp2_noftz(tp_43.y);
                            float2 p_43 = make_float2(_exp2_786, _exp2_787);
                            float2 d1_43 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_43);
                            float2 d2_43 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_43);
                            float2 dw_43 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_43);
                            float2 ng_43 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_43);
                            float2 nu_43 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_43);
                            float _rcp_788 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_43, d2_43), dw_43).x);
                            float _rcp_789 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_43, d2_43), dw_43).y);
                            float2 r_88 = make_float2(_rcp_788, _rcp_789);
                            float2 r_87 = r_88;
                            float2 _f32x2_mul_a_790 = mul_f32x2_noftz(ng_43, nu_43);
                            float2 _f32x2_mul_b_791 = mul_f32x2_noftz(r_87, make_float2(bl, bl));
                            float2 m_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_790), "l"(*(const unsigned long long*)&_f32x2_mul_b_791));
                            float2 ns_43 = make_float2(((x_44.x < 0.0f) ? p_43.x : 1.0f), ((x_44.y < 0.0f) ? p_43.y : 1.0f));
                            gate_packed_1[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_43, ns_43))) | sign_44;
                            float2 x_45 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[13]));
                            float2 v_45 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[13]));
                            unsigned int sign_45 = (gate_packed_1[13] ^ up_packed_1[13]) & 2147516416u;
                            float _abs_792;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_792) : "f"(x_45.x));
                            float _abs_793;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_793) : "f"(x_45.y));
                            float2 ax_44 = make_float2(_abs_792, _abs_793);
                            float _abs_794;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_794) : "f"(v_45.x));
                            float _abs_795;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_795) : "f"(v_45.y));
                            float2 av_44 = make_float2(_abs_794, _abs_795);
                            float2 _f32x2_mul_b_796 = make_float2(k1, k1);
                            float2 te_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_44) : "l"(*(const unsigned long long*)&ax_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_796));
                            float2 _f32x2_mul_b_797 = make_float2(k2, k2);
                            float2 tw_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_44) : "l"(*(const unsigned long long*)&av_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_797));
                            float _exp2_798 = approx_exp2(te_44.x);
                            float _exp2_799 = approx_exp2(te_44.y);
                            float2 e_44 = make_float2(_exp2_798, _exp2_799);
                            float _exp2_800 = approx_exp2(tw_44.x);
                            float _exp2_801 = approx_exp2(tw_44.y);
                            float2 w_44 = make_float2(_exp2_800, _exp2_801);
                            float2 _f32x2_mul_b_802 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_44) : "l"(*(const unsigned long long*)&ax_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_802));
                            float _exp2_803 = approx_exp2_noftz(tp_44.x);
                            float _exp2_804 = approx_exp2_noftz(tp_44.y);
                            float2 p_44 = make_float2(_exp2_803, _exp2_804);
                            float2 d1_44 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_44);
                            float2 d2_44 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_44);
                            float2 dw_44 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_44);
                            float2 ng_44 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_44);
                            float2 nu_44 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_44);
                            float _rcp_805 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_44, d2_44), dw_44).x);
                            float _rcp_806 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_44, d2_44), dw_44).y);
                            float2 r_90 = make_float2(_rcp_805, _rcp_806);
                            float2 r_89 = r_90;
                            float2 _f32x2_mul_a_807 = mul_f32x2_noftz(ng_44, nu_44);
                            float2 _f32x2_mul_b_808 = mul_f32x2_noftz(r_89, make_float2(bl, bl));
                            float2 m_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_807), "l"(*(const unsigned long long*)&_f32x2_mul_b_808));
                            float2 ns_44 = make_float2(((x_45.x < 0.0f) ? p_44.x : 1.0f), ((x_45.y < 0.0f) ? p_44.y : 1.0f));
                            gate_packed_1[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_44, ns_44))) | sign_45;
                            float2 x_46 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[14]));
                            float2 v_46 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[14]));
                            unsigned int sign_46 = (gate_packed_1[14] ^ up_packed_1[14]) & 2147516416u;
                            float _abs_809;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_809) : "f"(x_46.x));
                            float _abs_810;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_810) : "f"(x_46.y));
                            float2 ax_45 = make_float2(_abs_809, _abs_810);
                            float _abs_811;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_811) : "f"(v_46.x));
                            float _abs_812;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_812) : "f"(v_46.y));
                            float2 av_45 = make_float2(_abs_811, _abs_812);
                            float2 _f32x2_mul_b_813 = make_float2(k1, k1);
                            float2 te_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_45) : "l"(*(const unsigned long long*)&ax_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_813));
                            float2 _f32x2_mul_b_814 = make_float2(k2, k2);
                            float2 tw_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_45) : "l"(*(const unsigned long long*)&av_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_814));
                            float _exp2_815 = approx_exp2(te_45.x);
                            float _exp2_816 = approx_exp2(te_45.y);
                            float2 e_45 = make_float2(_exp2_815, _exp2_816);
                            float _exp2_817 = approx_exp2(tw_45.x);
                            float _exp2_818 = approx_exp2(tw_45.y);
                            float2 w_45 = make_float2(_exp2_817, _exp2_818);
                            float2 _f32x2_mul_b_819 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_45) : "l"(*(const unsigned long long*)&ax_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_819));
                            float _exp2_820 = approx_exp2_noftz(tp_45.x);
                            float _exp2_821 = approx_exp2_noftz(tp_45.y);
                            float2 p_45 = make_float2(_exp2_820, _exp2_821);
                            float2 d1_45 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_45);
                            float2 d2_45 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_45);
                            float2 dw_45 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_45);
                            float2 ng_45 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_45);
                            float2 nu_45 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_45);
                            float _rcp_822 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_45, d2_45), dw_45).x);
                            float _rcp_823 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_45, d2_45), dw_45).y);
                            float2 r_92 = make_float2(_rcp_822, _rcp_823);
                            float2 r_91 = r_92;
                            float2 _f32x2_mul_a_824 = mul_f32x2_noftz(ng_45, nu_45);
                            float2 _f32x2_mul_b_825 = mul_f32x2_noftz(r_91, make_float2(bl, bl));
                            float2 m_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_824), "l"(*(const unsigned long long*)&_f32x2_mul_b_825));
                            float2 ns_45 = make_float2(((x_46.x < 0.0f) ? p_45.x : 1.0f), ((x_46.y < 0.0f) ? p_45.y : 1.0f));
                            gate_packed_1[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_45, ns_45))) | sign_46;
                            float2 x_47 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[15]));
                            float2 v_47 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[15]));
                            unsigned int sign_47 = (gate_packed_1[15] ^ up_packed_1[15]) & 2147516416u;
                            float _abs_826;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_826) : "f"(x_47.x));
                            float _abs_827;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_827) : "f"(x_47.y));
                            float2 ax_46 = make_float2(_abs_826, _abs_827);
                            float _abs_828;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_828) : "f"(v_47.x));
                            float _abs_829;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_829) : "f"(v_47.y));
                            float2 av_46 = make_float2(_abs_828, _abs_829);
                            float2 _f32x2_mul_b_830 = make_float2(k1, k1);
                            float2 te_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_46) : "l"(*(const unsigned long long*)&ax_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_830));
                            float2 _f32x2_mul_b_831 = make_float2(k2, k2);
                            float2 tw_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_46) : "l"(*(const unsigned long long*)&av_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_831));
                            float _exp2_832 = approx_exp2(te_46.x);
                            float _exp2_833 = approx_exp2(te_46.y);
                            float2 e_46 = make_float2(_exp2_832, _exp2_833);
                            float _exp2_834 = approx_exp2(tw_46.x);
                            float _exp2_835 = approx_exp2(tw_46.y);
                            float2 w_46 = make_float2(_exp2_834, _exp2_835);
                            float2 _f32x2_mul_b_836 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_46) : "l"(*(const unsigned long long*)&ax_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_836));
                            float _exp2_837 = approx_exp2_noftz(tp_46.x);
                            float _exp2_838 = approx_exp2_noftz(tp_46.y);
                            float2 p_46 = make_float2(_exp2_837, _exp2_838);
                            float2 d1_46 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_46);
                            float2 d2_46 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_46);
                            float2 dw_46 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_46);
                            float2 ng_46 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_46);
                            float2 nu_46 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_46);
                            float _rcp_839 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_46, d2_46), dw_46).x);
                            float _rcp_840 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_46, d2_46), dw_46).y);
                            float2 r_94 = make_float2(_rcp_839, _rcp_840);
                            float2 r_93 = r_94;
                            float2 _f32x2_mul_a_841 = mul_f32x2_noftz(ng_46, nu_46);
                            float2 _f32x2_mul_b_842 = mul_f32x2_noftz(r_93, make_float2(bl, bl));
                            float2 m_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_841), "l"(*(const unsigned long long*)&_f32x2_mul_b_842));
                            float2 ns_46 = make_float2(((x_47.x < 0.0f) ? p_46.x : 1.0f), ((x_47.y < 0.0f) ? p_46.y : 1.0f));
                            gate_packed_1[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_46, ns_46))) | sign_47;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_9 = tid / 32;
                            int lane_9 = tid % 32;
                            #pragma unroll
                            for (int half_10 = 0; half_10 < 2; half_10++) {
                                #pragma unroll
                                for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                    int row_11 = warp_9 * 32 + half_10 * 16 + lane_9 % 16;
                                    int col_11 = col_tile_10 * 16 + lane_9 / 16 * 8;
                                    unsigned int address_10 = d_smem_addr + 8192 + (unsigned int)((row_11 * 32 + col_11) * 2);
                                    address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                    int offset_11 = half_10 * 8 + col_tile_10 * 4;
                                    uint32_t _stmatrix_addr_843 = static_cast<uint32_t>(address_10);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_843), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_48 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[16]));
                            float2 v_48 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[16]));
                            unsigned int sign_48 = (gate_packed_1[16] ^ up_packed_1[16]) & 2147516416u;
                            float _abs_844;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_844) : "f"(x_48.x));
                            float _abs_845;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_845) : "f"(x_48.y));
                            float2 ax_47 = make_float2(_abs_844, _abs_845);
                            float _abs_846;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_846) : "f"(v_48.x));
                            float _abs_847;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_847) : "f"(v_48.y));
                            float2 av_47 = make_float2(_abs_846, _abs_847);
                            float2 _f32x2_mul_b_848 = make_float2(k1, k1);
                            float2 te_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_47) : "l"(*(const unsigned long long*)&ax_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_848));
                            float2 _f32x2_mul_b_849 = make_float2(k2, k2);
                            float2 tw_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_47) : "l"(*(const unsigned long long*)&av_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_849));
                            float _exp2_850 = approx_exp2(te_47.x);
                            float _exp2_851 = approx_exp2(te_47.y);
                            float2 e_47 = make_float2(_exp2_850, _exp2_851);
                            float _exp2_852 = approx_exp2(tw_47.x);
                            float _exp2_853 = approx_exp2(tw_47.y);
                            float2 w_47 = make_float2(_exp2_852, _exp2_853);
                            float2 _f32x2_mul_b_854 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_47) : "l"(*(const unsigned long long*)&ax_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_854));
                            float _exp2_855 = approx_exp2_noftz(tp_47.x);
                            float _exp2_856 = approx_exp2_noftz(tp_47.y);
                            float2 p_47 = make_float2(_exp2_855, _exp2_856);
                            float2 d1_47 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_47);
                            float2 d2_47 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_47);
                            float2 dw_47 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_47);
                            float2 ng_47 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_47);
                            float2 nu_47 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_47);
                            float _rcp_857 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_47, d2_47), dw_47).x);
                            float _rcp_858 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_47, d2_47), dw_47).y);
                            float2 r_96 = make_float2(_rcp_857, _rcp_858);
                            float2 r_95 = r_96;
                            float2 _f32x2_mul_a_859 = mul_f32x2_noftz(ng_47, nu_47);
                            float2 _f32x2_mul_b_860 = mul_f32x2_noftz(r_95, make_float2(bl, bl));
                            float2 m_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_859), "l"(*(const unsigned long long*)&_f32x2_mul_b_860));
                            float2 ns_47 = make_float2(((x_48.x < 0.0f) ? p_47.x : 1.0f), ((x_48.y < 0.0f) ? p_47.y : 1.0f));
                            gate_packed_1[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_47, ns_47))) | sign_48;
                            float2 x_49 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[17]));
                            float2 v_49 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[17]));
                            unsigned int sign_49 = (gate_packed_1[17] ^ up_packed_1[17]) & 2147516416u;
                            float _abs_861;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_861) : "f"(x_49.x));
                            float _abs_862;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_862) : "f"(x_49.y));
                            float2 ax_48 = make_float2(_abs_861, _abs_862);
                            float _abs_863;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_863) : "f"(v_49.x));
                            float _abs_864;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_864) : "f"(v_49.y));
                            float2 av_48 = make_float2(_abs_863, _abs_864);
                            float2 _f32x2_mul_b_865 = make_float2(k1, k1);
                            float2 te_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_48) : "l"(*(const unsigned long long*)&ax_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_865));
                            float2 _f32x2_mul_b_866 = make_float2(k2, k2);
                            float2 tw_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_48) : "l"(*(const unsigned long long*)&av_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_866));
                            float _exp2_867 = approx_exp2(te_48.x);
                            float _exp2_868 = approx_exp2(te_48.y);
                            float2 e_48 = make_float2(_exp2_867, _exp2_868);
                            float _exp2_869 = approx_exp2(tw_48.x);
                            float _exp2_870 = approx_exp2(tw_48.y);
                            float2 w_48 = make_float2(_exp2_869, _exp2_870);
                            float2 _f32x2_mul_b_871 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_48) : "l"(*(const unsigned long long*)&ax_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_871));
                            float _exp2_872 = approx_exp2_noftz(tp_48.x);
                            float _exp2_873 = approx_exp2_noftz(tp_48.y);
                            float2 p_48 = make_float2(_exp2_872, _exp2_873);
                            float2 d1_48 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_48);
                            float2 d2_48 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_48);
                            float2 dw_48 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_48);
                            float2 ng_48 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_48);
                            float2 nu_48 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_48);
                            float _rcp_874 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_48, d2_48), dw_48).x);
                            float _rcp_875 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_48, d2_48), dw_48).y);
                            float2 r_98 = make_float2(_rcp_874, _rcp_875);
                            float2 r_97 = r_98;
                            float2 _f32x2_mul_a_876 = mul_f32x2_noftz(ng_48, nu_48);
                            float2 _f32x2_mul_b_877 = mul_f32x2_noftz(r_97, make_float2(bl, bl));
                            float2 m_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_876), "l"(*(const unsigned long long*)&_f32x2_mul_b_877));
                            float2 ns_48 = make_float2(((x_49.x < 0.0f) ? p_48.x : 1.0f), ((x_49.y < 0.0f) ? p_48.y : 1.0f));
                            gate_packed_1[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_48, ns_48))) | sign_49;
                            float2 x_50 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[18]));
                            float2 v_50 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[18]));
                            unsigned int sign_50 = (gate_packed_1[18] ^ up_packed_1[18]) & 2147516416u;
                            float _abs_878;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_878) : "f"(x_50.x));
                            float _abs_879;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_879) : "f"(x_50.y));
                            float2 ax_49 = make_float2(_abs_878, _abs_879);
                            float _abs_880;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_880) : "f"(v_50.x));
                            float _abs_881;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_881) : "f"(v_50.y));
                            float2 av_49 = make_float2(_abs_880, _abs_881);
                            float2 _f32x2_mul_b_882 = make_float2(k1, k1);
                            float2 te_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_49) : "l"(*(const unsigned long long*)&ax_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_882));
                            float2 _f32x2_mul_b_883 = make_float2(k2, k2);
                            float2 tw_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_49) : "l"(*(const unsigned long long*)&av_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_883));
                            float _exp2_884 = approx_exp2(te_49.x);
                            float _exp2_885 = approx_exp2(te_49.y);
                            float2 e_49 = make_float2(_exp2_884, _exp2_885);
                            float _exp2_886 = approx_exp2(tw_49.x);
                            float _exp2_887 = approx_exp2(tw_49.y);
                            float2 w_49 = make_float2(_exp2_886, _exp2_887);
                            float2 _f32x2_mul_b_888 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_49) : "l"(*(const unsigned long long*)&ax_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_888));
                            float _exp2_889 = approx_exp2_noftz(tp_49.x);
                            float _exp2_890 = approx_exp2_noftz(tp_49.y);
                            float2 p_49 = make_float2(_exp2_889, _exp2_890);
                            float2 d1_49 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_49);
                            float2 d2_49 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_49);
                            float2 dw_49 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_49);
                            float2 ng_49 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_49);
                            float2 nu_49 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_49);
                            float _rcp_891 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_49, d2_49), dw_49).x);
                            float _rcp_892 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_49, d2_49), dw_49).y);
                            float2 r_100 = make_float2(_rcp_891, _rcp_892);
                            float2 r_99 = r_100;
                            float2 _f32x2_mul_a_893 = mul_f32x2_noftz(ng_49, nu_49);
                            float2 _f32x2_mul_b_894 = mul_f32x2_noftz(r_99, make_float2(bl, bl));
                            float2 m_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_893), "l"(*(const unsigned long long*)&_f32x2_mul_b_894));
                            float2 ns_49 = make_float2(((x_50.x < 0.0f) ? p_49.x : 1.0f), ((x_50.y < 0.0f) ? p_49.y : 1.0f));
                            gate_packed_1[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_49, ns_49))) | sign_50;
                            float2 x_51 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[19]));
                            float2 v_51 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[19]));
                            unsigned int sign_51 = (gate_packed_1[19] ^ up_packed_1[19]) & 2147516416u;
                            float _abs_895;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_895) : "f"(x_51.x));
                            float _abs_896;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_896) : "f"(x_51.y));
                            float2 ax_50 = make_float2(_abs_895, _abs_896);
                            float _abs_897;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_897) : "f"(v_51.x));
                            float _abs_898;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_898) : "f"(v_51.y));
                            float2 av_50 = make_float2(_abs_897, _abs_898);
                            float2 _f32x2_mul_b_899 = make_float2(k1, k1);
                            float2 te_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_50) : "l"(*(const unsigned long long*)&ax_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_899));
                            float2 _f32x2_mul_b_900 = make_float2(k2, k2);
                            float2 tw_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_50) : "l"(*(const unsigned long long*)&av_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_900));
                            float _exp2_901 = approx_exp2(te_50.x);
                            float _exp2_902 = approx_exp2(te_50.y);
                            float2 e_50 = make_float2(_exp2_901, _exp2_902);
                            float _exp2_903 = approx_exp2(tw_50.x);
                            float _exp2_904 = approx_exp2(tw_50.y);
                            float2 w_50 = make_float2(_exp2_903, _exp2_904);
                            float2 _f32x2_mul_b_905 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_50) : "l"(*(const unsigned long long*)&ax_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_905));
                            float _exp2_906 = approx_exp2_noftz(tp_50.x);
                            float _exp2_907 = approx_exp2_noftz(tp_50.y);
                            float2 p_50 = make_float2(_exp2_906, _exp2_907);
                            float2 d1_50 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_50);
                            float2 d2_50 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_50);
                            float2 dw_50 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_50);
                            float2 ng_50 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_50);
                            float2 nu_50 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_50);
                            float _rcp_908 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_50, d2_50), dw_50).x);
                            float _rcp_909 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_50, d2_50), dw_50).y);
                            float2 r_102 = make_float2(_rcp_908, _rcp_909);
                            float2 r_101 = r_102;
                            float2 _f32x2_mul_a_910 = mul_f32x2_noftz(ng_50, nu_50);
                            float2 _f32x2_mul_b_911 = mul_f32x2_noftz(r_101, make_float2(bl, bl));
                            float2 m_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_910), "l"(*(const unsigned long long*)&_f32x2_mul_b_911));
                            float2 ns_50 = make_float2(((x_51.x < 0.0f) ? p_50.x : 1.0f), ((x_51.y < 0.0f) ? p_50.y : 1.0f));
                            gate_packed_1[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_50, ns_50))) | sign_51;
                            float2 x_52 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[20]));
                            float2 v_52 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[20]));
                            unsigned int sign_52 = (gate_packed_1[20] ^ up_packed_1[20]) & 2147516416u;
                            float _abs_912;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_912) : "f"(x_52.x));
                            float _abs_913;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_913) : "f"(x_52.y));
                            float2 ax_51 = make_float2(_abs_912, _abs_913);
                            float _abs_914;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_914) : "f"(v_52.x));
                            float _abs_915;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_915) : "f"(v_52.y));
                            float2 av_51 = make_float2(_abs_914, _abs_915);
                            float2 _f32x2_mul_b_916 = make_float2(k1, k1);
                            float2 te_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_51) : "l"(*(const unsigned long long*)&ax_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_916));
                            float2 _f32x2_mul_b_917 = make_float2(k2, k2);
                            float2 tw_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_51) : "l"(*(const unsigned long long*)&av_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_917));
                            float _exp2_918 = approx_exp2(te_51.x);
                            float _exp2_919 = approx_exp2(te_51.y);
                            float2 e_51 = make_float2(_exp2_918, _exp2_919);
                            float _exp2_920 = approx_exp2(tw_51.x);
                            float _exp2_921 = approx_exp2(tw_51.y);
                            float2 w_51 = make_float2(_exp2_920, _exp2_921);
                            float2 _f32x2_mul_b_922 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_51) : "l"(*(const unsigned long long*)&ax_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_922));
                            float _exp2_923 = approx_exp2_noftz(tp_51.x);
                            float _exp2_924 = approx_exp2_noftz(tp_51.y);
                            float2 p_51 = make_float2(_exp2_923, _exp2_924);
                            float2 d1_51 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_51);
                            float2 d2_51 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_51);
                            float2 dw_51 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_51);
                            float2 ng_51 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_51);
                            float2 nu_51 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_51);
                            float _rcp_925 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_51, d2_51), dw_51).x);
                            float _rcp_926 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_51, d2_51), dw_51).y);
                            float2 r_104 = make_float2(_rcp_925, _rcp_926);
                            float2 r_103 = r_104;
                            float2 _f32x2_mul_a_927 = mul_f32x2_noftz(ng_51, nu_51);
                            float2 _f32x2_mul_b_928 = mul_f32x2_noftz(r_103, make_float2(bl, bl));
                            float2 m_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_927), "l"(*(const unsigned long long*)&_f32x2_mul_b_928));
                            float2 ns_51 = make_float2(((x_52.x < 0.0f) ? p_51.x : 1.0f), ((x_52.y < 0.0f) ? p_51.y : 1.0f));
                            gate_packed_1[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_51, ns_51))) | sign_52;
                            float2 x_53 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[21]));
                            float2 v_53 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[21]));
                            unsigned int sign_53 = (gate_packed_1[21] ^ up_packed_1[21]) & 2147516416u;
                            float _abs_929;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_929) : "f"(x_53.x));
                            float _abs_930;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_930) : "f"(x_53.y));
                            float2 ax_52 = make_float2(_abs_929, _abs_930);
                            float _abs_931;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_931) : "f"(v_53.x));
                            float _abs_932;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_932) : "f"(v_53.y));
                            float2 av_52 = make_float2(_abs_931, _abs_932);
                            float2 _f32x2_mul_b_933 = make_float2(k1, k1);
                            float2 te_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_52) : "l"(*(const unsigned long long*)&ax_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_933));
                            float2 _f32x2_mul_b_934 = make_float2(k2, k2);
                            float2 tw_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_52) : "l"(*(const unsigned long long*)&av_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_934));
                            float _exp2_935 = approx_exp2(te_52.x);
                            float _exp2_936 = approx_exp2(te_52.y);
                            float2 e_52 = make_float2(_exp2_935, _exp2_936);
                            float _exp2_937 = approx_exp2(tw_52.x);
                            float _exp2_938 = approx_exp2(tw_52.y);
                            float2 w_52 = make_float2(_exp2_937, _exp2_938);
                            float2 _f32x2_mul_b_939 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_52) : "l"(*(const unsigned long long*)&ax_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_939));
                            float _exp2_940 = approx_exp2_noftz(tp_52.x);
                            float _exp2_941 = approx_exp2_noftz(tp_52.y);
                            float2 p_52 = make_float2(_exp2_940, _exp2_941);
                            float2 d1_52 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_52);
                            float2 d2_52 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_52);
                            float2 dw_52 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_52);
                            float2 ng_52 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_52);
                            float2 nu_52 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_52);
                            float _rcp_942 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_52, d2_52), dw_52).x);
                            float _rcp_943 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_52, d2_52), dw_52).y);
                            float2 r_106 = make_float2(_rcp_942, _rcp_943);
                            float2 r_105 = r_106;
                            float2 _f32x2_mul_a_944 = mul_f32x2_noftz(ng_52, nu_52);
                            float2 _f32x2_mul_b_945 = mul_f32x2_noftz(r_105, make_float2(bl, bl));
                            float2 m_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_944), "l"(*(const unsigned long long*)&_f32x2_mul_b_945));
                            float2 ns_52 = make_float2(((x_53.x < 0.0f) ? p_52.x : 1.0f), ((x_53.y < 0.0f) ? p_52.y : 1.0f));
                            gate_packed_1[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_52, ns_52))) | sign_53;
                            float2 x_54 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[22]));
                            float2 v_54 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[22]));
                            unsigned int sign_54 = (gate_packed_1[22] ^ up_packed_1[22]) & 2147516416u;
                            float _abs_946;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_946) : "f"(x_54.x));
                            float _abs_947;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_947) : "f"(x_54.y));
                            float2 ax_53 = make_float2(_abs_946, _abs_947);
                            float _abs_948;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_948) : "f"(v_54.x));
                            float _abs_949;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_949) : "f"(v_54.y));
                            float2 av_53 = make_float2(_abs_948, _abs_949);
                            float2 _f32x2_mul_b_950 = make_float2(k1, k1);
                            float2 te_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_53) : "l"(*(const unsigned long long*)&ax_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_950));
                            float2 _f32x2_mul_b_951 = make_float2(k2, k2);
                            float2 tw_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_53) : "l"(*(const unsigned long long*)&av_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_951));
                            float _exp2_952 = approx_exp2(te_53.x);
                            float _exp2_953 = approx_exp2(te_53.y);
                            float2 e_53 = make_float2(_exp2_952, _exp2_953);
                            float _exp2_954 = approx_exp2(tw_53.x);
                            float _exp2_955 = approx_exp2(tw_53.y);
                            float2 w_53 = make_float2(_exp2_954, _exp2_955);
                            float2 _f32x2_mul_b_956 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_53) : "l"(*(const unsigned long long*)&ax_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_956));
                            float _exp2_957 = approx_exp2_noftz(tp_53.x);
                            float _exp2_958 = approx_exp2_noftz(tp_53.y);
                            float2 p_53 = make_float2(_exp2_957, _exp2_958);
                            float2 d1_53 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_53);
                            float2 d2_53 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_53);
                            float2 dw_53 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_53);
                            float2 ng_53 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_53);
                            float2 nu_53 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_53);
                            float _rcp_959 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_53, d2_53), dw_53).x);
                            float _rcp_960 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_53, d2_53), dw_53).y);
                            float2 r_108 = make_float2(_rcp_959, _rcp_960);
                            float2 r_107 = r_108;
                            float2 _f32x2_mul_a_961 = mul_f32x2_noftz(ng_53, nu_53);
                            float2 _f32x2_mul_b_962 = mul_f32x2_noftz(r_107, make_float2(bl, bl));
                            float2 m_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_961), "l"(*(const unsigned long long*)&_f32x2_mul_b_962));
                            float2 ns_53 = make_float2(((x_54.x < 0.0f) ? p_53.x : 1.0f), ((x_54.y < 0.0f) ? p_53.y : 1.0f));
                            gate_packed_1[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_53, ns_53))) | sign_54;
                            float2 x_55 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[23]));
                            float2 v_55 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[23]));
                            unsigned int sign_55 = (gate_packed_1[23] ^ up_packed_1[23]) & 2147516416u;
                            float _abs_963;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_963) : "f"(x_55.x));
                            float _abs_964;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_964) : "f"(x_55.y));
                            float2 ax_54 = make_float2(_abs_963, _abs_964);
                            float _abs_965;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_965) : "f"(v_55.x));
                            float _abs_966;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_966) : "f"(v_55.y));
                            float2 av_54 = make_float2(_abs_965, _abs_966);
                            float2 _f32x2_mul_b_967 = make_float2(k1, k1);
                            float2 te_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_54) : "l"(*(const unsigned long long*)&ax_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_967));
                            float2 _f32x2_mul_b_968 = make_float2(k2, k2);
                            float2 tw_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_54) : "l"(*(const unsigned long long*)&av_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_968));
                            float _exp2_969 = approx_exp2(te_54.x);
                            float _exp2_970 = approx_exp2(te_54.y);
                            float2 e_54 = make_float2(_exp2_969, _exp2_970);
                            float _exp2_971 = approx_exp2(tw_54.x);
                            float _exp2_972 = approx_exp2(tw_54.y);
                            float2 w_54 = make_float2(_exp2_971, _exp2_972);
                            float2 _f32x2_mul_b_973 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_54) : "l"(*(const unsigned long long*)&ax_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_973));
                            float _exp2_974 = approx_exp2_noftz(tp_54.x);
                            float _exp2_975 = approx_exp2_noftz(tp_54.y);
                            float2 p_54 = make_float2(_exp2_974, _exp2_975);
                            float2 d1_54 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_54);
                            float2 d2_54 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_54);
                            float2 dw_54 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_54);
                            float2 ng_54 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_54);
                            float2 nu_54 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_54);
                            float _rcp_976 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_54, d2_54), dw_54).x);
                            float _rcp_977 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_54, d2_54), dw_54).y);
                            float2 r_110 = make_float2(_rcp_976, _rcp_977);
                            float2 r_109 = r_110;
                            float2 _f32x2_mul_a_978 = mul_f32x2_noftz(ng_54, nu_54);
                            float2 _f32x2_mul_b_979 = mul_f32x2_noftz(r_109, make_float2(bl, bl));
                            float2 m_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_978), "l"(*(const unsigned long long*)&_f32x2_mul_b_979));
                            float2 ns_54 = make_float2(((x_55.x < 0.0f) ? p_54.x : 1.0f), ((x_55.y < 0.0f) ? p_54.y : 1.0f));
                            gate_packed_1[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_54, ns_54))) | sign_55;
                            float2 x_56 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[24]));
                            float2 v_56 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[24]));
                            unsigned int sign_56 = (gate_packed_1[24] ^ up_packed_1[24]) & 2147516416u;
                            float _abs_980;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_980) : "f"(x_56.x));
                            float _abs_981;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_981) : "f"(x_56.y));
                            float2 ax_55 = make_float2(_abs_980, _abs_981);
                            float _abs_982;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_982) : "f"(v_56.x));
                            float _abs_983;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_983) : "f"(v_56.y));
                            float2 av_55 = make_float2(_abs_982, _abs_983);
                            float2 _f32x2_mul_b_984 = make_float2(k1, k1);
                            float2 te_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_55) : "l"(*(const unsigned long long*)&ax_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_984));
                            float2 _f32x2_mul_b_985 = make_float2(k2, k2);
                            float2 tw_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_55) : "l"(*(const unsigned long long*)&av_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_985));
                            float _exp2_986 = approx_exp2(te_55.x);
                            float _exp2_987 = approx_exp2(te_55.y);
                            float2 e_55 = make_float2(_exp2_986, _exp2_987);
                            float _exp2_988 = approx_exp2(tw_55.x);
                            float _exp2_989 = approx_exp2(tw_55.y);
                            float2 w_55 = make_float2(_exp2_988, _exp2_989);
                            float2 _f32x2_mul_b_990 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_55) : "l"(*(const unsigned long long*)&ax_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_990));
                            float _exp2_991 = approx_exp2_noftz(tp_55.x);
                            float _exp2_992 = approx_exp2_noftz(tp_55.y);
                            float2 p_55 = make_float2(_exp2_991, _exp2_992);
                            float2 d1_55 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_55);
                            float2 d2_55 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_55);
                            float2 dw_55 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_55);
                            float2 ng_55 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_55);
                            float2 nu_55 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_55);
                            float _rcp_993 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_55, d2_55), dw_55).x);
                            float _rcp_994 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_55, d2_55), dw_55).y);
                            float2 r_112 = make_float2(_rcp_993, _rcp_994);
                            float2 r_111 = r_112;
                            float2 _f32x2_mul_a_995 = mul_f32x2_noftz(ng_55, nu_55);
                            float2 _f32x2_mul_b_996 = mul_f32x2_noftz(r_111, make_float2(bl, bl));
                            float2 m_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_995), "l"(*(const unsigned long long*)&_f32x2_mul_b_996));
                            float2 ns_55 = make_float2(((x_56.x < 0.0f) ? p_55.x : 1.0f), ((x_56.y < 0.0f) ? p_55.y : 1.0f));
                            gate_packed_1[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_55, ns_55))) | sign_56;
                            float2 x_57 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[25]));
                            float2 v_57 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[25]));
                            unsigned int sign_57 = (gate_packed_1[25] ^ up_packed_1[25]) & 2147516416u;
                            float _abs_997;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_997) : "f"(x_57.x));
                            float _abs_998;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_998) : "f"(x_57.y));
                            float2 ax_56 = make_float2(_abs_997, _abs_998);
                            float _abs_999;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_999) : "f"(v_57.x));
                            float _abs_1000;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1000) : "f"(v_57.y));
                            float2 av_56 = make_float2(_abs_999, _abs_1000);
                            float2 _f32x2_mul_b_1001 = make_float2(k1, k1);
                            float2 te_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_56) : "l"(*(const unsigned long long*)&ax_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_1001));
                            float2 _f32x2_mul_b_1002 = make_float2(k2, k2);
                            float2 tw_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_56) : "l"(*(const unsigned long long*)&av_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_1002));
                            float _exp2_1003 = approx_exp2(te_56.x);
                            float _exp2_1004 = approx_exp2(te_56.y);
                            float2 e_56 = make_float2(_exp2_1003, _exp2_1004);
                            float _exp2_1005 = approx_exp2(tw_56.x);
                            float _exp2_1006 = approx_exp2(tw_56.y);
                            float2 w_56 = make_float2(_exp2_1005, _exp2_1006);
                            float2 _f32x2_mul_b_1007 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_56) : "l"(*(const unsigned long long*)&ax_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_1007));
                            float _exp2_1008 = approx_exp2_noftz(tp_56.x);
                            float _exp2_1009 = approx_exp2_noftz(tp_56.y);
                            float2 p_56 = make_float2(_exp2_1008, _exp2_1009);
                            float2 d1_56 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_56);
                            float2 d2_56 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_56);
                            float2 dw_56 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_56);
                            float2 ng_56 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_56);
                            float2 nu_56 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_56);
                            float _rcp_1010 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_56, d2_56), dw_56).x);
                            float _rcp_1011 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_56, d2_56), dw_56).y);
                            float2 r_114 = make_float2(_rcp_1010, _rcp_1011);
                            float2 r_113 = r_114;
                            float2 _f32x2_mul_a_1012 = mul_f32x2_noftz(ng_56, nu_56);
                            float2 _f32x2_mul_b_1013 = mul_f32x2_noftz(r_113, make_float2(bl, bl));
                            float2 m_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1012), "l"(*(const unsigned long long*)&_f32x2_mul_b_1013));
                            float2 ns_56 = make_float2(((x_57.x < 0.0f) ? p_56.x : 1.0f), ((x_57.y < 0.0f) ? p_56.y : 1.0f));
                            gate_packed_1[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_56, ns_56))) | sign_57;
                            float2 x_58 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[26]));
                            float2 v_58 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[26]));
                            unsigned int sign_58 = (gate_packed_1[26] ^ up_packed_1[26]) & 2147516416u;
                            float _abs_1014;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1014) : "f"(x_58.x));
                            float _abs_1015;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1015) : "f"(x_58.y));
                            float2 ax_57 = make_float2(_abs_1014, _abs_1015);
                            float _abs_1016;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1016) : "f"(v_58.x));
                            float _abs_1017;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1017) : "f"(v_58.y));
                            float2 av_57 = make_float2(_abs_1016, _abs_1017);
                            float2 _f32x2_mul_b_1018 = make_float2(k1, k1);
                            float2 te_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_57) : "l"(*(const unsigned long long*)&ax_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_1018));
                            float2 _f32x2_mul_b_1019 = make_float2(k2, k2);
                            float2 tw_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_57) : "l"(*(const unsigned long long*)&av_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_1019));
                            float _exp2_1020 = approx_exp2(te_57.x);
                            float _exp2_1021 = approx_exp2(te_57.y);
                            float2 e_57 = make_float2(_exp2_1020, _exp2_1021);
                            float _exp2_1022 = approx_exp2(tw_57.x);
                            float _exp2_1023 = approx_exp2(tw_57.y);
                            float2 w_57 = make_float2(_exp2_1022, _exp2_1023);
                            float2 _f32x2_mul_b_1024 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_57) : "l"(*(const unsigned long long*)&ax_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_1024));
                            float _exp2_1025 = approx_exp2_noftz(tp_57.x);
                            float _exp2_1026 = approx_exp2_noftz(tp_57.y);
                            float2 p_57 = make_float2(_exp2_1025, _exp2_1026);
                            float2 d1_57 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_57);
                            float2 d2_57 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_57);
                            float2 dw_57 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_57);
                            float2 ng_57 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_57);
                            float2 nu_57 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_57);
                            float _rcp_1027 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_57, d2_57), dw_57).x);
                            float _rcp_1028 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_57, d2_57), dw_57).y);
                            float2 r_116 = make_float2(_rcp_1027, _rcp_1028);
                            float2 r_115 = r_116;
                            float2 _f32x2_mul_a_1029 = mul_f32x2_noftz(ng_57, nu_57);
                            float2 _f32x2_mul_b_1030 = mul_f32x2_noftz(r_115, make_float2(bl, bl));
                            float2 m_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1029), "l"(*(const unsigned long long*)&_f32x2_mul_b_1030));
                            float2 ns_57 = make_float2(((x_58.x < 0.0f) ? p_57.x : 1.0f), ((x_58.y < 0.0f) ? p_57.y : 1.0f));
                            gate_packed_1[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_57, ns_57))) | sign_58;
                            float2 x_59 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[27]));
                            float2 v_59 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[27]));
                            unsigned int sign_59 = (gate_packed_1[27] ^ up_packed_1[27]) & 2147516416u;
                            float _abs_1031;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1031) : "f"(x_59.x));
                            float _abs_1032;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1032) : "f"(x_59.y));
                            float2 ax_58 = make_float2(_abs_1031, _abs_1032);
                            float _abs_1033;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1033) : "f"(v_59.x));
                            float _abs_1034;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1034) : "f"(v_59.y));
                            float2 av_58 = make_float2(_abs_1033, _abs_1034);
                            float2 _f32x2_mul_b_1035 = make_float2(k1, k1);
                            float2 te_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_58) : "l"(*(const unsigned long long*)&ax_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_1035));
                            float2 _f32x2_mul_b_1036 = make_float2(k2, k2);
                            float2 tw_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_58) : "l"(*(const unsigned long long*)&av_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_1036));
                            float _exp2_1037 = approx_exp2(te_58.x);
                            float _exp2_1038 = approx_exp2(te_58.y);
                            float2 e_58 = make_float2(_exp2_1037, _exp2_1038);
                            float _exp2_1039 = approx_exp2(tw_58.x);
                            float _exp2_1040 = approx_exp2(tw_58.y);
                            float2 w_58 = make_float2(_exp2_1039, _exp2_1040);
                            float2 _f32x2_mul_b_1041 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_58) : "l"(*(const unsigned long long*)&ax_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_1041));
                            float _exp2_1042 = approx_exp2_noftz(tp_58.x);
                            float _exp2_1043 = approx_exp2_noftz(tp_58.y);
                            float2 p_58 = make_float2(_exp2_1042, _exp2_1043);
                            float2 d1_58 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_58);
                            float2 d2_58 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_58);
                            float2 dw_58 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_58);
                            float2 ng_58 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_58);
                            float2 nu_58 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_58);
                            float _rcp_1044 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_58, d2_58), dw_58).x);
                            float _rcp_1045 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_58, d2_58), dw_58).y);
                            float2 r_118 = make_float2(_rcp_1044, _rcp_1045);
                            float2 r_117 = r_118;
                            float2 _f32x2_mul_a_1046 = mul_f32x2_noftz(ng_58, nu_58);
                            float2 _f32x2_mul_b_1047 = mul_f32x2_noftz(r_117, make_float2(bl, bl));
                            float2 m_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1046), "l"(*(const unsigned long long*)&_f32x2_mul_b_1047));
                            float2 ns_58 = make_float2(((x_59.x < 0.0f) ? p_58.x : 1.0f), ((x_59.y < 0.0f) ? p_58.y : 1.0f));
                            gate_packed_1[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_58, ns_58))) | sign_59;
                            float2 x_60 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[28]));
                            float2 v_60 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[28]));
                            unsigned int sign_60 = (gate_packed_1[28] ^ up_packed_1[28]) & 2147516416u;
                            float _abs_1048;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1048) : "f"(x_60.x));
                            float _abs_1049;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1049) : "f"(x_60.y));
                            float2 ax_59 = make_float2(_abs_1048, _abs_1049);
                            float _abs_1050;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1050) : "f"(v_60.x));
                            float _abs_1051;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1051) : "f"(v_60.y));
                            float2 av_59 = make_float2(_abs_1050, _abs_1051);
                            float2 _f32x2_mul_b_1052 = make_float2(k1, k1);
                            float2 te_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_59) : "l"(*(const unsigned long long*)&ax_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_1052));
                            float2 _f32x2_mul_b_1053 = make_float2(k2, k2);
                            float2 tw_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_59) : "l"(*(const unsigned long long*)&av_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_1053));
                            float _exp2_1054 = approx_exp2(te_59.x);
                            float _exp2_1055 = approx_exp2(te_59.y);
                            float2 e_59 = make_float2(_exp2_1054, _exp2_1055);
                            float _exp2_1056 = approx_exp2(tw_59.x);
                            float _exp2_1057 = approx_exp2(tw_59.y);
                            float2 w_59 = make_float2(_exp2_1056, _exp2_1057);
                            float2 _f32x2_mul_b_1058 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_59) : "l"(*(const unsigned long long*)&ax_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_1058));
                            float _exp2_1059 = approx_exp2_noftz(tp_59.x);
                            float _exp2_1060 = approx_exp2_noftz(tp_59.y);
                            float2 p_59 = make_float2(_exp2_1059, _exp2_1060);
                            float2 d1_59 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_59);
                            float2 d2_59 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_59);
                            float2 dw_59 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_59);
                            float2 ng_59 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_59);
                            float2 nu_59 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_59);
                            float _rcp_1061 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_59, d2_59), dw_59).x);
                            float _rcp_1062 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_59, d2_59), dw_59).y);
                            float2 r_120 = make_float2(_rcp_1061, _rcp_1062);
                            float2 r_119 = r_120;
                            float2 _f32x2_mul_a_1063 = mul_f32x2_noftz(ng_59, nu_59);
                            float2 _f32x2_mul_b_1064 = mul_f32x2_noftz(r_119, make_float2(bl, bl));
                            float2 m_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1063), "l"(*(const unsigned long long*)&_f32x2_mul_b_1064));
                            float2 ns_59 = make_float2(((x_60.x < 0.0f) ? p_59.x : 1.0f), ((x_60.y < 0.0f) ? p_59.y : 1.0f));
                            gate_packed_1[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_59, ns_59))) | sign_60;
                            float2 x_61 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[29]));
                            float2 v_61 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[29]));
                            unsigned int sign_61 = (gate_packed_1[29] ^ up_packed_1[29]) & 2147516416u;
                            float _abs_1065;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1065) : "f"(x_61.x));
                            float _abs_1066;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1066) : "f"(x_61.y));
                            float2 ax_60 = make_float2(_abs_1065, _abs_1066);
                            float _abs_1067;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1067) : "f"(v_61.x));
                            float _abs_1068;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1068) : "f"(v_61.y));
                            float2 av_60 = make_float2(_abs_1067, _abs_1068);
                            float2 _f32x2_mul_b_1069 = make_float2(k1, k1);
                            float2 te_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_60) : "l"(*(const unsigned long long*)&ax_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_1069));
                            float2 _f32x2_mul_b_1070 = make_float2(k2, k2);
                            float2 tw_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_60) : "l"(*(const unsigned long long*)&av_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_1070));
                            float _exp2_1071 = approx_exp2(te_60.x);
                            float _exp2_1072 = approx_exp2(te_60.y);
                            float2 e_60 = make_float2(_exp2_1071, _exp2_1072);
                            float _exp2_1073 = approx_exp2(tw_60.x);
                            float _exp2_1074 = approx_exp2(tw_60.y);
                            float2 w_60 = make_float2(_exp2_1073, _exp2_1074);
                            float2 _f32x2_mul_b_1075 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_60) : "l"(*(const unsigned long long*)&ax_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_1075));
                            float _exp2_1076 = approx_exp2_noftz(tp_60.x);
                            float _exp2_1077 = approx_exp2_noftz(tp_60.y);
                            float2 p_60 = make_float2(_exp2_1076, _exp2_1077);
                            float2 d1_60 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_60);
                            float2 d2_60 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_60);
                            float2 dw_60 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_60);
                            float2 ng_60 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_60);
                            float2 nu_60 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_60);
                            float _rcp_1078 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_60, d2_60), dw_60).x);
                            float _rcp_1079 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_60, d2_60), dw_60).y);
                            float2 r_122 = make_float2(_rcp_1078, _rcp_1079);
                            float2 r_121 = r_122;
                            float2 _f32x2_mul_a_1080 = mul_f32x2_noftz(ng_60, nu_60);
                            float2 _f32x2_mul_b_1081 = mul_f32x2_noftz(r_121, make_float2(bl, bl));
                            float2 m_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1080), "l"(*(const unsigned long long*)&_f32x2_mul_b_1081));
                            float2 ns_60 = make_float2(((x_61.x < 0.0f) ? p_60.x : 1.0f), ((x_61.y < 0.0f) ? p_60.y : 1.0f));
                            gate_packed_1[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_60, ns_60))) | sign_61;
                            float2 x_62 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[30]));
                            float2 v_62 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[30]));
                            unsigned int sign_62 = (gate_packed_1[30] ^ up_packed_1[30]) & 2147516416u;
                            float _abs_1082;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1082) : "f"(x_62.x));
                            float _abs_1083;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1083) : "f"(x_62.y));
                            float2 ax_61 = make_float2(_abs_1082, _abs_1083);
                            float _abs_1084;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1084) : "f"(v_62.x));
                            float _abs_1085;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1085) : "f"(v_62.y));
                            float2 av_61 = make_float2(_abs_1084, _abs_1085);
                            float2 _f32x2_mul_b_1086 = make_float2(k1, k1);
                            float2 te_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_61) : "l"(*(const unsigned long long*)&ax_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_1086));
                            float2 _f32x2_mul_b_1087 = make_float2(k2, k2);
                            float2 tw_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_61) : "l"(*(const unsigned long long*)&av_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_1087));
                            float _exp2_1088 = approx_exp2(te_61.x);
                            float _exp2_1089 = approx_exp2(te_61.y);
                            float2 e_61 = make_float2(_exp2_1088, _exp2_1089);
                            float _exp2_1090 = approx_exp2(tw_61.x);
                            float _exp2_1091 = approx_exp2(tw_61.y);
                            float2 w_61 = make_float2(_exp2_1090, _exp2_1091);
                            float2 _f32x2_mul_b_1092 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_61) : "l"(*(const unsigned long long*)&ax_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_1092));
                            float _exp2_1093 = approx_exp2_noftz(tp_61.x);
                            float _exp2_1094 = approx_exp2_noftz(tp_61.y);
                            float2 p_61 = make_float2(_exp2_1093, _exp2_1094);
                            float2 d1_61 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_61);
                            float2 d2_61 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_61);
                            float2 dw_61 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_61);
                            float2 ng_61 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_61);
                            float2 nu_61 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_61);
                            float _rcp_1095 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_61, d2_61), dw_61).x);
                            float _rcp_1096 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_61, d2_61), dw_61).y);
                            float2 r_124 = make_float2(_rcp_1095, _rcp_1096);
                            float2 r_123 = r_124;
                            float2 _f32x2_mul_a_1097 = mul_f32x2_noftz(ng_61, nu_61);
                            float2 _f32x2_mul_b_1098 = mul_f32x2_noftz(r_123, make_float2(bl, bl));
                            float2 m_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1097), "l"(*(const unsigned long long*)&_f32x2_mul_b_1098));
                            float2 ns_61 = make_float2(((x_62.x < 0.0f) ? p_61.x : 1.0f), ((x_62.y < 0.0f) ? p_61.y : 1.0f));
                            gate_packed_1[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_61, ns_61))) | sign_62;
                            float2 x_63 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[31]));
                            float2 v_63 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[31]));
                            unsigned int sign_63 = (gate_packed_1[31] ^ up_packed_1[31]) & 2147516416u;
                            float _abs_1099;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1099) : "f"(x_63.x));
                            float _abs_1100;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1100) : "f"(x_63.y));
                            float2 ax_62 = make_float2(_abs_1099, _abs_1100);
                            float _abs_1101;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1101) : "f"(v_63.x));
                            float _abs_1102;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1102) : "f"(v_63.y));
                            float2 av_62 = make_float2(_abs_1101, _abs_1102);
                            float2 _f32x2_mul_b_1103 = make_float2(k1, k1);
                            float2 te_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_62) : "l"(*(const unsigned long long*)&ax_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_1103));
                            float2 _f32x2_mul_b_1104 = make_float2(k2, k2);
                            float2 tw_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_62) : "l"(*(const unsigned long long*)&av_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_1104));
                            float _exp2_1105 = approx_exp2(te_62.x);
                            float _exp2_1106 = approx_exp2(te_62.y);
                            float2 e_62 = make_float2(_exp2_1105, _exp2_1106);
                            float _exp2_1107 = approx_exp2(tw_62.x);
                            float _exp2_1108 = approx_exp2(tw_62.y);
                            float2 w_62 = make_float2(_exp2_1107, _exp2_1108);
                            float2 _f32x2_mul_b_1109 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_62) : "l"(*(const unsigned long long*)&ax_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_1109));
                            float _exp2_1110 = approx_exp2_noftz(tp_62.x);
                            float _exp2_1111 = approx_exp2_noftz(tp_62.y);
                            float2 p_62 = make_float2(_exp2_1110, _exp2_1111);
                            float2 d1_62 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_62);
                            float2 d2_62 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_62);
                            float2 dw_62 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_62);
                            float2 ng_62 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_62);
                            float2 nu_62 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_62);
                            float _rcp_1112 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_62, d2_62), dw_62).x);
                            float _rcp_1113 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_62, d2_62), dw_62).y);
                            float2 r_126 = make_float2(_rcp_1112, _rcp_1113);
                            float2 r_125 = r_126;
                            float2 _f32x2_mul_a_1114 = mul_f32x2_noftz(ng_62, nu_62);
                            float2 _f32x2_mul_b_1115 = mul_f32x2_noftz(r_125, make_float2(bl, bl));
                            float2 m_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1114), "l"(*(const unsigned long long*)&_f32x2_mul_b_1115));
                            float2 ns_62 = make_float2(((x_63.x < 0.0f) ? p_62.x : 1.0f), ((x_63.y < 0.0f) ? p_62.y : 1.0f));
                            gate_packed_1[31] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_62, ns_62))) | sign_63;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_10 = tid / 32;
                            int lane_10 = tid % 32;
                            #pragma unroll
                            for (int half_11 = 0; half_11 < 2; half_11++) {
                                #pragma unroll
                                for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                    int row_12 = warp_10 * 32 + half_11 * 16 + lane_10 % 16;
                                    int col_12 = col_tile_11 * 16 + lane_10 / 16 * 8;
                                    unsigned int address_11 = d_smem_addr + 16384 + (unsigned int)((row_12 * 32 + col_12) * 2);
                                    address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                    int offset_12 = 16 + half_11 * 8 + col_tile_11 * 4;
                                    uint32_t _stmatrix_addr_1116 = static_cast<uint32_t>(address_11);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1116), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            int _parity_output_arrived_1 = (int)((unsigned int)gemm_out1 % 2);
                            mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_1);
                            unsigned int gate_packed_2[32];
                            unsigned int up_packed_2[32];
                            unsigned int address_3_8 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 256;
                            float _tmem_load_8[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                                : "r"(address_3_8));
                            float _tmem_load_32_8[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[15]))
                                : "r"(address_3_8 + 128));
                            gate_packed_2[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[0], _tmem_load_8[1])));
                            up_packed_2[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[0], _tmem_load_32_8[1])));
                            gate_packed_2[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[2], _tmem_load_8[3])));
                            up_packed_2[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[2], _tmem_load_32_8[3])));
                            gate_packed_2[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[4], _tmem_load_8[5])));
                            up_packed_2[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[4], _tmem_load_32_8[5])));
                            gate_packed_2[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[6], _tmem_load_8[7])));
                            up_packed_2[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[6], _tmem_load_32_8[7])));
                            gate_packed_2[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[8], _tmem_load_8[9])));
                            up_packed_2[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[8], _tmem_load_32_8[9])));
                            gate_packed_2[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[10], _tmem_load_8[11])));
                            up_packed_2[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[10], _tmem_load_32_8[11])));
                            gate_packed_2[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[12], _tmem_load_8[13])));
                            up_packed_2[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[12], _tmem_load_32_8[13])));
                            gate_packed_2[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[14], _tmem_load_8[15])));
                            up_packed_2[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[14], _tmem_load_32_8[15])));
                            unsigned int address_3_9 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 256;
                            float _tmem_load_9[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                                : "r"(address_3_9));
                            float _tmem_load_32_9[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[15]))
                                : "r"(address_3_9 + 128));
                            gate_packed_2[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[0], _tmem_load_9[1])));
                            up_packed_2[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[0], _tmem_load_32_9[1])));
                            gate_packed_2[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[2], _tmem_load_9[3])));
                            up_packed_2[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[2], _tmem_load_32_9[3])));
                            gate_packed_2[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[4], _tmem_load_9[5])));
                            up_packed_2[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[4], _tmem_load_32_9[5])));
                            gate_packed_2[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[6], _tmem_load_9[7])));
                            up_packed_2[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[6], _tmem_load_32_9[7])));
                            gate_packed_2[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[8], _tmem_load_9[9])));
                            up_packed_2[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[8], _tmem_load_32_9[9])));
                            gate_packed_2[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[10], _tmem_load_9[11])));
                            up_packed_2[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[10], _tmem_load_32_9[11])));
                            gate_packed_2[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[12], _tmem_load_9[13])));
                            up_packed_2[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[12], _tmem_load_32_9[13])));
                            gate_packed_2[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[14], _tmem_load_9[15])));
                            up_packed_2[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[14], _tmem_load_32_9[15])));
                            unsigned int address_3_10 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 288;
                            float _tmem_load_10[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                                : "r"(address_3_10));
                            float _tmem_load_32_10[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[15]))
                                : "r"(address_3_10 + 128));
                            gate_packed_2[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[0], _tmem_load_10[1])));
                            up_packed_2[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[0], _tmem_load_32_10[1])));
                            gate_packed_2[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[2], _tmem_load_10[3])));
                            up_packed_2[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[2], _tmem_load_32_10[3])));
                            gate_packed_2[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[4], _tmem_load_10[5])));
                            up_packed_2[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[4], _tmem_load_32_10[5])));
                            gate_packed_2[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[6], _tmem_load_10[7])));
                            up_packed_2[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[6], _tmem_load_32_10[7])));
                            gate_packed_2[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[8], _tmem_load_10[9])));
                            up_packed_2[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[8], _tmem_load_32_10[9])));
                            gate_packed_2[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[10], _tmem_load_10[11])));
                            up_packed_2[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[10], _tmem_load_32_10[11])));
                            gate_packed_2[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[12], _tmem_load_10[13])));
                            up_packed_2[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[12], _tmem_load_32_10[13])));
                            gate_packed_2[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[14], _tmem_load_10[15])));
                            up_packed_2[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[14], _tmem_load_32_10[15])));
                            unsigned int address_3_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 288;
                            float _tmem_load_11[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                                : "r"(address_3_11));
                            float _tmem_load_32_11[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[15]))
                                : "r"(address_3_11 + 128));
                            gate_packed_2[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[0], _tmem_load_11[1])));
                            up_packed_2[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[0], _tmem_load_32_11[1])));
                            gate_packed_2[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[2], _tmem_load_11[3])));
                            up_packed_2[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[2], _tmem_load_32_11[3])));
                            gate_packed_2[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[4], _tmem_load_11[5])));
                            up_packed_2[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[4], _tmem_load_32_11[5])));
                            gate_packed_2[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[6], _tmem_load_11[7])));
                            up_packed_2[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[6], _tmem_load_32_11[7])));
                            gate_packed_2[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[8], _tmem_load_11[9])));
                            up_packed_2[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[8], _tmem_load_32_11[9])));
                            gate_packed_2[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[10], _tmem_load_11[11])));
                            up_packed_2[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[10], _tmem_load_32_11[11])));
                            gate_packed_2[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[12], _tmem_load_11[13])));
                            up_packed_2[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[12], _tmem_load_32_11[13])));
                            gate_packed_2[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[14], _tmem_load_11[15])));
                            up_packed_2[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[14], _tmem_load_32_11[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_11 = tid / 32;
                            int lane_11 = tid % 32;
                            #pragma unroll
                            for (int half_12 = 0; half_12 < 2; half_12++) {
                                #pragma unroll
                                for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                    int row_13 = warp_11 * 32 + half_12 * 16 + lane_11 % 16;
                                    int col_13 = col_tile_12 * 16 + lane_11 / 16 * 8;
                                    unsigned int address_12 = d_smem_addr + (unsigned int)((row_13 * 32 + col_13) * 2);
                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                    int offset_13 = half_12 * 8 + col_tile_12 * 4;
                                    uint32_t _stmatrix_addr_1117 = static_cast<uint32_t>(address_12);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1117), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_12 = tid / 32;
                            int lane_12 = tid % 32;
                            #pragma unroll
                            for (int half_13 = 0; half_13 < 2; half_13++) {
                                #pragma unroll
                                for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                    int row_14 = warp_12 * 32 + half_13 * 16 + lane_12 % 16;
                                    int col_14 = col_tile_13 * 16 + lane_12 / 16 * 8;
                                    unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)((row_14 * 32 + col_14) * 2);
                                    address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                    int offset_14 = half_13 * 8 + col_tile_13 * 4;
                                    uint32_t _stmatrix_addr_1118 = static_cast<uint32_t>(address_13);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1118), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_13 = tid / 32;
                            int lane_13 = tid % 32;
                            #pragma unroll
                            for (int half_14 = 0; half_14 < 2; half_14++) {
                                #pragma unroll
                                for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                    int row_15 = warp_13 * 32 + half_14 * 16 + lane_13 % 16;
                                    int col_15 = col_tile_14 * 16 + lane_13 / 16 * 8;
                                    unsigned int address_14 = d_smem_addr + 16384 + (unsigned int)((row_15 * 32 + col_15) * 2);
                                    address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                    int offset_15 = 16 + half_14 * 8 + col_tile_14 * 4;
                                    uint32_t _stmatrix_addr_1119 = static_cast<uint32_t>(address_14);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1119), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_14 = tid / 32;
                            int lane_14 = tid % 32;
                            #pragma unroll
                            for (int half_15 = 0; half_15 < 2; half_15++) {
                                #pragma unroll
                                for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                    int row_16 = warp_14 * 32 + half_15 * 16 + lane_14 % 16;
                                    int col_16 = col_tile_15 * 16 + lane_14 / 16 * 8;
                                    unsigned int address_15 = d_smem_addr + (unsigned int)((row_16 * 32 + col_16) * 2);
                                    address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                    int offset_16 = 16 + half_15 * 8 + col_tile_15 * 4;
                                    uint32_t _stmatrix_addr_1120 = static_cast<uint32_t>(address_15);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1120), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_64 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[0]));
                            float2 v_64 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[0]));
                            unsigned int sign_64 = (gate_packed_2[0] ^ up_packed_2[0]) & 2147516416u;
                            float _abs_1121;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1121) : "f"(x_64.x));
                            float _abs_1122;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1122) : "f"(x_64.y));
                            float2 ax_63 = make_float2(_abs_1121, _abs_1122);
                            float _abs_1123;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1123) : "f"(v_64.x));
                            float _abs_1124;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1124) : "f"(v_64.y));
                            float2 av_63 = make_float2(_abs_1123, _abs_1124);
                            float2 _f32x2_mul_b_1125 = make_float2(k1, k1);
                            float2 te_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_63) : "l"(*(const unsigned long long*)&ax_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_1125));
                            float2 _f32x2_mul_b_1126 = make_float2(k2, k2);
                            float2 tw_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_63) : "l"(*(const unsigned long long*)&av_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_1126));
                            float _exp2_1127 = approx_exp2(te_63.x);
                            float _exp2_1128 = approx_exp2(te_63.y);
                            float2 e_63 = make_float2(_exp2_1127, _exp2_1128);
                            float _exp2_1129 = approx_exp2(tw_63.x);
                            float _exp2_1130 = approx_exp2(tw_63.y);
                            float2 w_63 = make_float2(_exp2_1129, _exp2_1130);
                            float2 _f32x2_mul_b_1131 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_63) : "l"(*(const unsigned long long*)&ax_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_1131));
                            float _exp2_1132 = approx_exp2_noftz(tp_63.x);
                            float _exp2_1133 = approx_exp2_noftz(tp_63.y);
                            float2 p_63 = make_float2(_exp2_1132, _exp2_1133);
                            float2 d1_63 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_63);
                            float2 d2_63 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_63);
                            float2 dw_63 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_63);
                            float2 ng_63 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_63);
                            float2 nu_63 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_63);
                            float _rcp_1134 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_63, d2_63), dw_63).x);
                            float _rcp_1135 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_63, d2_63), dw_63).y);
                            float2 r_128 = make_float2(_rcp_1134, _rcp_1135);
                            float2 r_127 = r_128;
                            float2 _f32x2_mul_a_1136 = mul_f32x2_noftz(ng_63, nu_63);
                            float2 _f32x2_mul_b_1137 = mul_f32x2_noftz(r_127, make_float2(bl, bl));
                            float2 m_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1136), "l"(*(const unsigned long long*)&_f32x2_mul_b_1137));
                            float2 ns_63 = make_float2(((x_64.x < 0.0f) ? p_63.x : 1.0f), ((x_64.y < 0.0f) ? p_63.y : 1.0f));
                            gate_packed_2[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_63, ns_63))) | sign_64;
                            float2 x_65 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[1]));
                            float2 v_65 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[1]));
                            unsigned int sign_65 = (gate_packed_2[1] ^ up_packed_2[1]) & 2147516416u;
                            float _abs_1138;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1138) : "f"(x_65.x));
                            float _abs_1139;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1139) : "f"(x_65.y));
                            float2 ax_64 = make_float2(_abs_1138, _abs_1139);
                            float _abs_1140;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1140) : "f"(v_65.x));
                            float _abs_1141;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1141) : "f"(v_65.y));
                            float2 av_64 = make_float2(_abs_1140, _abs_1141);
                            float2 _f32x2_mul_b_1142 = make_float2(k1, k1);
                            float2 te_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_64) : "l"(*(const unsigned long long*)&ax_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_1142));
                            float2 _f32x2_mul_b_1143 = make_float2(k2, k2);
                            float2 tw_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_64) : "l"(*(const unsigned long long*)&av_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_1143));
                            float _exp2_1144 = approx_exp2(te_64.x);
                            float _exp2_1145 = approx_exp2(te_64.y);
                            float2 e_64 = make_float2(_exp2_1144, _exp2_1145);
                            float _exp2_1146 = approx_exp2(tw_64.x);
                            float _exp2_1147 = approx_exp2(tw_64.y);
                            float2 w_64 = make_float2(_exp2_1146, _exp2_1147);
                            float2 _f32x2_mul_b_1148 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_64) : "l"(*(const unsigned long long*)&ax_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_1148));
                            float _exp2_1149 = approx_exp2_noftz(tp_64.x);
                            float _exp2_1150 = approx_exp2_noftz(tp_64.y);
                            float2 p_64 = make_float2(_exp2_1149, _exp2_1150);
                            float2 d1_64 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_64);
                            float2 d2_64 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_64);
                            float2 dw_64 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_64);
                            float2 ng_64 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_64);
                            float2 nu_64 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_64);
                            float _rcp_1151 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_64, d2_64), dw_64).x);
                            float _rcp_1152 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_64, d2_64), dw_64).y);
                            float2 r_130 = make_float2(_rcp_1151, _rcp_1152);
                            float2 r_129 = r_130;
                            float2 _f32x2_mul_a_1153 = mul_f32x2_noftz(ng_64, nu_64);
                            float2 _f32x2_mul_b_1154 = mul_f32x2_noftz(r_129, make_float2(bl, bl));
                            float2 m_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1153), "l"(*(const unsigned long long*)&_f32x2_mul_b_1154));
                            float2 ns_64 = make_float2(((x_65.x < 0.0f) ? p_64.x : 1.0f), ((x_65.y < 0.0f) ? p_64.y : 1.0f));
                            gate_packed_2[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_64, ns_64))) | sign_65;
                            float2 x_66 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[2]));
                            float2 v_66 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[2]));
                            unsigned int sign_66 = (gate_packed_2[2] ^ up_packed_2[2]) & 2147516416u;
                            float _abs_1155;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1155) : "f"(x_66.x));
                            float _abs_1156;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1156) : "f"(x_66.y));
                            float2 ax_65 = make_float2(_abs_1155, _abs_1156);
                            float _abs_1157;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1157) : "f"(v_66.x));
                            float _abs_1158;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1158) : "f"(v_66.y));
                            float2 av_65 = make_float2(_abs_1157, _abs_1158);
                            float2 _f32x2_mul_b_1159 = make_float2(k1, k1);
                            float2 te_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_65) : "l"(*(const unsigned long long*)&ax_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_1159));
                            float2 _f32x2_mul_b_1160 = make_float2(k2, k2);
                            float2 tw_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_65) : "l"(*(const unsigned long long*)&av_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_1160));
                            float _exp2_1161 = approx_exp2(te_65.x);
                            float _exp2_1162 = approx_exp2(te_65.y);
                            float2 e_65 = make_float2(_exp2_1161, _exp2_1162);
                            float _exp2_1163 = approx_exp2(tw_65.x);
                            float _exp2_1164 = approx_exp2(tw_65.y);
                            float2 w_65 = make_float2(_exp2_1163, _exp2_1164);
                            float2 _f32x2_mul_b_1165 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_65) : "l"(*(const unsigned long long*)&ax_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_1165));
                            float _exp2_1166 = approx_exp2_noftz(tp_65.x);
                            float _exp2_1167 = approx_exp2_noftz(tp_65.y);
                            float2 p_65 = make_float2(_exp2_1166, _exp2_1167);
                            float2 d1_65 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_65);
                            float2 d2_65 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_65);
                            float2 dw_65 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_65);
                            float2 ng_65 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_65);
                            float2 nu_65 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_65);
                            float _rcp_1168 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_65, d2_65), dw_65).x);
                            float _rcp_1169 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_65, d2_65), dw_65).y);
                            float2 r_132 = make_float2(_rcp_1168, _rcp_1169);
                            float2 r_131 = r_132;
                            float2 _f32x2_mul_a_1170 = mul_f32x2_noftz(ng_65, nu_65);
                            float2 _f32x2_mul_b_1171 = mul_f32x2_noftz(r_131, make_float2(bl, bl));
                            float2 m_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1170), "l"(*(const unsigned long long*)&_f32x2_mul_b_1171));
                            float2 ns_65 = make_float2(((x_66.x < 0.0f) ? p_65.x : 1.0f), ((x_66.y < 0.0f) ? p_65.y : 1.0f));
                            gate_packed_2[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_65, ns_65))) | sign_66;
                            float2 x_67 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[3]));
                            float2 v_67 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[3]));
                            unsigned int sign_67 = (gate_packed_2[3] ^ up_packed_2[3]) & 2147516416u;
                            float _abs_1172;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1172) : "f"(x_67.x));
                            float _abs_1173;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1173) : "f"(x_67.y));
                            float2 ax_66 = make_float2(_abs_1172, _abs_1173);
                            float _abs_1174;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1174) : "f"(v_67.x));
                            float _abs_1175;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1175) : "f"(v_67.y));
                            float2 av_66 = make_float2(_abs_1174, _abs_1175);
                            float2 _f32x2_mul_b_1176 = make_float2(k1, k1);
                            float2 te_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_66) : "l"(*(const unsigned long long*)&ax_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_1176));
                            float2 _f32x2_mul_b_1177 = make_float2(k2, k2);
                            float2 tw_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_66) : "l"(*(const unsigned long long*)&av_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_1177));
                            float _exp2_1178 = approx_exp2(te_66.x);
                            float _exp2_1179 = approx_exp2(te_66.y);
                            float2 e_66 = make_float2(_exp2_1178, _exp2_1179);
                            float _exp2_1180 = approx_exp2(tw_66.x);
                            float _exp2_1181 = approx_exp2(tw_66.y);
                            float2 w_66 = make_float2(_exp2_1180, _exp2_1181);
                            float2 _f32x2_mul_b_1182 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_66) : "l"(*(const unsigned long long*)&ax_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_1182));
                            float _exp2_1183 = approx_exp2_noftz(tp_66.x);
                            float _exp2_1184 = approx_exp2_noftz(tp_66.y);
                            float2 p_66 = make_float2(_exp2_1183, _exp2_1184);
                            float2 d1_66 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_66);
                            float2 d2_66 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_66);
                            float2 dw_66 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_66);
                            float2 ng_66 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_66);
                            float2 nu_66 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_66);
                            float _rcp_1185 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_66, d2_66), dw_66).x);
                            float _rcp_1186 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_66, d2_66), dw_66).y);
                            float2 r_134 = make_float2(_rcp_1185, _rcp_1186);
                            float2 r_133 = r_134;
                            float2 _f32x2_mul_a_1187 = mul_f32x2_noftz(ng_66, nu_66);
                            float2 _f32x2_mul_b_1188 = mul_f32x2_noftz(r_133, make_float2(bl, bl));
                            float2 m_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1187), "l"(*(const unsigned long long*)&_f32x2_mul_b_1188));
                            float2 ns_66 = make_float2(((x_67.x < 0.0f) ? p_66.x : 1.0f), ((x_67.y < 0.0f) ? p_66.y : 1.0f));
                            gate_packed_2[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_66, ns_66))) | sign_67;
                            float2 x_68 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[4]));
                            float2 v_68 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[4]));
                            unsigned int sign_68 = (gate_packed_2[4] ^ up_packed_2[4]) & 2147516416u;
                            float _abs_1189;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1189) : "f"(x_68.x));
                            float _abs_1190;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1190) : "f"(x_68.y));
                            float2 ax_67 = make_float2(_abs_1189, _abs_1190);
                            float _abs_1191;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1191) : "f"(v_68.x));
                            float _abs_1192;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1192) : "f"(v_68.y));
                            float2 av_67 = make_float2(_abs_1191, _abs_1192);
                            float2 _f32x2_mul_b_1193 = make_float2(k1, k1);
                            float2 te_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_67) : "l"(*(const unsigned long long*)&ax_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_1193));
                            float2 _f32x2_mul_b_1194 = make_float2(k2, k2);
                            float2 tw_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_67) : "l"(*(const unsigned long long*)&av_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_1194));
                            float _exp2_1195 = approx_exp2(te_67.x);
                            float _exp2_1196 = approx_exp2(te_67.y);
                            float2 e_67 = make_float2(_exp2_1195, _exp2_1196);
                            float _exp2_1197 = approx_exp2(tw_67.x);
                            float _exp2_1198 = approx_exp2(tw_67.y);
                            float2 w_67 = make_float2(_exp2_1197, _exp2_1198);
                            float2 _f32x2_mul_b_1199 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_67) : "l"(*(const unsigned long long*)&ax_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_1199));
                            float _exp2_1200 = approx_exp2_noftz(tp_67.x);
                            float _exp2_1201 = approx_exp2_noftz(tp_67.y);
                            float2 p_67 = make_float2(_exp2_1200, _exp2_1201);
                            float2 d1_67 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_67);
                            float2 d2_67 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_67);
                            float2 dw_67 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_67);
                            float2 ng_67 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_67);
                            float2 nu_67 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_67);
                            float _rcp_1202 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_67, d2_67), dw_67).x);
                            float _rcp_1203 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_67, d2_67), dw_67).y);
                            float2 r_136 = make_float2(_rcp_1202, _rcp_1203);
                            float2 r_135 = r_136;
                            float2 _f32x2_mul_a_1204 = mul_f32x2_noftz(ng_67, nu_67);
                            float2 _f32x2_mul_b_1205 = mul_f32x2_noftz(r_135, make_float2(bl, bl));
                            float2 m_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_67) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1204), "l"(*(const unsigned long long*)&_f32x2_mul_b_1205));
                            float2 ns_67 = make_float2(((x_68.x < 0.0f) ? p_67.x : 1.0f), ((x_68.y < 0.0f) ? p_67.y : 1.0f));
                            gate_packed_2[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_67, ns_67))) | sign_68;
                            float2 x_69 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[5]));
                            float2 v_69 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[5]));
                            unsigned int sign_69 = (gate_packed_2[5] ^ up_packed_2[5]) & 2147516416u;
                            float _abs_1206;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1206) : "f"(x_69.x));
                            float _abs_1207;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1207) : "f"(x_69.y));
                            float2 ax_68 = make_float2(_abs_1206, _abs_1207);
                            float _abs_1208;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1208) : "f"(v_69.x));
                            float _abs_1209;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1209) : "f"(v_69.y));
                            float2 av_68 = make_float2(_abs_1208, _abs_1209);
                            float2 _f32x2_mul_b_1210 = make_float2(k1, k1);
                            float2 te_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_68) : "l"(*(const unsigned long long*)&ax_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_1210));
                            float2 _f32x2_mul_b_1211 = make_float2(k2, k2);
                            float2 tw_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_68) : "l"(*(const unsigned long long*)&av_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_1211));
                            float _exp2_1212 = approx_exp2(te_68.x);
                            float _exp2_1213 = approx_exp2(te_68.y);
                            float2 e_68 = make_float2(_exp2_1212, _exp2_1213);
                            float _exp2_1214 = approx_exp2(tw_68.x);
                            float _exp2_1215 = approx_exp2(tw_68.y);
                            float2 w_68 = make_float2(_exp2_1214, _exp2_1215);
                            float2 _f32x2_mul_b_1216 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_68) : "l"(*(const unsigned long long*)&ax_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_1216));
                            float _exp2_1217 = approx_exp2_noftz(tp_68.x);
                            float _exp2_1218 = approx_exp2_noftz(tp_68.y);
                            float2 p_68 = make_float2(_exp2_1217, _exp2_1218);
                            float2 d1_68 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_68);
                            float2 d2_68 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_68);
                            float2 dw_68 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_68);
                            float2 ng_68 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_68);
                            float2 nu_68 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_68);
                            float _rcp_1219 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_68, d2_68), dw_68).x);
                            float _rcp_1220 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_68, d2_68), dw_68).y);
                            float2 r_138 = make_float2(_rcp_1219, _rcp_1220);
                            float2 r_137 = r_138;
                            float2 _f32x2_mul_a_1221 = mul_f32x2_noftz(ng_68, nu_68);
                            float2 _f32x2_mul_b_1222 = mul_f32x2_noftz(r_137, make_float2(bl, bl));
                            float2 m_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_68) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1221), "l"(*(const unsigned long long*)&_f32x2_mul_b_1222));
                            float2 ns_68 = make_float2(((x_69.x < 0.0f) ? p_68.x : 1.0f), ((x_69.y < 0.0f) ? p_68.y : 1.0f));
                            gate_packed_2[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_68, ns_68))) | sign_69;
                            float2 x_70 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[6]));
                            float2 v_70 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[6]));
                            unsigned int sign_70 = (gate_packed_2[6] ^ up_packed_2[6]) & 2147516416u;
                            float _abs_1223;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1223) : "f"(x_70.x));
                            float _abs_1224;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1224) : "f"(x_70.y));
                            float2 ax_69 = make_float2(_abs_1223, _abs_1224);
                            float _abs_1225;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1225) : "f"(v_70.x));
                            float _abs_1226;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1226) : "f"(v_70.y));
                            float2 av_69 = make_float2(_abs_1225, _abs_1226);
                            float2 _f32x2_mul_b_1227 = make_float2(k1, k1);
                            float2 te_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_69) : "l"(*(const unsigned long long*)&ax_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_1227));
                            float2 _f32x2_mul_b_1228 = make_float2(k2, k2);
                            float2 tw_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_69) : "l"(*(const unsigned long long*)&av_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_1228));
                            float _exp2_1229 = approx_exp2(te_69.x);
                            float _exp2_1230 = approx_exp2(te_69.y);
                            float2 e_69 = make_float2(_exp2_1229, _exp2_1230);
                            float _exp2_1231 = approx_exp2(tw_69.x);
                            float _exp2_1232 = approx_exp2(tw_69.y);
                            float2 w_69 = make_float2(_exp2_1231, _exp2_1232);
                            float2 _f32x2_mul_b_1233 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_69) : "l"(*(const unsigned long long*)&ax_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_1233));
                            float _exp2_1234 = approx_exp2_noftz(tp_69.x);
                            float _exp2_1235 = approx_exp2_noftz(tp_69.y);
                            float2 p_69 = make_float2(_exp2_1234, _exp2_1235);
                            float2 d1_69 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_69);
                            float2 d2_69 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_69);
                            float2 dw_69 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_69);
                            float2 ng_69 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_69);
                            float2 nu_69 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_69);
                            float _rcp_1236 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_69, d2_69), dw_69).x);
                            float _rcp_1237 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_69, d2_69), dw_69).y);
                            float2 r_140 = make_float2(_rcp_1236, _rcp_1237);
                            float2 r_139 = r_140;
                            float2 _f32x2_mul_a_1238 = mul_f32x2_noftz(ng_69, nu_69);
                            float2 _f32x2_mul_b_1239 = mul_f32x2_noftz(r_139, make_float2(bl, bl));
                            float2 m_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_69) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1238), "l"(*(const unsigned long long*)&_f32x2_mul_b_1239));
                            float2 ns_69 = make_float2(((x_70.x < 0.0f) ? p_69.x : 1.0f), ((x_70.y < 0.0f) ? p_69.y : 1.0f));
                            gate_packed_2[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_69, ns_69))) | sign_70;
                            float2 x_71 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[7]));
                            float2 v_71 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[7]));
                            unsigned int sign_71 = (gate_packed_2[7] ^ up_packed_2[7]) & 2147516416u;
                            float _abs_1240;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1240) : "f"(x_71.x));
                            float _abs_1241;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1241) : "f"(x_71.y));
                            float2 ax_70 = make_float2(_abs_1240, _abs_1241);
                            float _abs_1242;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1242) : "f"(v_71.x));
                            float _abs_1243;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1243) : "f"(v_71.y));
                            float2 av_70 = make_float2(_abs_1242, _abs_1243);
                            float2 _f32x2_mul_b_1244 = make_float2(k1, k1);
                            float2 te_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_70) : "l"(*(const unsigned long long*)&ax_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_1244));
                            float2 _f32x2_mul_b_1245 = make_float2(k2, k2);
                            float2 tw_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_70) : "l"(*(const unsigned long long*)&av_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_1245));
                            float _exp2_1246 = approx_exp2(te_70.x);
                            float _exp2_1247 = approx_exp2(te_70.y);
                            float2 e_70 = make_float2(_exp2_1246, _exp2_1247);
                            float _exp2_1248 = approx_exp2(tw_70.x);
                            float _exp2_1249 = approx_exp2(tw_70.y);
                            float2 w_70 = make_float2(_exp2_1248, _exp2_1249);
                            float2 _f32x2_mul_b_1250 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_70) : "l"(*(const unsigned long long*)&ax_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_1250));
                            float _exp2_1251 = approx_exp2_noftz(tp_70.x);
                            float _exp2_1252 = approx_exp2_noftz(tp_70.y);
                            float2 p_70 = make_float2(_exp2_1251, _exp2_1252);
                            float2 d1_70 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_70);
                            float2 d2_70 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_70);
                            float2 dw_70 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_70);
                            float2 ng_70 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_70);
                            float2 nu_70 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_70);
                            float _rcp_1253 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_70, d2_70), dw_70).x);
                            float _rcp_1254 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_70, d2_70), dw_70).y);
                            float2 r_142 = make_float2(_rcp_1253, _rcp_1254);
                            float2 r_141 = r_142;
                            float2 _f32x2_mul_a_1255 = mul_f32x2_noftz(ng_70, nu_70);
                            float2 _f32x2_mul_b_1256 = mul_f32x2_noftz(r_141, make_float2(bl, bl));
                            float2 m_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_70) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1255), "l"(*(const unsigned long long*)&_f32x2_mul_b_1256));
                            float2 ns_70 = make_float2(((x_71.x < 0.0f) ? p_70.x : 1.0f), ((x_71.y < 0.0f) ? p_70.y : 1.0f));
                            gate_packed_2[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_70, ns_70))) | sign_71;
                            float2 x_72 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[8]));
                            float2 v_72 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[8]));
                            unsigned int sign_72 = (gate_packed_2[8] ^ up_packed_2[8]) & 2147516416u;
                            float _abs_1257;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1257) : "f"(x_72.x));
                            float _abs_1258;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1258) : "f"(x_72.y));
                            float2 ax_71 = make_float2(_abs_1257, _abs_1258);
                            float _abs_1259;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1259) : "f"(v_72.x));
                            float _abs_1260;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1260) : "f"(v_72.y));
                            float2 av_71 = make_float2(_abs_1259, _abs_1260);
                            float2 _f32x2_mul_b_1261 = make_float2(k1, k1);
                            float2 te_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_71) : "l"(*(const unsigned long long*)&ax_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_1261));
                            float2 _f32x2_mul_b_1262 = make_float2(k2, k2);
                            float2 tw_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_71) : "l"(*(const unsigned long long*)&av_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_1262));
                            float _exp2_1263 = approx_exp2(te_71.x);
                            float _exp2_1264 = approx_exp2(te_71.y);
                            float2 e_71 = make_float2(_exp2_1263, _exp2_1264);
                            float _exp2_1265 = approx_exp2(tw_71.x);
                            float _exp2_1266 = approx_exp2(tw_71.y);
                            float2 w_71 = make_float2(_exp2_1265, _exp2_1266);
                            float2 _f32x2_mul_b_1267 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_71) : "l"(*(const unsigned long long*)&ax_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_1267));
                            float _exp2_1268 = approx_exp2_noftz(tp_71.x);
                            float _exp2_1269 = approx_exp2_noftz(tp_71.y);
                            float2 p_71 = make_float2(_exp2_1268, _exp2_1269);
                            float2 d1_71 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_71);
                            float2 d2_71 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_71);
                            float2 dw_71 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_71);
                            float2 ng_71 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_71);
                            float2 nu_71 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_71);
                            float _rcp_1270 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_71, d2_71), dw_71).x);
                            float _rcp_1271 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_71, d2_71), dw_71).y);
                            float2 r_144 = make_float2(_rcp_1270, _rcp_1271);
                            float2 r_143 = r_144;
                            float2 _f32x2_mul_a_1272 = mul_f32x2_noftz(ng_71, nu_71);
                            float2 _f32x2_mul_b_1273 = mul_f32x2_noftz(r_143, make_float2(bl, bl));
                            float2 m_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_71) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1272), "l"(*(const unsigned long long*)&_f32x2_mul_b_1273));
                            float2 ns_71 = make_float2(((x_72.x < 0.0f) ? p_71.x : 1.0f), ((x_72.y < 0.0f) ? p_71.y : 1.0f));
                            gate_packed_2[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_71, ns_71))) | sign_72;
                            float2 x_73 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[9]));
                            float2 v_73 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[9]));
                            unsigned int sign_73 = (gate_packed_2[9] ^ up_packed_2[9]) & 2147516416u;
                            float _abs_1274;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1274) : "f"(x_73.x));
                            float _abs_1275;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1275) : "f"(x_73.y));
                            float2 ax_72 = make_float2(_abs_1274, _abs_1275);
                            float _abs_1276;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1276) : "f"(v_73.x));
                            float _abs_1277;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1277) : "f"(v_73.y));
                            float2 av_72 = make_float2(_abs_1276, _abs_1277);
                            float2 _f32x2_mul_b_1278 = make_float2(k1, k1);
                            float2 te_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_72) : "l"(*(const unsigned long long*)&ax_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_1278));
                            float2 _f32x2_mul_b_1279 = make_float2(k2, k2);
                            float2 tw_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_72) : "l"(*(const unsigned long long*)&av_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_1279));
                            float _exp2_1280 = approx_exp2(te_72.x);
                            float _exp2_1281 = approx_exp2(te_72.y);
                            float2 e_72 = make_float2(_exp2_1280, _exp2_1281);
                            float _exp2_1282 = approx_exp2(tw_72.x);
                            float _exp2_1283 = approx_exp2(tw_72.y);
                            float2 w_72 = make_float2(_exp2_1282, _exp2_1283);
                            float2 _f32x2_mul_b_1284 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_72) : "l"(*(const unsigned long long*)&ax_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_1284));
                            float _exp2_1285 = approx_exp2_noftz(tp_72.x);
                            float _exp2_1286 = approx_exp2_noftz(tp_72.y);
                            float2 p_72 = make_float2(_exp2_1285, _exp2_1286);
                            float2 d1_72 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_72);
                            float2 d2_72 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_72);
                            float2 dw_72 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_72);
                            float2 ng_72 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_72);
                            float2 nu_72 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_72);
                            float _rcp_1287 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_72, d2_72), dw_72).x);
                            float _rcp_1288 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_72, d2_72), dw_72).y);
                            float2 r_146 = make_float2(_rcp_1287, _rcp_1288);
                            float2 r_145 = r_146;
                            float2 _f32x2_mul_a_1289 = mul_f32x2_noftz(ng_72, nu_72);
                            float2 _f32x2_mul_b_1290 = mul_f32x2_noftz(r_145, make_float2(bl, bl));
                            float2 m_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1289), "l"(*(const unsigned long long*)&_f32x2_mul_b_1290));
                            float2 ns_72 = make_float2(((x_73.x < 0.0f) ? p_72.x : 1.0f), ((x_73.y < 0.0f) ? p_72.y : 1.0f));
                            gate_packed_2[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_72, ns_72))) | sign_73;
                            float2 x_74 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[10]));
                            float2 v_74 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[10]));
                            unsigned int sign_74 = (gate_packed_2[10] ^ up_packed_2[10]) & 2147516416u;
                            float _abs_1291;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1291) : "f"(x_74.x));
                            float _abs_1292;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1292) : "f"(x_74.y));
                            float2 ax_73 = make_float2(_abs_1291, _abs_1292);
                            float _abs_1293;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1293) : "f"(v_74.x));
                            float _abs_1294;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1294) : "f"(v_74.y));
                            float2 av_73 = make_float2(_abs_1293, _abs_1294);
                            float2 _f32x2_mul_b_1295 = make_float2(k1, k1);
                            float2 te_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_73) : "l"(*(const unsigned long long*)&ax_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_1295));
                            float2 _f32x2_mul_b_1296 = make_float2(k2, k2);
                            float2 tw_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_73) : "l"(*(const unsigned long long*)&av_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_1296));
                            float _exp2_1297 = approx_exp2(te_73.x);
                            float _exp2_1298 = approx_exp2(te_73.y);
                            float2 e_73 = make_float2(_exp2_1297, _exp2_1298);
                            float _exp2_1299 = approx_exp2(tw_73.x);
                            float _exp2_1300 = approx_exp2(tw_73.y);
                            float2 w_73 = make_float2(_exp2_1299, _exp2_1300);
                            float2 _f32x2_mul_b_1301 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_73) : "l"(*(const unsigned long long*)&ax_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_1301));
                            float _exp2_1302 = approx_exp2_noftz(tp_73.x);
                            float _exp2_1303 = approx_exp2_noftz(tp_73.y);
                            float2 p_73 = make_float2(_exp2_1302, _exp2_1303);
                            float2 d1_73 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_73);
                            float2 d2_73 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_73);
                            float2 dw_73 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_73);
                            float2 ng_73 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_73);
                            float2 nu_73 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_73);
                            float _rcp_1304 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_73, d2_73), dw_73).x);
                            float _rcp_1305 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_73, d2_73), dw_73).y);
                            float2 r_148 = make_float2(_rcp_1304, _rcp_1305);
                            float2 r_147 = r_148;
                            float2 _f32x2_mul_a_1306 = mul_f32x2_noftz(ng_73, nu_73);
                            float2 _f32x2_mul_b_1307 = mul_f32x2_noftz(r_147, make_float2(bl, bl));
                            float2 m_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_73) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1306), "l"(*(const unsigned long long*)&_f32x2_mul_b_1307));
                            float2 ns_73 = make_float2(((x_74.x < 0.0f) ? p_73.x : 1.0f), ((x_74.y < 0.0f) ? p_73.y : 1.0f));
                            gate_packed_2[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_73, ns_73))) | sign_74;
                            float2 x_75 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[11]));
                            float2 v_75 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[11]));
                            unsigned int sign_75 = (gate_packed_2[11] ^ up_packed_2[11]) & 2147516416u;
                            float _abs_1308;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1308) : "f"(x_75.x));
                            float _abs_1309;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1309) : "f"(x_75.y));
                            float2 ax_74 = make_float2(_abs_1308, _abs_1309);
                            float _abs_1310;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1310) : "f"(v_75.x));
                            float _abs_1311;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1311) : "f"(v_75.y));
                            float2 av_74 = make_float2(_abs_1310, _abs_1311);
                            float2 _f32x2_mul_b_1312 = make_float2(k1, k1);
                            float2 te_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_74) : "l"(*(const unsigned long long*)&ax_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_1312));
                            float2 _f32x2_mul_b_1313 = make_float2(k2, k2);
                            float2 tw_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_74) : "l"(*(const unsigned long long*)&av_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_1313));
                            float _exp2_1314 = approx_exp2(te_74.x);
                            float _exp2_1315 = approx_exp2(te_74.y);
                            float2 e_74 = make_float2(_exp2_1314, _exp2_1315);
                            float _exp2_1316 = approx_exp2(tw_74.x);
                            float _exp2_1317 = approx_exp2(tw_74.y);
                            float2 w_74 = make_float2(_exp2_1316, _exp2_1317);
                            float2 _f32x2_mul_b_1318 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_74) : "l"(*(const unsigned long long*)&ax_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_1318));
                            float _exp2_1319 = approx_exp2_noftz(tp_74.x);
                            float _exp2_1320 = approx_exp2_noftz(tp_74.y);
                            float2 p_74 = make_float2(_exp2_1319, _exp2_1320);
                            float2 d1_74 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_74);
                            float2 d2_74 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_74);
                            float2 dw_74 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_74);
                            float2 ng_74 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_74);
                            float2 nu_74 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_74);
                            float _rcp_1321 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_74, d2_74), dw_74).x);
                            float _rcp_1322 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_74, d2_74), dw_74).y);
                            float2 r_150 = make_float2(_rcp_1321, _rcp_1322);
                            float2 r_149 = r_150;
                            float2 _f32x2_mul_a_1323 = mul_f32x2_noftz(ng_74, nu_74);
                            float2 _f32x2_mul_b_1324 = mul_f32x2_noftz(r_149, make_float2(bl, bl));
                            float2 m_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_74) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1323), "l"(*(const unsigned long long*)&_f32x2_mul_b_1324));
                            float2 ns_74 = make_float2(((x_75.x < 0.0f) ? p_74.x : 1.0f), ((x_75.y < 0.0f) ? p_74.y : 1.0f));
                            gate_packed_2[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_74, ns_74))) | sign_75;
                            float2 x_76 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[12]));
                            float2 v_76 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[12]));
                            unsigned int sign_76 = (gate_packed_2[12] ^ up_packed_2[12]) & 2147516416u;
                            float _abs_1325;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1325) : "f"(x_76.x));
                            float _abs_1326;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1326) : "f"(x_76.y));
                            float2 ax_75 = make_float2(_abs_1325, _abs_1326);
                            float _abs_1327;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1327) : "f"(v_76.x));
                            float _abs_1328;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1328) : "f"(v_76.y));
                            float2 av_75 = make_float2(_abs_1327, _abs_1328);
                            float2 _f32x2_mul_b_1329 = make_float2(k1, k1);
                            float2 te_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_75) : "l"(*(const unsigned long long*)&ax_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_1329));
                            float2 _f32x2_mul_b_1330 = make_float2(k2, k2);
                            float2 tw_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_75) : "l"(*(const unsigned long long*)&av_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_1330));
                            float _exp2_1331 = approx_exp2(te_75.x);
                            float _exp2_1332 = approx_exp2(te_75.y);
                            float2 e_75 = make_float2(_exp2_1331, _exp2_1332);
                            float _exp2_1333 = approx_exp2(tw_75.x);
                            float _exp2_1334 = approx_exp2(tw_75.y);
                            float2 w_75 = make_float2(_exp2_1333, _exp2_1334);
                            float2 _f32x2_mul_b_1335 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_75) : "l"(*(const unsigned long long*)&ax_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_1335));
                            float _exp2_1336 = approx_exp2_noftz(tp_75.x);
                            float _exp2_1337 = approx_exp2_noftz(tp_75.y);
                            float2 p_75 = make_float2(_exp2_1336, _exp2_1337);
                            float2 d1_75 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_75);
                            float2 d2_75 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_75);
                            float2 dw_75 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_75);
                            float2 ng_75 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_75);
                            float2 nu_75 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_75);
                            float _rcp_1338 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_75, d2_75), dw_75).x);
                            float _rcp_1339 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_75, d2_75), dw_75).y);
                            float2 r_152 = make_float2(_rcp_1338, _rcp_1339);
                            float2 r_151 = r_152;
                            float2 _f32x2_mul_a_1340 = mul_f32x2_noftz(ng_75, nu_75);
                            float2 _f32x2_mul_b_1341 = mul_f32x2_noftz(r_151, make_float2(bl, bl));
                            float2 m_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_75) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1340), "l"(*(const unsigned long long*)&_f32x2_mul_b_1341));
                            float2 ns_75 = make_float2(((x_76.x < 0.0f) ? p_75.x : 1.0f), ((x_76.y < 0.0f) ? p_75.y : 1.0f));
                            gate_packed_2[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_75, ns_75))) | sign_76;
                            float2 x_77 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[13]));
                            float2 v_77 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[13]));
                            unsigned int sign_77 = (gate_packed_2[13] ^ up_packed_2[13]) & 2147516416u;
                            float _abs_1342;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1342) : "f"(x_77.x));
                            float _abs_1343;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1343) : "f"(x_77.y));
                            float2 ax_76 = make_float2(_abs_1342, _abs_1343);
                            float _abs_1344;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1344) : "f"(v_77.x));
                            float _abs_1345;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1345) : "f"(v_77.y));
                            float2 av_76 = make_float2(_abs_1344, _abs_1345);
                            float2 _f32x2_mul_b_1346 = make_float2(k1, k1);
                            float2 te_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_76) : "l"(*(const unsigned long long*)&ax_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_1346));
                            float2 _f32x2_mul_b_1347 = make_float2(k2, k2);
                            float2 tw_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_76) : "l"(*(const unsigned long long*)&av_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_1347));
                            float _exp2_1348 = approx_exp2(te_76.x);
                            float _exp2_1349 = approx_exp2(te_76.y);
                            float2 e_76 = make_float2(_exp2_1348, _exp2_1349);
                            float _exp2_1350 = approx_exp2(tw_76.x);
                            float _exp2_1351 = approx_exp2(tw_76.y);
                            float2 w_76 = make_float2(_exp2_1350, _exp2_1351);
                            float2 _f32x2_mul_b_1352 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_76) : "l"(*(const unsigned long long*)&ax_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_1352));
                            float _exp2_1353 = approx_exp2_noftz(tp_76.x);
                            float _exp2_1354 = approx_exp2_noftz(tp_76.y);
                            float2 p_76 = make_float2(_exp2_1353, _exp2_1354);
                            float2 d1_76 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_76);
                            float2 d2_76 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_76);
                            float2 dw_76 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_76);
                            float2 ng_76 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_76);
                            float2 nu_76 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_76);
                            float _rcp_1355 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_76, d2_76), dw_76).x);
                            float _rcp_1356 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_76, d2_76), dw_76).y);
                            float2 r_154 = make_float2(_rcp_1355, _rcp_1356);
                            float2 r_153 = r_154;
                            float2 _f32x2_mul_a_1357 = mul_f32x2_noftz(ng_76, nu_76);
                            float2 _f32x2_mul_b_1358 = mul_f32x2_noftz(r_153, make_float2(bl, bl));
                            float2 m_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_76) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1357), "l"(*(const unsigned long long*)&_f32x2_mul_b_1358));
                            float2 ns_76 = make_float2(((x_77.x < 0.0f) ? p_76.x : 1.0f), ((x_77.y < 0.0f) ? p_76.y : 1.0f));
                            gate_packed_2[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_76, ns_76))) | sign_77;
                            float2 x_78 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[14]));
                            float2 v_78 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[14]));
                            unsigned int sign_78 = (gate_packed_2[14] ^ up_packed_2[14]) & 2147516416u;
                            float _abs_1359;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1359) : "f"(x_78.x));
                            float _abs_1360;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1360) : "f"(x_78.y));
                            float2 ax_77 = make_float2(_abs_1359, _abs_1360);
                            float _abs_1361;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1361) : "f"(v_78.x));
                            float _abs_1362;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1362) : "f"(v_78.y));
                            float2 av_77 = make_float2(_abs_1361, _abs_1362);
                            float2 _f32x2_mul_b_1363 = make_float2(k1, k1);
                            float2 te_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_77) : "l"(*(const unsigned long long*)&ax_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_1363));
                            float2 _f32x2_mul_b_1364 = make_float2(k2, k2);
                            float2 tw_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_77) : "l"(*(const unsigned long long*)&av_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_1364));
                            float _exp2_1365 = approx_exp2(te_77.x);
                            float _exp2_1366 = approx_exp2(te_77.y);
                            float2 e_77 = make_float2(_exp2_1365, _exp2_1366);
                            float _exp2_1367 = approx_exp2(tw_77.x);
                            float _exp2_1368 = approx_exp2(tw_77.y);
                            float2 w_77 = make_float2(_exp2_1367, _exp2_1368);
                            float2 _f32x2_mul_b_1369 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_77) : "l"(*(const unsigned long long*)&ax_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_1369));
                            float _exp2_1370 = approx_exp2_noftz(tp_77.x);
                            float _exp2_1371 = approx_exp2_noftz(tp_77.y);
                            float2 p_77 = make_float2(_exp2_1370, _exp2_1371);
                            float2 d1_77 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_77);
                            float2 d2_77 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_77);
                            float2 dw_77 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_77);
                            float2 ng_77 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_77);
                            float2 nu_77 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_77);
                            float _rcp_1372 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_77, d2_77), dw_77).x);
                            float _rcp_1373 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_77, d2_77), dw_77).y);
                            float2 r_156 = make_float2(_rcp_1372, _rcp_1373);
                            float2 r_155 = r_156;
                            float2 _f32x2_mul_a_1374 = mul_f32x2_noftz(ng_77, nu_77);
                            float2 _f32x2_mul_b_1375 = mul_f32x2_noftz(r_155, make_float2(bl, bl));
                            float2 m_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_77) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1374), "l"(*(const unsigned long long*)&_f32x2_mul_b_1375));
                            float2 ns_77 = make_float2(((x_78.x < 0.0f) ? p_77.x : 1.0f), ((x_78.y < 0.0f) ? p_77.y : 1.0f));
                            gate_packed_2[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_77, ns_77))) | sign_78;
                            float2 x_79 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[15]));
                            float2 v_79 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[15]));
                            unsigned int sign_79 = (gate_packed_2[15] ^ up_packed_2[15]) & 2147516416u;
                            float _abs_1376;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1376) : "f"(x_79.x));
                            float _abs_1377;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1377) : "f"(x_79.y));
                            float2 ax_78 = make_float2(_abs_1376, _abs_1377);
                            float _abs_1378;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1378) : "f"(v_79.x));
                            float _abs_1379;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1379) : "f"(v_79.y));
                            float2 av_78 = make_float2(_abs_1378, _abs_1379);
                            float2 _f32x2_mul_b_1380 = make_float2(k1, k1);
                            float2 te_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_78) : "l"(*(const unsigned long long*)&ax_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_1380));
                            float2 _f32x2_mul_b_1381 = make_float2(k2, k2);
                            float2 tw_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_78) : "l"(*(const unsigned long long*)&av_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_1381));
                            float _exp2_1382 = approx_exp2(te_78.x);
                            float _exp2_1383 = approx_exp2(te_78.y);
                            float2 e_78 = make_float2(_exp2_1382, _exp2_1383);
                            float _exp2_1384 = approx_exp2(tw_78.x);
                            float _exp2_1385 = approx_exp2(tw_78.y);
                            float2 w_78 = make_float2(_exp2_1384, _exp2_1385);
                            float2 _f32x2_mul_b_1386 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_78) : "l"(*(const unsigned long long*)&ax_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_1386));
                            float _exp2_1387 = approx_exp2_noftz(tp_78.x);
                            float _exp2_1388 = approx_exp2_noftz(tp_78.y);
                            float2 p_78 = make_float2(_exp2_1387, _exp2_1388);
                            float2 d1_78 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_78);
                            float2 d2_78 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_78);
                            float2 dw_78 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_78);
                            float2 ng_78 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_78);
                            float2 nu_78 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_78);
                            float _rcp_1389 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_78, d2_78), dw_78).x);
                            float _rcp_1390 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_78, d2_78), dw_78).y);
                            float2 r_158 = make_float2(_rcp_1389, _rcp_1390);
                            float2 r_157 = r_158;
                            float2 _f32x2_mul_a_1391 = mul_f32x2_noftz(ng_78, nu_78);
                            float2 _f32x2_mul_b_1392 = mul_f32x2_noftz(r_157, make_float2(bl, bl));
                            float2 m_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_78) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1391), "l"(*(const unsigned long long*)&_f32x2_mul_b_1392));
                            float2 ns_78 = make_float2(((x_79.x < 0.0f) ? p_78.x : 1.0f), ((x_79.y < 0.0f) ? p_78.y : 1.0f));
                            gate_packed_2[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_78, ns_78))) | sign_79;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_15 = tid / 32;
                            int lane_15 = tid % 32;
                            #pragma unroll
                            for (int half_16 = 0; half_16 < 2; half_16++) {
                                #pragma unroll
                                for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                    int row_17 = warp_15 * 32 + half_16 * 16 + lane_15 % 16;
                                    int col_17 = col_tile_16 * 16 + lane_15 / 16 * 8;
                                    unsigned int address_16 = d_smem_addr + 8192 + (unsigned int)((row_17 * 32 + col_17) * 2);
                                    address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                    int offset_17 = half_16 * 8 + col_tile_16 * 4;
                                    uint32_t _stmatrix_addr_1393 = static_cast<uint32_t>(address_16);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1393), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_80 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[16]));
                            float2 v_80 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[16]));
                            unsigned int sign_80 = (gate_packed_2[16] ^ up_packed_2[16]) & 2147516416u;
                            float _abs_1394;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1394) : "f"(x_80.x));
                            float _abs_1395;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1395) : "f"(x_80.y));
                            float2 ax_79 = make_float2(_abs_1394, _abs_1395);
                            float _abs_1396;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1396) : "f"(v_80.x));
                            float _abs_1397;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1397) : "f"(v_80.y));
                            float2 av_79 = make_float2(_abs_1396, _abs_1397);
                            float2 _f32x2_mul_b_1398 = make_float2(k1, k1);
                            float2 te_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_79) : "l"(*(const unsigned long long*)&ax_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_1398));
                            float2 _f32x2_mul_b_1399 = make_float2(k2, k2);
                            float2 tw_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_79) : "l"(*(const unsigned long long*)&av_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_1399));
                            float _exp2_1400 = approx_exp2(te_79.x);
                            float _exp2_1401 = approx_exp2(te_79.y);
                            float2 e_79 = make_float2(_exp2_1400, _exp2_1401);
                            float _exp2_1402 = approx_exp2(tw_79.x);
                            float _exp2_1403 = approx_exp2(tw_79.y);
                            float2 w_79 = make_float2(_exp2_1402, _exp2_1403);
                            float2 _f32x2_mul_b_1404 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_79) : "l"(*(const unsigned long long*)&ax_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_1404));
                            float _exp2_1405 = approx_exp2_noftz(tp_79.x);
                            float _exp2_1406 = approx_exp2_noftz(tp_79.y);
                            float2 p_79 = make_float2(_exp2_1405, _exp2_1406);
                            float2 d1_79 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_79);
                            float2 d2_79 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_79);
                            float2 dw_79 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_79);
                            float2 ng_79 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_79);
                            float2 nu_79 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_79);
                            float _rcp_1407 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_79, d2_79), dw_79).x);
                            float _rcp_1408 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_79, d2_79), dw_79).y);
                            float2 r_160 = make_float2(_rcp_1407, _rcp_1408);
                            float2 r_159 = r_160;
                            float2 _f32x2_mul_a_1409 = mul_f32x2_noftz(ng_79, nu_79);
                            float2 _f32x2_mul_b_1410 = mul_f32x2_noftz(r_159, make_float2(bl, bl));
                            float2 m_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_79) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1409), "l"(*(const unsigned long long*)&_f32x2_mul_b_1410));
                            float2 ns_79 = make_float2(((x_80.x < 0.0f) ? p_79.x : 1.0f), ((x_80.y < 0.0f) ? p_79.y : 1.0f));
                            gate_packed_2[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_79, ns_79))) | sign_80;
                            float2 x_81 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[17]));
                            float2 v_81 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[17]));
                            unsigned int sign_81 = (gate_packed_2[17] ^ up_packed_2[17]) & 2147516416u;
                            float _abs_1411;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1411) : "f"(x_81.x));
                            float _abs_1412;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1412) : "f"(x_81.y));
                            float2 ax_80 = make_float2(_abs_1411, _abs_1412);
                            float _abs_1413;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1413) : "f"(v_81.x));
                            float _abs_1414;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1414) : "f"(v_81.y));
                            float2 av_80 = make_float2(_abs_1413, _abs_1414);
                            float2 _f32x2_mul_b_1415 = make_float2(k1, k1);
                            float2 te_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_80) : "l"(*(const unsigned long long*)&ax_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_1415));
                            float2 _f32x2_mul_b_1416 = make_float2(k2, k2);
                            float2 tw_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_80) : "l"(*(const unsigned long long*)&av_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_1416));
                            float _exp2_1417 = approx_exp2(te_80.x);
                            float _exp2_1418 = approx_exp2(te_80.y);
                            float2 e_80 = make_float2(_exp2_1417, _exp2_1418);
                            float _exp2_1419 = approx_exp2(tw_80.x);
                            float _exp2_1420 = approx_exp2(tw_80.y);
                            float2 w_80 = make_float2(_exp2_1419, _exp2_1420);
                            float2 _f32x2_mul_b_1421 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_80) : "l"(*(const unsigned long long*)&ax_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_1421));
                            float _exp2_1422 = approx_exp2_noftz(tp_80.x);
                            float _exp2_1423 = approx_exp2_noftz(tp_80.y);
                            float2 p_80 = make_float2(_exp2_1422, _exp2_1423);
                            float2 d1_80 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_80);
                            float2 d2_80 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_80);
                            float2 dw_80 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_80);
                            float2 ng_80 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_80);
                            float2 nu_80 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_80);
                            float _rcp_1424 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_80, d2_80), dw_80).x);
                            float _rcp_1425 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_80, d2_80), dw_80).y);
                            float2 r_162 = make_float2(_rcp_1424, _rcp_1425);
                            float2 r_161 = r_162;
                            float2 _f32x2_mul_a_1426 = mul_f32x2_noftz(ng_80, nu_80);
                            float2 _f32x2_mul_b_1427 = mul_f32x2_noftz(r_161, make_float2(bl, bl));
                            float2 m_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_80) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1426), "l"(*(const unsigned long long*)&_f32x2_mul_b_1427));
                            float2 ns_80 = make_float2(((x_81.x < 0.0f) ? p_80.x : 1.0f), ((x_81.y < 0.0f) ? p_80.y : 1.0f));
                            gate_packed_2[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_80, ns_80))) | sign_81;
                            float2 x_82 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[18]));
                            float2 v_82 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[18]));
                            unsigned int sign_82 = (gate_packed_2[18] ^ up_packed_2[18]) & 2147516416u;
                            float _abs_1428;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1428) : "f"(x_82.x));
                            float _abs_1429;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1429) : "f"(x_82.y));
                            float2 ax_81 = make_float2(_abs_1428, _abs_1429);
                            float _abs_1430;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1430) : "f"(v_82.x));
                            float _abs_1431;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1431) : "f"(v_82.y));
                            float2 av_81 = make_float2(_abs_1430, _abs_1431);
                            float2 _f32x2_mul_b_1432 = make_float2(k1, k1);
                            float2 te_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_81) : "l"(*(const unsigned long long*)&ax_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_1432));
                            float2 _f32x2_mul_b_1433 = make_float2(k2, k2);
                            float2 tw_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_81) : "l"(*(const unsigned long long*)&av_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_1433));
                            float _exp2_1434 = approx_exp2(te_81.x);
                            float _exp2_1435 = approx_exp2(te_81.y);
                            float2 e_81 = make_float2(_exp2_1434, _exp2_1435);
                            float _exp2_1436 = approx_exp2(tw_81.x);
                            float _exp2_1437 = approx_exp2(tw_81.y);
                            float2 w_81 = make_float2(_exp2_1436, _exp2_1437);
                            float2 _f32x2_mul_b_1438 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_81) : "l"(*(const unsigned long long*)&ax_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_1438));
                            float _exp2_1439 = approx_exp2_noftz(tp_81.x);
                            float _exp2_1440 = approx_exp2_noftz(tp_81.y);
                            float2 p_81 = make_float2(_exp2_1439, _exp2_1440);
                            float2 d1_81 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_81);
                            float2 d2_81 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_81);
                            float2 dw_81 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_81);
                            float2 ng_81 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_81);
                            float2 nu_81 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_81);
                            float _rcp_1441 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_81, d2_81), dw_81).x);
                            float _rcp_1442 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_81, d2_81), dw_81).y);
                            float2 r_164 = make_float2(_rcp_1441, _rcp_1442);
                            float2 r_163 = r_164;
                            float2 _f32x2_mul_a_1443 = mul_f32x2_noftz(ng_81, nu_81);
                            float2 _f32x2_mul_b_1444 = mul_f32x2_noftz(r_163, make_float2(bl, bl));
                            float2 m_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_81) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1443), "l"(*(const unsigned long long*)&_f32x2_mul_b_1444));
                            float2 ns_81 = make_float2(((x_82.x < 0.0f) ? p_81.x : 1.0f), ((x_82.y < 0.0f) ? p_81.y : 1.0f));
                            gate_packed_2[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_81, ns_81))) | sign_82;
                            float2 x_83 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[19]));
                            float2 v_83 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[19]));
                            unsigned int sign_83 = (gate_packed_2[19] ^ up_packed_2[19]) & 2147516416u;
                            float _abs_1445;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1445) : "f"(x_83.x));
                            float _abs_1446;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1446) : "f"(x_83.y));
                            float2 ax_82 = make_float2(_abs_1445, _abs_1446);
                            float _abs_1447;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1447) : "f"(v_83.x));
                            float _abs_1448;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1448) : "f"(v_83.y));
                            float2 av_82 = make_float2(_abs_1447, _abs_1448);
                            float2 _f32x2_mul_b_1449 = make_float2(k1, k1);
                            float2 te_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_82) : "l"(*(const unsigned long long*)&ax_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_1449));
                            float2 _f32x2_mul_b_1450 = make_float2(k2, k2);
                            float2 tw_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_82) : "l"(*(const unsigned long long*)&av_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_1450));
                            float _exp2_1451 = approx_exp2(te_82.x);
                            float _exp2_1452 = approx_exp2(te_82.y);
                            float2 e_82 = make_float2(_exp2_1451, _exp2_1452);
                            float _exp2_1453 = approx_exp2(tw_82.x);
                            float _exp2_1454 = approx_exp2(tw_82.y);
                            float2 w_82 = make_float2(_exp2_1453, _exp2_1454);
                            float2 _f32x2_mul_b_1455 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_82) : "l"(*(const unsigned long long*)&ax_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_1455));
                            float _exp2_1456 = approx_exp2_noftz(tp_82.x);
                            float _exp2_1457 = approx_exp2_noftz(tp_82.y);
                            float2 p_82 = make_float2(_exp2_1456, _exp2_1457);
                            float2 d1_82 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_82);
                            float2 d2_82 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_82);
                            float2 dw_82 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_82);
                            float2 ng_82 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_82);
                            float2 nu_82 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_82);
                            float _rcp_1458 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_82, d2_82), dw_82).x);
                            float _rcp_1459 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_82, d2_82), dw_82).y);
                            float2 r_166 = make_float2(_rcp_1458, _rcp_1459);
                            float2 r_165 = r_166;
                            float2 _f32x2_mul_a_1460 = mul_f32x2_noftz(ng_82, nu_82);
                            float2 _f32x2_mul_b_1461 = mul_f32x2_noftz(r_165, make_float2(bl, bl));
                            float2 m_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_82) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1460), "l"(*(const unsigned long long*)&_f32x2_mul_b_1461));
                            float2 ns_82 = make_float2(((x_83.x < 0.0f) ? p_82.x : 1.0f), ((x_83.y < 0.0f) ? p_82.y : 1.0f));
                            gate_packed_2[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_82, ns_82))) | sign_83;
                            float2 x_84 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[20]));
                            float2 v_84 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[20]));
                            unsigned int sign_84 = (gate_packed_2[20] ^ up_packed_2[20]) & 2147516416u;
                            float _abs_1462;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1462) : "f"(x_84.x));
                            float _abs_1463;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1463) : "f"(x_84.y));
                            float2 ax_83 = make_float2(_abs_1462, _abs_1463);
                            float _abs_1464;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1464) : "f"(v_84.x));
                            float _abs_1465;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1465) : "f"(v_84.y));
                            float2 av_83 = make_float2(_abs_1464, _abs_1465);
                            float2 _f32x2_mul_b_1466 = make_float2(k1, k1);
                            float2 te_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_83) : "l"(*(const unsigned long long*)&ax_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_1466));
                            float2 _f32x2_mul_b_1467 = make_float2(k2, k2);
                            float2 tw_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_83) : "l"(*(const unsigned long long*)&av_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_1467));
                            float _exp2_1468 = approx_exp2(te_83.x);
                            float _exp2_1469 = approx_exp2(te_83.y);
                            float2 e_83 = make_float2(_exp2_1468, _exp2_1469);
                            float _exp2_1470 = approx_exp2(tw_83.x);
                            float _exp2_1471 = approx_exp2(tw_83.y);
                            float2 w_83 = make_float2(_exp2_1470, _exp2_1471);
                            float2 _f32x2_mul_b_1472 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_83) : "l"(*(const unsigned long long*)&ax_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_1472));
                            float _exp2_1473 = approx_exp2_noftz(tp_83.x);
                            float _exp2_1474 = approx_exp2_noftz(tp_83.y);
                            float2 p_83 = make_float2(_exp2_1473, _exp2_1474);
                            float2 d1_83 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_83);
                            float2 d2_83 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_83);
                            float2 dw_83 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_83);
                            float2 ng_83 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_83);
                            float2 nu_83 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_83);
                            float _rcp_1475 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_83, d2_83), dw_83).x);
                            float _rcp_1476 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_83, d2_83), dw_83).y);
                            float2 r_168 = make_float2(_rcp_1475, _rcp_1476);
                            float2 r_167 = r_168;
                            float2 _f32x2_mul_a_1477 = mul_f32x2_noftz(ng_83, nu_83);
                            float2 _f32x2_mul_b_1478 = mul_f32x2_noftz(r_167, make_float2(bl, bl));
                            float2 m_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_83) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1477), "l"(*(const unsigned long long*)&_f32x2_mul_b_1478));
                            float2 ns_83 = make_float2(((x_84.x < 0.0f) ? p_83.x : 1.0f), ((x_84.y < 0.0f) ? p_83.y : 1.0f));
                            gate_packed_2[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_83, ns_83))) | sign_84;
                            float2 x_85 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[21]));
                            float2 v_85 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[21]));
                            unsigned int sign_85 = (gate_packed_2[21] ^ up_packed_2[21]) & 2147516416u;
                            float _abs_1479;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1479) : "f"(x_85.x));
                            float _abs_1480;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1480) : "f"(x_85.y));
                            float2 ax_84 = make_float2(_abs_1479, _abs_1480);
                            float _abs_1481;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1481) : "f"(v_85.x));
                            float _abs_1482;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1482) : "f"(v_85.y));
                            float2 av_84 = make_float2(_abs_1481, _abs_1482);
                            float2 _f32x2_mul_b_1483 = make_float2(k1, k1);
                            float2 te_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_84) : "l"(*(const unsigned long long*)&ax_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_1483));
                            float2 _f32x2_mul_b_1484 = make_float2(k2, k2);
                            float2 tw_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_84) : "l"(*(const unsigned long long*)&av_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_1484));
                            float _exp2_1485 = approx_exp2(te_84.x);
                            float _exp2_1486 = approx_exp2(te_84.y);
                            float2 e_84 = make_float2(_exp2_1485, _exp2_1486);
                            float _exp2_1487 = approx_exp2(tw_84.x);
                            float _exp2_1488 = approx_exp2(tw_84.y);
                            float2 w_84 = make_float2(_exp2_1487, _exp2_1488);
                            float2 _f32x2_mul_b_1489 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_84) : "l"(*(const unsigned long long*)&ax_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_1489));
                            float _exp2_1490 = approx_exp2_noftz(tp_84.x);
                            float _exp2_1491 = approx_exp2_noftz(tp_84.y);
                            float2 p_84 = make_float2(_exp2_1490, _exp2_1491);
                            float2 d1_84 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_84);
                            float2 d2_84 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_84);
                            float2 dw_84 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_84);
                            float2 ng_84 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_84);
                            float2 nu_84 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_84);
                            float _rcp_1492 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_84, d2_84), dw_84).x);
                            float _rcp_1493 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_84, d2_84), dw_84).y);
                            float2 r_170 = make_float2(_rcp_1492, _rcp_1493);
                            float2 r_169 = r_170;
                            float2 _f32x2_mul_a_1494 = mul_f32x2_noftz(ng_84, nu_84);
                            float2 _f32x2_mul_b_1495 = mul_f32x2_noftz(r_169, make_float2(bl, bl));
                            float2 m_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1494), "l"(*(const unsigned long long*)&_f32x2_mul_b_1495));
                            float2 ns_84 = make_float2(((x_85.x < 0.0f) ? p_84.x : 1.0f), ((x_85.y < 0.0f) ? p_84.y : 1.0f));
                            gate_packed_2[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_84, ns_84))) | sign_85;
                            float2 x_86 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[22]));
                            float2 v_86 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[22]));
                            unsigned int sign_86 = (gate_packed_2[22] ^ up_packed_2[22]) & 2147516416u;
                            float _abs_1496;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1496) : "f"(x_86.x));
                            float _abs_1497;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1497) : "f"(x_86.y));
                            float2 ax_85 = make_float2(_abs_1496, _abs_1497);
                            float _abs_1498;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1498) : "f"(v_86.x));
                            float _abs_1499;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1499) : "f"(v_86.y));
                            float2 av_85 = make_float2(_abs_1498, _abs_1499);
                            float2 _f32x2_mul_b_1500 = make_float2(k1, k1);
                            float2 te_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_85) : "l"(*(const unsigned long long*)&ax_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_1500));
                            float2 _f32x2_mul_b_1501 = make_float2(k2, k2);
                            float2 tw_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_85) : "l"(*(const unsigned long long*)&av_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_1501));
                            float _exp2_1502 = approx_exp2(te_85.x);
                            float _exp2_1503 = approx_exp2(te_85.y);
                            float2 e_85 = make_float2(_exp2_1502, _exp2_1503);
                            float _exp2_1504 = approx_exp2(tw_85.x);
                            float _exp2_1505 = approx_exp2(tw_85.y);
                            float2 w_85 = make_float2(_exp2_1504, _exp2_1505);
                            float2 _f32x2_mul_b_1506 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_85) : "l"(*(const unsigned long long*)&ax_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_1506));
                            float _exp2_1507 = approx_exp2_noftz(tp_85.x);
                            float _exp2_1508 = approx_exp2_noftz(tp_85.y);
                            float2 p_85 = make_float2(_exp2_1507, _exp2_1508);
                            float2 d1_85 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_85);
                            float2 d2_85 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_85);
                            float2 dw_85 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_85);
                            float2 ng_85 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_85);
                            float2 nu_85 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_85);
                            float _rcp_1509 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_85, d2_85), dw_85).x);
                            float _rcp_1510 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_85, d2_85), dw_85).y);
                            float2 r_172 = make_float2(_rcp_1509, _rcp_1510);
                            float2 r_171 = r_172;
                            float2 _f32x2_mul_a_1511 = mul_f32x2_noftz(ng_85, nu_85);
                            float2 _f32x2_mul_b_1512 = mul_f32x2_noftz(r_171, make_float2(bl, bl));
                            float2 m_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_85) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1511), "l"(*(const unsigned long long*)&_f32x2_mul_b_1512));
                            float2 ns_85 = make_float2(((x_86.x < 0.0f) ? p_85.x : 1.0f), ((x_86.y < 0.0f) ? p_85.y : 1.0f));
                            gate_packed_2[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_85, ns_85))) | sign_86;
                            float2 x_87 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[23]));
                            float2 v_87 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[23]));
                            unsigned int sign_87 = (gate_packed_2[23] ^ up_packed_2[23]) & 2147516416u;
                            float _abs_1513;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1513) : "f"(x_87.x));
                            float _abs_1514;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1514) : "f"(x_87.y));
                            float2 ax_86 = make_float2(_abs_1513, _abs_1514);
                            float _abs_1515;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1515) : "f"(v_87.x));
                            float _abs_1516;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1516) : "f"(v_87.y));
                            float2 av_86 = make_float2(_abs_1515, _abs_1516);
                            float2 _f32x2_mul_b_1517 = make_float2(k1, k1);
                            float2 te_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_86) : "l"(*(const unsigned long long*)&ax_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_1517));
                            float2 _f32x2_mul_b_1518 = make_float2(k2, k2);
                            float2 tw_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_86) : "l"(*(const unsigned long long*)&av_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_1518));
                            float _exp2_1519 = approx_exp2(te_86.x);
                            float _exp2_1520 = approx_exp2(te_86.y);
                            float2 e_86 = make_float2(_exp2_1519, _exp2_1520);
                            float _exp2_1521 = approx_exp2(tw_86.x);
                            float _exp2_1522 = approx_exp2(tw_86.y);
                            float2 w_86 = make_float2(_exp2_1521, _exp2_1522);
                            float2 _f32x2_mul_b_1523 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_86) : "l"(*(const unsigned long long*)&ax_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_1523));
                            float _exp2_1524 = approx_exp2_noftz(tp_86.x);
                            float _exp2_1525 = approx_exp2_noftz(tp_86.y);
                            float2 p_86 = make_float2(_exp2_1524, _exp2_1525);
                            float2 d1_86 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_86);
                            float2 d2_86 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_86);
                            float2 dw_86 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_86);
                            float2 ng_86 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_86);
                            float2 nu_86 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_86);
                            float _rcp_1526 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_86, d2_86), dw_86).x);
                            float _rcp_1527 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_86, d2_86), dw_86).y);
                            float2 r_174 = make_float2(_rcp_1526, _rcp_1527);
                            float2 r_173 = r_174;
                            float2 _f32x2_mul_a_1528 = mul_f32x2_noftz(ng_86, nu_86);
                            float2 _f32x2_mul_b_1529 = mul_f32x2_noftz(r_173, make_float2(bl, bl));
                            float2 m_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1528), "l"(*(const unsigned long long*)&_f32x2_mul_b_1529));
                            float2 ns_86 = make_float2(((x_87.x < 0.0f) ? p_86.x : 1.0f), ((x_87.y < 0.0f) ? p_86.y : 1.0f));
                            gate_packed_2[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_86, ns_86))) | sign_87;
                            float2 x_88 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[24]));
                            float2 v_88 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[24]));
                            unsigned int sign_88 = (gate_packed_2[24] ^ up_packed_2[24]) & 2147516416u;
                            float _abs_1530;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1530) : "f"(x_88.x));
                            float _abs_1531;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1531) : "f"(x_88.y));
                            float2 ax_87 = make_float2(_abs_1530, _abs_1531);
                            float _abs_1532;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1532) : "f"(v_88.x));
                            float _abs_1533;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1533) : "f"(v_88.y));
                            float2 av_87 = make_float2(_abs_1532, _abs_1533);
                            float2 _f32x2_mul_b_1534 = make_float2(k1, k1);
                            float2 te_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_87) : "l"(*(const unsigned long long*)&ax_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_1534));
                            float2 _f32x2_mul_b_1535 = make_float2(k2, k2);
                            float2 tw_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_87) : "l"(*(const unsigned long long*)&av_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_1535));
                            float _exp2_1536 = approx_exp2(te_87.x);
                            float _exp2_1537 = approx_exp2(te_87.y);
                            float2 e_87 = make_float2(_exp2_1536, _exp2_1537);
                            float _exp2_1538 = approx_exp2(tw_87.x);
                            float _exp2_1539 = approx_exp2(tw_87.y);
                            float2 w_87 = make_float2(_exp2_1538, _exp2_1539);
                            float2 _f32x2_mul_b_1540 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_87) : "l"(*(const unsigned long long*)&ax_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_1540));
                            float _exp2_1541 = approx_exp2_noftz(tp_87.x);
                            float _exp2_1542 = approx_exp2_noftz(tp_87.y);
                            float2 p_87 = make_float2(_exp2_1541, _exp2_1542);
                            float2 d1_87 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_87);
                            float2 d2_87 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_87);
                            float2 dw_87 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_87);
                            float2 ng_87 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_87);
                            float2 nu_87 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_87);
                            float _rcp_1543 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_87, d2_87), dw_87).x);
                            float _rcp_1544 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_87, d2_87), dw_87).y);
                            float2 r_176 = make_float2(_rcp_1543, _rcp_1544);
                            float2 r_175 = r_176;
                            float2 _f32x2_mul_a_1545 = mul_f32x2_noftz(ng_87, nu_87);
                            float2 _f32x2_mul_b_1546 = mul_f32x2_noftz(r_175, make_float2(bl, bl));
                            float2 m_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_87) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1545), "l"(*(const unsigned long long*)&_f32x2_mul_b_1546));
                            float2 ns_87 = make_float2(((x_88.x < 0.0f) ? p_87.x : 1.0f), ((x_88.y < 0.0f) ? p_87.y : 1.0f));
                            gate_packed_2[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_87, ns_87))) | sign_88;
                            float2 x_89 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[25]));
                            float2 v_89 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[25]));
                            unsigned int sign_89 = (gate_packed_2[25] ^ up_packed_2[25]) & 2147516416u;
                            float _abs_1547;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1547) : "f"(x_89.x));
                            float _abs_1548;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1548) : "f"(x_89.y));
                            float2 ax_88 = make_float2(_abs_1547, _abs_1548);
                            float _abs_1549;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1549) : "f"(v_89.x));
                            float _abs_1550;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1550) : "f"(v_89.y));
                            float2 av_88 = make_float2(_abs_1549, _abs_1550);
                            float2 _f32x2_mul_b_1551 = make_float2(k1, k1);
                            float2 te_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_88) : "l"(*(const unsigned long long*)&ax_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_1551));
                            float2 _f32x2_mul_b_1552 = make_float2(k2, k2);
                            float2 tw_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_88) : "l"(*(const unsigned long long*)&av_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_1552));
                            float _exp2_1553 = approx_exp2(te_88.x);
                            float _exp2_1554 = approx_exp2(te_88.y);
                            float2 e_88 = make_float2(_exp2_1553, _exp2_1554);
                            float _exp2_1555 = approx_exp2(tw_88.x);
                            float _exp2_1556 = approx_exp2(tw_88.y);
                            float2 w_88 = make_float2(_exp2_1555, _exp2_1556);
                            float2 _f32x2_mul_b_1557 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_88) : "l"(*(const unsigned long long*)&ax_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_1557));
                            float _exp2_1558 = approx_exp2_noftz(tp_88.x);
                            float _exp2_1559 = approx_exp2_noftz(tp_88.y);
                            float2 p_88 = make_float2(_exp2_1558, _exp2_1559);
                            float2 d1_88 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_88);
                            float2 d2_88 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_88);
                            float2 dw_88 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_88);
                            float2 ng_88 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_88);
                            float2 nu_88 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_88);
                            float _rcp_1560 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_88, d2_88), dw_88).x);
                            float _rcp_1561 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_88, d2_88), dw_88).y);
                            float2 r_178 = make_float2(_rcp_1560, _rcp_1561);
                            float2 r_177 = r_178;
                            float2 _f32x2_mul_a_1562 = mul_f32x2_noftz(ng_88, nu_88);
                            float2 _f32x2_mul_b_1563 = mul_f32x2_noftz(r_177, make_float2(bl, bl));
                            float2 m_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1562), "l"(*(const unsigned long long*)&_f32x2_mul_b_1563));
                            float2 ns_88 = make_float2(((x_89.x < 0.0f) ? p_88.x : 1.0f), ((x_89.y < 0.0f) ? p_88.y : 1.0f));
                            gate_packed_2[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_88, ns_88))) | sign_89;
                            float2 x_90 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[26]));
                            float2 v_90 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[26]));
                            unsigned int sign_90 = (gate_packed_2[26] ^ up_packed_2[26]) & 2147516416u;
                            float _abs_1564;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1564) : "f"(x_90.x));
                            float _abs_1565;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1565) : "f"(x_90.y));
                            float2 ax_89 = make_float2(_abs_1564, _abs_1565);
                            float _abs_1566;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1566) : "f"(v_90.x));
                            float _abs_1567;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1567) : "f"(v_90.y));
                            float2 av_89 = make_float2(_abs_1566, _abs_1567);
                            float2 _f32x2_mul_b_1568 = make_float2(k1, k1);
                            float2 te_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_89) : "l"(*(const unsigned long long*)&ax_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_1568));
                            float2 _f32x2_mul_b_1569 = make_float2(k2, k2);
                            float2 tw_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_89) : "l"(*(const unsigned long long*)&av_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_1569));
                            float _exp2_1570 = approx_exp2(te_89.x);
                            float _exp2_1571 = approx_exp2(te_89.y);
                            float2 e_89 = make_float2(_exp2_1570, _exp2_1571);
                            float _exp2_1572 = approx_exp2(tw_89.x);
                            float _exp2_1573 = approx_exp2(tw_89.y);
                            float2 w_89 = make_float2(_exp2_1572, _exp2_1573);
                            float2 _f32x2_mul_b_1574 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_89) : "l"(*(const unsigned long long*)&ax_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_1574));
                            float _exp2_1575 = approx_exp2_noftz(tp_89.x);
                            float _exp2_1576 = approx_exp2_noftz(tp_89.y);
                            float2 p_89 = make_float2(_exp2_1575, _exp2_1576);
                            float2 d1_89 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_89);
                            float2 d2_89 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_89);
                            float2 dw_89 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_89);
                            float2 ng_89 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_89);
                            float2 nu_89 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_89);
                            float _rcp_1577 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_89, d2_89), dw_89).x);
                            float _rcp_1578 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_89, d2_89), dw_89).y);
                            float2 r_180 = make_float2(_rcp_1577, _rcp_1578);
                            float2 r_179 = r_180;
                            float2 _f32x2_mul_a_1579 = mul_f32x2_noftz(ng_89, nu_89);
                            float2 _f32x2_mul_b_1580 = mul_f32x2_noftz(r_179, make_float2(bl, bl));
                            float2 m_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1579), "l"(*(const unsigned long long*)&_f32x2_mul_b_1580));
                            float2 ns_89 = make_float2(((x_90.x < 0.0f) ? p_89.x : 1.0f), ((x_90.y < 0.0f) ? p_89.y : 1.0f));
                            gate_packed_2[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_89, ns_89))) | sign_90;
                            float2 x_91 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[27]));
                            float2 v_91 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[27]));
                            unsigned int sign_91 = (gate_packed_2[27] ^ up_packed_2[27]) & 2147516416u;
                            float _abs_1581;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1581) : "f"(x_91.x));
                            float _abs_1582;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1582) : "f"(x_91.y));
                            float2 ax_90 = make_float2(_abs_1581, _abs_1582);
                            float _abs_1583;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1583) : "f"(v_91.x));
                            float _abs_1584;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1584) : "f"(v_91.y));
                            float2 av_90 = make_float2(_abs_1583, _abs_1584);
                            float2 _f32x2_mul_b_1585 = make_float2(k1, k1);
                            float2 te_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_90) : "l"(*(const unsigned long long*)&ax_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_1585));
                            float2 _f32x2_mul_b_1586 = make_float2(k2, k2);
                            float2 tw_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_90) : "l"(*(const unsigned long long*)&av_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_1586));
                            float _exp2_1587 = approx_exp2(te_90.x);
                            float _exp2_1588 = approx_exp2(te_90.y);
                            float2 e_90 = make_float2(_exp2_1587, _exp2_1588);
                            float _exp2_1589 = approx_exp2(tw_90.x);
                            float _exp2_1590 = approx_exp2(tw_90.y);
                            float2 w_90 = make_float2(_exp2_1589, _exp2_1590);
                            float2 _f32x2_mul_b_1591 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_90) : "l"(*(const unsigned long long*)&ax_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_1591));
                            float _exp2_1592 = approx_exp2_noftz(tp_90.x);
                            float _exp2_1593 = approx_exp2_noftz(tp_90.y);
                            float2 p_90 = make_float2(_exp2_1592, _exp2_1593);
                            float2 d1_90 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_90);
                            float2 d2_90 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_90);
                            float2 dw_90 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_90);
                            float2 ng_90 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_90);
                            float2 nu_90 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_90);
                            float _rcp_1594 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_90, d2_90), dw_90).x);
                            float _rcp_1595 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_90, d2_90), dw_90).y);
                            float2 r_182 = make_float2(_rcp_1594, _rcp_1595);
                            float2 r_181 = r_182;
                            float2 _f32x2_mul_a_1596 = mul_f32x2_noftz(ng_90, nu_90);
                            float2 _f32x2_mul_b_1597 = mul_f32x2_noftz(r_181, make_float2(bl, bl));
                            float2 m_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1596), "l"(*(const unsigned long long*)&_f32x2_mul_b_1597));
                            float2 ns_90 = make_float2(((x_91.x < 0.0f) ? p_90.x : 1.0f), ((x_91.y < 0.0f) ? p_90.y : 1.0f));
                            gate_packed_2[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_90, ns_90))) | sign_91;
                            float2 x_92 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[28]));
                            float2 v_92 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[28]));
                            unsigned int sign_92 = (gate_packed_2[28] ^ up_packed_2[28]) & 2147516416u;
                            float _abs_1598;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1598) : "f"(x_92.x));
                            float _abs_1599;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1599) : "f"(x_92.y));
                            float2 ax_91 = make_float2(_abs_1598, _abs_1599);
                            float _abs_1600;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1600) : "f"(v_92.x));
                            float _abs_1601;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1601) : "f"(v_92.y));
                            float2 av_91 = make_float2(_abs_1600, _abs_1601);
                            float2 _f32x2_mul_b_1602 = make_float2(k1, k1);
                            float2 te_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_91) : "l"(*(const unsigned long long*)&ax_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_1602));
                            float2 _f32x2_mul_b_1603 = make_float2(k2, k2);
                            float2 tw_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_91) : "l"(*(const unsigned long long*)&av_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_1603));
                            float _exp2_1604 = approx_exp2(te_91.x);
                            float _exp2_1605 = approx_exp2(te_91.y);
                            float2 e_91 = make_float2(_exp2_1604, _exp2_1605);
                            float _exp2_1606 = approx_exp2(tw_91.x);
                            float _exp2_1607 = approx_exp2(tw_91.y);
                            float2 w_91 = make_float2(_exp2_1606, _exp2_1607);
                            float2 _f32x2_mul_b_1608 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_91) : "l"(*(const unsigned long long*)&ax_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_1608));
                            float _exp2_1609 = approx_exp2_noftz(tp_91.x);
                            float _exp2_1610 = approx_exp2_noftz(tp_91.y);
                            float2 p_91 = make_float2(_exp2_1609, _exp2_1610);
                            float2 d1_91 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_91);
                            float2 d2_91 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_91);
                            float2 dw_91 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_91);
                            float2 ng_91 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_91);
                            float2 nu_91 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_91);
                            float _rcp_1611 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_91, d2_91), dw_91).x);
                            float _rcp_1612 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_91, d2_91), dw_91).y);
                            float2 r_184 = make_float2(_rcp_1611, _rcp_1612);
                            float2 r_183 = r_184;
                            float2 _f32x2_mul_a_1613 = mul_f32x2_noftz(ng_91, nu_91);
                            float2 _f32x2_mul_b_1614 = mul_f32x2_noftz(r_183, make_float2(bl, bl));
                            float2 m_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_91) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1613), "l"(*(const unsigned long long*)&_f32x2_mul_b_1614));
                            float2 ns_91 = make_float2(((x_92.x < 0.0f) ? p_91.x : 1.0f), ((x_92.y < 0.0f) ? p_91.y : 1.0f));
                            gate_packed_2[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_91, ns_91))) | sign_92;
                            float2 x_93 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[29]));
                            float2 v_93 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[29]));
                            unsigned int sign_93 = (gate_packed_2[29] ^ up_packed_2[29]) & 2147516416u;
                            float _abs_1615;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1615) : "f"(x_93.x));
                            float _abs_1616;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1616) : "f"(x_93.y));
                            float2 ax_92 = make_float2(_abs_1615, _abs_1616);
                            float _abs_1617;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1617) : "f"(v_93.x));
                            float _abs_1618;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1618) : "f"(v_93.y));
                            float2 av_92 = make_float2(_abs_1617, _abs_1618);
                            float2 _f32x2_mul_b_1619 = make_float2(k1, k1);
                            float2 te_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_92) : "l"(*(const unsigned long long*)&ax_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_1619));
                            float2 _f32x2_mul_b_1620 = make_float2(k2, k2);
                            float2 tw_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_92) : "l"(*(const unsigned long long*)&av_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_1620));
                            float _exp2_1621 = approx_exp2(te_92.x);
                            float _exp2_1622 = approx_exp2(te_92.y);
                            float2 e_92 = make_float2(_exp2_1621, _exp2_1622);
                            float _exp2_1623 = approx_exp2(tw_92.x);
                            float _exp2_1624 = approx_exp2(tw_92.y);
                            float2 w_92 = make_float2(_exp2_1623, _exp2_1624);
                            float2 _f32x2_mul_b_1625 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_92) : "l"(*(const unsigned long long*)&ax_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_1625));
                            float _exp2_1626 = approx_exp2_noftz(tp_92.x);
                            float _exp2_1627 = approx_exp2_noftz(tp_92.y);
                            float2 p_92 = make_float2(_exp2_1626, _exp2_1627);
                            float2 d1_92 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_92);
                            float2 d2_92 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_92);
                            float2 dw_92 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_92);
                            float2 ng_92 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_92);
                            float2 nu_92 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_92);
                            float _rcp_1628 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_92, d2_92), dw_92).x);
                            float _rcp_1629 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_92, d2_92), dw_92).y);
                            float2 r_186 = make_float2(_rcp_1628, _rcp_1629);
                            float2 r_185 = r_186;
                            float2 _f32x2_mul_a_1630 = mul_f32x2_noftz(ng_92, nu_92);
                            float2 _f32x2_mul_b_1631 = mul_f32x2_noftz(r_185, make_float2(bl, bl));
                            float2 m_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1630), "l"(*(const unsigned long long*)&_f32x2_mul_b_1631));
                            float2 ns_92 = make_float2(((x_93.x < 0.0f) ? p_92.x : 1.0f), ((x_93.y < 0.0f) ? p_92.y : 1.0f));
                            gate_packed_2[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_92, ns_92))) | sign_93;
                            float2 x_94 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[30]));
                            float2 v_94 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[30]));
                            unsigned int sign_94 = (gate_packed_2[30] ^ up_packed_2[30]) & 2147516416u;
                            float _abs_1632;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1632) : "f"(x_94.x));
                            float _abs_1633;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1633) : "f"(x_94.y));
                            float2 ax_93 = make_float2(_abs_1632, _abs_1633);
                            float _abs_1634;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1634) : "f"(v_94.x));
                            float _abs_1635;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1635) : "f"(v_94.y));
                            float2 av_93 = make_float2(_abs_1634, _abs_1635);
                            float2 _f32x2_mul_b_1636 = make_float2(k1, k1);
                            float2 te_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_93) : "l"(*(const unsigned long long*)&ax_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_1636));
                            float2 _f32x2_mul_b_1637 = make_float2(k2, k2);
                            float2 tw_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_93) : "l"(*(const unsigned long long*)&av_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_1637));
                            float _exp2_1638 = approx_exp2(te_93.x);
                            float _exp2_1639 = approx_exp2(te_93.y);
                            float2 e_93 = make_float2(_exp2_1638, _exp2_1639);
                            float _exp2_1640 = approx_exp2(tw_93.x);
                            float _exp2_1641 = approx_exp2(tw_93.y);
                            float2 w_93 = make_float2(_exp2_1640, _exp2_1641);
                            float2 _f32x2_mul_b_1642 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_93) : "l"(*(const unsigned long long*)&ax_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_1642));
                            float _exp2_1643 = approx_exp2_noftz(tp_93.x);
                            float _exp2_1644 = approx_exp2_noftz(tp_93.y);
                            float2 p_93 = make_float2(_exp2_1643, _exp2_1644);
                            float2 d1_93 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_93);
                            float2 d2_93 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_93);
                            float2 dw_93 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_93);
                            float2 ng_93 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_93);
                            float2 nu_93 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_93);
                            float _rcp_1645 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_93, d2_93), dw_93).x);
                            float _rcp_1646 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_93, d2_93), dw_93).y);
                            float2 r_188 = make_float2(_rcp_1645, _rcp_1646);
                            float2 r_187 = r_188;
                            float2 _f32x2_mul_a_1647 = mul_f32x2_noftz(ng_93, nu_93);
                            float2 _f32x2_mul_b_1648 = mul_f32x2_noftz(r_187, make_float2(bl, bl));
                            float2 m_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_93) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1647), "l"(*(const unsigned long long*)&_f32x2_mul_b_1648));
                            float2 ns_93 = make_float2(((x_94.x < 0.0f) ? p_93.x : 1.0f), ((x_94.y < 0.0f) ? p_93.y : 1.0f));
                            gate_packed_2[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_93, ns_93))) | sign_94;
                            float2 x_95 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[31]));
                            float2 v_95 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[31]));
                            unsigned int sign_95 = (gate_packed_2[31] ^ up_packed_2[31]) & 2147516416u;
                            float _abs_1649;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1649) : "f"(x_95.x));
                            float _abs_1650;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1650) : "f"(x_95.y));
                            float2 ax_94 = make_float2(_abs_1649, _abs_1650);
                            float _abs_1651;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1651) : "f"(v_95.x));
                            float _abs_1652;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1652) : "f"(v_95.y));
                            float2 av_94 = make_float2(_abs_1651, _abs_1652);
                            float2 _f32x2_mul_b_1653 = make_float2(k1, k1);
                            float2 te_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_94) : "l"(*(const unsigned long long*)&ax_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_1653));
                            float2 _f32x2_mul_b_1654 = make_float2(k2, k2);
                            float2 tw_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_94) : "l"(*(const unsigned long long*)&av_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_1654));
                            float _exp2_1655 = approx_exp2(te_94.x);
                            float _exp2_1656 = approx_exp2(te_94.y);
                            float2 e_94 = make_float2(_exp2_1655, _exp2_1656);
                            float _exp2_1657 = approx_exp2(tw_94.x);
                            float _exp2_1658 = approx_exp2(tw_94.y);
                            float2 w_94 = make_float2(_exp2_1657, _exp2_1658);
                            float2 _f32x2_mul_b_1659 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_94) : "l"(*(const unsigned long long*)&ax_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_1659));
                            float _exp2_1660 = approx_exp2_noftz(tp_94.x);
                            float _exp2_1661 = approx_exp2_noftz(tp_94.y);
                            float2 p_94 = make_float2(_exp2_1660, _exp2_1661);
                            float2 d1_94 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_94);
                            float2 d2_94 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_94);
                            float2 dw_94 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_94);
                            float2 ng_94 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_94);
                            float2 nu_94 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_94);
                            float _rcp_1662 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_94, d2_94), dw_94).x);
                            float _rcp_1663 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_94, d2_94), dw_94).y);
                            float2 r_190 = make_float2(_rcp_1662, _rcp_1663);
                            float2 r_189 = r_190;
                            float2 _f32x2_mul_a_1664 = mul_f32x2_noftz(ng_94, nu_94);
                            float2 _f32x2_mul_b_1665 = mul_f32x2_noftz(r_189, make_float2(bl, bl));
                            float2 m_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1664), "l"(*(const unsigned long long*)&_f32x2_mul_b_1665));
                            float2 ns_94 = make_float2(((x_95.x < 0.0f) ? p_94.x : 1.0f), ((x_95.y < 0.0f) ? p_94.y : 1.0f));
                            gate_packed_2[31] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_94, ns_94))) | sign_95;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_16 = tid / 32;
                            int lane_16 = tid % 32;
                            #pragma unroll
                            for (int half_17 = 0; half_17 < 2; half_17++) {
                                #pragma unroll
                                for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                    int row_18 = warp_16 * 32 + half_17 * 16 + lane_16 % 16;
                                    int col_18 = col_tile_17 * 16 + lane_16 / 16 * 8;
                                    unsigned int address_17 = d_smem_addr + 16384 + (unsigned int)((row_18 * 32 + col_18) * 2);
                                    address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                    int offset_18 = 16 + half_17 * 8 + col_tile_17 * 4;
                                    uint32_t _stmatrix_addr_1666 = static_cast<uint32_t>(address_17);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1666), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int gate_packed_3[32];
                            unsigned int up_packed_3[32];
                            unsigned int address_3_12 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 320;
                            float _tmem_load_12[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                                : "r"(address_3_12));
                            float _tmem_load_32_12[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[15]))
                                : "r"(address_3_12 + 128));
                            gate_packed_3[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[0], _tmem_load_12[1])));
                            up_packed_3[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[0], _tmem_load_32_12[1])));
                            gate_packed_3[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[2], _tmem_load_12[3])));
                            up_packed_3[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[2], _tmem_load_32_12[3])));
                            gate_packed_3[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[4], _tmem_load_12[5])));
                            up_packed_3[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[4], _tmem_load_32_12[5])));
                            gate_packed_3[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[6], _tmem_load_12[7])));
                            up_packed_3[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[6], _tmem_load_32_12[7])));
                            gate_packed_3[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[8], _tmem_load_12[9])));
                            up_packed_3[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[8], _tmem_load_32_12[9])));
                            gate_packed_3[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[10], _tmem_load_12[11])));
                            up_packed_3[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[10], _tmem_load_32_12[11])));
                            gate_packed_3[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[12], _tmem_load_12[13])));
                            up_packed_3[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[12], _tmem_load_32_12[13])));
                            gate_packed_3[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[14], _tmem_load_12[15])));
                            up_packed_3[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[14], _tmem_load_32_12[15])));
                            unsigned int address_3_13 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 320;
                            float _tmem_load_13[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                                : "r"(address_3_13));
                            float _tmem_load_32_13[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[15]))
                                : "r"(address_3_13 + 128));
                            gate_packed_3[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[0], _tmem_load_13[1])));
                            up_packed_3[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[0], _tmem_load_32_13[1])));
                            gate_packed_3[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[2], _tmem_load_13[3])));
                            up_packed_3[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[2], _tmem_load_32_13[3])));
                            gate_packed_3[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[4], _tmem_load_13[5])));
                            up_packed_3[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[4], _tmem_load_32_13[5])));
                            gate_packed_3[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[6], _tmem_load_13[7])));
                            up_packed_3[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[6], _tmem_load_32_13[7])));
                            gate_packed_3[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[8], _tmem_load_13[9])));
                            up_packed_3[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[8], _tmem_load_32_13[9])));
                            gate_packed_3[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[10], _tmem_load_13[11])));
                            up_packed_3[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[10], _tmem_load_32_13[11])));
                            gate_packed_3[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[12], _tmem_load_13[13])));
                            up_packed_3[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[12], _tmem_load_32_13[13])));
                            gate_packed_3[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[14], _tmem_load_13[15])));
                            up_packed_3[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[14], _tmem_load_32_13[15])));
                            unsigned int address_3_14 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 352;
                            float _tmem_load_14[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                                : "r"(address_3_14));
                            float _tmem_load_32_14[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[15]))
                                : "r"(address_3_14 + 128));
                            gate_packed_3[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[0], _tmem_load_14[1])));
                            up_packed_3[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[0], _tmem_load_32_14[1])));
                            gate_packed_3[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[2], _tmem_load_14[3])));
                            up_packed_3[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[2], _tmem_load_32_14[3])));
                            gate_packed_3[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[4], _tmem_load_14[5])));
                            up_packed_3[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[4], _tmem_load_32_14[5])));
                            gate_packed_3[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[6], _tmem_load_14[7])));
                            up_packed_3[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[6], _tmem_load_32_14[7])));
                            gate_packed_3[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[8], _tmem_load_14[9])));
                            up_packed_3[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[8], _tmem_load_32_14[9])));
                            gate_packed_3[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[10], _tmem_load_14[11])));
                            up_packed_3[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[10], _tmem_load_32_14[11])));
                            gate_packed_3[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[12], _tmem_load_14[13])));
                            up_packed_3[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[12], _tmem_load_32_14[13])));
                            gate_packed_3[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[14], _tmem_load_14[15])));
                            up_packed_3[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[14], _tmem_load_32_14[15])));
                            unsigned int address_3_15 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 352;
                            float _tmem_load_15[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                                : "r"(address_3_15));
                            float _tmem_load_32_15[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[15]))
                                : "r"(address_3_15 + 128));
                            gate_packed_3[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[0], _tmem_load_15[1])));
                            up_packed_3[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[0], _tmem_load_32_15[1])));
                            gate_packed_3[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[2], _tmem_load_15[3])));
                            up_packed_3[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[2], _tmem_load_32_15[3])));
                            gate_packed_3[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[4], _tmem_load_15[5])));
                            up_packed_3[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[4], _tmem_load_32_15[5])));
                            gate_packed_3[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[6], _tmem_load_15[7])));
                            up_packed_3[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[6], _tmem_load_32_15[7])));
                            gate_packed_3[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[8], _tmem_load_15[9])));
                            up_packed_3[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[8], _tmem_load_32_15[9])));
                            gate_packed_3[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[10], _tmem_load_15[11])));
                            up_packed_3[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[10], _tmem_load_32_15[11])));
                            gate_packed_3[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[12], _tmem_load_15[13])));
                            up_packed_3[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[12], _tmem_load_32_15[13])));
                            gate_packed_3[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[14], _tmem_load_15[15])));
                            up_packed_3[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[14], _tmem_load_32_15[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_17 = tid / 32;
                            int lane_17 = tid % 32;
                            #pragma unroll
                            for (int half_18 = 0; half_18 < 2; half_18++) {
                                #pragma unroll
                                for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                    int row_19 = warp_17 * 32 + half_18 * 16 + lane_17 % 16;
                                    int col_19 = col_tile_18 * 16 + lane_17 / 16 * 8;
                                    unsigned int address_18 = d_smem_addr + (unsigned int)((row_19 * 32 + col_19) * 2);
                                    address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                    int offset_19 = half_18 * 8 + col_tile_18 * 4;
                                    uint32_t _stmatrix_addr_1667 = static_cast<uint32_t>(address_18);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1667), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_18 = tid / 32;
                            int lane_18 = tid % 32;
                            #pragma unroll
                            for (int half_19 = 0; half_19 < 2; half_19++) {
                                #pragma unroll
                                for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                    int row_20 = warp_18 * 32 + half_19 * 16 + lane_18 % 16;
                                    int col_20 = col_tile_19 * 16 + lane_18 / 16 * 8;
                                    unsigned int address_19 = d_smem_addr + 8192 + (unsigned int)((row_20 * 32 + col_20) * 2);
                                    address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                    int offset_20 = half_19 * 8 + col_tile_19 * 4;
                                    uint32_t _stmatrix_addr_1668 = static_cast<uint32_t>(address_19);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1668), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_19 = tid / 32;
                            int lane_19 = tid % 32;
                            #pragma unroll
                            for (int half_20 = 0; half_20 < 2; half_20++) {
                                #pragma unroll
                                for (int col_tile_20 = 0; col_tile_20 < 2; col_tile_20++) {
                                    int row_21 = warp_19 * 32 + half_20 * 16 + lane_19 % 16;
                                    int col_21 = col_tile_20 * 16 + lane_19 / 16 * 8;
                                    unsigned int address_20 = d_smem_addr + 16384 + (unsigned int)((row_21 * 32 + col_21) * 2);
                                    address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                    int offset_21 = 16 + half_20 * 8 + col_tile_20 * 4;
                                    uint32_t _stmatrix_addr_1669 = static_cast<uint32_t>(address_20);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1669), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_20 = tid / 32;
                            int lane_20 = tid % 32;
                            #pragma unroll
                            for (int half_21 = 0; half_21 < 2; half_21++) {
                                #pragma unroll
                                for (int col_tile_21 = 0; col_tile_21 < 2; col_tile_21++) {
                                    int row_22 = warp_20 * 32 + half_21 * 16 + lane_20 % 16;
                                    int col_22 = col_tile_21 * 16 + lane_20 / 16 * 8;
                                    unsigned int address_21 = d_smem_addr + (unsigned int)((row_22 * 32 + col_22) * 2);
                                    address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                    int offset_22 = 16 + half_21 * 8 + col_tile_21 * 4;
                                    uint32_t _stmatrix_addr_1670 = static_cast<uint32_t>(address_21);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1670), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_96 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[0]));
                            float2 v_96 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[0]));
                            unsigned int sign_96 = (gate_packed_3[0] ^ up_packed_3[0]) & 2147516416u;
                            float _abs_1671;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1671) : "f"(x_96.x));
                            float _abs_1672;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1672) : "f"(x_96.y));
                            float2 ax_95 = make_float2(_abs_1671, _abs_1672);
                            float _abs_1673;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1673) : "f"(v_96.x));
                            float _abs_1674;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1674) : "f"(v_96.y));
                            float2 av_95 = make_float2(_abs_1673, _abs_1674);
                            float2 _f32x2_mul_b_1675 = make_float2(k1, k1);
                            float2 te_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_95) : "l"(*(const unsigned long long*)&ax_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_1675));
                            float2 _f32x2_mul_b_1676 = make_float2(k2, k2);
                            float2 tw_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_95) : "l"(*(const unsigned long long*)&av_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_1676));
                            float _exp2_1677 = approx_exp2(te_95.x);
                            float _exp2_1678 = approx_exp2(te_95.y);
                            float2 e_95 = make_float2(_exp2_1677, _exp2_1678);
                            float _exp2_1679 = approx_exp2(tw_95.x);
                            float _exp2_1680 = approx_exp2(tw_95.y);
                            float2 w_95 = make_float2(_exp2_1679, _exp2_1680);
                            float2 _f32x2_mul_b_1681 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_95) : "l"(*(const unsigned long long*)&ax_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_1681));
                            float _exp2_1682 = approx_exp2_noftz(tp_95.x);
                            float _exp2_1683 = approx_exp2_noftz(tp_95.y);
                            float2 p_95 = make_float2(_exp2_1682, _exp2_1683);
                            float2 d1_95 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_95);
                            float2 d2_95 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_95);
                            float2 dw_95 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_95);
                            float2 ng_95 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_95);
                            float2 nu_95 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_95);
                            float _rcp_1684 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_95, d2_95), dw_95).x);
                            float _rcp_1685 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_95, d2_95), dw_95).y);
                            float2 r_192 = make_float2(_rcp_1684, _rcp_1685);
                            float2 r_191 = r_192;
                            float2 _f32x2_mul_a_1686 = mul_f32x2_noftz(ng_95, nu_95);
                            float2 _f32x2_mul_b_1687 = mul_f32x2_noftz(r_191, make_float2(bl, bl));
                            float2 m_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_95) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1686), "l"(*(const unsigned long long*)&_f32x2_mul_b_1687));
                            float2 ns_95 = make_float2(((x_96.x < 0.0f) ? p_95.x : 1.0f), ((x_96.y < 0.0f) ? p_95.y : 1.0f));
                            gate_packed_3[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_95, ns_95))) | sign_96;
                            float2 x_97 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[1]));
                            float2 v_97 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[1]));
                            unsigned int sign_97 = (gate_packed_3[1] ^ up_packed_3[1]) & 2147516416u;
                            float _abs_1688;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1688) : "f"(x_97.x));
                            float _abs_1689;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1689) : "f"(x_97.y));
                            float2 ax_96 = make_float2(_abs_1688, _abs_1689);
                            float _abs_1690;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1690) : "f"(v_97.x));
                            float _abs_1691;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1691) : "f"(v_97.y));
                            float2 av_96 = make_float2(_abs_1690, _abs_1691);
                            float2 _f32x2_mul_b_1692 = make_float2(k1, k1);
                            float2 te_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_96) : "l"(*(const unsigned long long*)&ax_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_1692));
                            float2 _f32x2_mul_b_1693 = make_float2(k2, k2);
                            float2 tw_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_96) : "l"(*(const unsigned long long*)&av_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_1693));
                            float _exp2_1694 = approx_exp2(te_96.x);
                            float _exp2_1695 = approx_exp2(te_96.y);
                            float2 e_96 = make_float2(_exp2_1694, _exp2_1695);
                            float _exp2_1696 = approx_exp2(tw_96.x);
                            float _exp2_1697 = approx_exp2(tw_96.y);
                            float2 w_96 = make_float2(_exp2_1696, _exp2_1697);
                            float2 _f32x2_mul_b_1698 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_96) : "l"(*(const unsigned long long*)&ax_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_1698));
                            float _exp2_1699 = approx_exp2_noftz(tp_96.x);
                            float _exp2_1700 = approx_exp2_noftz(tp_96.y);
                            float2 p_96 = make_float2(_exp2_1699, _exp2_1700);
                            float2 d1_96 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_96);
                            float2 d2_96 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_96);
                            float2 dw_96 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_96);
                            float2 ng_96 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_96);
                            float2 nu_96 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_96);
                            float _rcp_1701 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_96, d2_96), dw_96).x);
                            float _rcp_1702 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_96, d2_96), dw_96).y);
                            float2 r_194 = make_float2(_rcp_1701, _rcp_1702);
                            float2 r_193 = r_194;
                            float2 _f32x2_mul_a_1703 = mul_f32x2_noftz(ng_96, nu_96);
                            float2 _f32x2_mul_b_1704 = mul_f32x2_noftz(r_193, make_float2(bl, bl));
                            float2 m_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_96) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1703), "l"(*(const unsigned long long*)&_f32x2_mul_b_1704));
                            float2 ns_96 = make_float2(((x_97.x < 0.0f) ? p_96.x : 1.0f), ((x_97.y < 0.0f) ? p_96.y : 1.0f));
                            gate_packed_3[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_96, ns_96))) | sign_97;
                            float2 x_98 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[2]));
                            float2 v_98 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[2]));
                            unsigned int sign_98 = (gate_packed_3[2] ^ up_packed_3[2]) & 2147516416u;
                            float _abs_1705;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1705) : "f"(x_98.x));
                            float _abs_1706;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1706) : "f"(x_98.y));
                            float2 ax_97 = make_float2(_abs_1705, _abs_1706);
                            float _abs_1707;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1707) : "f"(v_98.x));
                            float _abs_1708;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1708) : "f"(v_98.y));
                            float2 av_97 = make_float2(_abs_1707, _abs_1708);
                            float2 _f32x2_mul_b_1709 = make_float2(k1, k1);
                            float2 te_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_97) : "l"(*(const unsigned long long*)&ax_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_1709));
                            float2 _f32x2_mul_b_1710 = make_float2(k2, k2);
                            float2 tw_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_97) : "l"(*(const unsigned long long*)&av_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_1710));
                            float _exp2_1711 = approx_exp2(te_97.x);
                            float _exp2_1712 = approx_exp2(te_97.y);
                            float2 e_97 = make_float2(_exp2_1711, _exp2_1712);
                            float _exp2_1713 = approx_exp2(tw_97.x);
                            float _exp2_1714 = approx_exp2(tw_97.y);
                            float2 w_97 = make_float2(_exp2_1713, _exp2_1714);
                            float2 _f32x2_mul_b_1715 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_97) : "l"(*(const unsigned long long*)&ax_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_1715));
                            float _exp2_1716 = approx_exp2_noftz(tp_97.x);
                            float _exp2_1717 = approx_exp2_noftz(tp_97.y);
                            float2 p_97 = make_float2(_exp2_1716, _exp2_1717);
                            float2 d1_97 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_97);
                            float2 d2_97 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_97);
                            float2 dw_97 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_97);
                            float2 ng_97 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_97);
                            float2 nu_97 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_97);
                            float _rcp_1718 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_97, d2_97), dw_97).x);
                            float _rcp_1719 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_97, d2_97), dw_97).y);
                            float2 r_196 = make_float2(_rcp_1718, _rcp_1719);
                            float2 r_195 = r_196;
                            float2 _f32x2_mul_a_1720 = mul_f32x2_noftz(ng_97, nu_97);
                            float2 _f32x2_mul_b_1721 = mul_f32x2_noftz(r_195, make_float2(bl, bl));
                            float2 m_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_97) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1720), "l"(*(const unsigned long long*)&_f32x2_mul_b_1721));
                            float2 ns_97 = make_float2(((x_98.x < 0.0f) ? p_97.x : 1.0f), ((x_98.y < 0.0f) ? p_97.y : 1.0f));
                            gate_packed_3[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_97, ns_97))) | sign_98;
                            float2 x_99 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[3]));
                            float2 v_99 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[3]));
                            unsigned int sign_99 = (gate_packed_3[3] ^ up_packed_3[3]) & 2147516416u;
                            float _abs_1722;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1722) : "f"(x_99.x));
                            float _abs_1723;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1723) : "f"(x_99.y));
                            float2 ax_98 = make_float2(_abs_1722, _abs_1723);
                            float _abs_1724;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1724) : "f"(v_99.x));
                            float _abs_1725;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1725) : "f"(v_99.y));
                            float2 av_98 = make_float2(_abs_1724, _abs_1725);
                            float2 _f32x2_mul_b_1726 = make_float2(k1, k1);
                            float2 te_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_98) : "l"(*(const unsigned long long*)&ax_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_1726));
                            float2 _f32x2_mul_b_1727 = make_float2(k2, k2);
                            float2 tw_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_98) : "l"(*(const unsigned long long*)&av_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_1727));
                            float _exp2_1728 = approx_exp2(te_98.x);
                            float _exp2_1729 = approx_exp2(te_98.y);
                            float2 e_98 = make_float2(_exp2_1728, _exp2_1729);
                            float _exp2_1730 = approx_exp2(tw_98.x);
                            float _exp2_1731 = approx_exp2(tw_98.y);
                            float2 w_98 = make_float2(_exp2_1730, _exp2_1731);
                            float2 _f32x2_mul_b_1732 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_98) : "l"(*(const unsigned long long*)&ax_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_1732));
                            float _exp2_1733 = approx_exp2_noftz(tp_98.x);
                            float _exp2_1734 = approx_exp2_noftz(tp_98.y);
                            float2 p_98 = make_float2(_exp2_1733, _exp2_1734);
                            float2 d1_98 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_98);
                            float2 d2_98 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_98);
                            float2 dw_98 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_98);
                            float2 ng_98 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_98);
                            float2 nu_98 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_98);
                            float _rcp_1735 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_98, d2_98), dw_98).x);
                            float _rcp_1736 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_98, d2_98), dw_98).y);
                            float2 r_198 = make_float2(_rcp_1735, _rcp_1736);
                            float2 r_197 = r_198;
                            float2 _f32x2_mul_a_1737 = mul_f32x2_noftz(ng_98, nu_98);
                            float2 _f32x2_mul_b_1738 = mul_f32x2_noftz(r_197, make_float2(bl, bl));
                            float2 m_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1737), "l"(*(const unsigned long long*)&_f32x2_mul_b_1738));
                            float2 ns_98 = make_float2(((x_99.x < 0.0f) ? p_98.x : 1.0f), ((x_99.y < 0.0f) ? p_98.y : 1.0f));
                            gate_packed_3[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_98, ns_98))) | sign_99;
                            float2 x_100 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[4]));
                            float2 v_100 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[4]));
                            unsigned int sign_100 = (gate_packed_3[4] ^ up_packed_3[4]) & 2147516416u;
                            float _abs_1739;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1739) : "f"(x_100.x));
                            float _abs_1740;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1740) : "f"(x_100.y));
                            float2 ax_99 = make_float2(_abs_1739, _abs_1740);
                            float _abs_1741;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1741) : "f"(v_100.x));
                            float _abs_1742;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1742) : "f"(v_100.y));
                            float2 av_99 = make_float2(_abs_1741, _abs_1742);
                            float2 _f32x2_mul_b_1743 = make_float2(k1, k1);
                            float2 te_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_99) : "l"(*(const unsigned long long*)&ax_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_1743));
                            float2 _f32x2_mul_b_1744 = make_float2(k2, k2);
                            float2 tw_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_99) : "l"(*(const unsigned long long*)&av_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_1744));
                            float _exp2_1745 = approx_exp2(te_99.x);
                            float _exp2_1746 = approx_exp2(te_99.y);
                            float2 e_99 = make_float2(_exp2_1745, _exp2_1746);
                            float _exp2_1747 = approx_exp2(tw_99.x);
                            float _exp2_1748 = approx_exp2(tw_99.y);
                            float2 w_99 = make_float2(_exp2_1747, _exp2_1748);
                            float2 _f32x2_mul_b_1749 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_99) : "l"(*(const unsigned long long*)&ax_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_1749));
                            float _exp2_1750 = approx_exp2_noftz(tp_99.x);
                            float _exp2_1751 = approx_exp2_noftz(tp_99.y);
                            float2 p_99 = make_float2(_exp2_1750, _exp2_1751);
                            float2 d1_99 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_99);
                            float2 d2_99 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_99);
                            float2 dw_99 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_99);
                            float2 ng_99 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_99);
                            float2 nu_99 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_99);
                            float _rcp_1752 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_99, d2_99), dw_99).x);
                            float _rcp_1753 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_99, d2_99), dw_99).y);
                            float2 r_200 = make_float2(_rcp_1752, _rcp_1753);
                            float2 r_199 = r_200;
                            float2 _f32x2_mul_a_1754 = mul_f32x2_noftz(ng_99, nu_99);
                            float2 _f32x2_mul_b_1755 = mul_f32x2_noftz(r_199, make_float2(bl, bl));
                            float2 m_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_99) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1754), "l"(*(const unsigned long long*)&_f32x2_mul_b_1755));
                            float2 ns_99 = make_float2(((x_100.x < 0.0f) ? p_99.x : 1.0f), ((x_100.y < 0.0f) ? p_99.y : 1.0f));
                            gate_packed_3[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_99, ns_99))) | sign_100;
                            float2 x_101 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[5]));
                            float2 v_101 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[5]));
                            unsigned int sign_101 = (gate_packed_3[5] ^ up_packed_3[5]) & 2147516416u;
                            float _abs_1756;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1756) : "f"(x_101.x));
                            float _abs_1757;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1757) : "f"(x_101.y));
                            float2 ax_100 = make_float2(_abs_1756, _abs_1757);
                            float _abs_1758;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1758) : "f"(v_101.x));
                            float _abs_1759;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1759) : "f"(v_101.y));
                            float2 av_100 = make_float2(_abs_1758, _abs_1759);
                            float2 _f32x2_mul_b_1760 = make_float2(k1, k1);
                            float2 te_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_100) : "l"(*(const unsigned long long*)&ax_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_1760));
                            float2 _f32x2_mul_b_1761 = make_float2(k2, k2);
                            float2 tw_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_100) : "l"(*(const unsigned long long*)&av_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_1761));
                            float _exp2_1762 = approx_exp2(te_100.x);
                            float _exp2_1763 = approx_exp2(te_100.y);
                            float2 e_100 = make_float2(_exp2_1762, _exp2_1763);
                            float _exp2_1764 = approx_exp2(tw_100.x);
                            float _exp2_1765 = approx_exp2(tw_100.y);
                            float2 w_100 = make_float2(_exp2_1764, _exp2_1765);
                            float2 _f32x2_mul_b_1766 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_100) : "l"(*(const unsigned long long*)&ax_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_1766));
                            float _exp2_1767 = approx_exp2_noftz(tp_100.x);
                            float _exp2_1768 = approx_exp2_noftz(tp_100.y);
                            float2 p_100 = make_float2(_exp2_1767, _exp2_1768);
                            float2 d1_100 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_100);
                            float2 d2_100 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_100);
                            float2 dw_100 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_100);
                            float2 ng_100 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_100);
                            float2 nu_100 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_100);
                            float _rcp_1769 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_100, d2_100), dw_100).x);
                            float _rcp_1770 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_100, d2_100), dw_100).y);
                            float2 r_202 = make_float2(_rcp_1769, _rcp_1770);
                            float2 r_201 = r_202;
                            float2 _f32x2_mul_a_1771 = mul_f32x2_noftz(ng_100, nu_100);
                            float2 _f32x2_mul_b_1772 = mul_f32x2_noftz(r_201, make_float2(bl, bl));
                            float2 m_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_100) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1771), "l"(*(const unsigned long long*)&_f32x2_mul_b_1772));
                            float2 ns_100 = make_float2(((x_101.x < 0.0f) ? p_100.x : 1.0f), ((x_101.y < 0.0f) ? p_100.y : 1.0f));
                            gate_packed_3[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_100, ns_100))) | sign_101;
                            float2 x_102 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[6]));
                            float2 v_102 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[6]));
                            unsigned int sign_102 = (gate_packed_3[6] ^ up_packed_3[6]) & 2147516416u;
                            float _abs_1773;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1773) : "f"(x_102.x));
                            float _abs_1774;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1774) : "f"(x_102.y));
                            float2 ax_101 = make_float2(_abs_1773, _abs_1774);
                            float _abs_1775;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1775) : "f"(v_102.x));
                            float _abs_1776;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1776) : "f"(v_102.y));
                            float2 av_101 = make_float2(_abs_1775, _abs_1776);
                            float2 _f32x2_mul_b_1777 = make_float2(k1, k1);
                            float2 te_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_101) : "l"(*(const unsigned long long*)&ax_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_1777));
                            float2 _f32x2_mul_b_1778 = make_float2(k2, k2);
                            float2 tw_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_101) : "l"(*(const unsigned long long*)&av_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_1778));
                            float _exp2_1779 = approx_exp2(te_101.x);
                            float _exp2_1780 = approx_exp2(te_101.y);
                            float2 e_101 = make_float2(_exp2_1779, _exp2_1780);
                            float _exp2_1781 = approx_exp2(tw_101.x);
                            float _exp2_1782 = approx_exp2(tw_101.y);
                            float2 w_101 = make_float2(_exp2_1781, _exp2_1782);
                            float2 _f32x2_mul_b_1783 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_101) : "l"(*(const unsigned long long*)&ax_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_1783));
                            float _exp2_1784 = approx_exp2_noftz(tp_101.x);
                            float _exp2_1785 = approx_exp2_noftz(tp_101.y);
                            float2 p_101 = make_float2(_exp2_1784, _exp2_1785);
                            float2 d1_101 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_101);
                            float2 d2_101 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_101);
                            float2 dw_101 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_101);
                            float2 ng_101 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_101);
                            float2 nu_101 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_101);
                            float _rcp_1786 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_101, d2_101), dw_101).x);
                            float _rcp_1787 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_101, d2_101), dw_101).y);
                            float2 r_204 = make_float2(_rcp_1786, _rcp_1787);
                            float2 r_203 = r_204;
                            float2 _f32x2_mul_a_1788 = mul_f32x2_noftz(ng_101, nu_101);
                            float2 _f32x2_mul_b_1789 = mul_f32x2_noftz(r_203, make_float2(bl, bl));
                            float2 m_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_101) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1788), "l"(*(const unsigned long long*)&_f32x2_mul_b_1789));
                            float2 ns_101 = make_float2(((x_102.x < 0.0f) ? p_101.x : 1.0f), ((x_102.y < 0.0f) ? p_101.y : 1.0f));
                            gate_packed_3[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_101, ns_101))) | sign_102;
                            float2 x_103 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[7]));
                            float2 v_103 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[7]));
                            unsigned int sign_103 = (gate_packed_3[7] ^ up_packed_3[7]) & 2147516416u;
                            float _abs_1790;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1790) : "f"(x_103.x));
                            float _abs_1791;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1791) : "f"(x_103.y));
                            float2 ax_102 = make_float2(_abs_1790, _abs_1791);
                            float _abs_1792;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1792) : "f"(v_103.x));
                            float _abs_1793;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1793) : "f"(v_103.y));
                            float2 av_102 = make_float2(_abs_1792, _abs_1793);
                            float2 _f32x2_mul_b_1794 = make_float2(k1, k1);
                            float2 te_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_102) : "l"(*(const unsigned long long*)&ax_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_1794));
                            float2 _f32x2_mul_b_1795 = make_float2(k2, k2);
                            float2 tw_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_102) : "l"(*(const unsigned long long*)&av_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_1795));
                            float _exp2_1796 = approx_exp2(te_102.x);
                            float _exp2_1797 = approx_exp2(te_102.y);
                            float2 e_102 = make_float2(_exp2_1796, _exp2_1797);
                            float _exp2_1798 = approx_exp2(tw_102.x);
                            float _exp2_1799 = approx_exp2(tw_102.y);
                            float2 w_102 = make_float2(_exp2_1798, _exp2_1799);
                            float2 _f32x2_mul_b_1800 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_102) : "l"(*(const unsigned long long*)&ax_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_1800));
                            float _exp2_1801 = approx_exp2_noftz(tp_102.x);
                            float _exp2_1802 = approx_exp2_noftz(tp_102.y);
                            float2 p_102 = make_float2(_exp2_1801, _exp2_1802);
                            float2 d1_102 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_102);
                            float2 d2_102 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_102);
                            float2 dw_102 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_102);
                            float2 ng_102 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_102);
                            float2 nu_102 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_102);
                            float _rcp_1803 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_102, d2_102), dw_102).x);
                            float _rcp_1804 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_102, d2_102), dw_102).y);
                            float2 r_206 = make_float2(_rcp_1803, _rcp_1804);
                            float2 r_205 = r_206;
                            float2 _f32x2_mul_a_1805 = mul_f32x2_noftz(ng_102, nu_102);
                            float2 _f32x2_mul_b_1806 = mul_f32x2_noftz(r_205, make_float2(bl, bl));
                            float2 m_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_102) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1805), "l"(*(const unsigned long long*)&_f32x2_mul_b_1806));
                            float2 ns_102 = make_float2(((x_103.x < 0.0f) ? p_102.x : 1.0f), ((x_103.y < 0.0f) ? p_102.y : 1.0f));
                            gate_packed_3[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_102, ns_102))) | sign_103;
                            float2 x_104 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[8]));
                            float2 v_104 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[8]));
                            unsigned int sign_104 = (gate_packed_3[8] ^ up_packed_3[8]) & 2147516416u;
                            float _abs_1807;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1807) : "f"(x_104.x));
                            float _abs_1808;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1808) : "f"(x_104.y));
                            float2 ax_103 = make_float2(_abs_1807, _abs_1808);
                            float _abs_1809;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1809) : "f"(v_104.x));
                            float _abs_1810;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1810) : "f"(v_104.y));
                            float2 av_103 = make_float2(_abs_1809, _abs_1810);
                            float2 _f32x2_mul_b_1811 = make_float2(k1, k1);
                            float2 te_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_103) : "l"(*(const unsigned long long*)&ax_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_1811));
                            float2 _f32x2_mul_b_1812 = make_float2(k2, k2);
                            float2 tw_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_103) : "l"(*(const unsigned long long*)&av_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_1812));
                            float _exp2_1813 = approx_exp2(te_103.x);
                            float _exp2_1814 = approx_exp2(te_103.y);
                            float2 e_103 = make_float2(_exp2_1813, _exp2_1814);
                            float _exp2_1815 = approx_exp2(tw_103.x);
                            float _exp2_1816 = approx_exp2(tw_103.y);
                            float2 w_103 = make_float2(_exp2_1815, _exp2_1816);
                            float2 _f32x2_mul_b_1817 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_103) : "l"(*(const unsigned long long*)&ax_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_1817));
                            float _exp2_1818 = approx_exp2_noftz(tp_103.x);
                            float _exp2_1819 = approx_exp2_noftz(tp_103.y);
                            float2 p_103 = make_float2(_exp2_1818, _exp2_1819);
                            float2 d1_103 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_103);
                            float2 d2_103 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_103);
                            float2 dw_103 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_103);
                            float2 ng_103 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_103);
                            float2 nu_103 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_103);
                            float _rcp_1820 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_103, d2_103), dw_103).x);
                            float _rcp_1821 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_103, d2_103), dw_103).y);
                            float2 r_208 = make_float2(_rcp_1820, _rcp_1821);
                            float2 r_207 = r_208;
                            float2 _f32x2_mul_a_1822 = mul_f32x2_noftz(ng_103, nu_103);
                            float2 _f32x2_mul_b_1823 = mul_f32x2_noftz(r_207, make_float2(bl, bl));
                            float2 m_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_103) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1822), "l"(*(const unsigned long long*)&_f32x2_mul_b_1823));
                            float2 ns_103 = make_float2(((x_104.x < 0.0f) ? p_103.x : 1.0f), ((x_104.y < 0.0f) ? p_103.y : 1.0f));
                            gate_packed_3[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_103, ns_103))) | sign_104;
                            float2 x_105 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[9]));
                            float2 v_105 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[9]));
                            unsigned int sign_105 = (gate_packed_3[9] ^ up_packed_3[9]) & 2147516416u;
                            float _abs_1824;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1824) : "f"(x_105.x));
                            float _abs_1825;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1825) : "f"(x_105.y));
                            float2 ax_104 = make_float2(_abs_1824, _abs_1825);
                            float _abs_1826;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1826) : "f"(v_105.x));
                            float _abs_1827;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1827) : "f"(v_105.y));
                            float2 av_104 = make_float2(_abs_1826, _abs_1827);
                            float2 _f32x2_mul_b_1828 = make_float2(k1, k1);
                            float2 te_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_104) : "l"(*(const unsigned long long*)&ax_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_1828));
                            float2 _f32x2_mul_b_1829 = make_float2(k2, k2);
                            float2 tw_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_104) : "l"(*(const unsigned long long*)&av_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_1829));
                            float _exp2_1830 = approx_exp2(te_104.x);
                            float _exp2_1831 = approx_exp2(te_104.y);
                            float2 e_104 = make_float2(_exp2_1830, _exp2_1831);
                            float _exp2_1832 = approx_exp2(tw_104.x);
                            float _exp2_1833 = approx_exp2(tw_104.y);
                            float2 w_104 = make_float2(_exp2_1832, _exp2_1833);
                            float2 _f32x2_mul_b_1834 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_104) : "l"(*(const unsigned long long*)&ax_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_1834));
                            float _exp2_1835 = approx_exp2_noftz(tp_104.x);
                            float _exp2_1836 = approx_exp2_noftz(tp_104.y);
                            float2 p_104 = make_float2(_exp2_1835, _exp2_1836);
                            float2 d1_104 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_104);
                            float2 d2_104 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_104);
                            float2 dw_104 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_104);
                            float2 ng_104 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_104);
                            float2 nu_104 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_104);
                            float _rcp_1837 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_104, d2_104), dw_104).x);
                            float _rcp_1838 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_104, d2_104), dw_104).y);
                            float2 r_210 = make_float2(_rcp_1837, _rcp_1838);
                            float2 r_209 = r_210;
                            float2 _f32x2_mul_a_1839 = mul_f32x2_noftz(ng_104, nu_104);
                            float2 _f32x2_mul_b_1840 = mul_f32x2_noftz(r_209, make_float2(bl, bl));
                            float2 m_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_104) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1839), "l"(*(const unsigned long long*)&_f32x2_mul_b_1840));
                            float2 ns_104 = make_float2(((x_105.x < 0.0f) ? p_104.x : 1.0f), ((x_105.y < 0.0f) ? p_104.y : 1.0f));
                            gate_packed_3[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_104, ns_104))) | sign_105;
                            float2 x_106 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[10]));
                            float2 v_106 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[10]));
                            unsigned int sign_106 = (gate_packed_3[10] ^ up_packed_3[10]) & 2147516416u;
                            float _abs_1841;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1841) : "f"(x_106.x));
                            float _abs_1842;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1842) : "f"(x_106.y));
                            float2 ax_105 = make_float2(_abs_1841, _abs_1842);
                            float _abs_1843;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1843) : "f"(v_106.x));
                            float _abs_1844;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1844) : "f"(v_106.y));
                            float2 av_105 = make_float2(_abs_1843, _abs_1844);
                            float2 _f32x2_mul_b_1845 = make_float2(k1, k1);
                            float2 te_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_105) : "l"(*(const unsigned long long*)&ax_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_1845));
                            float2 _f32x2_mul_b_1846 = make_float2(k2, k2);
                            float2 tw_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_105) : "l"(*(const unsigned long long*)&av_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_1846));
                            float _exp2_1847 = approx_exp2(te_105.x);
                            float _exp2_1848 = approx_exp2(te_105.y);
                            float2 e_105 = make_float2(_exp2_1847, _exp2_1848);
                            float _exp2_1849 = approx_exp2(tw_105.x);
                            float _exp2_1850 = approx_exp2(tw_105.y);
                            float2 w_105 = make_float2(_exp2_1849, _exp2_1850);
                            float2 _f32x2_mul_b_1851 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_105) : "l"(*(const unsigned long long*)&ax_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_1851));
                            float _exp2_1852 = approx_exp2_noftz(tp_105.x);
                            float _exp2_1853 = approx_exp2_noftz(tp_105.y);
                            float2 p_105 = make_float2(_exp2_1852, _exp2_1853);
                            float2 d1_105 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_105);
                            float2 d2_105 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_105);
                            float2 dw_105 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_105);
                            float2 ng_105 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_105);
                            float2 nu_105 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_105);
                            float _rcp_1854 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_105, d2_105), dw_105).x);
                            float _rcp_1855 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_105, d2_105), dw_105).y);
                            float2 r_212 = make_float2(_rcp_1854, _rcp_1855);
                            float2 r_211 = r_212;
                            float2 _f32x2_mul_a_1856 = mul_f32x2_noftz(ng_105, nu_105);
                            float2 _f32x2_mul_b_1857 = mul_f32x2_noftz(r_211, make_float2(bl, bl));
                            float2 m_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1856), "l"(*(const unsigned long long*)&_f32x2_mul_b_1857));
                            float2 ns_105 = make_float2(((x_106.x < 0.0f) ? p_105.x : 1.0f), ((x_106.y < 0.0f) ? p_105.y : 1.0f));
                            gate_packed_3[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_105, ns_105))) | sign_106;
                            float2 x_107 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[11]));
                            float2 v_107 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[11]));
                            unsigned int sign_107 = (gate_packed_3[11] ^ up_packed_3[11]) & 2147516416u;
                            float _abs_1858;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1858) : "f"(x_107.x));
                            float _abs_1859;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1859) : "f"(x_107.y));
                            float2 ax_106 = make_float2(_abs_1858, _abs_1859);
                            float _abs_1860;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1860) : "f"(v_107.x));
                            float _abs_1861;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1861) : "f"(v_107.y));
                            float2 av_106 = make_float2(_abs_1860, _abs_1861);
                            float2 _f32x2_mul_b_1862 = make_float2(k1, k1);
                            float2 te_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_106) : "l"(*(const unsigned long long*)&ax_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_1862));
                            float2 _f32x2_mul_b_1863 = make_float2(k2, k2);
                            float2 tw_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_106) : "l"(*(const unsigned long long*)&av_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_1863));
                            float _exp2_1864 = approx_exp2(te_106.x);
                            float _exp2_1865 = approx_exp2(te_106.y);
                            float2 e_106 = make_float2(_exp2_1864, _exp2_1865);
                            float _exp2_1866 = approx_exp2(tw_106.x);
                            float _exp2_1867 = approx_exp2(tw_106.y);
                            float2 w_106 = make_float2(_exp2_1866, _exp2_1867);
                            float2 _f32x2_mul_b_1868 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_106) : "l"(*(const unsigned long long*)&ax_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_1868));
                            float _exp2_1869 = approx_exp2_noftz(tp_106.x);
                            float _exp2_1870 = approx_exp2_noftz(tp_106.y);
                            float2 p_106 = make_float2(_exp2_1869, _exp2_1870);
                            float2 d1_106 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_106);
                            float2 d2_106 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_106);
                            float2 dw_106 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_106);
                            float2 ng_106 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_106);
                            float2 nu_106 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_106);
                            float _rcp_1871 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_106, d2_106), dw_106).x);
                            float _rcp_1872 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_106, d2_106), dw_106).y);
                            float2 r_214 = make_float2(_rcp_1871, _rcp_1872);
                            float2 r_213 = r_214;
                            float2 _f32x2_mul_a_1873 = mul_f32x2_noftz(ng_106, nu_106);
                            float2 _f32x2_mul_b_1874 = mul_f32x2_noftz(r_213, make_float2(bl, bl));
                            float2 m_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_106) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1873), "l"(*(const unsigned long long*)&_f32x2_mul_b_1874));
                            float2 ns_106 = make_float2(((x_107.x < 0.0f) ? p_106.x : 1.0f), ((x_107.y < 0.0f) ? p_106.y : 1.0f));
                            gate_packed_3[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_106, ns_106))) | sign_107;
                            float2 x_108 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[12]));
                            float2 v_108 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[12]));
                            unsigned int sign_108 = (gate_packed_3[12] ^ up_packed_3[12]) & 2147516416u;
                            float _abs_1875;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1875) : "f"(x_108.x));
                            float _abs_1876;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1876) : "f"(x_108.y));
                            float2 ax_107 = make_float2(_abs_1875, _abs_1876);
                            float _abs_1877;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1877) : "f"(v_108.x));
                            float _abs_1878;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1878) : "f"(v_108.y));
                            float2 av_107 = make_float2(_abs_1877, _abs_1878);
                            float2 _f32x2_mul_b_1879 = make_float2(k1, k1);
                            float2 te_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_107) : "l"(*(const unsigned long long*)&ax_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1879));
                            float2 _f32x2_mul_b_1880 = make_float2(k2, k2);
                            float2 tw_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_107) : "l"(*(const unsigned long long*)&av_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1880));
                            float _exp2_1881 = approx_exp2(te_107.x);
                            float _exp2_1882 = approx_exp2(te_107.y);
                            float2 e_107 = make_float2(_exp2_1881, _exp2_1882);
                            float _exp2_1883 = approx_exp2(tw_107.x);
                            float _exp2_1884 = approx_exp2(tw_107.y);
                            float2 w_107 = make_float2(_exp2_1883, _exp2_1884);
                            float2 _f32x2_mul_b_1885 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_107) : "l"(*(const unsigned long long*)&ax_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1885));
                            float _exp2_1886 = approx_exp2_noftz(tp_107.x);
                            float _exp2_1887 = approx_exp2_noftz(tp_107.y);
                            float2 p_107 = make_float2(_exp2_1886, _exp2_1887);
                            float2 d1_107 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_107);
                            float2 d2_107 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_107);
                            float2 dw_107 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_107);
                            float2 ng_107 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_107);
                            float2 nu_107 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_107);
                            float _rcp_1888 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_107, d2_107), dw_107).x);
                            float _rcp_1889 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_107, d2_107), dw_107).y);
                            float2 r_216 = make_float2(_rcp_1888, _rcp_1889);
                            float2 r_215 = r_216;
                            float2 _f32x2_mul_a_1890 = mul_f32x2_noftz(ng_107, nu_107);
                            float2 _f32x2_mul_b_1891 = mul_f32x2_noftz(r_215, make_float2(bl, bl));
                            float2 m_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_107) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1890), "l"(*(const unsigned long long*)&_f32x2_mul_b_1891));
                            float2 ns_107 = make_float2(((x_108.x < 0.0f) ? p_107.x : 1.0f), ((x_108.y < 0.0f) ? p_107.y : 1.0f));
                            gate_packed_3[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_107, ns_107))) | sign_108;
                            float2 x_109 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[13]));
                            float2 v_109 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[13]));
                            unsigned int sign_109 = (gate_packed_3[13] ^ up_packed_3[13]) & 2147516416u;
                            float _abs_1892;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1892) : "f"(x_109.x));
                            float _abs_1893;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1893) : "f"(x_109.y));
                            float2 ax_108 = make_float2(_abs_1892, _abs_1893);
                            float _abs_1894;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1894) : "f"(v_109.x));
                            float _abs_1895;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1895) : "f"(v_109.y));
                            float2 av_108 = make_float2(_abs_1894, _abs_1895);
                            float2 _f32x2_mul_b_1896 = make_float2(k1, k1);
                            float2 te_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_108) : "l"(*(const unsigned long long*)&ax_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_1896));
                            float2 _f32x2_mul_b_1897 = make_float2(k2, k2);
                            float2 tw_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_108) : "l"(*(const unsigned long long*)&av_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_1897));
                            float _exp2_1898 = approx_exp2(te_108.x);
                            float _exp2_1899 = approx_exp2(te_108.y);
                            float2 e_108 = make_float2(_exp2_1898, _exp2_1899);
                            float _exp2_1900 = approx_exp2(tw_108.x);
                            float _exp2_1901 = approx_exp2(tw_108.y);
                            float2 w_108 = make_float2(_exp2_1900, _exp2_1901);
                            float2 _f32x2_mul_b_1902 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_108) : "l"(*(const unsigned long long*)&ax_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_1902));
                            float _exp2_1903 = approx_exp2_noftz(tp_108.x);
                            float _exp2_1904 = approx_exp2_noftz(tp_108.y);
                            float2 p_108 = make_float2(_exp2_1903, _exp2_1904);
                            float2 d1_108 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_108);
                            float2 d2_108 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_108);
                            float2 dw_108 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_108);
                            float2 ng_108 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_108);
                            float2 nu_108 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_108);
                            float _rcp_1905 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_108, d2_108), dw_108).x);
                            float _rcp_1906 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_108, d2_108), dw_108).y);
                            float2 r_218 = make_float2(_rcp_1905, _rcp_1906);
                            float2 r_217 = r_218;
                            float2 _f32x2_mul_a_1907 = mul_f32x2_noftz(ng_108, nu_108);
                            float2 _f32x2_mul_b_1908 = mul_f32x2_noftz(r_217, make_float2(bl, bl));
                            float2 m_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_108) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1907), "l"(*(const unsigned long long*)&_f32x2_mul_b_1908));
                            float2 ns_108 = make_float2(((x_109.x < 0.0f) ? p_108.x : 1.0f), ((x_109.y < 0.0f) ? p_108.y : 1.0f));
                            gate_packed_3[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_108, ns_108))) | sign_109;
                            float2 x_110 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[14]));
                            float2 v_110 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[14]));
                            unsigned int sign_110 = (gate_packed_3[14] ^ up_packed_3[14]) & 2147516416u;
                            float _abs_1909;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1909) : "f"(x_110.x));
                            float _abs_1910;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1910) : "f"(x_110.y));
                            float2 ax_109 = make_float2(_abs_1909, _abs_1910);
                            float _abs_1911;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1911) : "f"(v_110.x));
                            float _abs_1912;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1912) : "f"(v_110.y));
                            float2 av_109 = make_float2(_abs_1911, _abs_1912);
                            float2 _f32x2_mul_b_1913 = make_float2(k1, k1);
                            float2 te_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_109) : "l"(*(const unsigned long long*)&ax_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_1913));
                            float2 _f32x2_mul_b_1914 = make_float2(k2, k2);
                            float2 tw_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_109) : "l"(*(const unsigned long long*)&av_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_1914));
                            float _exp2_1915 = approx_exp2(te_109.x);
                            float _exp2_1916 = approx_exp2(te_109.y);
                            float2 e_109 = make_float2(_exp2_1915, _exp2_1916);
                            float _exp2_1917 = approx_exp2(tw_109.x);
                            float _exp2_1918 = approx_exp2(tw_109.y);
                            float2 w_109 = make_float2(_exp2_1917, _exp2_1918);
                            float2 _f32x2_mul_b_1919 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_109) : "l"(*(const unsigned long long*)&ax_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_1919));
                            float _exp2_1920 = approx_exp2_noftz(tp_109.x);
                            float _exp2_1921 = approx_exp2_noftz(tp_109.y);
                            float2 p_109 = make_float2(_exp2_1920, _exp2_1921);
                            float2 d1_109 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_109);
                            float2 d2_109 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_109);
                            float2 dw_109 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_109);
                            float2 ng_109 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_109);
                            float2 nu_109 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_109);
                            float _rcp_1922 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_109, d2_109), dw_109).x);
                            float _rcp_1923 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_109, d2_109), dw_109).y);
                            float2 r_220 = make_float2(_rcp_1922, _rcp_1923);
                            float2 r_219 = r_220;
                            float2 _f32x2_mul_a_1924 = mul_f32x2_noftz(ng_109, nu_109);
                            float2 _f32x2_mul_b_1925 = mul_f32x2_noftz(r_219, make_float2(bl, bl));
                            float2 m_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_109) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1924), "l"(*(const unsigned long long*)&_f32x2_mul_b_1925));
                            float2 ns_109 = make_float2(((x_110.x < 0.0f) ? p_109.x : 1.0f), ((x_110.y < 0.0f) ? p_109.y : 1.0f));
                            gate_packed_3[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_109, ns_109))) | sign_110;
                            float2 x_111 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[15]));
                            float2 v_111 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[15]));
                            unsigned int sign_111 = (gate_packed_3[15] ^ up_packed_3[15]) & 2147516416u;
                            float _abs_1926;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1926) : "f"(x_111.x));
                            float _abs_1927;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1927) : "f"(x_111.y));
                            float2 ax_110 = make_float2(_abs_1926, _abs_1927);
                            float _abs_1928;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1928) : "f"(v_111.x));
                            float _abs_1929;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1929) : "f"(v_111.y));
                            float2 av_110 = make_float2(_abs_1928, _abs_1929);
                            float2 _f32x2_mul_b_1930 = make_float2(k1, k1);
                            float2 te_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_110) : "l"(*(const unsigned long long*)&ax_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_1930));
                            float2 _f32x2_mul_b_1931 = make_float2(k2, k2);
                            float2 tw_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_110) : "l"(*(const unsigned long long*)&av_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_1931));
                            float _exp2_1932 = approx_exp2(te_110.x);
                            float _exp2_1933 = approx_exp2(te_110.y);
                            float2 e_110 = make_float2(_exp2_1932, _exp2_1933);
                            float _exp2_1934 = approx_exp2(tw_110.x);
                            float _exp2_1935 = approx_exp2(tw_110.y);
                            float2 w_110 = make_float2(_exp2_1934, _exp2_1935);
                            float2 _f32x2_mul_b_1936 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_110) : "l"(*(const unsigned long long*)&ax_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_1936));
                            float _exp2_1937 = approx_exp2_noftz(tp_110.x);
                            float _exp2_1938 = approx_exp2_noftz(tp_110.y);
                            float2 p_110 = make_float2(_exp2_1937, _exp2_1938);
                            float2 d1_110 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_110);
                            float2 d2_110 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_110);
                            float2 dw_110 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_110);
                            float2 ng_110 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_110);
                            float2 nu_110 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_110);
                            float _rcp_1939 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_110, d2_110), dw_110).x);
                            float _rcp_1940 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_110, d2_110), dw_110).y);
                            float2 r_222 = make_float2(_rcp_1939, _rcp_1940);
                            float2 r_221 = r_222;
                            float2 _f32x2_mul_a_1941 = mul_f32x2_noftz(ng_110, nu_110);
                            float2 _f32x2_mul_b_1942 = mul_f32x2_noftz(r_221, make_float2(bl, bl));
                            float2 m_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1941), "l"(*(const unsigned long long*)&_f32x2_mul_b_1942));
                            float2 ns_110 = make_float2(((x_111.x < 0.0f) ? p_110.x : 1.0f), ((x_111.y < 0.0f) ? p_110.y : 1.0f));
                            gate_packed_3[15] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_110, ns_110))) | sign_111;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_21 = tid / 32;
                            int lane_21 = tid % 32;
                            #pragma unroll
                            for (int half_22 = 0; half_22 < 2; half_22++) {
                                #pragma unroll
                                for (int col_tile_22 = 0; col_tile_22 < 2; col_tile_22++) {
                                    int row_23 = warp_21 * 32 + half_22 * 16 + lane_21 % 16;
                                    int col_23 = col_tile_22 * 16 + lane_21 / 16 * 8;
                                    unsigned int address_22 = d_smem_addr + 8192 + (unsigned int)((row_23 * 32 + col_23) * 2);
                                    address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                    int offset_23 = half_22 * 8 + col_tile_22 * 4;
                                    uint32_t _stmatrix_addr_1943 = static_cast<uint32_t>(address_22);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1943), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            float2 x_112 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[16]));
                            float2 v_112 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[16]));
                            unsigned int sign_112 = (gate_packed_3[16] ^ up_packed_3[16]) & 2147516416u;
                            float _abs_1944;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1944) : "f"(x_112.x));
                            float _abs_1945;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1945) : "f"(x_112.y));
                            float2 ax_111 = make_float2(_abs_1944, _abs_1945);
                            float _abs_1946;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1946) : "f"(v_112.x));
                            float _abs_1947;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1947) : "f"(v_112.y));
                            float2 av_111 = make_float2(_abs_1946, _abs_1947);
                            float2 _f32x2_mul_b_1948 = make_float2(k1, k1);
                            float2 te_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_111) : "l"(*(const unsigned long long*)&ax_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_1948));
                            float2 _f32x2_mul_b_1949 = make_float2(k2, k2);
                            float2 tw_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_111) : "l"(*(const unsigned long long*)&av_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_1949));
                            float _exp2_1950 = approx_exp2(te_111.x);
                            float _exp2_1951 = approx_exp2(te_111.y);
                            float2 e_111 = make_float2(_exp2_1950, _exp2_1951);
                            float _exp2_1952 = approx_exp2(tw_111.x);
                            float _exp2_1953 = approx_exp2(tw_111.y);
                            float2 w_111 = make_float2(_exp2_1952, _exp2_1953);
                            float2 _f32x2_mul_b_1954 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_111) : "l"(*(const unsigned long long*)&ax_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_1954));
                            float _exp2_1955 = approx_exp2_noftz(tp_111.x);
                            float _exp2_1956 = approx_exp2_noftz(tp_111.y);
                            float2 p_111 = make_float2(_exp2_1955, _exp2_1956);
                            float2 d1_111 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_111);
                            float2 d2_111 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_111);
                            float2 dw_111 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_111);
                            float2 ng_111 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_111);
                            float2 nu_111 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_111);
                            float _rcp_1957 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_111, d2_111), dw_111).x);
                            float _rcp_1958 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_111, d2_111), dw_111).y);
                            float2 r_224 = make_float2(_rcp_1957, _rcp_1958);
                            float2 r_223 = r_224;
                            float2 _f32x2_mul_a_1959 = mul_f32x2_noftz(ng_111, nu_111);
                            float2 _f32x2_mul_b_1960 = mul_f32x2_noftz(r_223, make_float2(bl, bl));
                            float2 m_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_111) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1959), "l"(*(const unsigned long long*)&_f32x2_mul_b_1960));
                            float2 ns_111 = make_float2(((x_112.x < 0.0f) ? p_111.x : 1.0f), ((x_112.y < 0.0f) ? p_111.y : 1.0f));
                            gate_packed_3[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_111, ns_111))) | sign_112;
                            float2 x_113 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[17]));
                            float2 v_113 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[17]));
                            unsigned int sign_113 = (gate_packed_3[17] ^ up_packed_3[17]) & 2147516416u;
                            float _abs_1961;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1961) : "f"(x_113.x));
                            float _abs_1962;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1962) : "f"(x_113.y));
                            float2 ax_112 = make_float2(_abs_1961, _abs_1962);
                            float _abs_1963;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1963) : "f"(v_113.x));
                            float _abs_1964;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1964) : "f"(v_113.y));
                            float2 av_112 = make_float2(_abs_1963, _abs_1964);
                            float2 _f32x2_mul_b_1965 = make_float2(k1, k1);
                            float2 te_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_112) : "l"(*(const unsigned long long*)&ax_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_1965));
                            float2 _f32x2_mul_b_1966 = make_float2(k2, k2);
                            float2 tw_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_112) : "l"(*(const unsigned long long*)&av_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_1966));
                            float _exp2_1967 = approx_exp2(te_112.x);
                            float _exp2_1968 = approx_exp2(te_112.y);
                            float2 e_112 = make_float2(_exp2_1967, _exp2_1968);
                            float _exp2_1969 = approx_exp2(tw_112.x);
                            float _exp2_1970 = approx_exp2(tw_112.y);
                            float2 w_112 = make_float2(_exp2_1969, _exp2_1970);
                            float2 _f32x2_mul_b_1971 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_112) : "l"(*(const unsigned long long*)&ax_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_1971));
                            float _exp2_1972 = approx_exp2_noftz(tp_112.x);
                            float _exp2_1973 = approx_exp2_noftz(tp_112.y);
                            float2 p_112 = make_float2(_exp2_1972, _exp2_1973);
                            float2 d1_112 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_112);
                            float2 d2_112 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_112);
                            float2 dw_112 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_112);
                            float2 ng_112 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_112);
                            float2 nu_112 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_112);
                            float _rcp_1974 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_112, d2_112), dw_112).x);
                            float _rcp_1975 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_112, d2_112), dw_112).y);
                            float2 r_226 = make_float2(_rcp_1974, _rcp_1975);
                            float2 r_225 = r_226;
                            float2 _f32x2_mul_a_1976 = mul_f32x2_noftz(ng_112, nu_112);
                            float2 _f32x2_mul_b_1977 = mul_f32x2_noftz(r_225, make_float2(bl, bl));
                            float2 m_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_112) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1976), "l"(*(const unsigned long long*)&_f32x2_mul_b_1977));
                            float2 ns_112 = make_float2(((x_113.x < 0.0f) ? p_112.x : 1.0f), ((x_113.y < 0.0f) ? p_112.y : 1.0f));
                            gate_packed_3[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_112, ns_112))) | sign_113;
                            float2 x_114 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[18]));
                            float2 v_114 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[18]));
                            unsigned int sign_114 = (gate_packed_3[18] ^ up_packed_3[18]) & 2147516416u;
                            float _abs_1978;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1978) : "f"(x_114.x));
                            float _abs_1979;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1979) : "f"(x_114.y));
                            float2 ax_113 = make_float2(_abs_1978, _abs_1979);
                            float _abs_1980;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1980) : "f"(v_114.x));
                            float _abs_1981;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1981) : "f"(v_114.y));
                            float2 av_113 = make_float2(_abs_1980, _abs_1981);
                            float2 _f32x2_mul_b_1982 = make_float2(k1, k1);
                            float2 te_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_113) : "l"(*(const unsigned long long*)&ax_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_1982));
                            float2 _f32x2_mul_b_1983 = make_float2(k2, k2);
                            float2 tw_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_113) : "l"(*(const unsigned long long*)&av_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_1983));
                            float _exp2_1984 = approx_exp2(te_113.x);
                            float _exp2_1985 = approx_exp2(te_113.y);
                            float2 e_113 = make_float2(_exp2_1984, _exp2_1985);
                            float _exp2_1986 = approx_exp2(tw_113.x);
                            float _exp2_1987 = approx_exp2(tw_113.y);
                            float2 w_113 = make_float2(_exp2_1986, _exp2_1987);
                            float2 _f32x2_mul_b_1988 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_113) : "l"(*(const unsigned long long*)&ax_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_1988));
                            float _exp2_1989 = approx_exp2_noftz(tp_113.x);
                            float _exp2_1990 = approx_exp2_noftz(tp_113.y);
                            float2 p_113 = make_float2(_exp2_1989, _exp2_1990);
                            float2 d1_113 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_113);
                            float2 d2_113 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_113);
                            float2 dw_113 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_113);
                            float2 ng_113 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_113);
                            float2 nu_113 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_113);
                            float _rcp_1991 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_113, d2_113), dw_113).x);
                            float _rcp_1992 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_113, d2_113), dw_113).y);
                            float2 r_228 = make_float2(_rcp_1991, _rcp_1992);
                            float2 r_227 = r_228;
                            float2 _f32x2_mul_a_1993 = mul_f32x2_noftz(ng_113, nu_113);
                            float2 _f32x2_mul_b_1994 = mul_f32x2_noftz(r_227, make_float2(bl, bl));
                            float2 m_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_113) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1993), "l"(*(const unsigned long long*)&_f32x2_mul_b_1994));
                            float2 ns_113 = make_float2(((x_114.x < 0.0f) ? p_113.x : 1.0f), ((x_114.y < 0.0f) ? p_113.y : 1.0f));
                            gate_packed_3[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_113, ns_113))) | sign_114;
                            float2 x_115 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[19]));
                            float2 v_115 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[19]));
                            unsigned int sign_115 = (gate_packed_3[19] ^ up_packed_3[19]) & 2147516416u;
                            float _abs_1995;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1995) : "f"(x_115.x));
                            float _abs_1996;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1996) : "f"(x_115.y));
                            float2 ax_114 = make_float2(_abs_1995, _abs_1996);
                            float _abs_1997;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1997) : "f"(v_115.x));
                            float _abs_1998;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1998) : "f"(v_115.y));
                            float2 av_114 = make_float2(_abs_1997, _abs_1998);
                            float2 _f32x2_mul_b_1999 = make_float2(k1, k1);
                            float2 te_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_114) : "l"(*(const unsigned long long*)&ax_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_1999));
                            float2 _f32x2_mul_b_2000 = make_float2(k2, k2);
                            float2 tw_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_114) : "l"(*(const unsigned long long*)&av_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_2000));
                            float _exp2_2001 = approx_exp2(te_114.x);
                            float _exp2_2002 = approx_exp2(te_114.y);
                            float2 e_114 = make_float2(_exp2_2001, _exp2_2002);
                            float _exp2_2003 = approx_exp2(tw_114.x);
                            float _exp2_2004 = approx_exp2(tw_114.y);
                            float2 w_114 = make_float2(_exp2_2003, _exp2_2004);
                            float2 _f32x2_mul_b_2005 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_114) : "l"(*(const unsigned long long*)&ax_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_2005));
                            float _exp2_2006 = approx_exp2_noftz(tp_114.x);
                            float _exp2_2007 = approx_exp2_noftz(tp_114.y);
                            float2 p_114 = make_float2(_exp2_2006, _exp2_2007);
                            float2 d1_114 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_114);
                            float2 d2_114 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_114);
                            float2 dw_114 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_114);
                            float2 ng_114 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_114);
                            float2 nu_114 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_114);
                            float _rcp_2008 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_114, d2_114), dw_114).x);
                            float _rcp_2009 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_114, d2_114), dw_114).y);
                            float2 r_230 = make_float2(_rcp_2008, _rcp_2009);
                            float2 r_229 = r_230;
                            float2 _f32x2_mul_a_2010 = mul_f32x2_noftz(ng_114, nu_114);
                            float2 _f32x2_mul_b_2011 = mul_f32x2_noftz(r_229, make_float2(bl, bl));
                            float2 m_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_114) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2010), "l"(*(const unsigned long long*)&_f32x2_mul_b_2011));
                            float2 ns_114 = make_float2(((x_115.x < 0.0f) ? p_114.x : 1.0f), ((x_115.y < 0.0f) ? p_114.y : 1.0f));
                            gate_packed_3[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_114, ns_114))) | sign_115;
                            float2 x_116 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[20]));
                            float2 v_116 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[20]));
                            unsigned int sign_116 = (gate_packed_3[20] ^ up_packed_3[20]) & 2147516416u;
                            float _abs_2012;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2012) : "f"(x_116.x));
                            float _abs_2013;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2013) : "f"(x_116.y));
                            float2 ax_115 = make_float2(_abs_2012, _abs_2013);
                            float _abs_2014;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2014) : "f"(v_116.x));
                            float _abs_2015;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2015) : "f"(v_116.y));
                            float2 av_115 = make_float2(_abs_2014, _abs_2015);
                            float2 _f32x2_mul_b_2016 = make_float2(k1, k1);
                            float2 te_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_115) : "l"(*(const unsigned long long*)&ax_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_2016));
                            float2 _f32x2_mul_b_2017 = make_float2(k2, k2);
                            float2 tw_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_115) : "l"(*(const unsigned long long*)&av_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_2017));
                            float _exp2_2018 = approx_exp2(te_115.x);
                            float _exp2_2019 = approx_exp2(te_115.y);
                            float2 e_115 = make_float2(_exp2_2018, _exp2_2019);
                            float _exp2_2020 = approx_exp2(tw_115.x);
                            float _exp2_2021 = approx_exp2(tw_115.y);
                            float2 w_115 = make_float2(_exp2_2020, _exp2_2021);
                            float2 _f32x2_mul_b_2022 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_115) : "l"(*(const unsigned long long*)&ax_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_2022));
                            float _exp2_2023 = approx_exp2_noftz(tp_115.x);
                            float _exp2_2024 = approx_exp2_noftz(tp_115.y);
                            float2 p_115 = make_float2(_exp2_2023, _exp2_2024);
                            float2 d1_115 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_115);
                            float2 d2_115 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_115);
                            float2 dw_115 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_115);
                            float2 ng_115 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_115);
                            float2 nu_115 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_115);
                            float _rcp_2025 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_115, d2_115), dw_115).x);
                            float _rcp_2026 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_115, d2_115), dw_115).y);
                            float2 r_232 = make_float2(_rcp_2025, _rcp_2026);
                            float2 r_231 = r_232;
                            float2 _f32x2_mul_a_2027 = mul_f32x2_noftz(ng_115, nu_115);
                            float2 _f32x2_mul_b_2028 = mul_f32x2_noftz(r_231, make_float2(bl, bl));
                            float2 m_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_115) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2027), "l"(*(const unsigned long long*)&_f32x2_mul_b_2028));
                            float2 ns_115 = make_float2(((x_116.x < 0.0f) ? p_115.x : 1.0f), ((x_116.y < 0.0f) ? p_115.y : 1.0f));
                            gate_packed_3[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_115, ns_115))) | sign_116;
                            float2 x_117 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[21]));
                            float2 v_117 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[21]));
                            unsigned int sign_117 = (gate_packed_3[21] ^ up_packed_3[21]) & 2147516416u;
                            float _abs_2029;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2029) : "f"(x_117.x));
                            float _abs_2030;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2030) : "f"(x_117.y));
                            float2 ax_116 = make_float2(_abs_2029, _abs_2030);
                            float _abs_2031;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2031) : "f"(v_117.x));
                            float _abs_2032;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2032) : "f"(v_117.y));
                            float2 av_116 = make_float2(_abs_2031, _abs_2032);
                            float2 _f32x2_mul_b_2033 = make_float2(k1, k1);
                            float2 te_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_116) : "l"(*(const unsigned long long*)&ax_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_2033));
                            float2 _f32x2_mul_b_2034 = make_float2(k2, k2);
                            float2 tw_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_116) : "l"(*(const unsigned long long*)&av_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_2034));
                            float _exp2_2035 = approx_exp2(te_116.x);
                            float _exp2_2036 = approx_exp2(te_116.y);
                            float2 e_116 = make_float2(_exp2_2035, _exp2_2036);
                            float _exp2_2037 = approx_exp2(tw_116.x);
                            float _exp2_2038 = approx_exp2(tw_116.y);
                            float2 w_116 = make_float2(_exp2_2037, _exp2_2038);
                            float2 _f32x2_mul_b_2039 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_116) : "l"(*(const unsigned long long*)&ax_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_2039));
                            float _exp2_2040 = approx_exp2_noftz(tp_116.x);
                            float _exp2_2041 = approx_exp2_noftz(tp_116.y);
                            float2 p_116 = make_float2(_exp2_2040, _exp2_2041);
                            float2 d1_116 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_116);
                            float2 d2_116 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_116);
                            float2 dw_116 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_116);
                            float2 ng_116 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_116);
                            float2 nu_116 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_116);
                            float _rcp_2042 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_116, d2_116), dw_116).x);
                            float _rcp_2043 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_116, d2_116), dw_116).y);
                            float2 r_234 = make_float2(_rcp_2042, _rcp_2043);
                            float2 r_233 = r_234;
                            float2 _f32x2_mul_a_2044 = mul_f32x2_noftz(ng_116, nu_116);
                            float2 _f32x2_mul_b_2045 = mul_f32x2_noftz(r_233, make_float2(bl, bl));
                            float2 m_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2044), "l"(*(const unsigned long long*)&_f32x2_mul_b_2045));
                            float2 ns_116 = make_float2(((x_117.x < 0.0f) ? p_116.x : 1.0f), ((x_117.y < 0.0f) ? p_116.y : 1.0f));
                            gate_packed_3[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_116, ns_116))) | sign_117;
                            float2 x_118 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[22]));
                            float2 v_118 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[22]));
                            unsigned int sign_118 = (gate_packed_3[22] ^ up_packed_3[22]) & 2147516416u;
                            float _abs_2046;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2046) : "f"(x_118.x));
                            float _abs_2047;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2047) : "f"(x_118.y));
                            float2 ax_117 = make_float2(_abs_2046, _abs_2047);
                            float _abs_2048;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2048) : "f"(v_118.x));
                            float _abs_2049;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2049) : "f"(v_118.y));
                            float2 av_117 = make_float2(_abs_2048, _abs_2049);
                            float2 _f32x2_mul_b_2050 = make_float2(k1, k1);
                            float2 te_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_117) : "l"(*(const unsigned long long*)&ax_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_2050));
                            float2 _f32x2_mul_b_2051 = make_float2(k2, k2);
                            float2 tw_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_117) : "l"(*(const unsigned long long*)&av_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_2051));
                            float _exp2_2052 = approx_exp2(te_117.x);
                            float _exp2_2053 = approx_exp2(te_117.y);
                            float2 e_117 = make_float2(_exp2_2052, _exp2_2053);
                            float _exp2_2054 = approx_exp2(tw_117.x);
                            float _exp2_2055 = approx_exp2(tw_117.y);
                            float2 w_117 = make_float2(_exp2_2054, _exp2_2055);
                            float2 _f32x2_mul_b_2056 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_117) : "l"(*(const unsigned long long*)&ax_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_2056));
                            float _exp2_2057 = approx_exp2_noftz(tp_117.x);
                            float _exp2_2058 = approx_exp2_noftz(tp_117.y);
                            float2 p_117 = make_float2(_exp2_2057, _exp2_2058);
                            float2 d1_117 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_117);
                            float2 d2_117 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_117);
                            float2 dw_117 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_117);
                            float2 ng_117 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_117);
                            float2 nu_117 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_117);
                            float _rcp_2059 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_117, d2_117), dw_117).x);
                            float _rcp_2060 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_117, d2_117), dw_117).y);
                            float2 r_236 = make_float2(_rcp_2059, _rcp_2060);
                            float2 r_235 = r_236;
                            float2 _f32x2_mul_a_2061 = mul_f32x2_noftz(ng_117, nu_117);
                            float2 _f32x2_mul_b_2062 = mul_f32x2_noftz(r_235, make_float2(bl, bl));
                            float2 m_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_117) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2061), "l"(*(const unsigned long long*)&_f32x2_mul_b_2062));
                            float2 ns_117 = make_float2(((x_118.x < 0.0f) ? p_117.x : 1.0f), ((x_118.y < 0.0f) ? p_117.y : 1.0f));
                            gate_packed_3[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_117, ns_117))) | sign_118;
                            float2 x_119 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[23]));
                            float2 v_119 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[23]));
                            unsigned int sign_119 = (gate_packed_3[23] ^ up_packed_3[23]) & 2147516416u;
                            float _abs_2063;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2063) : "f"(x_119.x));
                            float _abs_2064;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2064) : "f"(x_119.y));
                            float2 ax_118 = make_float2(_abs_2063, _abs_2064);
                            float _abs_2065;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2065) : "f"(v_119.x));
                            float _abs_2066;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2066) : "f"(v_119.y));
                            float2 av_118 = make_float2(_abs_2065, _abs_2066);
                            float2 _f32x2_mul_b_2067 = make_float2(k1, k1);
                            float2 te_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_118) : "l"(*(const unsigned long long*)&ax_118), "l"(*(const unsigned long long*)&_f32x2_mul_b_2067));
                            float2 _f32x2_mul_b_2068 = make_float2(k2, k2);
                            float2 tw_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_118) : "l"(*(const unsigned long long*)&av_118), "l"(*(const unsigned long long*)&_f32x2_mul_b_2068));
                            float _exp2_2069 = approx_exp2(te_118.x);
                            float _exp2_2070 = approx_exp2(te_118.y);
                            float2 e_118 = make_float2(_exp2_2069, _exp2_2070);
                            float _exp2_2071 = approx_exp2(tw_118.x);
                            float _exp2_2072 = approx_exp2(tw_118.y);
                            float2 w_118 = make_float2(_exp2_2071, _exp2_2072);
                            float2 _f32x2_mul_b_2073 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_118) : "l"(*(const unsigned long long*)&ax_118), "l"(*(const unsigned long long*)&_f32x2_mul_b_2073));
                            float _exp2_2074 = approx_exp2_noftz(tp_118.x);
                            float _exp2_2075 = approx_exp2_noftz(tp_118.y);
                            float2 p_118 = make_float2(_exp2_2074, _exp2_2075);
                            float2 d1_118 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_118);
                            float2 d2_118 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_118);
                            float2 dw_118 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_118);
                            float2 ng_118 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_118);
                            float2 nu_118 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_118);
                            float _rcp_2076 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_118, d2_118), dw_118).x);
                            float _rcp_2077 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_118, d2_118), dw_118).y);
                            float2 r_238 = make_float2(_rcp_2076, _rcp_2077);
                            float2 r_237 = r_238;
                            float2 _f32x2_mul_a_2078 = mul_f32x2_noftz(ng_118, nu_118);
                            float2 _f32x2_mul_b_2079 = mul_f32x2_noftz(r_237, make_float2(bl, bl));
                            float2 m_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_118) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2078), "l"(*(const unsigned long long*)&_f32x2_mul_b_2079));
                            float2 ns_118 = make_float2(((x_119.x < 0.0f) ? p_118.x : 1.0f), ((x_119.y < 0.0f) ? p_118.y : 1.0f));
                            gate_packed_3[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_118, ns_118))) | sign_119;
                            float2 x_120 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[24]));
                            float2 v_120 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[24]));
                            unsigned int sign_120 = (gate_packed_3[24] ^ up_packed_3[24]) & 2147516416u;
                            float _abs_2080;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2080) : "f"(x_120.x));
                            float _abs_2081;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2081) : "f"(x_120.y));
                            float2 ax_119 = make_float2(_abs_2080, _abs_2081);
                            float _abs_2082;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2082) : "f"(v_120.x));
                            float _abs_2083;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2083) : "f"(v_120.y));
                            float2 av_119 = make_float2(_abs_2082, _abs_2083);
                            float2 _f32x2_mul_b_2084 = make_float2(k1, k1);
                            float2 te_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_119) : "l"(*(const unsigned long long*)&ax_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_2084));
                            float2 _f32x2_mul_b_2085 = make_float2(k2, k2);
                            float2 tw_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_119) : "l"(*(const unsigned long long*)&av_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_2085));
                            float _exp2_2086 = approx_exp2(te_119.x);
                            float _exp2_2087 = approx_exp2(te_119.y);
                            float2 e_119 = make_float2(_exp2_2086, _exp2_2087);
                            float _exp2_2088 = approx_exp2(tw_119.x);
                            float _exp2_2089 = approx_exp2(tw_119.y);
                            float2 w_119 = make_float2(_exp2_2088, _exp2_2089);
                            float2 _f32x2_mul_b_2090 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_119) : "l"(*(const unsigned long long*)&ax_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_2090));
                            float _exp2_2091 = approx_exp2_noftz(tp_119.x);
                            float _exp2_2092 = approx_exp2_noftz(tp_119.y);
                            float2 p_119 = make_float2(_exp2_2091, _exp2_2092);
                            float2 d1_119 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_119);
                            float2 d2_119 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_119);
                            float2 dw_119 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_119);
                            float2 ng_119 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_119);
                            float2 nu_119 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_119);
                            float _rcp_2093 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_119, d2_119), dw_119).x);
                            float _rcp_2094 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_119, d2_119), dw_119).y);
                            float2 r_240 = make_float2(_rcp_2093, _rcp_2094);
                            float2 r_239 = r_240;
                            float2 _f32x2_mul_a_2095 = mul_f32x2_noftz(ng_119, nu_119);
                            float2 _f32x2_mul_b_2096 = mul_f32x2_noftz(r_239, make_float2(bl, bl));
                            float2 m_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_119) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2095), "l"(*(const unsigned long long*)&_f32x2_mul_b_2096));
                            float2 ns_119 = make_float2(((x_120.x < 0.0f) ? p_119.x : 1.0f), ((x_120.y < 0.0f) ? p_119.y : 1.0f));
                            gate_packed_3[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_119, ns_119))) | sign_120;
                            float2 x_121 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[25]));
                            float2 v_121 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[25]));
                            unsigned int sign_121 = (gate_packed_3[25] ^ up_packed_3[25]) & 2147516416u;
                            float _abs_2097;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2097) : "f"(x_121.x));
                            float _abs_2098;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2098) : "f"(x_121.y));
                            float2 ax_120 = make_float2(_abs_2097, _abs_2098);
                            float _abs_2099;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2099) : "f"(v_121.x));
                            float _abs_2100;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2100) : "f"(v_121.y));
                            float2 av_120 = make_float2(_abs_2099, _abs_2100);
                            float2 _f32x2_mul_b_2101 = make_float2(k1, k1);
                            float2 te_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_120) : "l"(*(const unsigned long long*)&ax_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_2101));
                            float2 _f32x2_mul_b_2102 = make_float2(k2, k2);
                            float2 tw_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_120) : "l"(*(const unsigned long long*)&av_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_2102));
                            float _exp2_2103 = approx_exp2(te_120.x);
                            float _exp2_2104 = approx_exp2(te_120.y);
                            float2 e_120 = make_float2(_exp2_2103, _exp2_2104);
                            float _exp2_2105 = approx_exp2(tw_120.x);
                            float _exp2_2106 = approx_exp2(tw_120.y);
                            float2 w_120 = make_float2(_exp2_2105, _exp2_2106);
                            float2 _f32x2_mul_b_2107 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_120) : "l"(*(const unsigned long long*)&ax_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_2107));
                            float _exp2_2108 = approx_exp2_noftz(tp_120.x);
                            float _exp2_2109 = approx_exp2_noftz(tp_120.y);
                            float2 p_120 = make_float2(_exp2_2108, _exp2_2109);
                            float2 d1_120 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_120);
                            float2 d2_120 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_120);
                            float2 dw_120 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_120);
                            float2 ng_120 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_120);
                            float2 nu_120 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_120);
                            float _rcp_2110 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_120, d2_120), dw_120).x);
                            float _rcp_2111 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_120, d2_120), dw_120).y);
                            float2 r_242 = make_float2(_rcp_2110, _rcp_2111);
                            float2 r_241 = r_242;
                            float2 _f32x2_mul_a_2112 = mul_f32x2_noftz(ng_120, nu_120);
                            float2 _f32x2_mul_b_2113 = mul_f32x2_noftz(r_241, make_float2(bl, bl));
                            float2 m_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_120) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2112), "l"(*(const unsigned long long*)&_f32x2_mul_b_2113));
                            float2 ns_120 = make_float2(((x_121.x < 0.0f) ? p_120.x : 1.0f), ((x_121.y < 0.0f) ? p_120.y : 1.0f));
                            gate_packed_3[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_120, ns_120))) | sign_121;
                            float2 x_122 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[26]));
                            float2 v_122 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[26]));
                            unsigned int sign_122 = (gate_packed_3[26] ^ up_packed_3[26]) & 2147516416u;
                            float _abs_2114;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2114) : "f"(x_122.x));
                            float _abs_2115;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2115) : "f"(x_122.y));
                            float2 ax_121 = make_float2(_abs_2114, _abs_2115);
                            float _abs_2116;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2116) : "f"(v_122.x));
                            float _abs_2117;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2117) : "f"(v_122.y));
                            float2 av_121 = make_float2(_abs_2116, _abs_2117);
                            float2 _f32x2_mul_b_2118 = make_float2(k1, k1);
                            float2 te_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_121) : "l"(*(const unsigned long long*)&ax_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_2118));
                            float2 _f32x2_mul_b_2119 = make_float2(k2, k2);
                            float2 tw_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_121) : "l"(*(const unsigned long long*)&av_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_2119));
                            float _exp2_2120 = approx_exp2(te_121.x);
                            float _exp2_2121 = approx_exp2(te_121.y);
                            float2 e_121 = make_float2(_exp2_2120, _exp2_2121);
                            float _exp2_2122 = approx_exp2(tw_121.x);
                            float _exp2_2123 = approx_exp2(tw_121.y);
                            float2 w_121 = make_float2(_exp2_2122, _exp2_2123);
                            float2 _f32x2_mul_b_2124 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_121) : "l"(*(const unsigned long long*)&ax_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_2124));
                            float _exp2_2125 = approx_exp2_noftz(tp_121.x);
                            float _exp2_2126 = approx_exp2_noftz(tp_121.y);
                            float2 p_121 = make_float2(_exp2_2125, _exp2_2126);
                            float2 d1_121 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_121);
                            float2 d2_121 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_121);
                            float2 dw_121 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_121);
                            float2 ng_121 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_121);
                            float2 nu_121 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_121);
                            float _rcp_2127 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_121, d2_121), dw_121).x);
                            float _rcp_2128 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_121, d2_121), dw_121).y);
                            float2 r_244 = make_float2(_rcp_2127, _rcp_2128);
                            float2 r_243 = r_244;
                            float2 _f32x2_mul_a_2129 = mul_f32x2_noftz(ng_121, nu_121);
                            float2 _f32x2_mul_b_2130 = mul_f32x2_noftz(r_243, make_float2(bl, bl));
                            float2 m_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_121) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2129), "l"(*(const unsigned long long*)&_f32x2_mul_b_2130));
                            float2 ns_121 = make_float2(((x_122.x < 0.0f) ? p_121.x : 1.0f), ((x_122.y < 0.0f) ? p_121.y : 1.0f));
                            gate_packed_3[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_121, ns_121))) | sign_122;
                            float2 x_123 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[27]));
                            float2 v_123 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[27]));
                            unsigned int sign_123 = (gate_packed_3[27] ^ up_packed_3[27]) & 2147516416u;
                            float _abs_2131;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2131) : "f"(x_123.x));
                            float _abs_2132;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2132) : "f"(x_123.y));
                            float2 ax_122 = make_float2(_abs_2131, _abs_2132);
                            float _abs_2133;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2133) : "f"(v_123.x));
                            float _abs_2134;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2134) : "f"(v_123.y));
                            float2 av_122 = make_float2(_abs_2133, _abs_2134);
                            float2 _f32x2_mul_b_2135 = make_float2(k1, k1);
                            float2 te_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_122) : "l"(*(const unsigned long long*)&ax_122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2135));
                            float2 _f32x2_mul_b_2136 = make_float2(k2, k2);
                            float2 tw_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_122) : "l"(*(const unsigned long long*)&av_122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2136));
                            float _exp2_2137 = approx_exp2(te_122.x);
                            float _exp2_2138 = approx_exp2(te_122.y);
                            float2 e_122 = make_float2(_exp2_2137, _exp2_2138);
                            float _exp2_2139 = approx_exp2(tw_122.x);
                            float _exp2_2140 = approx_exp2(tw_122.y);
                            float2 w_122 = make_float2(_exp2_2139, _exp2_2140);
                            float2 _f32x2_mul_b_2141 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_122) : "l"(*(const unsigned long long*)&ax_122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2141));
                            float _exp2_2142 = approx_exp2_noftz(tp_122.x);
                            float _exp2_2143 = approx_exp2_noftz(tp_122.y);
                            float2 p_122 = make_float2(_exp2_2142, _exp2_2143);
                            float2 d1_122 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_122);
                            float2 d2_122 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_122);
                            float2 dw_122 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_122);
                            float2 ng_122 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_122);
                            float2 nu_122 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_122);
                            float _rcp_2144 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_122, d2_122), dw_122).x);
                            float _rcp_2145 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_122, d2_122), dw_122).y);
                            float2 r_246 = make_float2(_rcp_2144, _rcp_2145);
                            float2 r_245 = r_246;
                            float2 _f32x2_mul_a_2146 = mul_f32x2_noftz(ng_122, nu_122);
                            float2 _f32x2_mul_b_2147 = mul_f32x2_noftz(r_245, make_float2(bl, bl));
                            float2 m_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_122) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2146), "l"(*(const unsigned long long*)&_f32x2_mul_b_2147));
                            float2 ns_122 = make_float2(((x_123.x < 0.0f) ? p_122.x : 1.0f), ((x_123.y < 0.0f) ? p_122.y : 1.0f));
                            gate_packed_3[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_122, ns_122))) | sign_123;
                            float2 x_124 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[28]));
                            float2 v_124 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[28]));
                            unsigned int sign_124 = (gate_packed_3[28] ^ up_packed_3[28]) & 2147516416u;
                            float _abs_2148;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2148) : "f"(x_124.x));
                            float _abs_2149;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2149) : "f"(x_124.y));
                            float2 ax_123 = make_float2(_abs_2148, _abs_2149);
                            float _abs_2150;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2150) : "f"(v_124.x));
                            float _abs_2151;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2151) : "f"(v_124.y));
                            float2 av_123 = make_float2(_abs_2150, _abs_2151);
                            float2 _f32x2_mul_b_2152 = make_float2(k1, k1);
                            float2 te_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_123) : "l"(*(const unsigned long long*)&ax_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_2152));
                            float2 _f32x2_mul_b_2153 = make_float2(k2, k2);
                            float2 tw_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_123) : "l"(*(const unsigned long long*)&av_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_2153));
                            float _exp2_2154 = approx_exp2(te_123.x);
                            float _exp2_2155 = approx_exp2(te_123.y);
                            float2 e_123 = make_float2(_exp2_2154, _exp2_2155);
                            float _exp2_2156 = approx_exp2(tw_123.x);
                            float _exp2_2157 = approx_exp2(tw_123.y);
                            float2 w_123 = make_float2(_exp2_2156, _exp2_2157);
                            float2 _f32x2_mul_b_2158 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_123) : "l"(*(const unsigned long long*)&ax_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_2158));
                            float _exp2_2159 = approx_exp2_noftz(tp_123.x);
                            float _exp2_2160 = approx_exp2_noftz(tp_123.y);
                            float2 p_123 = make_float2(_exp2_2159, _exp2_2160);
                            float2 d1_123 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_123);
                            float2 d2_123 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_123);
                            float2 dw_123 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_123);
                            float2 ng_123 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_123);
                            float2 nu_123 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_123);
                            float _rcp_2161 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_123, d2_123), dw_123).x);
                            float _rcp_2162 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_123, d2_123), dw_123).y);
                            float2 r_248 = make_float2(_rcp_2161, _rcp_2162);
                            float2 r_247 = r_248;
                            float2 _f32x2_mul_a_2163 = mul_f32x2_noftz(ng_123, nu_123);
                            float2 _f32x2_mul_b_2164 = mul_f32x2_noftz(r_247, make_float2(bl, bl));
                            float2 m_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_123) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2163), "l"(*(const unsigned long long*)&_f32x2_mul_b_2164));
                            float2 ns_123 = make_float2(((x_124.x < 0.0f) ? p_123.x : 1.0f), ((x_124.y < 0.0f) ? p_123.y : 1.0f));
                            gate_packed_3[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_123, ns_123))) | sign_124;
                            float2 x_125 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[29]));
                            float2 v_125 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[29]));
                            unsigned int sign_125 = (gate_packed_3[29] ^ up_packed_3[29]) & 2147516416u;
                            float _abs_2165;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2165) : "f"(x_125.x));
                            float _abs_2166;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2166) : "f"(x_125.y));
                            float2 ax_124 = make_float2(_abs_2165, _abs_2166);
                            float _abs_2167;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2167) : "f"(v_125.x));
                            float _abs_2168;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2168) : "f"(v_125.y));
                            float2 av_124 = make_float2(_abs_2167, _abs_2168);
                            float2 _f32x2_mul_b_2169 = make_float2(k1, k1);
                            float2 te_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_124) : "l"(*(const unsigned long long*)&ax_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_2169));
                            float2 _f32x2_mul_b_2170 = make_float2(k2, k2);
                            float2 tw_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_124) : "l"(*(const unsigned long long*)&av_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_2170));
                            float _exp2_2171 = approx_exp2(te_124.x);
                            float _exp2_2172 = approx_exp2(te_124.y);
                            float2 e_124 = make_float2(_exp2_2171, _exp2_2172);
                            float _exp2_2173 = approx_exp2(tw_124.x);
                            float _exp2_2174 = approx_exp2(tw_124.y);
                            float2 w_124 = make_float2(_exp2_2173, _exp2_2174);
                            float2 _f32x2_mul_b_2175 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_124) : "l"(*(const unsigned long long*)&ax_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_2175));
                            float _exp2_2176 = approx_exp2_noftz(tp_124.x);
                            float _exp2_2177 = approx_exp2_noftz(tp_124.y);
                            float2 p_124 = make_float2(_exp2_2176, _exp2_2177);
                            float2 d1_124 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_124);
                            float2 d2_124 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_124);
                            float2 dw_124 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_124);
                            float2 ng_124 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_124);
                            float2 nu_124 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_124);
                            float _rcp_2178 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_124, d2_124), dw_124).x);
                            float _rcp_2179 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_124, d2_124), dw_124).y);
                            float2 r_250 = make_float2(_rcp_2178, _rcp_2179);
                            float2 r_249 = r_250;
                            float2 _f32x2_mul_a_2180 = mul_f32x2_noftz(ng_124, nu_124);
                            float2 _f32x2_mul_b_2181 = mul_f32x2_noftz(r_249, make_float2(bl, bl));
                            float2 m_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_124) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2180), "l"(*(const unsigned long long*)&_f32x2_mul_b_2181));
                            float2 ns_124 = make_float2(((x_125.x < 0.0f) ? p_124.x : 1.0f), ((x_125.y < 0.0f) ? p_124.y : 1.0f));
                            gate_packed_3[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_124, ns_124))) | sign_125;
                            float2 x_126 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[30]));
                            float2 v_126 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[30]));
                            unsigned int sign_126 = (gate_packed_3[30] ^ up_packed_3[30]) & 2147516416u;
                            float _abs_2182;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2182) : "f"(x_126.x));
                            float _abs_2183;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2183) : "f"(x_126.y));
                            float2 ax_125 = make_float2(_abs_2182, _abs_2183);
                            float _abs_2184;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2184) : "f"(v_126.x));
                            float _abs_2185;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2185) : "f"(v_126.y));
                            float2 av_125 = make_float2(_abs_2184, _abs_2185);
                            float2 _f32x2_mul_b_2186 = make_float2(k1, k1);
                            float2 te_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_125) : "l"(*(const unsigned long long*)&ax_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_2186));
                            float2 _f32x2_mul_b_2187 = make_float2(k2, k2);
                            float2 tw_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_125) : "l"(*(const unsigned long long*)&av_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_2187));
                            float _exp2_2188 = approx_exp2(te_125.x);
                            float _exp2_2189 = approx_exp2(te_125.y);
                            float2 e_125 = make_float2(_exp2_2188, _exp2_2189);
                            float _exp2_2190 = approx_exp2(tw_125.x);
                            float _exp2_2191 = approx_exp2(tw_125.y);
                            float2 w_125 = make_float2(_exp2_2190, _exp2_2191);
                            float2 _f32x2_mul_b_2192 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_125) : "l"(*(const unsigned long long*)&ax_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_2192));
                            float _exp2_2193 = approx_exp2_noftz(tp_125.x);
                            float _exp2_2194 = approx_exp2_noftz(tp_125.y);
                            float2 p_125 = make_float2(_exp2_2193, _exp2_2194);
                            float2 d1_125 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_125);
                            float2 d2_125 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_125);
                            float2 dw_125 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_125);
                            float2 ng_125 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_125);
                            float2 nu_125 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_125);
                            float _rcp_2195 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_125, d2_125), dw_125).x);
                            float _rcp_2196 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_125, d2_125), dw_125).y);
                            float2 r_252 = make_float2(_rcp_2195, _rcp_2196);
                            float2 r_251 = r_252;
                            float2 _f32x2_mul_a_2197 = mul_f32x2_noftz(ng_125, nu_125);
                            float2 _f32x2_mul_b_2198 = mul_f32x2_noftz(r_251, make_float2(bl, bl));
                            float2 m_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_125) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2197), "l"(*(const unsigned long long*)&_f32x2_mul_b_2198));
                            float2 ns_125 = make_float2(((x_126.x < 0.0f) ? p_125.x : 1.0f), ((x_126.y < 0.0f) ? p_125.y : 1.0f));
                            gate_packed_3[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_125, ns_125))) | sign_126;
                            float2 x_127 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[31]));
                            float2 v_127 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[31]));
                            unsigned int sign_127 = (gate_packed_3[31] ^ up_packed_3[31]) & 2147516416u;
                            float _abs_2199;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2199) : "f"(x_127.x));
                            float _abs_2200;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2200) : "f"(x_127.y));
                            float2 ax_126 = make_float2(_abs_2199, _abs_2200);
                            float _abs_2201;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2201) : "f"(v_127.x));
                            float _abs_2202;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2202) : "f"(v_127.y));
                            float2 av_126 = make_float2(_abs_2201, _abs_2202);
                            float2 _f32x2_mul_b_2203 = make_float2(k1, k1);
                            float2 te_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_126) : "l"(*(const unsigned long long*)&ax_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_2203));
                            float2 _f32x2_mul_b_2204 = make_float2(k2, k2);
                            float2 tw_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_126) : "l"(*(const unsigned long long*)&av_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_2204));
                            float _exp2_2205 = approx_exp2(te_126.x);
                            float _exp2_2206 = approx_exp2(te_126.y);
                            float2 e_126 = make_float2(_exp2_2205, _exp2_2206);
                            float _exp2_2207 = approx_exp2(tw_126.x);
                            float _exp2_2208 = approx_exp2(tw_126.y);
                            float2 w_126 = make_float2(_exp2_2207, _exp2_2208);
                            float2 _f32x2_mul_b_2209 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_126) : "l"(*(const unsigned long long*)&ax_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_2209));
                            float _exp2_2210 = approx_exp2_noftz(tp_126.x);
                            float _exp2_2211 = approx_exp2_noftz(tp_126.y);
                            float2 p_126 = make_float2(_exp2_2210, _exp2_2211);
                            float2 d1_126 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_126);
                            float2 d2_126 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_126);
                            float2 dw_126 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_126);
                            float2 ng_126 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_126);
                            float2 nu_126 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_126);
                            float _rcp_2212 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_126, d2_126), dw_126).x);
                            float _rcp_2213 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_126, d2_126), dw_126).y);
                            float2 r_254 = make_float2(_rcp_2212, _rcp_2213);
                            float2 r_253 = r_254;
                            float2 _f32x2_mul_a_2214 = mul_f32x2_noftz(ng_126, nu_126);
                            float2 _f32x2_mul_b_2215 = mul_f32x2_noftz(r_253, make_float2(bl, bl));
                            float2 m_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_126) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2214), "l"(*(const unsigned long long*)&_f32x2_mul_b_2215));
                            float2 ns_126 = make_float2(((x_127.x < 0.0f) ? p_126.x : 1.0f), ((x_127.y < 0.0f) ? p_126.y : 1.0f));
                            gate_packed_3[31] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_126, ns_126))) | sign_127;
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_22 = tid / 32;
                            int lane_22 = tid % 32;
                            #pragma unroll
                            for (int half_23 = 0; half_23 < 2; half_23++) {
                                #pragma unroll
                                for (int col_tile_23 = 0; col_tile_23 < 2; col_tile_23++) {
                                    int row_24 = warp_22 * 32 + half_23 * 16 + lane_22 % 16;
                                    int col_24 = col_tile_23 * 16 + lane_22 / 16 * 8;
                                    unsigned int address_23 = d_smem_addr + 16384 + (unsigned int)((row_24 * 32 + col_24) * 2);
                                    address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                    int offset_24 = 16 + half_23 * 8 + col_tile_23 * 4;
                                    uint32_t _stmatrix_addr_2216 = static_cast<uint32_t>(address_23);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_2216), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (tid / 32 == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + x))), "r"(static_cast<unsigned int>(2)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    gemm_wide += iterations;
                    gemm_out0 += 1;
                    gemm_out1 += 1;
                    gemm_acc_tiles += 2;
                }
            }
            int _parity_schedule_arrived = (int)((unsigned int)iteration % 2);
            mbarrier_wait(schedule_arrived_addr, _parity_schedule_arrived);
            uint32_t _clc_valid_0 = 0;
            uint32_t _clc_ctaid_x_0;
            uint32_t _clc_ctaid_y_0;
            uint32_t _clc_ctaid_z_0;
            asm volatile(
                "{\n\t"
                ".reg .pred p1;\n\t"
                ".reg .b128 clc_r;\n\t"
                "ld.shared.b128 clc_r, [%4];\n\t"
                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                "selp.u32 %3, 1, 0, p1;\n\t"
                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                "}\n"
                : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                : "r"(smem + 512 + 0 * 16 + 0 * 16)
                : "memory");
            cluster = -1;
            if (_clc_valid_0 != 0) {
                cluster = (int)(_clc_ctaid_x_0 / 2);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            }
            __syncwarp();
            bool _elect_sync_0 = elect_sync();
            if (_elect_sync_0) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((schedule_finished_addr) & 0xFEFFFFFF) : "memory");
            }
            iteration += 1;
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (cluster >= 0 && tid == 0) {
            int drain_iteration = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 528 + 0 * 16 + 0 * 16), "r"(drain_arrived_0_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_0_addr, 16);
            while (1) {
                int _parity_drain_arrived_0 = (int)((unsigned int)drain_iteration % 2);
                mbarrier_wait(drain_arrived_0_addr, _parity_drain_arrived_0);
                uint32_t clc_valid = 0;
                uint32_t clc_ctaid_x;
                uint32_t clc_ctaid_y;
                uint32_t clc_ctaid_z;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x), "=r"(clc_ctaid_y), "=r"(clc_ctaid_z), "=r"(clc_valid)
                    : "r"(smem + 528 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_0_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_0 = (int)((unsigned int)drain_iteration % 2);
                    mbarrier_wait(drain_finished_0_addr, _parity_drain_finished_0);
                }
                if (clc_valid == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 528 + 0 * 16 + 0 * 16), "r"(drain_arrived_0_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_0_addr, 16);
                drain_iteration += 1;
            }
        }
        if (cluster >= 0 && tid == 32) {
            int drain_iteration_1 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 544 + 0 * 16 + 0 * 16), "r"(drain_arrived_1_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_1_addr, 16);
            while (1) {
                int _parity_drain_arrived_1 = (int)((unsigned int)drain_iteration_1 % 2);
                mbarrier_wait(drain_arrived_1_addr, _parity_drain_arrived_1);
                uint32_t clc_valid_1 = 0;
                uint32_t clc_ctaid_x_1;
                uint32_t clc_ctaid_y_1;
                uint32_t clc_ctaid_z_1;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_1), "=r"(clc_ctaid_y_1), "=r"(clc_ctaid_z_1), "=r"(clc_valid_1)
                    : "r"(smem + 544 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_1 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_1_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_1 = (int)((unsigned int)drain_iteration_1 % 2);
                    mbarrier_wait(drain_finished_1_addr, _parity_drain_finished_1);
                }
                if (clc_valid_1 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 544 + 0 * 16 + 0 * 16), "r"(drain_arrived_1_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_1_addr, 16);
                drain_iteration_1 += 1;
            }
        }
        if (cluster >= 0 && tid == 64) {
            int drain_iteration_2 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 560 + 0 * 16 + 0 * 16), "r"(drain_arrived_2_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_2_addr, 16);
            while (1) {
                int _parity_drain_arrived_2 = (int)((unsigned int)drain_iteration_2 % 2);
                mbarrier_wait(drain_arrived_2_addr, _parity_drain_arrived_2);
                uint32_t clc_valid_2 = 0;
                uint32_t clc_ctaid_x_2;
                uint32_t clc_ctaid_y_2;
                uint32_t clc_ctaid_z_2;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_2), "=r"(clc_ctaid_y_2), "=r"(clc_ctaid_z_2), "=r"(clc_valid_2)
                    : "r"(smem + 560 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_2 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_2_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_2 = (int)((unsigned int)drain_iteration_2 % 2);
                    mbarrier_wait(drain_finished_2_addr, _parity_drain_finished_2);
                }
                if (clc_valid_2 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 560 + 0 * 16 + 0 * 16), "r"(drain_arrived_2_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_2_addr, 16);
                drain_iteration_2 += 1;
            }
        }
        if (cluster >= 0 && tid == 96) {
            int drain_iteration_3 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 576 + 0 * 16 + 0 * 16), "r"(drain_arrived_3_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_3_addr, 16);
            while (1) {
                int _parity_drain_arrived_3 = (int)((unsigned int)drain_iteration_3 % 2);
                mbarrier_wait(drain_arrived_3_addr, _parity_drain_arrived_3);
                uint32_t clc_valid_3 = 0;
                uint32_t clc_ctaid_x_3;
                uint32_t clc_ctaid_y_3;
                uint32_t clc_ctaid_z_3;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_3), "=r"(clc_ctaid_y_3), "=r"(clc_ctaid_z_3), "=r"(clc_valid_3)
                    : "r"(smem + 576 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_3 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_3_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_3 = (int)((unsigned int)drain_iteration_3 % 2);
                    mbarrier_wait(drain_finished_3_addr, _parity_drain_finished_3);
                }
                if (clc_valid_3 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 576 + 0 * 16 + 0 * 16), "r"(drain_arrived_3_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_3_addr, 16);
                drain_iteration_3 += 1;
            }
        }
        if (cluster >= 0 && tid == 128) {
            int drain_iteration_4 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 592 + 0 * 16 + 0 * 16), "r"(drain_arrived_4_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_4_addr, 16);
            while (1) {
                int _parity_drain_arrived_4 = (int)((unsigned int)drain_iteration_4 % 2);
                mbarrier_wait(drain_arrived_4_addr, _parity_drain_arrived_4);
                uint32_t clc_valid_4 = 0;
                uint32_t clc_ctaid_x_4;
                uint32_t clc_ctaid_y_4;
                uint32_t clc_ctaid_z_4;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_4), "=r"(clc_ctaid_y_4), "=r"(clc_ctaid_z_4), "=r"(clc_valid_4)
                    : "r"(smem + 592 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_4 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_4_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_4 = (int)((unsigned int)drain_iteration_4 % 2);
                    mbarrier_wait(drain_finished_4_addr, _parity_drain_finished_4);
                }
                if (clc_valid_4 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 592 + 0 * 16 + 0 * 16), "r"(drain_arrived_4_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_4_addr, 16);
                drain_iteration_4 += 1;
            }
        }
        if (cluster >= 0 && tid == 160) {
            int drain_iteration_5 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 608 + 0 * 16 + 0 * 16), "r"(drain_arrived_5_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_5_addr, 16);
            while (1) {
                int _parity_drain_arrived_5 = (int)((unsigned int)drain_iteration_5 % 2);
                mbarrier_wait(drain_arrived_5_addr, _parity_drain_arrived_5);
                uint32_t clc_valid_5 = 0;
                uint32_t clc_ctaid_x_5;
                uint32_t clc_ctaid_y_5;
                uint32_t clc_ctaid_z_5;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_5), "=r"(clc_ctaid_y_5), "=r"(clc_ctaid_z_5), "=r"(clc_valid_5)
                    : "r"(smem + 608 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_5 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_5_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_5 = (int)((unsigned int)drain_iteration_5 % 2);
                    mbarrier_wait(drain_finished_5_addr, _parity_drain_finished_5);
                }
                if (clc_valid_5 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 608 + 0 * 16 + 0 * 16), "r"(drain_arrived_5_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_5_addr, 16);
                drain_iteration_5 += 1;
            }
        }
        if (cluster >= 0 && tid == 192) {
            int drain_iteration_6 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 624 + 0 * 16 + 0 * 16), "r"(drain_arrived_6_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_6_addr, 16);
            while (1) {
                int _parity_drain_arrived_6 = (int)((unsigned int)drain_iteration_6 % 2);
                mbarrier_wait(drain_arrived_6_addr, _parity_drain_arrived_6);
                uint32_t clc_valid_6 = 0;
                uint32_t clc_ctaid_x_6;
                uint32_t clc_ctaid_y_6;
                uint32_t clc_ctaid_z_6;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_6), "=r"(clc_ctaid_y_6), "=r"(clc_ctaid_z_6), "=r"(clc_valid_6)
                    : "r"(smem + 624 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_6 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_6_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_6 = (int)((unsigned int)drain_iteration_6 % 2);
                    mbarrier_wait(drain_finished_6_addr, _parity_drain_finished_6);
                }
                if (clc_valid_6 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 624 + 0 * 16 + 0 * 16), "r"(drain_arrived_6_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_6_addr, 16);
                drain_iteration_6 += 1;
            }
        }
        if (cluster >= 0 && tid == 224) {
            int drain_iteration_7 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 640 + 0 * 16 + 0 * 16), "r"(drain_arrived_7_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_7_addr, 16);
            while (1) {
                int _parity_drain_arrived_7 = (int)((unsigned int)drain_iteration_7 % 2);
                mbarrier_wait(drain_arrived_7_addr, _parity_drain_arrived_7);
                uint32_t clc_valid_7 = 0;
                uint32_t clc_ctaid_x_7;
                uint32_t clc_ctaid_y_7;
                uint32_t clc_ctaid_z_7;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_7), "=r"(clc_ctaid_y_7), "=r"(clc_ctaid_z_7), "=r"(clc_valid_7)
                    : "r"(smem + 640 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_7 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_7_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_7 = (int)((unsigned int)drain_iteration_7 % 2);
                    mbarrier_wait(drain_finished_7_addr, _parity_drain_finished_7);
                }
                if (clc_valid_7 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 640 + 0 * 16 + 0 * 16), "r"(drain_arrived_7_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_7_addr, 16);
                drain_iteration_7 += 1;
            }
        }
    }

    // ---- Role: warp_role_0 ----
    if (warp == 0) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_1 ----
    if (warp == 1) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_2 ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_3 ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_4 ----
    if (warp == 4) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_5 ----
    if (warp == 5) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_6 ----
    if (warp == 6) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_7 ----
    if (warp == 7) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
