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





__device__ __forceinline__ void tma_store_5d(
    const void *tmap, int x, int y, int z, int w, int v, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr) : "memory");
}


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
kernel_cake_mok_forward_situ_noshared(const __grid_constant__ CUtensorMap x_routed, const __grid_constant__ CUtensorMap wg_routed, const __grid_constant__ CUtensorMap wu_routed, const __grid_constant__ CUtensorMap wd_routed, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap hidden_routed_out, const __grid_constant__ CUtensorMap hidden_routed_in, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ x_routed_ptr, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float situ_beta, float situ_lb, CakeFastDivmod mini_tasks_div, CakeFastDivmod minis_per_macro_div, CakeFastDivmod fused_col_blocks_div, CakeFastDivmod down_col_blocks_div)
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
    #define dispatch_arrived_addr (mbar_base + 240)
    #define range_arrived_addr (mbar_base + 248)
    #define rows_arrived_addr (mbar_base + 272)
    #define combine_arrived_addr (mbar_base + 296)

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
    int* flat_view_3 = reinterpret_cast<int*>(smem_raw + 197632);
    const int flat_view_addr_3 = smem + 197632;
    float* flat_view_4 = reinterpret_cast<float*>(smem_raw + 199680);
    const int flat_view_addr_4 = smem + 199680;
    __nv_bfloat16* flat_view_5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr_5 = smem + 1024;

    // Kernel setup ops
    float k1 = __fdiv_rn(-2.88539f, situ_beta);
    float k2 = __fdiv_rn(-2.88539f, situ_lb);
    float bl;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(bl) : "f"(situ_beta), "f"(situ_lb));
    int tokens = num_tokens[0];
    int shared_rows = 0;
    int shared_fused = shared_rows * (intermediate / 256);
    int mini_fused = mini_size / 256 * (intermediate / 256);
    int shared_tasks = shared_fused + shared_rows * ((hidden + 511) / 512);
    int mini_tasks = mini_fused + mini_size / 256 * ((hidden + 511) / 512);
    int comm_clusters = comm_sms / 2;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis_per_macro = macro_size / mini_size;
    int true_minis = (tokens + mini_size - 1) / mini_size;
    int last_minis = true_minis - (macros - 1) * minis_per_macro;
    int true_clusters = comm_clusters + shared_tasks + true_minis * mini_tasks;
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (44 barriers at smem_raw[0..352))

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
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 240, 1);
        // range_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 248, 1);
        mbarrier_init(smem + 256, 1);
        mbarrier_init(smem + 264, 1);
        // rows_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 272, 1);
        mbarrier_init(smem + 280, 1);
        mbarrier_init(smem + 288, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 296, 1);
        mbarrier_init(smem + 304, 1);
        mbarrier_init(smem + 312, 1);
        mbarrier_init(smem + 320, 1);
        mbarrier_init(smem + 328, 1);
        mbarrier_init(smem + 336, 1);
        mbarrier_init(smem + 344, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    if (warp == 0) {
        int _tmem_hold = smem + 352;
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
    int dispatch_seq = 0;
    int combine_seq_0 = 0;
    int combine_seq_1 = 0;
    int combine_seq_2 = 0;
    int combine_seq_3 = 0;
    int combine_seq_4 = 0;
    int combine_seq_5 = 0;
    int combine_seq_6 = 0;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        if (macros > 0) {
            int col_blocks = (hidden + 511) / 512;
            int unit_rows = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int block_units = (128 + unit_rows - 1) / unit_rows;
            int macro_offset = (macros - 1) * macro_size;
            int _min_0 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
            int macro_blocks = _min_0 / 128;
            int blocks = 0;
            if (comm_cta < macro_blocks) {
                blocks = (macro_blocks - comm_cta + comm_sms - 1) / comm_sms;
            }
            int units = blocks * block_units;
            int _min_0_1 = ((units) < (2) ? (units) : (2));
            #pragma unroll 1
            for (int unit = 0; unit < _min_0_1; unit++) {
                int buffer = unit % 3;
                int block = comm_cta + unit / block_units * comm_sms;
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
                int block_1 = comm_cta + unit_1 / block_units * comm_sms;
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
                        int done_block = comm_cta + (unit_1 - 2) / block_units * comm_sms;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + done_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                    }
                }
                if (units > unit_1 + 2) {
                    int buffer_0 = (unit_1 + 2) % 3;
                    int block_0 = comm_cta + (unit_1 + 2) / block_units * comm_sms;
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
                        int tail_block = comm_cta + tail / block_units * comm_sms;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + tail_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                    }
                }
            }
            int macro = macros - 1;
            while (macro > 0) {
                int macro_rows = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                int previous_rows = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                int dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                int _max_1 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task = comm_cta; task < _max_1; task += comm_sms) {
                    if (combine_tasks > task) {
                        int col_blocks_0 = (hidden + 1023) / 1024;
                        int first_tile = task * 7;
                        int macro_offset_1 = macro * macro_size;
                        int macro_tokens = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                        int valid_tiles = ((7) < (macro_tokens / 16 * col_blocks_0 - first_tile) ? (7) : (macro_tokens / 16 * col_blocks_0 - first_tile));
                        if (valid_tiles > 0) {
                            int first_row_2 = first_tile / col_blocks_0 * 16 + tid;
                            int first_col = first_tile % col_blocks_0;
                            int rows_2[7];
                            int columns[7];
                            int peers[7];
                            int tokens_0[7];
                            unsigned int counts_1[7];
                            int row = first_row_2;
                            int column = first_col;
                            #pragma unroll
                            for (int stage = 0; stage < 7; stage++) {
                                rows_2[stage] = row;
                                columns[stage] = column;
                                peers[stage] = -1;
                                tokens_0[stage] = -1;
                                if (valid_tiles > stage && tid < 16) {
                                    peers[stage] = schedule_rank[macro_offset_1 + row];
                                    tokens_0[stage] = schedule_token[macro_offset_1 + row];
                                }
                                counts_1[stage] = 0;
                                if (valid_tiles > stage) {
                                    if (stage == 0 || column == 0) {
                                        uint32_t _barrier_vote_2 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _barrier_vote_2;
                                    } else {
                                        counts_1[stage] = counts_1[stage - 1];
                                    }
                                }
                                column = column + 1;
                                if (column == col_blocks_0) {
                                    column = 0;
                                    row = row + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini = (macro_offset_1 + first_row_2) / mini_size;
                                int last_mini = (macro_offset_1 + (first_tile + valid_tiles - 1) / col_blocks_0 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini = first_mini; mini < last_mini + 1; mini++) {
                                    int mini_rows = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                                    int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                                    int poll_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_first) : "l"(y_ready + mini) : "memory");
                                    int poll = poll_first;
                                    while (poll < required) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_next) : "l"(y_ready + mini) : "memory");
                                        poll = poll_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                if (valid_tiles > 0) {
                                    int _min_3 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                    unsigned int chunk_bytes = (unsigned int)(_min_3 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr, counts_1[0] * chunk_bytes);
                                }
                                if (valid_tiles > 1) {
                                    int _min_4 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                    unsigned int chunk_bytes_1 = (unsigned int)(_min_4 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 8, counts_1[1] * chunk_bytes_1);
                                }
                                if (valid_tiles > 2) {
                                    int _min_5 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                    unsigned int chunk_bytes_2 = (unsigned int)(_min_5 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 16, counts_1[2] * chunk_bytes_2);
                                }
                                if (valid_tiles > 3) {
                                    int _min_6 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                    unsigned int chunk_bytes_3 = (unsigned int)(_min_6 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 24, counts_1[3] * chunk_bytes_3);
                                }
                                if (valid_tiles > 4) {
                                    int _min_7 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                    unsigned int chunk_bytes_4 = (unsigned int)(_min_7 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 32, counts_1[4] * chunk_bytes_4);
                                }
                                if (valid_tiles > 5) {
                                    int _min_8 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                    unsigned int chunk_bytes_5 = (unsigned int)(_min_8 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 40, counts_1[5] * chunk_bytes_5);
                                }
                                if (valid_tiles > 6) {
                                    int _min_9 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                    unsigned int chunk_bytes_6 = (unsigned int)(_min_9 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 48, counts_1[6] * chunk_bytes_6);
                                }
                            }
                            __syncthreads();
                            if (peers[0] >= 0) {
                                int chunk_cols = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)(tid * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)) * (unsigned long long)2), chunk_cols * 2, combine_arrived_addr);
                            }
                            if (peers[1] >= 0) {
                                int chunk_cols_1 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)) * (unsigned long long)2), chunk_cols_1 * 2, combine_arrived_addr + 8);
                            }
                            if (peers[2] >= 0) {
                                int chunk_cols_2 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((32 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)) * (unsigned long long)2), chunk_cols_2 * 2, combine_arrived_addr + 16);
                            }
                            if (peers[3] >= 0) {
                                int chunk_cols_3 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((48 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)) * (unsigned long long)2), chunk_cols_3 * 2, combine_arrived_addr + 24);
                            }
                            if (peers[4] >= 0) {
                                int chunk_cols_4 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((64 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)) * (unsigned long long)2), chunk_cols_4 * 2, combine_arrived_addr + 32);
                            }
                            if (peers[5] >= 0) {
                                int chunk_cols_5 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((80 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)) * (unsigned long long)2), chunk_cols_5 * 2, combine_arrived_addr + 40);
                            }
                            if (peers[6] >= 0) {
                                int chunk_cols_6 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((96 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)) * (unsigned long long)2), chunk_cols_6 * 2, combine_arrived_addr + 48);
                            }
                            if (valid_tiles > 0) {
                                int _parity_combine_arrived = (int)((unsigned int)combine_seq_0 % 2);
                                mbarrier_wait(combine_arrived_addr, _parity_combine_arrived);
                                combine_seq_0 += 1;
                                if (peers[0] >= 0) {
                                    int _min_10 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                    unsigned int chunk_bytes_7 = (unsigned int)(_min_10 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_11 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[0]]) + ((unsigned long long)tokens_0[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_11), "r"(flat_view_addr_5 + (unsigned int)(tid * 2048)), "r"((uint32_t)(chunk_bytes_7))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 1) {
                                int _parity_combine_arrived_1 = (int)((unsigned int)combine_seq_1 % 2);
                                mbarrier_wait(combine_arrived_addr + 8, _parity_combine_arrived_1);
                                combine_seq_1 += 1;
                                if (peers[1] >= 0) {
                                    int _min_12 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                    unsigned int chunk_bytes_8 = (unsigned int)(_min_12 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_13 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[1]]) + ((unsigned long long)tokens_0[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_13), "r"(flat_view_addr_5 + (unsigned int)((16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_8))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 2) {
                                int _parity_combine_arrived_2 = (int)((unsigned int)combine_seq_2 % 2);
                                mbarrier_wait(combine_arrived_addr + 16, _parity_combine_arrived_2);
                                combine_seq_2 += 1;
                                if (peers[2] >= 0) {
                                    int _min_14 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                    unsigned int chunk_bytes_9 = (unsigned int)(_min_14 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_15 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[2]]) + ((unsigned long long)tokens_0[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_15), "r"(flat_view_addr_5 + (unsigned int)((32 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_9))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 3) {
                                int _parity_combine_arrived_3 = (int)((unsigned int)combine_seq_3 % 2);
                                mbarrier_wait(combine_arrived_addr + 24, _parity_combine_arrived_3);
                                combine_seq_3 += 1;
                                if (peers[3] >= 0) {
                                    int _min_16 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                    unsigned int chunk_bytes_10 = (unsigned int)(_min_16 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_17 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[3]]) + ((unsigned long long)tokens_0[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_17), "r"(flat_view_addr_5 + (unsigned int)((48 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_10))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 4) {
                                int _parity_combine_arrived_4 = (int)((unsigned int)combine_seq_4 % 2);
                                mbarrier_wait(combine_arrived_addr + 32, _parity_combine_arrived_4);
                                combine_seq_4 += 1;
                                if (peers[4] >= 0) {
                                    int _min_18 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                    unsigned int chunk_bytes_11 = (unsigned int)(_min_18 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_19 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[4]]) + ((unsigned long long)tokens_0[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_19), "r"(flat_view_addr_5 + (unsigned int)((64 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_11))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 5) {
                                int _parity_combine_arrived_5 = (int)((unsigned int)combine_seq_5 % 2);
                                mbarrier_wait(combine_arrived_addr + 40, _parity_combine_arrived_5);
                                combine_seq_5 += 1;
                                if (peers[5] >= 0) {
                                    int _min_20 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                    unsigned int chunk_bytes_12 = (unsigned int)(_min_20 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_21 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[5]]) + ((unsigned long long)tokens_0[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_21), "r"(flat_view_addr_5 + (unsigned int)((80 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_12))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 6) {
                                int _parity_combine_arrived_6 = (int)((unsigned int)combine_seq_6 % 2);
                                mbarrier_wait(combine_arrived_addr + 48, _parity_combine_arrived_6);
                                combine_seq_6 += 1;
                                if (peers[6] >= 0) {
                                    int _min_22 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                    unsigned int chunk_bytes_13 = (unsigned int)(_min_22 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_23 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[6]]) + ((unsigned long long)tokens_0[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_23), "r"(flat_view_addr_5 + (unsigned int)((96 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_13))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            int warp_1 = tid / 32;
                            if (tid % 32 == 0 && warp_1 < valid_tiles) {
                                int row_done = (first_tile + warp_1) / col_blocks_0 * 16;
                                bool enabled_value = 1;
                                if (enabled_value != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_1 + row_done) / 128))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                    }
                    if (dispatch_tasks > task) {
                        int col_blocks_0_1 = (hidden + 511) / 512;
                        int macro_offset_1_1 = (macro - 1) * macro_size;
                        int macro_tokens_1 = ((macro_size) < (tokens - macro_offset_1_1) ? (macro_size) : (tokens - macro_offset_1_1));
                        if (task < macro_tokens_1 / 128 * col_blocks_0_1) {
                            int row_1 = task / col_blocks_0_1 * 128;
                            int col_block = task % col_blocks_0_1;
                            int chunk_cols_7 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                            unsigned int chunk_bytes_14 = (unsigned int)(chunk_cols_7 * 2);
                            int peer_2 = -1;
                            int peer_token_2 = -1;
                            if (tid < 128) {
                                peer_2 = schedule_rank[macro_offset_1_1 + row_1 + tid];
                                peer_token_2 = schedule_token[macro_offset_1_1 + row_1 + tid];
                            }
                            uint32_t _cta_count_2 = __syncthreads_count(peer_2 >= 0);
                            if (tid == 0) {
                                int previous_offset = macro * macro_size;
                                int previous_tokens = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                if (row_1 < previous_tokens) {
                                    int previous_mini = (previous_offset + row_1) / mini_size;
                                    int mini_rows_1 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    bool enabled_value_1 = 1;
                                    if (enabled_value_1 != 0) {
                                        int poll_1_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_first) : "l"(y_ready + previous_mini) : "memory");
                                        int poll_1 = poll_1_first;
                                        while (poll_1 < required_1) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_1_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_next) : "l"(y_ready + previous_mini) : "memory");
                                            poll_1 = poll_1_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * chunk_bytes_14);
                            }
                            __syncthreads();
                            if (peer_2 >= 0) {
                                cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + (unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2), chunk_cols_7 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view) + (tid * 1024 + vec_2 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                            int _parity_dispatch_arrived = (int)((unsigned int)dispatch_seq % 2);
                            mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived);
                            dispatch_seq += 1;
                            if (tid < 128) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_24 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row_1 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_24), "r"(flat_view_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes_14))
                                        : "memory");
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncthreads();
                            if (tid == 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_1_1 + row_1) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                }
                macro = macro - 1;
            }
            int unit_rows_0 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int block_units_1 = (128 + unit_rows_0 - 1) / unit_rows_0;
            int macro_offset_2 = 0;
            int _min_25 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
            int macro_blocks_3 = _min_25 / 128;
            int blocks_4 = 0;
            if (comm_cta < macro_blocks_3) {
                blocks_4 = (macro_blocks_3 - comm_cta + comm_sms - 1) / comm_sms;
            }
            int units_5 = blocks_4 * block_units_1;
            int _min_1 = ((units_5) < (2) ? (units_5) : (2));
            #pragma unroll 1
            for (int unit_2 = 0; unit_2 < _min_1; unit_2++) {
                int slot = unit_2 % 3;
                int block_2 = comm_cta + unit_2 / block_units_1 * comm_sms;
                int first_row_3 = block_2 * 128 + unit_2 % block_units_1 * unit_rows_0;
                int _min_26 = ((unit_rows_0) < (block_2 * 128 + 128 - first_row_3) ? (unit_rows_0) : (block_2 * 128 + 128 - first_row_3));
                if (_min_26 > tid) {
                    flat_view_3[slot * 128 + tid] = schedule_rank[macro_offset_2 + first_row_3 + tid];
                    flat_view_3[384 + slot * 128 + tid] = schedule_token[macro_offset_2 + first_row_3 + tid];
                }
            }
            if (tid == 0) {
                int _min_2 = ((units_5) < (2) ? (units_5) : (2));
                #pragma unroll 1
                for (int unit_3 = 0; unit_3 < _min_2; unit_3++) {
                    int buffer_2 = unit_3 % 3;
                    int block_3 = comm_cta + unit_3 / block_units_1 * comm_sms;
                    int first_row_4 = block_3 * 128 + unit_3 % block_units_1 * unit_rows_0;
                    int rows_3 = ((unit_rows_0) < (block_3 * 128 + 128 - first_row_4) ? (unit_rows_0) : (block_3 * 128 + 128 - first_row_4));
                    int mini_1 = (macro_offset_2 + first_row_4) / mini_size;
                    int mini_rows_2 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                    int poll_2_first;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_first) : "l"(y_ready + mini_1) : "memory");
                    int poll_2 = poll_2_first;
                    while (poll_2 < (mini_rows_2 + 255) / 256 * (hidden / 256) * 2) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int poll_2_next;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_next) : "l"(y_ready + mini_1) : "memory");
                        poll_2 = poll_2_next;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_2) * 8, (unsigned int)(rows_3 * hidden * 2));
                    cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)(buffer_2 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)first_row_4 * (unsigned long long)hidden) * (unsigned long long)2), rows_3 * hidden * 2, rows_arrived_addr + (buffer_2) * 8);
                }
            }
            __syncthreads();
            #pragma unroll 1
            for (int unit_4 = 0; unit_4 < units_5; unit_4++) {
                int buffer_3 = unit_4 % 3;
                int block_4 = comm_cta + unit_4 / block_units_1 * comm_sms;
                int first_row_5 = block_4 * 128 + unit_4 % block_units_1 * unit_rows_0;
                int rows_4 = ((unit_rows_0) < (block_4 * 128 + 128 - first_row_5) ? (unit_rows_0) : (block_4 * 128 + 128 - first_row_5));
                int _parity_rows_arrived = (int)((unsigned int)unit_4 / 3 % 2);
                mbarrier_wait(rows_arrived_addr + (buffer_3) * 8, _parity_rows_arrived);
                if (rows_4 > tid) {
                    int peer_3 = flat_view_3[buffer_3 * 128 + tid];
                    int token = flat_view_3[384 + buffer_3 * 128 + tid];
                    if (peer_3 >= 0) {
                    }
                }
                if (tid == 0) {
                    #pragma unroll 1
                    for (int row_2 = 0; row_2 < rows_4; row_2++) {
                        int row_peer = flat_view_3[buffer_3 * 128 + row_2];
                        int row_token = flat_view_3[384 + buffer_3 * 128 + row_2];
                        if (row_peer >= 0) {
                            {
                                void* _cpbulk_dst_27 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[row_peer]) + (unsigned long long)row_token * (unsigned long long)hidden);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_27), "r"(flat_view_addr + (unsigned int)(buffer_3 * 65536) + (unsigned int)(row_2 * hidden * 2)), "r"((uint32_t)((unsigned int)(hidden * 2)))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    if (unit_4 % block_units_1 == block_units_1 - 1) {
                        bool enabled_value_2 = 0;
                        if (enabled_value_2 != 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_2 + block_4 * 128) / 128))), "r"(static_cast<unsigned int>(8 * ((hidden + 1023) / 1024))) : "memory");
                        }
                    }
                    if (units_5 > unit_4 + 2) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        int buffer_1_1 = (unit_4 + 2) % 3;
                        int block_1_1 = comm_cta + (unit_4 + 2) / block_units_1 * comm_sms;
                        int first_row_1_1 = block_1_1 * 128 + (unit_4 + 2) % block_units_1 * unit_rows_0;
                        int rows_1_1 = ((unit_rows_0) < (block_1_1 * 128 + 128 - first_row_1_1) ? (unit_rows_0) : (block_1_1 * 128 + 128 - first_row_1_1));
                        int mini_2 = (macro_offset_2 + first_row_1_1) / mini_size;
                        int mini_rows_3 = ((mini_size) < (tokens - mini_2 * mini_size) ? (mini_size) : (tokens - mini_2 * mini_size));
                        int poll_3_first;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_first) : "l"(y_ready + mini_2) : "memory");
                        int poll_3 = poll_3_first;
                        while (poll_3 < (mini_rows_3 + 255) / 256 * (hidden / 256) * 2) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int poll_3_next;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_next) : "l"(y_ready + mini_2) : "memory");
                            poll_3 = poll_3_next;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_1_1) * 8, (unsigned int)(rows_1_1 * hidden * 2));
                        cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)(buffer_1_1 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)first_row_1_1 * (unsigned long long)hidden) * (unsigned long long)2), rows_1_1 * hidden * 2, rows_arrived_addr + (buffer_1_1) * 8);
                    }
                }
                if (units_5 > unit_4 + 2) {
                    int slot_1 = (unit_4 + 2) % 3;
                    int block_2_1 = comm_cta + (unit_4 + 2) / block_units_1 * comm_sms;
                    int first_row_2_1 = block_2_1 * 128 + (unit_4 + 2) % block_units_1 * unit_rows_0;
                    int _min_28 = ((unit_rows_0) < (block_2_1 * 128 + 128 - first_row_2_1) ? (unit_rows_0) : (block_2_1 * 128 + 128 - first_row_2_1));
                    if (_min_28 > tid) {
                        flat_view_3[slot_1 * 128 + tid] = schedule_rank[macro_offset_2 + first_row_2_1 + tid];
                        flat_view_3[384 + slot_1 * 128 + tid] = schedule_token[macro_offset_2 + first_row_2_1 + tid];
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    } else {
        int iteration = 0;
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
            if (compute < shared_fused) {
            } else if (compute < shared_tasks) {
            } else {
                int task_1 = 0;
                int macro_1 = 0;
                int mini_3 = 0;
                int swiglu = 0;
                if (compute >= shared_tasks) {
                    uint32_t _fast_div_q_0 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(compute - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(compute - shared_tasks));
                    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)(compute - shared_tasks)) - _fast_div_q_0 * (uint32_t)(mini_tasks_div.divisor);
                    int ordered_mini = (int)_fast_div_q_0;
                    task_1 = (int)_fast_div_r_0;
                    macro_1 = macros - 1;
                    mini_3 = ordered_mini;
                    if (ordered_mini >= last_minis) {
                        uint32_t _fast_div_q_1 = (minis_per_macro_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(ordered_mini - last_minis)), minis_per_macro_div.multiplier) >> minis_per_macro_div.shift_right) : (uint32_t)((unsigned int)(ordered_mini - last_minis));
                        uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)(ordered_mini - last_minis)) - _fast_div_q_1 * (uint32_t)(minis_per_macro_div.divisor);
                        macro_1 = macros - 2 - (int)_fast_div_q_1;
                        mini_3 = (int)_fast_div_r_1;
                    }
                }
                if (task_1 < mini_fused) {
                    int col_blocks_1 = (intermediate + 256 - 1) / 256;
                    int x = -1;
                    int y = -1;
                    int expert = -1;
                    int k_start = 0;
                    int k_end = 0;
                    int first = 0;
                    int first_block = (macro_1 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                    int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    int block_5 = first_block;
                    uint32_t fast_div_q = (fused_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_1), fused_col_blocks_div.multiplier) >> fused_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_1);
                    uint32_t fast_div_r = (uint32_t)((unsigned int)task_1) - fast_div_q * (uint32_t)(fused_col_blocks_div.divisor);
                    block_5 = first_block + (int)fast_div_q;
                    if (block_5 < end_block) {
                        int index = counts[3 * experts + block_5];
                        int offset = counts[experts + index] / 256;
                        int first_row_6 = ((first_block) > (offset) ? (first_block) : (offset));
                        int _min_29 = ((end_block) < (offset + counts[index] / 256) ? (end_block) : (offset + counts[index] / 256));
                        int rows_5 = _min_29 - first_row_6;
                        int supergroup = (task_1 - (first_row_6 - first_block) * col_blocks_1) / (rows_5 * 8);
                        int full_cols = col_blocks_1 / 8 * 8;
                        int row_3 = 0;
                        int col = 0;
                        if (task_1 - (first_row_6 - first_block) * col_blocks_1 < rows_5 * full_cols) {
                            row_3 = (task_1 - (first_row_6 - first_block) * col_blocks_1) % (rows_5 * 8) / 8;
                            col = supergroup * 8 + (task_1 - (first_row_6 - first_block) * col_blocks_1) % 8;
                        } else {
                            row_3 = (task_1 - (first_row_6 - first_block) * col_blocks_1 - rows_5 * full_cols) / (col_blocks_1 - full_cols);
                            col = full_cols + (task_1 - (first_row_6 - first_block) * col_blocks_1 - rows_5 * full_cols) % (col_blocks_1 - full_cols);
                        }
                        if ((supergroup & 1) != 0) {
                            row_3 = rows_5 - row_3 - 1;
                        }
                        x = first_row_6 + row_3 - macro_1 * (macro_size / 256);
                        y = col;
                        expert = index;
                    }
                    int has_hi = 0;
                    has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                    int global_mini = macro_1 * (macro_size / mini_size) + mini_3;
                    int macro_rows_1 = macro_1 * (macro_size / 256);
                    int iterations = hidden / 64;
                    if (expert < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_30 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int mini_rows_4 = ((0) > (_min_30) ? (0) : (_min_30));
                                    int required_2 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_3 = 1;
                                    if (enabled_value_3 != 0) {
                                        int poll_4_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_first) : "l"(x_ready + global_mini) : "memory");
                                        int poll_4 = poll_4_first;
                                        while (poll_4 < required_2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_4_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_next) : "l"(x_ready + global_mini) : "memory");
                                            poll_4 = poll_4_next;
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
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int head = ((3) < (iterations) ? (3) : (iterations));
                                    int tail_1 = 0;
                                    int _min_31 = ((3) < (iterations - head) ? (3) : (iterations - head));
                                    tail_1 = _min_31;
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
                                        int _smem_desc_lo_32 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                        int _smem_desc_lo_33 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_32), "r"(_smem_desc_lo_33), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                    }
                                    int _stage_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                    int _parity_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_1) * 8, _parity_output_finished_1);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_2 = 0; idx_2 < head; idx_2++) {
                                        int _smem_desc_lo_34 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
                                        int _smem_desc_lo_35 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_34), "r"(_smem_desc_lo_35), "r"((tmem_accumulator + (256))), "r"(((idx_2 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_3 = head; idx_3 < iterations - tail_1; idx_3++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) % 4);
                                        int _parity_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_1) * 8, _parity_gemm_arrived_1);
                                        int _smem_desc_lo_36 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                        int _smem_desc_lo_37 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_36), "r"(_smem_desc_lo_37), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                        int _smem_desc_lo_38 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                        int _smem_desc_lo_39 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_38), "r"(_smem_desc_lo_39), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_4 = iterations - tail_1; idx_4 < iterations; idx_4++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) % 4);
                                        int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
                                        int _smem_desc_lo_40 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
                                        int _smem_desc_lo_41 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_40), "r"(_smem_desc_lo_41), "r"(tmem_accumulator), "r"(((idx_4 == 0) ? 0 : 1)) : "memory");
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    #pragma unroll 1
                                    for (int idx_5 = iterations - tail_1; idx_5 < iterations; idx_5++) {
                                        int _smem_desc_lo_42 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
                                        int _smem_desc_lo_43 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_42), "r"(_smem_desc_lo_43), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_5) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
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
                            int warp_2 = tid / 32;
                            int lane_1 = tid % 32;
                            #pragma unroll
                            for (int half = 0; half < 2; half++) {
                                #pragma unroll
                                for (int col_tile = 0; col_tile < 2; col_tile++) {
                                    int row_4 = warp_2 * 32 + half * 16 + lane_1 % 16;
                                    int col_1 = col_tile * 16 + lane_1 / 16 * 8;
                                    unsigned int address = d_smem_addr + (unsigned int)((row_4 * 32 + col_1) * 2);
                                    address = address ^ (address & 511) >> 7 << 4;
                                    int offset_1 = half * 8 + col_tile * 4;
                                    uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(address);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_1 + 3]))
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
                                    int row_5 = warp_0 * 32 + half_1 * 16 + lane_0 % 16;
                                    int col_2 = col_tile_1 * 16 + lane_0 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + 8192 + (unsigned int)((row_5 * 32 + col_2) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset_2 = half_1 * 8 + col_tile_1 * 4;
                                    uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_2 + 3]))
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
                                    int row_6 = warp_1_1 * 32 + half_2 * 16 + lane_1_1 % 16;
                                    int col_3 = col_tile_2 * 16 + lane_1_1 / 16 * 8;
                                    unsigned int address_2 = d_smem_addr + 16384 + (unsigned int)((row_6 * 32 + col_3) * 2);
                                    address_2 = address_2 ^ (address_2 & 511) >> 7 << 4;
                                    int offset_3 = 16 + half_2 * 8 + col_tile_2 * 4;
                                    uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(address_2);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 3]))
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
                            int warp_2_1 = tid / 32;
                            int lane_2 = tid % 32;
                            #pragma unroll
                            for (int half_3 = 0; half_3 < 2; half_3++) {
                                #pragma unroll
                                for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                    int row_7 = warp_2_1 * 32 + half_3 * 16 + lane_2 % 16;
                                    int col_4 = col_tile_3 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_3 = d_smem_addr + (unsigned int)((row_7 * 32 + col_4) * 2);
                                    address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                    int offset_4 = 16 + half_3 * 8 + col_tile_3 * 4;
                                    uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(address_3);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 3]))
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
                            float _abs_48;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(x_0.x));
                            float _abs_49;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(x_0.y));
                            float2 ax = make_float2(_abs_48, _abs_49);
                            float _abs_50;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(v_0.x));
                            float _abs_51;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(v_0.y));
                            float2 av = make_float2(_abs_50, _abs_51);
                            float2 _f32x2_mul_b_52 = make_float2(k1, k1);
                            float2 te;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
                            float2 _f32x2_mul_b_53 = make_float2(k2, k2);
                            float2 tw;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw) : "l"(*(const unsigned long long*)&av), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
                            float _exp2_54 = approx_exp2(te.x);
                            float _exp2_55 = approx_exp2(te.y);
                            float2 e = make_float2(_exp2_54, _exp2_55);
                            float _exp2_56 = approx_exp2(tw.x);
                            float _exp2_57 = approx_exp2(tw.y);
                            float2 w = make_float2(_exp2_56, _exp2_57);
                            float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
                            float2 tp;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp) : "l"(*(const unsigned long long*)&ax), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
                            float _exp2_59 = approx_exp2_noftz(tp.x);
                            float _exp2_60 = approx_exp2_noftz(tp.y);
                            float2 p = make_float2(_exp2_59, _exp2_60);
                            float2 d1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                            float2 d2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p);
                            float2 dw = add_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                            float2 ng = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e);
                            float2 nu = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w);
                            float _rcp_61 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1, d2), dw).x);
                            float _rcp_62 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1, d2), dw).y);
                            float2 r_0 = make_float2(_rcp_61, _rcp_62);
                            float2 r = r_0;
                            float2 _f32x2_mul_a_63 = mul_f32x2_noftz(ng, nu);
                            float2 _f32x2_mul_b_64 = mul_f32x2_noftz(r, make_float2(bl, bl));
                            float2 m;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                            float2 ns = make_float2(((x_0.x < 0.0f) ? p.x : 1.0f), ((x_0.y < 0.0f) ? p.y : 1.0f));
                            gate_packed_0[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m, ns))) | sign_0;
                            float2 x_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[1]));
                            float2 v_1 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[1]));
                            unsigned int sign_1 = (gate_packed_0[1] ^ up_packed_0[1]) & 2147516416u;
                            float _abs_65;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_65) : "f"(x_1.x));
                            float _abs_66;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(x_1.y));
                            float2 ax_0 = make_float2(_abs_65, _abs_66);
                            float _abs_67;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(v_1.x));
                            float _abs_68;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_68) : "f"(v_1.y));
                            float2 av_0 = make_float2(_abs_67, _abs_68);
                            float2 _f32x2_mul_b_69 = make_float2(k1, k1);
                            float2 te_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_0) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                            float2 _f32x2_mul_b_70 = make_float2(k2, k2);
                            float2 tw_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_0) : "l"(*(const unsigned long long*)&av_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
                            float _exp2_71 = approx_exp2(te_0.x);
                            float _exp2_72 = approx_exp2(te_0.y);
                            float2 e_0 = make_float2(_exp2_71, _exp2_72);
                            float _exp2_73 = approx_exp2(tw_0.x);
                            float _exp2_74 = approx_exp2(tw_0.y);
                            float2 w_0 = make_float2(_exp2_73, _exp2_74);
                            float2 _f32x2_mul_b_75 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_0) : "l"(*(const unsigned long long*)&ax_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                            float _exp2_76 = approx_exp2_noftz(tp_0.x);
                            float _exp2_77 = approx_exp2_noftz(tp_0.y);
                            float2 p_0 = make_float2(_exp2_76, _exp2_77);
                            float2 d1_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                            float2 d2_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_0);
                            float2 dw_0 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                            float2 ng_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_0);
                            float2 nu_0 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_0);
                            float _rcp_78 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_0, d2_0), dw_0).x);
                            float _rcp_79 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_0, d2_0), dw_0).y);
                            float2 r_2 = make_float2(_rcp_78, _rcp_79);
                            float2 r_1 = r_2;
                            float2 _f32x2_mul_a_80 = mul_f32x2_noftz(ng_0, nu_0);
                            float2 _f32x2_mul_b_81 = mul_f32x2_noftz(r_1, make_float2(bl, bl));
                            float2 m_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
                            float2 ns_0 = make_float2(((x_1.x < 0.0f) ? p_0.x : 1.0f), ((x_1.y < 0.0f) ? p_0.y : 1.0f));
                            gate_packed_0[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_0, ns_0))) | sign_1;
                            float2 x_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[2]));
                            float2 v_2 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[2]));
                            unsigned int sign_2 = (gate_packed_0[2] ^ up_packed_0[2]) & 2147516416u;
                            float _abs_82;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(x_2.x));
                            float _abs_83;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(x_2.y));
                            float2 ax_1 = make_float2(_abs_82, _abs_83);
                            float _abs_84;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(v_2.x));
                            float _abs_85;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(v_2.y));
                            float2 av_1 = make_float2(_abs_84, _abs_85);
                            float2 _f32x2_mul_b_86 = make_float2(k1, k1);
                            float2 te_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_1) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                            float2 _f32x2_mul_b_87 = make_float2(k2, k2);
                            float2 tw_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_1) : "l"(*(const unsigned long long*)&av_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
                            float _exp2_88 = approx_exp2(te_1.x);
                            float _exp2_89 = approx_exp2(te_1.y);
                            float2 e_1 = make_float2(_exp2_88, _exp2_89);
                            float _exp2_90 = approx_exp2(tw_1.x);
                            float _exp2_91 = approx_exp2(tw_1.y);
                            float2 w_1 = make_float2(_exp2_90, _exp2_91);
                            float2 _f32x2_mul_b_92 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_1) : "l"(*(const unsigned long long*)&ax_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
                            float _exp2_93 = approx_exp2_noftz(tp_1.x);
                            float _exp2_94 = approx_exp2_noftz(tp_1.y);
                            float2 p_1 = make_float2(_exp2_93, _exp2_94);
                            float2 d1_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                            float2 d2_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_1);
                            float2 dw_1 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                            float2 ng_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_1);
                            float2 nu_1 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_1);
                            float _rcp_95 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_1, d2_1), dw_1).x);
                            float _rcp_96 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_1, d2_1), dw_1).y);
                            float2 r_4 = make_float2(_rcp_95, _rcp_96);
                            float2 r_3 = r_4;
                            float2 _f32x2_mul_a_97 = mul_f32x2_noftz(ng_1, nu_1);
                            float2 _f32x2_mul_b_98 = mul_f32x2_noftz(r_3, make_float2(bl, bl));
                            float2 m_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
                            float2 ns_1 = make_float2(((x_2.x < 0.0f) ? p_1.x : 1.0f), ((x_2.y < 0.0f) ? p_1.y : 1.0f));
                            gate_packed_0[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_1, ns_1))) | sign_2;
                            float2 x_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[3]));
                            float2 v_3 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[3]));
                            unsigned int sign_3 = (gate_packed_0[3] ^ up_packed_0[3]) & 2147516416u;
                            float _abs_99;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_99) : "f"(x_3.x));
                            float _abs_100;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_100) : "f"(x_3.y));
                            float2 ax_2 = make_float2(_abs_99, _abs_100);
                            float _abs_101;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(v_3.x));
                            float _abs_102;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_102) : "f"(v_3.y));
                            float2 av_2 = make_float2(_abs_101, _abs_102);
                            float2 _f32x2_mul_b_103 = make_float2(k1, k1);
                            float2 te_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_2) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
                            float2 _f32x2_mul_b_104 = make_float2(k2, k2);
                            float2 tw_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_2) : "l"(*(const unsigned long long*)&av_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_104));
                            float _exp2_105 = approx_exp2(te_2.x);
                            float _exp2_106 = approx_exp2(te_2.y);
                            float2 e_2 = make_float2(_exp2_105, _exp2_106);
                            float _exp2_107 = approx_exp2(tw_2.x);
                            float _exp2_108 = approx_exp2(tw_2.y);
                            float2 w_2 = make_float2(_exp2_107, _exp2_108);
                            float2 _f32x2_mul_b_109 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_2) : "l"(*(const unsigned long long*)&ax_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_109));
                            float _exp2_110 = approx_exp2_noftz(tp_2.x);
                            float _exp2_111 = approx_exp2_noftz(tp_2.y);
                            float2 p_2 = make_float2(_exp2_110, _exp2_111);
                            float2 d1_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                            float2 d2_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_2);
                            float2 dw_2 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                            float2 ng_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_2);
                            float2 nu_2 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_2);
                            float _rcp_112 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_2, d2_2), dw_2).x);
                            float _rcp_113 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_2, d2_2), dw_2).y);
                            float2 r_6 = make_float2(_rcp_112, _rcp_113);
                            float2 r_5 = r_6;
                            float2 _f32x2_mul_a_114 = mul_f32x2_noftz(ng_2, nu_2);
                            float2 _f32x2_mul_b_115 = mul_f32x2_noftz(r_5, make_float2(bl, bl));
                            float2 m_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_115));
                            float2 ns_2 = make_float2(((x_3.x < 0.0f) ? p_2.x : 1.0f), ((x_3.y < 0.0f) ? p_2.y : 1.0f));
                            gate_packed_0[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_2, ns_2))) | sign_3;
                            float2 x_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[4]));
                            float2 v_4 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[4]));
                            unsigned int sign_4 = (gate_packed_0[4] ^ up_packed_0[4]) & 2147516416u;
                            float _abs_116;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_116) : "f"(x_4.x));
                            float _abs_117;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_117) : "f"(x_4.y));
                            float2 ax_3 = make_float2(_abs_116, _abs_117);
                            float _abs_118;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_118) : "f"(v_4.x));
                            float _abs_119;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(v_4.y));
                            float2 av_3 = make_float2(_abs_118, _abs_119);
                            float2 _f32x2_mul_b_120 = make_float2(k1, k1);
                            float2 te_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_3) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_120));
                            float2 _f32x2_mul_b_121 = make_float2(k2, k2);
                            float2 tw_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_3) : "l"(*(const unsigned long long*)&av_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
                            float _exp2_122 = approx_exp2(te_3.x);
                            float _exp2_123 = approx_exp2(te_3.y);
                            float2 e_3 = make_float2(_exp2_122, _exp2_123);
                            float _exp2_124 = approx_exp2(tw_3.x);
                            float _exp2_125 = approx_exp2(tw_3.y);
                            float2 w_3 = make_float2(_exp2_124, _exp2_125);
                            float2 _f32x2_mul_b_126 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_3) : "l"(*(const unsigned long long*)&ax_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_126));
                            float _exp2_127 = approx_exp2_noftz(tp_3.x);
                            float _exp2_128 = approx_exp2_noftz(tp_3.y);
                            float2 p_3 = make_float2(_exp2_127, _exp2_128);
                            float2 d1_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                            float2 d2_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_3);
                            float2 dw_3 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                            float2 ng_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_3);
                            float2 nu_3 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_3);
                            float _rcp_129 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_3, d2_3), dw_3).x);
                            float _rcp_130 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_3, d2_3), dw_3).y);
                            float2 r_8 = make_float2(_rcp_129, _rcp_130);
                            float2 r_7 = r_8;
                            float2 _f32x2_mul_a_131 = mul_f32x2_noftz(ng_3, nu_3);
                            float2 _f32x2_mul_b_132 = mul_f32x2_noftz(r_7, make_float2(bl, bl));
                            float2 m_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_131), "l"(*(const unsigned long long*)&_f32x2_mul_b_132));
                            float2 ns_3 = make_float2(((x_4.x < 0.0f) ? p_3.x : 1.0f), ((x_4.y < 0.0f) ? p_3.y : 1.0f));
                            gate_packed_0[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_3, ns_3))) | sign_4;
                            float2 x_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[5]));
                            float2 v_5 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[5]));
                            unsigned int sign_5 = (gate_packed_0[5] ^ up_packed_0[5]) & 2147516416u;
                            float _abs_133;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_133) : "f"(x_5.x));
                            float _abs_134;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_134) : "f"(x_5.y));
                            float2 ax_4 = make_float2(_abs_133, _abs_134);
                            float _abs_135;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_135) : "f"(v_5.x));
                            float _abs_136;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_136) : "f"(v_5.y));
                            float2 av_4 = make_float2(_abs_135, _abs_136);
                            float2 _f32x2_mul_b_137 = make_float2(k1, k1);
                            float2 te_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_4) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
                            float2 _f32x2_mul_b_138 = make_float2(k2, k2);
                            float2 tw_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_4) : "l"(*(const unsigned long long*)&av_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
                            float _exp2_139 = approx_exp2(te_4.x);
                            float _exp2_140 = approx_exp2(te_4.y);
                            float2 e_4 = make_float2(_exp2_139, _exp2_140);
                            float _exp2_141 = approx_exp2(tw_4.x);
                            float _exp2_142 = approx_exp2(tw_4.y);
                            float2 w_4 = make_float2(_exp2_141, _exp2_142);
                            float2 _f32x2_mul_b_143 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_4) : "l"(*(const unsigned long long*)&ax_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
                            float _exp2_144 = approx_exp2_noftz(tp_4.x);
                            float _exp2_145 = approx_exp2_noftz(tp_4.y);
                            float2 p_4 = make_float2(_exp2_144, _exp2_145);
                            float2 d1_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                            float2 d2_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_4);
                            float2 dw_4 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                            float2 ng_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_4);
                            float2 nu_4 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_4);
                            float _rcp_146 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_4, d2_4), dw_4).x);
                            float _rcp_147 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_4, d2_4), dw_4).y);
                            float2 r_10 = make_float2(_rcp_146, _rcp_147);
                            float2 r_9 = r_10;
                            float2 _f32x2_mul_a_148 = mul_f32x2_noftz(ng_4, nu_4);
                            float2 _f32x2_mul_b_149 = mul_f32x2_noftz(r_9, make_float2(bl, bl));
                            float2 m_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
                            float2 ns_4 = make_float2(((x_5.x < 0.0f) ? p_4.x : 1.0f), ((x_5.y < 0.0f) ? p_4.y : 1.0f));
                            gate_packed_0[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_4, ns_4))) | sign_5;
                            float2 x_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[6]));
                            float2 v_6 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[6]));
                            unsigned int sign_6 = (gate_packed_0[6] ^ up_packed_0[6]) & 2147516416u;
                            float _abs_150;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(x_6.x));
                            float _abs_151;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_151) : "f"(x_6.y));
                            float2 ax_5 = make_float2(_abs_150, _abs_151);
                            float _abs_152;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(v_6.x));
                            float _abs_153;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(v_6.y));
                            float2 av_5 = make_float2(_abs_152, _abs_153);
                            float2 _f32x2_mul_b_154 = make_float2(k1, k1);
                            float2 te_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_5) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
                            float2 _f32x2_mul_b_155 = make_float2(k2, k2);
                            float2 tw_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_5) : "l"(*(const unsigned long long*)&av_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
                            float _exp2_156 = approx_exp2(te_5.x);
                            float _exp2_157 = approx_exp2(te_5.y);
                            float2 e_5 = make_float2(_exp2_156, _exp2_157);
                            float _exp2_158 = approx_exp2(tw_5.x);
                            float _exp2_159 = approx_exp2(tw_5.y);
                            float2 w_5 = make_float2(_exp2_158, _exp2_159);
                            float2 _f32x2_mul_b_160 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_5) : "l"(*(const unsigned long long*)&ax_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_160));
                            float _exp2_161 = approx_exp2_noftz(tp_5.x);
                            float _exp2_162 = approx_exp2_noftz(tp_5.y);
                            float2 p_5 = make_float2(_exp2_161, _exp2_162);
                            float2 d1_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                            float2 d2_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_5);
                            float2 dw_5 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                            float2 ng_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_5);
                            float2 nu_5 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_5);
                            float _rcp_163 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_5, d2_5), dw_5).x);
                            float _rcp_164 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_5, d2_5), dw_5).y);
                            float2 r_12 = make_float2(_rcp_163, _rcp_164);
                            float2 r_11 = r_12;
                            float2 _f32x2_mul_a_165 = mul_f32x2_noftz(ng_5, nu_5);
                            float2 _f32x2_mul_b_166 = mul_f32x2_noftz(r_11, make_float2(bl, bl));
                            float2 m_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&_f32x2_mul_b_166));
                            float2 ns_5 = make_float2(((x_6.x < 0.0f) ? p_5.x : 1.0f), ((x_6.y < 0.0f) ? p_5.y : 1.0f));
                            gate_packed_0[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_5, ns_5))) | sign_6;
                            float2 x_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[7]));
                            float2 v_7 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[7]));
                            unsigned int sign_7 = (gate_packed_0[7] ^ up_packed_0[7]) & 2147516416u;
                            float _abs_167;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_167) : "f"(x_7.x));
                            float _abs_168;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_168) : "f"(x_7.y));
                            float2 ax_6 = make_float2(_abs_167, _abs_168);
                            float _abs_169;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_169) : "f"(v_7.x));
                            float _abs_170;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_170) : "f"(v_7.y));
                            float2 av_6 = make_float2(_abs_169, _abs_170);
                            float2 _f32x2_mul_b_171 = make_float2(k1, k1);
                            float2 te_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_6) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
                            float2 _f32x2_mul_b_172 = make_float2(k2, k2);
                            float2 tw_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_6) : "l"(*(const unsigned long long*)&av_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_172));
                            float _exp2_173 = approx_exp2(te_6.x);
                            float _exp2_174 = approx_exp2(te_6.y);
                            float2 e_6 = make_float2(_exp2_173, _exp2_174);
                            float _exp2_175 = approx_exp2(tw_6.x);
                            float _exp2_176 = approx_exp2(tw_6.y);
                            float2 w_6 = make_float2(_exp2_175, _exp2_176);
                            float2 _f32x2_mul_b_177 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_6) : "l"(*(const unsigned long long*)&ax_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
                            float _exp2_178 = approx_exp2_noftz(tp_6.x);
                            float _exp2_179 = approx_exp2_noftz(tp_6.y);
                            float2 p_6 = make_float2(_exp2_178, _exp2_179);
                            float2 d1_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_6);
                            float2 d2_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_6);
                            float2 dw_6 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_6);
                            float2 ng_6 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_6);
                            float2 nu_6 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_6);
                            float _rcp_180 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_6, d2_6), dw_6).x);
                            float _rcp_181 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_6, d2_6), dw_6).y);
                            float2 r_14 = make_float2(_rcp_180, _rcp_181);
                            float2 r_13 = r_14;
                            float2 _f32x2_mul_a_182 = mul_f32x2_noftz(ng_6, nu_6);
                            float2 _f32x2_mul_b_183 = mul_f32x2_noftz(r_13, make_float2(bl, bl));
                            float2 m_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&_f32x2_mul_b_183));
                            float2 ns_6 = make_float2(((x_7.x < 0.0f) ? p_6.x : 1.0f), ((x_7.y < 0.0f) ? p_6.y : 1.0f));
                            gate_packed_0[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_6, ns_6))) | sign_7;
                            float2 x_8 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[8]));
                            float2 v_8 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[8]));
                            unsigned int sign_8 = (gate_packed_0[8] ^ up_packed_0[8]) & 2147516416u;
                            float _abs_184;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_184) : "f"(x_8.x));
                            float _abs_185;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_185) : "f"(x_8.y));
                            float2 ax_7 = make_float2(_abs_184, _abs_185);
                            float _abs_186;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_186) : "f"(v_8.x));
                            float _abs_187;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_187) : "f"(v_8.y));
                            float2 av_7 = make_float2(_abs_186, _abs_187);
                            float2 _f32x2_mul_b_188 = make_float2(k1, k1);
                            float2 te_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_7) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
                            float2 _f32x2_mul_b_189 = make_float2(k2, k2);
                            float2 tw_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_7) : "l"(*(const unsigned long long*)&av_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
                            float _exp2_190 = approx_exp2(te_7.x);
                            float _exp2_191 = approx_exp2(te_7.y);
                            float2 e_7 = make_float2(_exp2_190, _exp2_191);
                            float _exp2_192 = approx_exp2(tw_7.x);
                            float _exp2_193 = approx_exp2(tw_7.y);
                            float2 w_7 = make_float2(_exp2_192, _exp2_193);
                            float2 _f32x2_mul_b_194 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_7) : "l"(*(const unsigned long long*)&ax_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_194));
                            float _exp2_195 = approx_exp2_noftz(tp_7.x);
                            float _exp2_196 = approx_exp2_noftz(tp_7.y);
                            float2 p_7 = make_float2(_exp2_195, _exp2_196);
                            float2 d1_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_7);
                            float2 d2_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_7);
                            float2 dw_7 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_7);
                            float2 ng_7 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_7);
                            float2 nu_7 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_7);
                            float _rcp_197 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_7, d2_7), dw_7).x);
                            float _rcp_198 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_7, d2_7), dw_7).y);
                            float2 r_16 = make_float2(_rcp_197, _rcp_198);
                            float2 r_15 = r_16;
                            float2 _f32x2_mul_a_199 = mul_f32x2_noftz(ng_7, nu_7);
                            float2 _f32x2_mul_b_200 = mul_f32x2_noftz(r_15, make_float2(bl, bl));
                            float2 m_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_199), "l"(*(const unsigned long long*)&_f32x2_mul_b_200));
                            float2 ns_7 = make_float2(((x_8.x < 0.0f) ? p_7.x : 1.0f), ((x_8.y < 0.0f) ? p_7.y : 1.0f));
                            gate_packed_0[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_7, ns_7))) | sign_8;
                            float2 x_9 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[9]));
                            float2 v_9 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[9]));
                            unsigned int sign_9 = (gate_packed_0[9] ^ up_packed_0[9]) & 2147516416u;
                            float _abs_201;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_201) : "f"(x_9.x));
                            float _abs_202;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_202) : "f"(x_9.y));
                            float2 ax_8 = make_float2(_abs_201, _abs_202);
                            float _abs_203;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(v_9.x));
                            float _abs_204;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_204) : "f"(v_9.y));
                            float2 av_8 = make_float2(_abs_203, _abs_204);
                            float2 _f32x2_mul_b_205 = make_float2(k1, k1);
                            float2 te_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_8) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
                            float2 _f32x2_mul_b_206 = make_float2(k2, k2);
                            float2 tw_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_8) : "l"(*(const unsigned long long*)&av_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_206));
                            float _exp2_207 = approx_exp2(te_8.x);
                            float _exp2_208 = approx_exp2(te_8.y);
                            float2 e_8 = make_float2(_exp2_207, _exp2_208);
                            float _exp2_209 = approx_exp2(tw_8.x);
                            float _exp2_210 = approx_exp2(tw_8.y);
                            float2 w_8 = make_float2(_exp2_209, _exp2_210);
                            float2 _f32x2_mul_b_211 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_8) : "l"(*(const unsigned long long*)&ax_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_211));
                            float _exp2_212 = approx_exp2_noftz(tp_8.x);
                            float _exp2_213 = approx_exp2_noftz(tp_8.y);
                            float2 p_8 = make_float2(_exp2_212, _exp2_213);
                            float2 d1_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_8);
                            float2 d2_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_8);
                            float2 dw_8 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_8);
                            float2 ng_8 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_8);
                            float2 nu_8 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_8);
                            float _rcp_214 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_8, d2_8), dw_8).x);
                            float _rcp_215 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_8, d2_8), dw_8).y);
                            float2 r_18 = make_float2(_rcp_214, _rcp_215);
                            float2 r_17 = r_18;
                            float2 _f32x2_mul_a_216 = mul_f32x2_noftz(ng_8, nu_8);
                            float2 _f32x2_mul_b_217 = mul_f32x2_noftz(r_17, make_float2(bl, bl));
                            float2 m_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
                            float2 ns_8 = make_float2(((x_9.x < 0.0f) ? p_8.x : 1.0f), ((x_9.y < 0.0f) ? p_8.y : 1.0f));
                            gate_packed_0[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_8, ns_8))) | sign_9;
                            float2 x_10 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[10]));
                            float2 v_10 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[10]));
                            unsigned int sign_10 = (gate_packed_0[10] ^ up_packed_0[10]) & 2147516416u;
                            float _abs_218;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_218) : "f"(x_10.x));
                            float _abs_219;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_219) : "f"(x_10.y));
                            float2 ax_9 = make_float2(_abs_218, _abs_219);
                            float _abs_220;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_220) : "f"(v_10.x));
                            float _abs_221;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_221) : "f"(v_10.y));
                            float2 av_9 = make_float2(_abs_220, _abs_221);
                            float2 _f32x2_mul_b_222 = make_float2(k1, k1);
                            float2 te_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_9) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_222));
                            float2 _f32x2_mul_b_223 = make_float2(k2, k2);
                            float2 tw_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_9) : "l"(*(const unsigned long long*)&av_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_223));
                            float _exp2_224 = approx_exp2(te_9.x);
                            float _exp2_225 = approx_exp2(te_9.y);
                            float2 e_9 = make_float2(_exp2_224, _exp2_225);
                            float _exp2_226 = approx_exp2(tw_9.x);
                            float _exp2_227 = approx_exp2(tw_9.y);
                            float2 w_9 = make_float2(_exp2_226, _exp2_227);
                            float2 _f32x2_mul_b_228 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_9) : "l"(*(const unsigned long long*)&ax_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_228));
                            float _exp2_229 = approx_exp2_noftz(tp_9.x);
                            float _exp2_230 = approx_exp2_noftz(tp_9.y);
                            float2 p_9 = make_float2(_exp2_229, _exp2_230);
                            float2 d1_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_9);
                            float2 d2_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_9);
                            float2 dw_9 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_9);
                            float2 ng_9 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_9);
                            float2 nu_9 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_9);
                            float _rcp_231 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_9, d2_9), dw_9).x);
                            float _rcp_232 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_9, d2_9), dw_9).y);
                            float2 r_20 = make_float2(_rcp_231, _rcp_232);
                            float2 r_19 = r_20;
                            float2 _f32x2_mul_a_233 = mul_f32x2_noftz(ng_9, nu_9);
                            float2 _f32x2_mul_b_234 = mul_f32x2_noftz(r_19, make_float2(bl, bl));
                            float2 m_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_233), "l"(*(const unsigned long long*)&_f32x2_mul_b_234));
                            float2 ns_9 = make_float2(((x_10.x < 0.0f) ? p_9.x : 1.0f), ((x_10.y < 0.0f) ? p_9.y : 1.0f));
                            gate_packed_0[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_9, ns_9))) | sign_10;
                            float2 x_11 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[11]));
                            float2 v_11 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[11]));
                            unsigned int sign_11 = (gate_packed_0[11] ^ up_packed_0[11]) & 2147516416u;
                            float _abs_235;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_235) : "f"(x_11.x));
                            float _abs_236;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_236) : "f"(x_11.y));
                            float2 ax_10 = make_float2(_abs_235, _abs_236);
                            float _abs_237;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_237) : "f"(v_11.x));
                            float _abs_238;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_238) : "f"(v_11.y));
                            float2 av_10 = make_float2(_abs_237, _abs_238);
                            float2 _f32x2_mul_b_239 = make_float2(k1, k1);
                            float2 te_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_10) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_239));
                            float2 _f32x2_mul_b_240 = make_float2(k2, k2);
                            float2 tw_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_10) : "l"(*(const unsigned long long*)&av_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_240));
                            float _exp2_241 = approx_exp2(te_10.x);
                            float _exp2_242 = approx_exp2(te_10.y);
                            float2 e_10 = make_float2(_exp2_241, _exp2_242);
                            float _exp2_243 = approx_exp2(tw_10.x);
                            float _exp2_244 = approx_exp2(tw_10.y);
                            float2 w_10 = make_float2(_exp2_243, _exp2_244);
                            float2 _f32x2_mul_b_245 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_10) : "l"(*(const unsigned long long*)&ax_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_245));
                            float _exp2_246 = approx_exp2_noftz(tp_10.x);
                            float _exp2_247 = approx_exp2_noftz(tp_10.y);
                            float2 p_10 = make_float2(_exp2_246, _exp2_247);
                            float2 d1_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_10);
                            float2 d2_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_10);
                            float2 dw_10 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_10);
                            float2 ng_10 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_10);
                            float2 nu_10 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_10);
                            float _rcp_248 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_10, d2_10), dw_10).x);
                            float _rcp_249 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_10, d2_10), dw_10).y);
                            float2 r_22 = make_float2(_rcp_248, _rcp_249);
                            float2 r_21 = r_22;
                            float2 _f32x2_mul_a_250 = mul_f32x2_noftz(ng_10, nu_10);
                            float2 _f32x2_mul_b_251 = mul_f32x2_noftz(r_21, make_float2(bl, bl));
                            float2 m_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_250), "l"(*(const unsigned long long*)&_f32x2_mul_b_251));
                            float2 ns_10 = make_float2(((x_11.x < 0.0f) ? p_10.x : 1.0f), ((x_11.y < 0.0f) ? p_10.y : 1.0f));
                            gate_packed_0[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_10, ns_10))) | sign_11;
                            float2 x_12 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[12]));
                            float2 v_12 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[12]));
                            unsigned int sign_12 = (gate_packed_0[12] ^ up_packed_0[12]) & 2147516416u;
                            float _abs_252;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_252) : "f"(x_12.x));
                            float _abs_253;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_253) : "f"(x_12.y));
                            float2 ax_11 = make_float2(_abs_252, _abs_253);
                            float _abs_254;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_254) : "f"(v_12.x));
                            float _abs_255;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_255) : "f"(v_12.y));
                            float2 av_11 = make_float2(_abs_254, _abs_255);
                            float2 _f32x2_mul_b_256 = make_float2(k1, k1);
                            float2 te_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_11) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_256));
                            float2 _f32x2_mul_b_257 = make_float2(k2, k2);
                            float2 tw_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_11) : "l"(*(const unsigned long long*)&av_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_257));
                            float _exp2_258 = approx_exp2(te_11.x);
                            float _exp2_259 = approx_exp2(te_11.y);
                            float2 e_11 = make_float2(_exp2_258, _exp2_259);
                            float _exp2_260 = approx_exp2(tw_11.x);
                            float _exp2_261 = approx_exp2(tw_11.y);
                            float2 w_11 = make_float2(_exp2_260, _exp2_261);
                            float2 _f32x2_mul_b_262 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_11) : "l"(*(const unsigned long long*)&ax_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_262));
                            float _exp2_263 = approx_exp2_noftz(tp_11.x);
                            float _exp2_264 = approx_exp2_noftz(tp_11.y);
                            float2 p_11 = make_float2(_exp2_263, _exp2_264);
                            float2 d1_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_11);
                            float2 d2_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_11);
                            float2 dw_11 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_11);
                            float2 ng_11 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_11);
                            float2 nu_11 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_11);
                            float _rcp_265 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_11, d2_11), dw_11).x);
                            float _rcp_266 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_11, d2_11), dw_11).y);
                            float2 r_24 = make_float2(_rcp_265, _rcp_266);
                            float2 r_23 = r_24;
                            float2 _f32x2_mul_a_267 = mul_f32x2_noftz(ng_11, nu_11);
                            float2 _f32x2_mul_b_268 = mul_f32x2_noftz(r_23, make_float2(bl, bl));
                            float2 m_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&_f32x2_mul_b_268));
                            float2 ns_11 = make_float2(((x_12.x < 0.0f) ? p_11.x : 1.0f), ((x_12.y < 0.0f) ? p_11.y : 1.0f));
                            gate_packed_0[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_11, ns_11))) | sign_12;
                            float2 x_13 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[13]));
                            float2 v_13 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[13]));
                            unsigned int sign_13 = (gate_packed_0[13] ^ up_packed_0[13]) & 2147516416u;
                            float _abs_269;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_269) : "f"(x_13.x));
                            float _abs_270;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_270) : "f"(x_13.y));
                            float2 ax_12 = make_float2(_abs_269, _abs_270);
                            float _abs_271;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_271) : "f"(v_13.x));
                            float _abs_272;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_272) : "f"(v_13.y));
                            float2 av_12 = make_float2(_abs_271, _abs_272);
                            float2 _f32x2_mul_b_273 = make_float2(k1, k1);
                            float2 te_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_12) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_273));
                            float2 _f32x2_mul_b_274 = make_float2(k2, k2);
                            float2 tw_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_12) : "l"(*(const unsigned long long*)&av_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_274));
                            float _exp2_275 = approx_exp2(te_12.x);
                            float _exp2_276 = approx_exp2(te_12.y);
                            float2 e_12 = make_float2(_exp2_275, _exp2_276);
                            float _exp2_277 = approx_exp2(tw_12.x);
                            float _exp2_278 = approx_exp2(tw_12.y);
                            float2 w_12 = make_float2(_exp2_277, _exp2_278);
                            float2 _f32x2_mul_b_279 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_12) : "l"(*(const unsigned long long*)&ax_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_279));
                            float _exp2_280 = approx_exp2_noftz(tp_12.x);
                            float _exp2_281 = approx_exp2_noftz(tp_12.y);
                            float2 p_12 = make_float2(_exp2_280, _exp2_281);
                            float2 d1_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_12);
                            float2 d2_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_12);
                            float2 dw_12 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_12);
                            float2 ng_12 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_12);
                            float2 nu_12 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_12);
                            float _rcp_282 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_12, d2_12), dw_12).x);
                            float _rcp_283 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_12, d2_12), dw_12).y);
                            float2 r_26 = make_float2(_rcp_282, _rcp_283);
                            float2 r_25 = r_26;
                            float2 _f32x2_mul_a_284 = mul_f32x2_noftz(ng_12, nu_12);
                            float2 _f32x2_mul_b_285 = mul_f32x2_noftz(r_25, make_float2(bl, bl));
                            float2 m_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_284), "l"(*(const unsigned long long*)&_f32x2_mul_b_285));
                            float2 ns_12 = make_float2(((x_13.x < 0.0f) ? p_12.x : 1.0f), ((x_13.y < 0.0f) ? p_12.y : 1.0f));
                            gate_packed_0[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_12, ns_12))) | sign_13;
                            float2 x_14 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[14]));
                            float2 v_14 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[14]));
                            unsigned int sign_14 = (gate_packed_0[14] ^ up_packed_0[14]) & 2147516416u;
                            float _abs_286;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_286) : "f"(x_14.x));
                            float _abs_287;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_287) : "f"(x_14.y));
                            float2 ax_13 = make_float2(_abs_286, _abs_287);
                            float _abs_288;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_288) : "f"(v_14.x));
                            float _abs_289;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_289) : "f"(v_14.y));
                            float2 av_13 = make_float2(_abs_288, _abs_289);
                            float2 _f32x2_mul_b_290 = make_float2(k1, k1);
                            float2 te_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_13) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_290));
                            float2 _f32x2_mul_b_291 = make_float2(k2, k2);
                            float2 tw_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_13) : "l"(*(const unsigned long long*)&av_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_291));
                            float _exp2_292 = approx_exp2(te_13.x);
                            float _exp2_293 = approx_exp2(te_13.y);
                            float2 e_13 = make_float2(_exp2_292, _exp2_293);
                            float _exp2_294 = approx_exp2(tw_13.x);
                            float _exp2_295 = approx_exp2(tw_13.y);
                            float2 w_13 = make_float2(_exp2_294, _exp2_295);
                            float2 _f32x2_mul_b_296 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_13) : "l"(*(const unsigned long long*)&ax_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_296));
                            float _exp2_297 = approx_exp2_noftz(tp_13.x);
                            float _exp2_298 = approx_exp2_noftz(tp_13.y);
                            float2 p_13 = make_float2(_exp2_297, _exp2_298);
                            float2 d1_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_13);
                            float2 d2_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_13);
                            float2 dw_13 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_13);
                            float2 ng_13 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_13);
                            float2 nu_13 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_13);
                            float _rcp_299 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_13, d2_13), dw_13).x);
                            float _rcp_300 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_13, d2_13), dw_13).y);
                            float2 r_28 = make_float2(_rcp_299, _rcp_300);
                            float2 r_27 = r_28;
                            float2 _f32x2_mul_a_301 = mul_f32x2_noftz(ng_13, nu_13);
                            float2 _f32x2_mul_b_302 = mul_f32x2_noftz(r_27, make_float2(bl, bl));
                            float2 m_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_301), "l"(*(const unsigned long long*)&_f32x2_mul_b_302));
                            float2 ns_13 = make_float2(((x_14.x < 0.0f) ? p_13.x : 1.0f), ((x_14.y < 0.0f) ? p_13.y : 1.0f));
                            gate_packed_0[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_13, ns_13))) | sign_14;
                            float2 x_15 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[15]));
                            float2 v_15 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[15]));
                            unsigned int sign_15 = (gate_packed_0[15] ^ up_packed_0[15]) & 2147516416u;
                            float _abs_303;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_303) : "f"(x_15.x));
                            float _abs_304;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_304) : "f"(x_15.y));
                            float2 ax_14 = make_float2(_abs_303, _abs_304);
                            float _abs_305;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_305) : "f"(v_15.x));
                            float _abs_306;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_306) : "f"(v_15.y));
                            float2 av_14 = make_float2(_abs_305, _abs_306);
                            float2 _f32x2_mul_b_307 = make_float2(k1, k1);
                            float2 te_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_14) : "l"(*(const unsigned long long*)&ax_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_307));
                            float2 _f32x2_mul_b_308 = make_float2(k2, k2);
                            float2 tw_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_14) : "l"(*(const unsigned long long*)&av_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_308));
                            float _exp2_309 = approx_exp2(te_14.x);
                            float _exp2_310 = approx_exp2(te_14.y);
                            float2 e_14 = make_float2(_exp2_309, _exp2_310);
                            float _exp2_311 = approx_exp2(tw_14.x);
                            float _exp2_312 = approx_exp2(tw_14.y);
                            float2 w_14 = make_float2(_exp2_311, _exp2_312);
                            float2 _f32x2_mul_b_313 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_14) : "l"(*(const unsigned long long*)&ax_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_313));
                            float _exp2_314 = approx_exp2_noftz(tp_14.x);
                            float _exp2_315 = approx_exp2_noftz(tp_14.y);
                            float2 p_14 = make_float2(_exp2_314, _exp2_315);
                            float2 d1_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_14);
                            float2 d2_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_14);
                            float2 dw_14 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_14);
                            float2 ng_14 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_14);
                            float2 nu_14 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_14);
                            float _rcp_316 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_14, d2_14), dw_14).x);
                            float _rcp_317 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_14, d2_14), dw_14).y);
                            float2 r_30 = make_float2(_rcp_316, _rcp_317);
                            float2 r_29 = r_30;
                            float2 _f32x2_mul_a_318 = mul_f32x2_noftz(ng_14, nu_14);
                            float2 _f32x2_mul_b_319 = mul_f32x2_noftz(r_29, make_float2(bl, bl));
                            float2 m_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_318), "l"(*(const unsigned long long*)&_f32x2_mul_b_319));
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
                                    int row_8 = warp_3 * 32 + half_4 * 16 + lane_3 % 16;
                                    int col_5 = col_tile_4 * 16 + lane_3 / 16 * 8;
                                    unsigned int address_4 = d_smem_addr + 8192 + (unsigned int)((row_8 * 32 + col_5) * 2);
                                    address_4 = address_4 ^ (address_4 & 511) >> 7 << 4;
                                    int offset_5 = half_4 * 8 + col_tile_4 * 4;
                                    uint32_t _stmatrix_addr_320 = static_cast<uint32_t>(address_4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_320), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_5 + 3]))
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
                            float _abs_321;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_321) : "f"(x_16.x));
                            float _abs_322;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_322) : "f"(x_16.y));
                            float2 ax_15 = make_float2(_abs_321, _abs_322);
                            float _abs_323;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_323) : "f"(v_16.x));
                            float _abs_324;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_324) : "f"(v_16.y));
                            float2 av_15 = make_float2(_abs_323, _abs_324);
                            float2 _f32x2_mul_b_325 = make_float2(k1, k1);
                            float2 te_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_15) : "l"(*(const unsigned long long*)&ax_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_325));
                            float2 _f32x2_mul_b_326 = make_float2(k2, k2);
                            float2 tw_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_15) : "l"(*(const unsigned long long*)&av_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_326));
                            float _exp2_327 = approx_exp2(te_15.x);
                            float _exp2_328 = approx_exp2(te_15.y);
                            float2 e_15 = make_float2(_exp2_327, _exp2_328);
                            float _exp2_329 = approx_exp2(tw_15.x);
                            float _exp2_330 = approx_exp2(tw_15.y);
                            float2 w_15 = make_float2(_exp2_329, _exp2_330);
                            float2 _f32x2_mul_b_331 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_15) : "l"(*(const unsigned long long*)&ax_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_331));
                            float _exp2_332 = approx_exp2_noftz(tp_15.x);
                            float _exp2_333 = approx_exp2_noftz(tp_15.y);
                            float2 p_15 = make_float2(_exp2_332, _exp2_333);
                            float2 d1_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_15);
                            float2 d2_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_15);
                            float2 dw_15 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_15);
                            float2 ng_15 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_15);
                            float2 nu_15 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_15);
                            float _rcp_334 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_15, d2_15), dw_15).x);
                            float _rcp_335 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_15, d2_15), dw_15).y);
                            float2 r_32 = make_float2(_rcp_334, _rcp_335);
                            float2 r_31 = r_32;
                            float2 _f32x2_mul_a_336 = mul_f32x2_noftz(ng_15, nu_15);
                            float2 _f32x2_mul_b_337 = mul_f32x2_noftz(r_31, make_float2(bl, bl));
                            float2 m_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_336), "l"(*(const unsigned long long*)&_f32x2_mul_b_337));
                            float2 ns_15 = make_float2(((x_16.x < 0.0f) ? p_15.x : 1.0f), ((x_16.y < 0.0f) ? p_15.y : 1.0f));
                            gate_packed_0[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_15, ns_15))) | sign_16;
                            float2 x_17 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[17]));
                            float2 v_17 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[17]));
                            unsigned int sign_17 = (gate_packed_0[17] ^ up_packed_0[17]) & 2147516416u;
                            float _abs_338;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_338) : "f"(x_17.x));
                            float _abs_339;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_339) : "f"(x_17.y));
                            float2 ax_16 = make_float2(_abs_338, _abs_339);
                            float _abs_340;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_340) : "f"(v_17.x));
                            float _abs_341;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_341) : "f"(v_17.y));
                            float2 av_16 = make_float2(_abs_340, _abs_341);
                            float2 _f32x2_mul_b_342 = make_float2(k1, k1);
                            float2 te_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_16) : "l"(*(const unsigned long long*)&ax_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_342));
                            float2 _f32x2_mul_b_343 = make_float2(k2, k2);
                            float2 tw_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_16) : "l"(*(const unsigned long long*)&av_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_343));
                            float _exp2_344 = approx_exp2(te_16.x);
                            float _exp2_345 = approx_exp2(te_16.y);
                            float2 e_16 = make_float2(_exp2_344, _exp2_345);
                            float _exp2_346 = approx_exp2(tw_16.x);
                            float _exp2_347 = approx_exp2(tw_16.y);
                            float2 w_16 = make_float2(_exp2_346, _exp2_347);
                            float2 _f32x2_mul_b_348 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_16) : "l"(*(const unsigned long long*)&ax_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_348));
                            float _exp2_349 = approx_exp2_noftz(tp_16.x);
                            float _exp2_350 = approx_exp2_noftz(tp_16.y);
                            float2 p_16 = make_float2(_exp2_349, _exp2_350);
                            float2 d1_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_16);
                            float2 d2_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_16);
                            float2 dw_16 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_16);
                            float2 ng_16 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_16);
                            float2 nu_16 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_16);
                            float _rcp_351 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_16, d2_16), dw_16).x);
                            float _rcp_352 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_16, d2_16), dw_16).y);
                            float2 r_34 = make_float2(_rcp_351, _rcp_352);
                            float2 r_33 = r_34;
                            float2 _f32x2_mul_a_353 = mul_f32x2_noftz(ng_16, nu_16);
                            float2 _f32x2_mul_b_354 = mul_f32x2_noftz(r_33, make_float2(bl, bl));
                            float2 m_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_353), "l"(*(const unsigned long long*)&_f32x2_mul_b_354));
                            float2 ns_16 = make_float2(((x_17.x < 0.0f) ? p_16.x : 1.0f), ((x_17.y < 0.0f) ? p_16.y : 1.0f));
                            gate_packed_0[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_16, ns_16))) | sign_17;
                            float2 x_18 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[18]));
                            float2 v_18 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[18]));
                            unsigned int sign_18 = (gate_packed_0[18] ^ up_packed_0[18]) & 2147516416u;
                            float _abs_355;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_355) : "f"(x_18.x));
                            float _abs_356;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_356) : "f"(x_18.y));
                            float2 ax_17 = make_float2(_abs_355, _abs_356);
                            float _abs_357;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_357) : "f"(v_18.x));
                            float _abs_358;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_358) : "f"(v_18.y));
                            float2 av_17 = make_float2(_abs_357, _abs_358);
                            float2 _f32x2_mul_b_359 = make_float2(k1, k1);
                            float2 te_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_17) : "l"(*(const unsigned long long*)&ax_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_359));
                            float2 _f32x2_mul_b_360 = make_float2(k2, k2);
                            float2 tw_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_17) : "l"(*(const unsigned long long*)&av_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_360));
                            float _exp2_361 = approx_exp2(te_17.x);
                            float _exp2_362 = approx_exp2(te_17.y);
                            float2 e_17 = make_float2(_exp2_361, _exp2_362);
                            float _exp2_363 = approx_exp2(tw_17.x);
                            float _exp2_364 = approx_exp2(tw_17.y);
                            float2 w_17 = make_float2(_exp2_363, _exp2_364);
                            float2 _f32x2_mul_b_365 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_17) : "l"(*(const unsigned long long*)&ax_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_365));
                            float _exp2_366 = approx_exp2_noftz(tp_17.x);
                            float _exp2_367 = approx_exp2_noftz(tp_17.y);
                            float2 p_17 = make_float2(_exp2_366, _exp2_367);
                            float2 d1_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_17);
                            float2 d2_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_17);
                            float2 dw_17 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_17);
                            float2 ng_17 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_17);
                            float2 nu_17 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_17);
                            float _rcp_368 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_17, d2_17), dw_17).x);
                            float _rcp_369 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_17, d2_17), dw_17).y);
                            float2 r_36 = make_float2(_rcp_368, _rcp_369);
                            float2 r_35 = r_36;
                            float2 _f32x2_mul_a_370 = mul_f32x2_noftz(ng_17, nu_17);
                            float2 _f32x2_mul_b_371 = mul_f32x2_noftz(r_35, make_float2(bl, bl));
                            float2 m_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_370), "l"(*(const unsigned long long*)&_f32x2_mul_b_371));
                            float2 ns_17 = make_float2(((x_18.x < 0.0f) ? p_17.x : 1.0f), ((x_18.y < 0.0f) ? p_17.y : 1.0f));
                            gate_packed_0[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_17, ns_17))) | sign_18;
                            float2 x_19 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[19]));
                            float2 v_19 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[19]));
                            unsigned int sign_19 = (gate_packed_0[19] ^ up_packed_0[19]) & 2147516416u;
                            float _abs_372;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_372) : "f"(x_19.x));
                            float _abs_373;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_373) : "f"(x_19.y));
                            float2 ax_18 = make_float2(_abs_372, _abs_373);
                            float _abs_374;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_374) : "f"(v_19.x));
                            float _abs_375;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_375) : "f"(v_19.y));
                            float2 av_18 = make_float2(_abs_374, _abs_375);
                            float2 _f32x2_mul_b_376 = make_float2(k1, k1);
                            float2 te_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_18) : "l"(*(const unsigned long long*)&ax_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_376));
                            float2 _f32x2_mul_b_377 = make_float2(k2, k2);
                            float2 tw_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_18) : "l"(*(const unsigned long long*)&av_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_377));
                            float _exp2_378 = approx_exp2(te_18.x);
                            float _exp2_379 = approx_exp2(te_18.y);
                            float2 e_18 = make_float2(_exp2_378, _exp2_379);
                            float _exp2_380 = approx_exp2(tw_18.x);
                            float _exp2_381 = approx_exp2(tw_18.y);
                            float2 w_18 = make_float2(_exp2_380, _exp2_381);
                            float2 _f32x2_mul_b_382 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_18) : "l"(*(const unsigned long long*)&ax_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_382));
                            float _exp2_383 = approx_exp2_noftz(tp_18.x);
                            float _exp2_384 = approx_exp2_noftz(tp_18.y);
                            float2 p_18 = make_float2(_exp2_383, _exp2_384);
                            float2 d1_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_18);
                            float2 d2_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_18);
                            float2 dw_18 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_18);
                            float2 ng_18 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_18);
                            float2 nu_18 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_18);
                            float _rcp_385 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_18, d2_18), dw_18).x);
                            float _rcp_386 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_18, d2_18), dw_18).y);
                            float2 r_38 = make_float2(_rcp_385, _rcp_386);
                            float2 r_37 = r_38;
                            float2 _f32x2_mul_a_387 = mul_f32x2_noftz(ng_18, nu_18);
                            float2 _f32x2_mul_b_388 = mul_f32x2_noftz(r_37, make_float2(bl, bl));
                            float2 m_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_387), "l"(*(const unsigned long long*)&_f32x2_mul_b_388));
                            float2 ns_18 = make_float2(((x_19.x < 0.0f) ? p_18.x : 1.0f), ((x_19.y < 0.0f) ? p_18.y : 1.0f));
                            gate_packed_0[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_18, ns_18))) | sign_19;
                            float2 x_20 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[20]));
                            float2 v_20 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[20]));
                            unsigned int sign_20 = (gate_packed_0[20] ^ up_packed_0[20]) & 2147516416u;
                            float _abs_389;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_389) : "f"(x_20.x));
                            float _abs_390;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_390) : "f"(x_20.y));
                            float2 ax_19 = make_float2(_abs_389, _abs_390);
                            float _abs_391;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_391) : "f"(v_20.x));
                            float _abs_392;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_392) : "f"(v_20.y));
                            float2 av_19 = make_float2(_abs_391, _abs_392);
                            float2 _f32x2_mul_b_393 = make_float2(k1, k1);
                            float2 te_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_19) : "l"(*(const unsigned long long*)&ax_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_393));
                            float2 _f32x2_mul_b_394 = make_float2(k2, k2);
                            float2 tw_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_19) : "l"(*(const unsigned long long*)&av_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_394));
                            float _exp2_395 = approx_exp2(te_19.x);
                            float _exp2_396 = approx_exp2(te_19.y);
                            float2 e_19 = make_float2(_exp2_395, _exp2_396);
                            float _exp2_397 = approx_exp2(tw_19.x);
                            float _exp2_398 = approx_exp2(tw_19.y);
                            float2 w_19 = make_float2(_exp2_397, _exp2_398);
                            float2 _f32x2_mul_b_399 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_19) : "l"(*(const unsigned long long*)&ax_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_399));
                            float _exp2_400 = approx_exp2_noftz(tp_19.x);
                            float _exp2_401 = approx_exp2_noftz(tp_19.y);
                            float2 p_19 = make_float2(_exp2_400, _exp2_401);
                            float2 d1_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_19);
                            float2 d2_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_19);
                            float2 dw_19 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_19);
                            float2 ng_19 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_19);
                            float2 nu_19 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_19);
                            float _rcp_402 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_19, d2_19), dw_19).x);
                            float _rcp_403 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_19, d2_19), dw_19).y);
                            float2 r_40 = make_float2(_rcp_402, _rcp_403);
                            float2 r_39 = r_40;
                            float2 _f32x2_mul_a_404 = mul_f32x2_noftz(ng_19, nu_19);
                            float2 _f32x2_mul_b_405 = mul_f32x2_noftz(r_39, make_float2(bl, bl));
                            float2 m_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_404), "l"(*(const unsigned long long*)&_f32x2_mul_b_405));
                            float2 ns_19 = make_float2(((x_20.x < 0.0f) ? p_19.x : 1.0f), ((x_20.y < 0.0f) ? p_19.y : 1.0f));
                            gate_packed_0[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_19, ns_19))) | sign_20;
                            float2 x_21 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[21]));
                            float2 v_21 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[21]));
                            unsigned int sign_21 = (gate_packed_0[21] ^ up_packed_0[21]) & 2147516416u;
                            float _abs_406;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_406) : "f"(x_21.x));
                            float _abs_407;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_407) : "f"(x_21.y));
                            float2 ax_20 = make_float2(_abs_406, _abs_407);
                            float _abs_408;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_408) : "f"(v_21.x));
                            float _abs_409;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_409) : "f"(v_21.y));
                            float2 av_20 = make_float2(_abs_408, _abs_409);
                            float2 _f32x2_mul_b_410 = make_float2(k1, k1);
                            float2 te_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_20) : "l"(*(const unsigned long long*)&ax_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_410));
                            float2 _f32x2_mul_b_411 = make_float2(k2, k2);
                            float2 tw_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_20) : "l"(*(const unsigned long long*)&av_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_411));
                            float _exp2_412 = approx_exp2(te_20.x);
                            float _exp2_413 = approx_exp2(te_20.y);
                            float2 e_20 = make_float2(_exp2_412, _exp2_413);
                            float _exp2_414 = approx_exp2(tw_20.x);
                            float _exp2_415 = approx_exp2(tw_20.y);
                            float2 w_20 = make_float2(_exp2_414, _exp2_415);
                            float2 _f32x2_mul_b_416 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_20) : "l"(*(const unsigned long long*)&ax_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_416));
                            float _exp2_417 = approx_exp2_noftz(tp_20.x);
                            float _exp2_418 = approx_exp2_noftz(tp_20.y);
                            float2 p_20 = make_float2(_exp2_417, _exp2_418);
                            float2 d1_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_20);
                            float2 d2_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_20);
                            float2 dw_20 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_20);
                            float2 ng_20 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_20);
                            float2 nu_20 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_20);
                            float _rcp_419 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_20, d2_20), dw_20).x);
                            float _rcp_420 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_20, d2_20), dw_20).y);
                            float2 r_42 = make_float2(_rcp_419, _rcp_420);
                            float2 r_41 = r_42;
                            float2 _f32x2_mul_a_421 = mul_f32x2_noftz(ng_20, nu_20);
                            float2 _f32x2_mul_b_422 = mul_f32x2_noftz(r_41, make_float2(bl, bl));
                            float2 m_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_421), "l"(*(const unsigned long long*)&_f32x2_mul_b_422));
                            float2 ns_20 = make_float2(((x_21.x < 0.0f) ? p_20.x : 1.0f), ((x_21.y < 0.0f) ? p_20.y : 1.0f));
                            gate_packed_0[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_20, ns_20))) | sign_21;
                            float2 x_22 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[22]));
                            float2 v_22 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[22]));
                            unsigned int sign_22 = (gate_packed_0[22] ^ up_packed_0[22]) & 2147516416u;
                            float _abs_423;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_423) : "f"(x_22.x));
                            float _abs_424;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_424) : "f"(x_22.y));
                            float2 ax_21 = make_float2(_abs_423, _abs_424);
                            float _abs_425;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_425) : "f"(v_22.x));
                            float _abs_426;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_426) : "f"(v_22.y));
                            float2 av_21 = make_float2(_abs_425, _abs_426);
                            float2 _f32x2_mul_b_427 = make_float2(k1, k1);
                            float2 te_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_21) : "l"(*(const unsigned long long*)&ax_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_427));
                            float2 _f32x2_mul_b_428 = make_float2(k2, k2);
                            float2 tw_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_21) : "l"(*(const unsigned long long*)&av_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_428));
                            float _exp2_429 = approx_exp2(te_21.x);
                            float _exp2_430 = approx_exp2(te_21.y);
                            float2 e_21 = make_float2(_exp2_429, _exp2_430);
                            float _exp2_431 = approx_exp2(tw_21.x);
                            float _exp2_432 = approx_exp2(tw_21.y);
                            float2 w_21 = make_float2(_exp2_431, _exp2_432);
                            float2 _f32x2_mul_b_433 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_21) : "l"(*(const unsigned long long*)&ax_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_433));
                            float _exp2_434 = approx_exp2_noftz(tp_21.x);
                            float _exp2_435 = approx_exp2_noftz(tp_21.y);
                            float2 p_21 = make_float2(_exp2_434, _exp2_435);
                            float2 d1_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_21);
                            float2 d2_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_21);
                            float2 dw_21 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_21);
                            float2 ng_21 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_21);
                            float2 nu_21 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_21);
                            float _rcp_436 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_21, d2_21), dw_21).x);
                            float _rcp_437 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_21, d2_21), dw_21).y);
                            float2 r_44 = make_float2(_rcp_436, _rcp_437);
                            float2 r_43 = r_44;
                            float2 _f32x2_mul_a_438 = mul_f32x2_noftz(ng_21, nu_21);
                            float2 _f32x2_mul_b_439 = mul_f32x2_noftz(r_43, make_float2(bl, bl));
                            float2 m_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_438), "l"(*(const unsigned long long*)&_f32x2_mul_b_439));
                            float2 ns_21 = make_float2(((x_22.x < 0.0f) ? p_21.x : 1.0f), ((x_22.y < 0.0f) ? p_21.y : 1.0f));
                            gate_packed_0[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_21, ns_21))) | sign_22;
                            float2 x_23 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[23]));
                            float2 v_23 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[23]));
                            unsigned int sign_23 = (gate_packed_0[23] ^ up_packed_0[23]) & 2147516416u;
                            float _abs_440;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_440) : "f"(x_23.x));
                            float _abs_441;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_441) : "f"(x_23.y));
                            float2 ax_22 = make_float2(_abs_440, _abs_441);
                            float _abs_442;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_442) : "f"(v_23.x));
                            float _abs_443;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_443) : "f"(v_23.y));
                            float2 av_22 = make_float2(_abs_442, _abs_443);
                            float2 _f32x2_mul_b_444 = make_float2(k1, k1);
                            float2 te_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_22) : "l"(*(const unsigned long long*)&ax_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_444));
                            float2 _f32x2_mul_b_445 = make_float2(k2, k2);
                            float2 tw_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_22) : "l"(*(const unsigned long long*)&av_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_445));
                            float _exp2_446 = approx_exp2(te_22.x);
                            float _exp2_447 = approx_exp2(te_22.y);
                            float2 e_22 = make_float2(_exp2_446, _exp2_447);
                            float _exp2_448 = approx_exp2(tw_22.x);
                            float _exp2_449 = approx_exp2(tw_22.y);
                            float2 w_22 = make_float2(_exp2_448, _exp2_449);
                            float2 _f32x2_mul_b_450 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_22) : "l"(*(const unsigned long long*)&ax_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_450));
                            float _exp2_451 = approx_exp2_noftz(tp_22.x);
                            float _exp2_452 = approx_exp2_noftz(tp_22.y);
                            float2 p_22 = make_float2(_exp2_451, _exp2_452);
                            float2 d1_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_22);
                            float2 d2_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_22);
                            float2 dw_22 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_22);
                            float2 ng_22 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_22);
                            float2 nu_22 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_22);
                            float _rcp_453 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_22, d2_22), dw_22).x);
                            float _rcp_454 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_22, d2_22), dw_22).y);
                            float2 r_46 = make_float2(_rcp_453, _rcp_454);
                            float2 r_45 = r_46;
                            float2 _f32x2_mul_a_455 = mul_f32x2_noftz(ng_22, nu_22);
                            float2 _f32x2_mul_b_456 = mul_f32x2_noftz(r_45, make_float2(bl, bl));
                            float2 m_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_455), "l"(*(const unsigned long long*)&_f32x2_mul_b_456));
                            float2 ns_22 = make_float2(((x_23.x < 0.0f) ? p_22.x : 1.0f), ((x_23.y < 0.0f) ? p_22.y : 1.0f));
                            gate_packed_0[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_22, ns_22))) | sign_23;
                            float2 x_24 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[24]));
                            float2 v_24 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[24]));
                            unsigned int sign_24 = (gate_packed_0[24] ^ up_packed_0[24]) & 2147516416u;
                            float _abs_457;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_457) : "f"(x_24.x));
                            float _abs_458;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_458) : "f"(x_24.y));
                            float2 ax_23 = make_float2(_abs_457, _abs_458);
                            float _abs_459;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_459) : "f"(v_24.x));
                            float _abs_460;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_460) : "f"(v_24.y));
                            float2 av_23 = make_float2(_abs_459, _abs_460);
                            float2 _f32x2_mul_b_461 = make_float2(k1, k1);
                            float2 te_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_23) : "l"(*(const unsigned long long*)&ax_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_461));
                            float2 _f32x2_mul_b_462 = make_float2(k2, k2);
                            float2 tw_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_23) : "l"(*(const unsigned long long*)&av_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_462));
                            float _exp2_463 = approx_exp2(te_23.x);
                            float _exp2_464 = approx_exp2(te_23.y);
                            float2 e_23 = make_float2(_exp2_463, _exp2_464);
                            float _exp2_465 = approx_exp2(tw_23.x);
                            float _exp2_466 = approx_exp2(tw_23.y);
                            float2 w_23 = make_float2(_exp2_465, _exp2_466);
                            float2 _f32x2_mul_b_467 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_23) : "l"(*(const unsigned long long*)&ax_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_467));
                            float _exp2_468 = approx_exp2_noftz(tp_23.x);
                            float _exp2_469 = approx_exp2_noftz(tp_23.y);
                            float2 p_23 = make_float2(_exp2_468, _exp2_469);
                            float2 d1_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_23);
                            float2 d2_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_23);
                            float2 dw_23 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_23);
                            float2 ng_23 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_23);
                            float2 nu_23 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_23);
                            float _rcp_470 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_23, d2_23), dw_23).x);
                            float _rcp_471 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_23, d2_23), dw_23).y);
                            float2 r_48 = make_float2(_rcp_470, _rcp_471);
                            float2 r_47 = r_48;
                            float2 _f32x2_mul_a_472 = mul_f32x2_noftz(ng_23, nu_23);
                            float2 _f32x2_mul_b_473 = mul_f32x2_noftz(r_47, make_float2(bl, bl));
                            float2 m_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_472), "l"(*(const unsigned long long*)&_f32x2_mul_b_473));
                            float2 ns_23 = make_float2(((x_24.x < 0.0f) ? p_23.x : 1.0f), ((x_24.y < 0.0f) ? p_23.y : 1.0f));
                            gate_packed_0[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_23, ns_23))) | sign_24;
                            float2 x_25 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[25]));
                            float2 v_25 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[25]));
                            unsigned int sign_25 = (gate_packed_0[25] ^ up_packed_0[25]) & 2147516416u;
                            float _abs_474;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_474) : "f"(x_25.x));
                            float _abs_475;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_475) : "f"(x_25.y));
                            float2 ax_24 = make_float2(_abs_474, _abs_475);
                            float _abs_476;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_476) : "f"(v_25.x));
                            float _abs_477;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_477) : "f"(v_25.y));
                            float2 av_24 = make_float2(_abs_476, _abs_477);
                            float2 _f32x2_mul_b_478 = make_float2(k1, k1);
                            float2 te_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_24) : "l"(*(const unsigned long long*)&ax_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_478));
                            float2 _f32x2_mul_b_479 = make_float2(k2, k2);
                            float2 tw_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_24) : "l"(*(const unsigned long long*)&av_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_479));
                            float _exp2_480 = approx_exp2(te_24.x);
                            float _exp2_481 = approx_exp2(te_24.y);
                            float2 e_24 = make_float2(_exp2_480, _exp2_481);
                            float _exp2_482 = approx_exp2(tw_24.x);
                            float _exp2_483 = approx_exp2(tw_24.y);
                            float2 w_24 = make_float2(_exp2_482, _exp2_483);
                            float2 _f32x2_mul_b_484 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_24) : "l"(*(const unsigned long long*)&ax_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_484));
                            float _exp2_485 = approx_exp2_noftz(tp_24.x);
                            float _exp2_486 = approx_exp2_noftz(tp_24.y);
                            float2 p_24 = make_float2(_exp2_485, _exp2_486);
                            float2 d1_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_24);
                            float2 d2_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_24);
                            float2 dw_24 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_24);
                            float2 ng_24 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_24);
                            float2 nu_24 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_24);
                            float _rcp_487 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_24, d2_24), dw_24).x);
                            float _rcp_488 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_24, d2_24), dw_24).y);
                            float2 r_50 = make_float2(_rcp_487, _rcp_488);
                            float2 r_49 = r_50;
                            float2 _f32x2_mul_a_489 = mul_f32x2_noftz(ng_24, nu_24);
                            float2 _f32x2_mul_b_490 = mul_f32x2_noftz(r_49, make_float2(bl, bl));
                            float2 m_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_489), "l"(*(const unsigned long long*)&_f32x2_mul_b_490));
                            float2 ns_24 = make_float2(((x_25.x < 0.0f) ? p_24.x : 1.0f), ((x_25.y < 0.0f) ? p_24.y : 1.0f));
                            gate_packed_0[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_24, ns_24))) | sign_25;
                            float2 x_26 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[26]));
                            float2 v_26 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[26]));
                            unsigned int sign_26 = (gate_packed_0[26] ^ up_packed_0[26]) & 2147516416u;
                            float _abs_491;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_491) : "f"(x_26.x));
                            float _abs_492;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_492) : "f"(x_26.y));
                            float2 ax_25 = make_float2(_abs_491, _abs_492);
                            float _abs_493;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_493) : "f"(v_26.x));
                            float _abs_494;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_494) : "f"(v_26.y));
                            float2 av_25 = make_float2(_abs_493, _abs_494);
                            float2 _f32x2_mul_b_495 = make_float2(k1, k1);
                            float2 te_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_25) : "l"(*(const unsigned long long*)&ax_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_495));
                            float2 _f32x2_mul_b_496 = make_float2(k2, k2);
                            float2 tw_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_25) : "l"(*(const unsigned long long*)&av_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_496));
                            float _exp2_497 = approx_exp2(te_25.x);
                            float _exp2_498 = approx_exp2(te_25.y);
                            float2 e_25 = make_float2(_exp2_497, _exp2_498);
                            float _exp2_499 = approx_exp2(tw_25.x);
                            float _exp2_500 = approx_exp2(tw_25.y);
                            float2 w_25 = make_float2(_exp2_499, _exp2_500);
                            float2 _f32x2_mul_b_501 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_25) : "l"(*(const unsigned long long*)&ax_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_501));
                            float _exp2_502 = approx_exp2_noftz(tp_25.x);
                            float _exp2_503 = approx_exp2_noftz(tp_25.y);
                            float2 p_25 = make_float2(_exp2_502, _exp2_503);
                            float2 d1_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_25);
                            float2 d2_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_25);
                            float2 dw_25 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_25);
                            float2 ng_25 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_25);
                            float2 nu_25 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_25);
                            float _rcp_504 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_25, d2_25), dw_25).x);
                            float _rcp_505 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_25, d2_25), dw_25).y);
                            float2 r_52 = make_float2(_rcp_504, _rcp_505);
                            float2 r_51 = r_52;
                            float2 _f32x2_mul_a_506 = mul_f32x2_noftz(ng_25, nu_25);
                            float2 _f32x2_mul_b_507 = mul_f32x2_noftz(r_51, make_float2(bl, bl));
                            float2 m_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_506), "l"(*(const unsigned long long*)&_f32x2_mul_b_507));
                            float2 ns_25 = make_float2(((x_26.x < 0.0f) ? p_25.x : 1.0f), ((x_26.y < 0.0f) ? p_25.y : 1.0f));
                            gate_packed_0[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_25, ns_25))) | sign_26;
                            float2 x_27 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[27]));
                            float2 v_27 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[27]));
                            unsigned int sign_27 = (gate_packed_0[27] ^ up_packed_0[27]) & 2147516416u;
                            float _abs_508;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_508) : "f"(x_27.x));
                            float _abs_509;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_509) : "f"(x_27.y));
                            float2 ax_26 = make_float2(_abs_508, _abs_509);
                            float _abs_510;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_510) : "f"(v_27.x));
                            float _abs_511;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_511) : "f"(v_27.y));
                            float2 av_26 = make_float2(_abs_510, _abs_511);
                            float2 _f32x2_mul_b_512 = make_float2(k1, k1);
                            float2 te_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_26) : "l"(*(const unsigned long long*)&ax_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_512));
                            float2 _f32x2_mul_b_513 = make_float2(k2, k2);
                            float2 tw_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_26) : "l"(*(const unsigned long long*)&av_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_513));
                            float _exp2_514 = approx_exp2(te_26.x);
                            float _exp2_515 = approx_exp2(te_26.y);
                            float2 e_26 = make_float2(_exp2_514, _exp2_515);
                            float _exp2_516 = approx_exp2(tw_26.x);
                            float _exp2_517 = approx_exp2(tw_26.y);
                            float2 w_26 = make_float2(_exp2_516, _exp2_517);
                            float2 _f32x2_mul_b_518 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_26) : "l"(*(const unsigned long long*)&ax_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_518));
                            float _exp2_519 = approx_exp2_noftz(tp_26.x);
                            float _exp2_520 = approx_exp2_noftz(tp_26.y);
                            float2 p_26 = make_float2(_exp2_519, _exp2_520);
                            float2 d1_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_26);
                            float2 d2_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_26);
                            float2 dw_26 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_26);
                            float2 ng_26 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_26);
                            float2 nu_26 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_26);
                            float _rcp_521 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_26, d2_26), dw_26).x);
                            float _rcp_522 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_26, d2_26), dw_26).y);
                            float2 r_54 = make_float2(_rcp_521, _rcp_522);
                            float2 r_53 = r_54;
                            float2 _f32x2_mul_a_523 = mul_f32x2_noftz(ng_26, nu_26);
                            float2 _f32x2_mul_b_524 = mul_f32x2_noftz(r_53, make_float2(bl, bl));
                            float2 m_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_523), "l"(*(const unsigned long long*)&_f32x2_mul_b_524));
                            float2 ns_26 = make_float2(((x_27.x < 0.0f) ? p_26.x : 1.0f), ((x_27.y < 0.0f) ? p_26.y : 1.0f));
                            gate_packed_0[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_26, ns_26))) | sign_27;
                            float2 x_28 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[28]));
                            float2 v_28 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[28]));
                            unsigned int sign_28 = (gate_packed_0[28] ^ up_packed_0[28]) & 2147516416u;
                            float _abs_525;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_525) : "f"(x_28.x));
                            float _abs_526;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_526) : "f"(x_28.y));
                            float2 ax_27 = make_float2(_abs_525, _abs_526);
                            float _abs_527;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_527) : "f"(v_28.x));
                            float _abs_528;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_528) : "f"(v_28.y));
                            float2 av_27 = make_float2(_abs_527, _abs_528);
                            float2 _f32x2_mul_b_529 = make_float2(k1, k1);
                            float2 te_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_27) : "l"(*(const unsigned long long*)&ax_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_529));
                            float2 _f32x2_mul_b_530 = make_float2(k2, k2);
                            float2 tw_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_27) : "l"(*(const unsigned long long*)&av_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_530));
                            float _exp2_531 = approx_exp2(te_27.x);
                            float _exp2_532 = approx_exp2(te_27.y);
                            float2 e_27 = make_float2(_exp2_531, _exp2_532);
                            float _exp2_533 = approx_exp2(tw_27.x);
                            float _exp2_534 = approx_exp2(tw_27.y);
                            float2 w_27 = make_float2(_exp2_533, _exp2_534);
                            float2 _f32x2_mul_b_535 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_27) : "l"(*(const unsigned long long*)&ax_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_535));
                            float _exp2_536 = approx_exp2_noftz(tp_27.x);
                            float _exp2_537 = approx_exp2_noftz(tp_27.y);
                            float2 p_27 = make_float2(_exp2_536, _exp2_537);
                            float2 d1_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_27);
                            float2 d2_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_27);
                            float2 dw_27 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_27);
                            float2 ng_27 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_27);
                            float2 nu_27 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_27);
                            float _rcp_538 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_27, d2_27), dw_27).x);
                            float _rcp_539 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_27, d2_27), dw_27).y);
                            float2 r_56 = make_float2(_rcp_538, _rcp_539);
                            float2 r_55 = r_56;
                            float2 _f32x2_mul_a_540 = mul_f32x2_noftz(ng_27, nu_27);
                            float2 _f32x2_mul_b_541 = mul_f32x2_noftz(r_55, make_float2(bl, bl));
                            float2 m_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_540), "l"(*(const unsigned long long*)&_f32x2_mul_b_541));
                            float2 ns_27 = make_float2(((x_28.x < 0.0f) ? p_27.x : 1.0f), ((x_28.y < 0.0f) ? p_27.y : 1.0f));
                            gate_packed_0[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_27, ns_27))) | sign_28;
                            float2 x_29 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[29]));
                            float2 v_29 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[29]));
                            unsigned int sign_29 = (gate_packed_0[29] ^ up_packed_0[29]) & 2147516416u;
                            float _abs_542;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_542) : "f"(x_29.x));
                            float _abs_543;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_543) : "f"(x_29.y));
                            float2 ax_28 = make_float2(_abs_542, _abs_543);
                            float _abs_544;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_544) : "f"(v_29.x));
                            float _abs_545;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_545) : "f"(v_29.y));
                            float2 av_28 = make_float2(_abs_544, _abs_545);
                            float2 _f32x2_mul_b_546 = make_float2(k1, k1);
                            float2 te_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_28) : "l"(*(const unsigned long long*)&ax_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_546));
                            float2 _f32x2_mul_b_547 = make_float2(k2, k2);
                            float2 tw_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_28) : "l"(*(const unsigned long long*)&av_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_547));
                            float _exp2_548 = approx_exp2(te_28.x);
                            float _exp2_549 = approx_exp2(te_28.y);
                            float2 e_28 = make_float2(_exp2_548, _exp2_549);
                            float _exp2_550 = approx_exp2(tw_28.x);
                            float _exp2_551 = approx_exp2(tw_28.y);
                            float2 w_28 = make_float2(_exp2_550, _exp2_551);
                            float2 _f32x2_mul_b_552 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_28) : "l"(*(const unsigned long long*)&ax_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_552));
                            float _exp2_553 = approx_exp2_noftz(tp_28.x);
                            float _exp2_554 = approx_exp2_noftz(tp_28.y);
                            float2 p_28 = make_float2(_exp2_553, _exp2_554);
                            float2 d1_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_28);
                            float2 d2_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_28);
                            float2 dw_28 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_28);
                            float2 ng_28 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_28);
                            float2 nu_28 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_28);
                            float _rcp_555 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_28, d2_28), dw_28).x);
                            float _rcp_556 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_28, d2_28), dw_28).y);
                            float2 r_58 = make_float2(_rcp_555, _rcp_556);
                            float2 r_57 = r_58;
                            float2 _f32x2_mul_a_557 = mul_f32x2_noftz(ng_28, nu_28);
                            float2 _f32x2_mul_b_558 = mul_f32x2_noftz(r_57, make_float2(bl, bl));
                            float2 m_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_557), "l"(*(const unsigned long long*)&_f32x2_mul_b_558));
                            float2 ns_28 = make_float2(((x_29.x < 0.0f) ? p_28.x : 1.0f), ((x_29.y < 0.0f) ? p_28.y : 1.0f));
                            gate_packed_0[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_28, ns_28))) | sign_29;
                            float2 x_30 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[30]));
                            float2 v_30 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[30]));
                            unsigned int sign_30 = (gate_packed_0[30] ^ up_packed_0[30]) & 2147516416u;
                            float _abs_559;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_559) : "f"(x_30.x));
                            float _abs_560;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_560) : "f"(x_30.y));
                            float2 ax_29 = make_float2(_abs_559, _abs_560);
                            float _abs_561;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_561) : "f"(v_30.x));
                            float _abs_562;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_562) : "f"(v_30.y));
                            float2 av_29 = make_float2(_abs_561, _abs_562);
                            float2 _f32x2_mul_b_563 = make_float2(k1, k1);
                            float2 te_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_29) : "l"(*(const unsigned long long*)&ax_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_563));
                            float2 _f32x2_mul_b_564 = make_float2(k2, k2);
                            float2 tw_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_29) : "l"(*(const unsigned long long*)&av_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_564));
                            float _exp2_565 = approx_exp2(te_29.x);
                            float _exp2_566 = approx_exp2(te_29.y);
                            float2 e_29 = make_float2(_exp2_565, _exp2_566);
                            float _exp2_567 = approx_exp2(tw_29.x);
                            float _exp2_568 = approx_exp2(tw_29.y);
                            float2 w_29 = make_float2(_exp2_567, _exp2_568);
                            float2 _f32x2_mul_b_569 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_29) : "l"(*(const unsigned long long*)&ax_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_569));
                            float _exp2_570 = approx_exp2_noftz(tp_29.x);
                            float _exp2_571 = approx_exp2_noftz(tp_29.y);
                            float2 p_29 = make_float2(_exp2_570, _exp2_571);
                            float2 d1_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_29);
                            float2 d2_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_29);
                            float2 dw_29 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_29);
                            float2 ng_29 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_29);
                            float2 nu_29 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_29);
                            float _rcp_572 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_29, d2_29), dw_29).x);
                            float _rcp_573 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_29, d2_29), dw_29).y);
                            float2 r_60 = make_float2(_rcp_572, _rcp_573);
                            float2 r_59 = r_60;
                            float2 _f32x2_mul_a_574 = mul_f32x2_noftz(ng_29, nu_29);
                            float2 _f32x2_mul_b_575 = mul_f32x2_noftz(r_59, make_float2(bl, bl));
                            float2 m_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_574), "l"(*(const unsigned long long*)&_f32x2_mul_b_575));
                            float2 ns_29 = make_float2(((x_30.x < 0.0f) ? p_29.x : 1.0f), ((x_30.y < 0.0f) ? p_29.y : 1.0f));
                            gate_packed_0[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_29, ns_29))) | sign_30;
                            float2 x_31 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[31]));
                            float2 v_31 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[31]));
                            unsigned int sign_31 = (gate_packed_0[31] ^ up_packed_0[31]) & 2147516416u;
                            float _abs_576;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_576) : "f"(x_31.x));
                            float _abs_577;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_577) : "f"(x_31.y));
                            float2 ax_30 = make_float2(_abs_576, _abs_577);
                            float _abs_578;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_578) : "f"(v_31.x));
                            float _abs_579;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_579) : "f"(v_31.y));
                            float2 av_30 = make_float2(_abs_578, _abs_579);
                            float2 _f32x2_mul_b_580 = make_float2(k1, k1);
                            float2 te_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_30) : "l"(*(const unsigned long long*)&ax_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_580));
                            float2 _f32x2_mul_b_581 = make_float2(k2, k2);
                            float2 tw_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_30) : "l"(*(const unsigned long long*)&av_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_581));
                            float _exp2_582 = approx_exp2(te_30.x);
                            float _exp2_583 = approx_exp2(te_30.y);
                            float2 e_30 = make_float2(_exp2_582, _exp2_583);
                            float _exp2_584 = approx_exp2(tw_30.x);
                            float _exp2_585 = approx_exp2(tw_30.y);
                            float2 w_30 = make_float2(_exp2_584, _exp2_585);
                            float2 _f32x2_mul_b_586 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_30) : "l"(*(const unsigned long long*)&ax_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_586));
                            float _exp2_587 = approx_exp2_noftz(tp_30.x);
                            float _exp2_588 = approx_exp2_noftz(tp_30.y);
                            float2 p_30 = make_float2(_exp2_587, _exp2_588);
                            float2 d1_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_30);
                            float2 d2_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_30);
                            float2 dw_30 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_30);
                            float2 ng_30 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_30);
                            float2 nu_30 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_30);
                            float _rcp_589 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_30, d2_30), dw_30).x);
                            float _rcp_590 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_30, d2_30), dw_30).y);
                            float2 r_62 = make_float2(_rcp_589, _rcp_590);
                            float2 r_61 = r_62;
                            float2 _f32x2_mul_a_591 = mul_f32x2_noftz(ng_30, nu_30);
                            float2 _f32x2_mul_b_592 = mul_f32x2_noftz(r_61, make_float2(bl, bl));
                            float2 m_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_591), "l"(*(const unsigned long long*)&_f32x2_mul_b_592));
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
                                    int row_9 = warp_4 * 32 + half_5 * 16 + lane_4 % 16;
                                    int col_6 = col_tile_5 * 16 + lane_4 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + 16384 + (unsigned int)((row_9 * 32 + col_6) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_6 = 16 + half_5 * 8 + col_tile_5 * 4;
                                    uint32_t _stmatrix_addr_593 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_593), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_6 + 3]))
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
                                    int row_10 = warp_5 * 32 + half_6 * 16 + lane_5 % 16;
                                    int col_7 = col_tile_6 * 16 + lane_5 / 16 * 8;
                                    unsigned int address_6 = d_smem_addr + (unsigned int)((row_10 * 32 + col_7) * 2);
                                    address_6 = address_6 ^ (address_6 & 511) >> 7 << 4;
                                    int offset_7 = half_6 * 8 + col_tile_6 * 4;
                                    uint32_t _stmatrix_addr_594 = static_cast<uint32_t>(address_6);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_594), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_7 + 3]))
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
                                    int row_11 = warp_6 * 32 + half_7 * 16 + lane_6 % 16;
                                    int col_8 = col_tile_7 * 16 + lane_6 / 16 * 8;
                                    unsigned int address_7 = d_smem_addr + 8192 + (unsigned int)((row_11 * 32 + col_8) * 2);
                                    address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                    int offset_8 = half_7 * 8 + col_tile_7 * 4;
                                    uint32_t _stmatrix_addr_595 = static_cast<uint32_t>(address_7);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_595), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_8 + 3]))
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
                                    int row_12 = warp_7 * 32 + half_8 * 16 + lane_7 % 16;
                                    int col_9 = col_tile_8 * 16 + lane_7 / 16 * 8;
                                    unsigned int address_8 = d_smem_addr + 16384 + (unsigned int)((row_12 * 32 + col_9) * 2);
                                    address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                    int offset_9 = 16 + half_8 * 8 + col_tile_8 * 4;
                                    uint32_t _stmatrix_addr_596 = static_cast<uint32_t>(address_8);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_596), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 3]))
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
                                    int row_13 = warp_8 * 32 + half_9 * 16 + lane_8 % 16;
                                    int col_10 = col_tile_9 * 16 + lane_8 / 16 * 8;
                                    unsigned int address_9 = d_smem_addr + (unsigned int)((row_13 * 32 + col_10) * 2);
                                    address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                    int offset_10 = 16 + half_9 * 8 + col_tile_9 * 4;
                                    uint32_t _stmatrix_addr_597 = static_cast<uint32_t>(address_9);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_597), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 3]))
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
                            float _abs_598;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_598) : "f"(x_32.x));
                            float _abs_599;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_599) : "f"(x_32.y));
                            float2 ax_31 = make_float2(_abs_598, _abs_599);
                            float _abs_600;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_600) : "f"(v_32.x));
                            float _abs_601;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_601) : "f"(v_32.y));
                            float2 av_31 = make_float2(_abs_600, _abs_601);
                            float2 _f32x2_mul_b_602 = make_float2(k1, k1);
                            float2 te_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_31) : "l"(*(const unsigned long long*)&ax_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_602));
                            float2 _f32x2_mul_b_603 = make_float2(k2, k2);
                            float2 tw_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_31) : "l"(*(const unsigned long long*)&av_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_603));
                            float _exp2_604 = approx_exp2(te_31.x);
                            float _exp2_605 = approx_exp2(te_31.y);
                            float2 e_31 = make_float2(_exp2_604, _exp2_605);
                            float _exp2_606 = approx_exp2(tw_31.x);
                            float _exp2_607 = approx_exp2(tw_31.y);
                            float2 w_31 = make_float2(_exp2_606, _exp2_607);
                            float2 _f32x2_mul_b_608 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_31) : "l"(*(const unsigned long long*)&ax_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_608));
                            float _exp2_609 = approx_exp2_noftz(tp_31.x);
                            float _exp2_610 = approx_exp2_noftz(tp_31.y);
                            float2 p_31 = make_float2(_exp2_609, _exp2_610);
                            float2 d1_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_31);
                            float2 d2_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_31);
                            float2 dw_31 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_31);
                            float2 ng_31 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_31);
                            float2 nu_31 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_31);
                            float _rcp_611 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_31, d2_31), dw_31).x);
                            float _rcp_612 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_31, d2_31), dw_31).y);
                            float2 r_64 = make_float2(_rcp_611, _rcp_612);
                            float2 r_63 = r_64;
                            float2 _f32x2_mul_a_613 = mul_f32x2_noftz(ng_31, nu_31);
                            float2 _f32x2_mul_b_614 = mul_f32x2_noftz(r_63, make_float2(bl, bl));
                            float2 m_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_613), "l"(*(const unsigned long long*)&_f32x2_mul_b_614));
                            float2 ns_31 = make_float2(((x_32.x < 0.0f) ? p_31.x : 1.0f), ((x_32.y < 0.0f) ? p_31.y : 1.0f));
                            gate_packed_1[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_31, ns_31))) | sign_32;
                            float2 x_33 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[1]));
                            float2 v_33 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[1]));
                            unsigned int sign_33 = (gate_packed_1[1] ^ up_packed_1[1]) & 2147516416u;
                            float _abs_615;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_615) : "f"(x_33.x));
                            float _abs_616;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_616) : "f"(x_33.y));
                            float2 ax_32 = make_float2(_abs_615, _abs_616);
                            float _abs_617;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_617) : "f"(v_33.x));
                            float _abs_618;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_618) : "f"(v_33.y));
                            float2 av_32 = make_float2(_abs_617, _abs_618);
                            float2 _f32x2_mul_b_619 = make_float2(k1, k1);
                            float2 te_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_32) : "l"(*(const unsigned long long*)&ax_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_619));
                            float2 _f32x2_mul_b_620 = make_float2(k2, k2);
                            float2 tw_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_32) : "l"(*(const unsigned long long*)&av_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_620));
                            float _exp2_621 = approx_exp2(te_32.x);
                            float _exp2_622 = approx_exp2(te_32.y);
                            float2 e_32 = make_float2(_exp2_621, _exp2_622);
                            float _exp2_623 = approx_exp2(tw_32.x);
                            float _exp2_624 = approx_exp2(tw_32.y);
                            float2 w_32 = make_float2(_exp2_623, _exp2_624);
                            float2 _f32x2_mul_b_625 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_32) : "l"(*(const unsigned long long*)&ax_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_625));
                            float _exp2_626 = approx_exp2_noftz(tp_32.x);
                            float _exp2_627 = approx_exp2_noftz(tp_32.y);
                            float2 p_32 = make_float2(_exp2_626, _exp2_627);
                            float2 d1_32 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_32);
                            float2 d2_32 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_32);
                            float2 dw_32 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_32);
                            float2 ng_32 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_32);
                            float2 nu_32 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_32);
                            float _rcp_628 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_32, d2_32), dw_32).x);
                            float _rcp_629 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_32, d2_32), dw_32).y);
                            float2 r_66 = make_float2(_rcp_628, _rcp_629);
                            float2 r_65 = r_66;
                            float2 _f32x2_mul_a_630 = mul_f32x2_noftz(ng_32, nu_32);
                            float2 _f32x2_mul_b_631 = mul_f32x2_noftz(r_65, make_float2(bl, bl));
                            float2 m_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_630), "l"(*(const unsigned long long*)&_f32x2_mul_b_631));
                            float2 ns_32 = make_float2(((x_33.x < 0.0f) ? p_32.x : 1.0f), ((x_33.y < 0.0f) ? p_32.y : 1.0f));
                            gate_packed_1[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_32, ns_32))) | sign_33;
                            float2 x_34 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[2]));
                            float2 v_34 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[2]));
                            unsigned int sign_34 = (gate_packed_1[2] ^ up_packed_1[2]) & 2147516416u;
                            float _abs_632;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_632) : "f"(x_34.x));
                            float _abs_633;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_633) : "f"(x_34.y));
                            float2 ax_33 = make_float2(_abs_632, _abs_633);
                            float _abs_634;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_634) : "f"(v_34.x));
                            float _abs_635;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_635) : "f"(v_34.y));
                            float2 av_33 = make_float2(_abs_634, _abs_635);
                            float2 _f32x2_mul_b_636 = make_float2(k1, k1);
                            float2 te_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_33) : "l"(*(const unsigned long long*)&ax_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_636));
                            float2 _f32x2_mul_b_637 = make_float2(k2, k2);
                            float2 tw_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_33) : "l"(*(const unsigned long long*)&av_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_637));
                            float _exp2_638 = approx_exp2(te_33.x);
                            float _exp2_639 = approx_exp2(te_33.y);
                            float2 e_33 = make_float2(_exp2_638, _exp2_639);
                            float _exp2_640 = approx_exp2(tw_33.x);
                            float _exp2_641 = approx_exp2(tw_33.y);
                            float2 w_33 = make_float2(_exp2_640, _exp2_641);
                            float2 _f32x2_mul_b_642 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_33) : "l"(*(const unsigned long long*)&ax_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_642));
                            float _exp2_643 = approx_exp2_noftz(tp_33.x);
                            float _exp2_644 = approx_exp2_noftz(tp_33.y);
                            float2 p_33 = make_float2(_exp2_643, _exp2_644);
                            float2 d1_33 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_33);
                            float2 d2_33 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_33);
                            float2 dw_33 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_33);
                            float2 ng_33 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_33);
                            float2 nu_33 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_33);
                            float _rcp_645 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_33, d2_33), dw_33).x);
                            float _rcp_646 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_33, d2_33), dw_33).y);
                            float2 r_68 = make_float2(_rcp_645, _rcp_646);
                            float2 r_67 = r_68;
                            float2 _f32x2_mul_a_647 = mul_f32x2_noftz(ng_33, nu_33);
                            float2 _f32x2_mul_b_648 = mul_f32x2_noftz(r_67, make_float2(bl, bl));
                            float2 m_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_647), "l"(*(const unsigned long long*)&_f32x2_mul_b_648));
                            float2 ns_33 = make_float2(((x_34.x < 0.0f) ? p_33.x : 1.0f), ((x_34.y < 0.0f) ? p_33.y : 1.0f));
                            gate_packed_1[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_33, ns_33))) | sign_34;
                            float2 x_35 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[3]));
                            float2 v_35 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[3]));
                            unsigned int sign_35 = (gate_packed_1[3] ^ up_packed_1[3]) & 2147516416u;
                            float _abs_649;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_649) : "f"(x_35.x));
                            float _abs_650;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_650) : "f"(x_35.y));
                            float2 ax_34 = make_float2(_abs_649, _abs_650);
                            float _abs_651;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_651) : "f"(v_35.x));
                            float _abs_652;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_652) : "f"(v_35.y));
                            float2 av_34 = make_float2(_abs_651, _abs_652);
                            float2 _f32x2_mul_b_653 = make_float2(k1, k1);
                            float2 te_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_34) : "l"(*(const unsigned long long*)&ax_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_653));
                            float2 _f32x2_mul_b_654 = make_float2(k2, k2);
                            float2 tw_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_34) : "l"(*(const unsigned long long*)&av_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_654));
                            float _exp2_655 = approx_exp2(te_34.x);
                            float _exp2_656 = approx_exp2(te_34.y);
                            float2 e_34 = make_float2(_exp2_655, _exp2_656);
                            float _exp2_657 = approx_exp2(tw_34.x);
                            float _exp2_658 = approx_exp2(tw_34.y);
                            float2 w_34 = make_float2(_exp2_657, _exp2_658);
                            float2 _f32x2_mul_b_659 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_34) : "l"(*(const unsigned long long*)&ax_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_659));
                            float _exp2_660 = approx_exp2_noftz(tp_34.x);
                            float _exp2_661 = approx_exp2_noftz(tp_34.y);
                            float2 p_34 = make_float2(_exp2_660, _exp2_661);
                            float2 d1_34 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_34);
                            float2 d2_34 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_34);
                            float2 dw_34 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_34);
                            float2 ng_34 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_34);
                            float2 nu_34 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_34);
                            float _rcp_662 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_34, d2_34), dw_34).x);
                            float _rcp_663 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_34, d2_34), dw_34).y);
                            float2 r_70 = make_float2(_rcp_662, _rcp_663);
                            float2 r_69 = r_70;
                            float2 _f32x2_mul_a_664 = mul_f32x2_noftz(ng_34, nu_34);
                            float2 _f32x2_mul_b_665 = mul_f32x2_noftz(r_69, make_float2(bl, bl));
                            float2 m_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_664), "l"(*(const unsigned long long*)&_f32x2_mul_b_665));
                            float2 ns_34 = make_float2(((x_35.x < 0.0f) ? p_34.x : 1.0f), ((x_35.y < 0.0f) ? p_34.y : 1.0f));
                            gate_packed_1[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_34, ns_34))) | sign_35;
                            float2 x_36 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[4]));
                            float2 v_36 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[4]));
                            unsigned int sign_36 = (gate_packed_1[4] ^ up_packed_1[4]) & 2147516416u;
                            float _abs_666;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_666) : "f"(x_36.x));
                            float _abs_667;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_667) : "f"(x_36.y));
                            float2 ax_35 = make_float2(_abs_666, _abs_667);
                            float _abs_668;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_668) : "f"(v_36.x));
                            float _abs_669;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_669) : "f"(v_36.y));
                            float2 av_35 = make_float2(_abs_668, _abs_669);
                            float2 _f32x2_mul_b_670 = make_float2(k1, k1);
                            float2 te_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_35) : "l"(*(const unsigned long long*)&ax_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_670));
                            float2 _f32x2_mul_b_671 = make_float2(k2, k2);
                            float2 tw_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_35) : "l"(*(const unsigned long long*)&av_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_671));
                            float _exp2_672 = approx_exp2(te_35.x);
                            float _exp2_673 = approx_exp2(te_35.y);
                            float2 e_35 = make_float2(_exp2_672, _exp2_673);
                            float _exp2_674 = approx_exp2(tw_35.x);
                            float _exp2_675 = approx_exp2(tw_35.y);
                            float2 w_35 = make_float2(_exp2_674, _exp2_675);
                            float2 _f32x2_mul_b_676 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_35) : "l"(*(const unsigned long long*)&ax_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_676));
                            float _exp2_677 = approx_exp2_noftz(tp_35.x);
                            float _exp2_678 = approx_exp2_noftz(tp_35.y);
                            float2 p_35 = make_float2(_exp2_677, _exp2_678);
                            float2 d1_35 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_35);
                            float2 d2_35 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_35);
                            float2 dw_35 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_35);
                            float2 ng_35 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_35);
                            float2 nu_35 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_35);
                            float _rcp_679 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_35, d2_35), dw_35).x);
                            float _rcp_680 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_35, d2_35), dw_35).y);
                            float2 r_72 = make_float2(_rcp_679, _rcp_680);
                            float2 r_71 = r_72;
                            float2 _f32x2_mul_a_681 = mul_f32x2_noftz(ng_35, nu_35);
                            float2 _f32x2_mul_b_682 = mul_f32x2_noftz(r_71, make_float2(bl, bl));
                            float2 m_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_681), "l"(*(const unsigned long long*)&_f32x2_mul_b_682));
                            float2 ns_35 = make_float2(((x_36.x < 0.0f) ? p_35.x : 1.0f), ((x_36.y < 0.0f) ? p_35.y : 1.0f));
                            gate_packed_1[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_35, ns_35))) | sign_36;
                            float2 x_37 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[5]));
                            float2 v_37 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[5]));
                            unsigned int sign_37 = (gate_packed_1[5] ^ up_packed_1[5]) & 2147516416u;
                            float _abs_683;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_683) : "f"(x_37.x));
                            float _abs_684;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_684) : "f"(x_37.y));
                            float2 ax_36 = make_float2(_abs_683, _abs_684);
                            float _abs_685;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_685) : "f"(v_37.x));
                            float _abs_686;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_686) : "f"(v_37.y));
                            float2 av_36 = make_float2(_abs_685, _abs_686);
                            float2 _f32x2_mul_b_687 = make_float2(k1, k1);
                            float2 te_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_36) : "l"(*(const unsigned long long*)&ax_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_687));
                            float2 _f32x2_mul_b_688 = make_float2(k2, k2);
                            float2 tw_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_36) : "l"(*(const unsigned long long*)&av_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_688));
                            float _exp2_689 = approx_exp2(te_36.x);
                            float _exp2_690 = approx_exp2(te_36.y);
                            float2 e_36 = make_float2(_exp2_689, _exp2_690);
                            float _exp2_691 = approx_exp2(tw_36.x);
                            float _exp2_692 = approx_exp2(tw_36.y);
                            float2 w_36 = make_float2(_exp2_691, _exp2_692);
                            float2 _f32x2_mul_b_693 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_36) : "l"(*(const unsigned long long*)&ax_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_693));
                            float _exp2_694 = approx_exp2_noftz(tp_36.x);
                            float _exp2_695 = approx_exp2_noftz(tp_36.y);
                            float2 p_36 = make_float2(_exp2_694, _exp2_695);
                            float2 d1_36 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_36);
                            float2 d2_36 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_36);
                            float2 dw_36 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_36);
                            float2 ng_36 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_36);
                            float2 nu_36 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_36);
                            float _rcp_696 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_36, d2_36), dw_36).x);
                            float _rcp_697 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_36, d2_36), dw_36).y);
                            float2 r_74 = make_float2(_rcp_696, _rcp_697);
                            float2 r_73 = r_74;
                            float2 _f32x2_mul_a_698 = mul_f32x2_noftz(ng_36, nu_36);
                            float2 _f32x2_mul_b_699 = mul_f32x2_noftz(r_73, make_float2(bl, bl));
                            float2 m_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_698), "l"(*(const unsigned long long*)&_f32x2_mul_b_699));
                            float2 ns_36 = make_float2(((x_37.x < 0.0f) ? p_36.x : 1.0f), ((x_37.y < 0.0f) ? p_36.y : 1.0f));
                            gate_packed_1[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_36, ns_36))) | sign_37;
                            float2 x_38 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[6]));
                            float2 v_38 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[6]));
                            unsigned int sign_38 = (gate_packed_1[6] ^ up_packed_1[6]) & 2147516416u;
                            float _abs_700;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_700) : "f"(x_38.x));
                            float _abs_701;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_701) : "f"(x_38.y));
                            float2 ax_37 = make_float2(_abs_700, _abs_701);
                            float _abs_702;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_702) : "f"(v_38.x));
                            float _abs_703;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_703) : "f"(v_38.y));
                            float2 av_37 = make_float2(_abs_702, _abs_703);
                            float2 _f32x2_mul_b_704 = make_float2(k1, k1);
                            float2 te_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_37) : "l"(*(const unsigned long long*)&ax_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_704));
                            float2 _f32x2_mul_b_705 = make_float2(k2, k2);
                            float2 tw_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_37) : "l"(*(const unsigned long long*)&av_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_705));
                            float _exp2_706 = approx_exp2(te_37.x);
                            float _exp2_707 = approx_exp2(te_37.y);
                            float2 e_37 = make_float2(_exp2_706, _exp2_707);
                            float _exp2_708 = approx_exp2(tw_37.x);
                            float _exp2_709 = approx_exp2(tw_37.y);
                            float2 w_37 = make_float2(_exp2_708, _exp2_709);
                            float2 _f32x2_mul_b_710 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_37) : "l"(*(const unsigned long long*)&ax_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_710));
                            float _exp2_711 = approx_exp2_noftz(tp_37.x);
                            float _exp2_712 = approx_exp2_noftz(tp_37.y);
                            float2 p_37 = make_float2(_exp2_711, _exp2_712);
                            float2 d1_37 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_37);
                            float2 d2_37 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_37);
                            float2 dw_37 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_37);
                            float2 ng_37 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_37);
                            float2 nu_37 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_37);
                            float _rcp_713 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_37, d2_37), dw_37).x);
                            float _rcp_714 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_37, d2_37), dw_37).y);
                            float2 r_76 = make_float2(_rcp_713, _rcp_714);
                            float2 r_75 = r_76;
                            float2 _f32x2_mul_a_715 = mul_f32x2_noftz(ng_37, nu_37);
                            float2 _f32x2_mul_b_716 = mul_f32x2_noftz(r_75, make_float2(bl, bl));
                            float2 m_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_715), "l"(*(const unsigned long long*)&_f32x2_mul_b_716));
                            float2 ns_37 = make_float2(((x_38.x < 0.0f) ? p_37.x : 1.0f), ((x_38.y < 0.0f) ? p_37.y : 1.0f));
                            gate_packed_1[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_37, ns_37))) | sign_38;
                            float2 x_39 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[7]));
                            float2 v_39 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[7]));
                            unsigned int sign_39 = (gate_packed_1[7] ^ up_packed_1[7]) & 2147516416u;
                            float _abs_717;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_717) : "f"(x_39.x));
                            float _abs_718;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_718) : "f"(x_39.y));
                            float2 ax_38 = make_float2(_abs_717, _abs_718);
                            float _abs_719;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_719) : "f"(v_39.x));
                            float _abs_720;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_720) : "f"(v_39.y));
                            float2 av_38 = make_float2(_abs_719, _abs_720);
                            float2 _f32x2_mul_b_721 = make_float2(k1, k1);
                            float2 te_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_38) : "l"(*(const unsigned long long*)&ax_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_721));
                            float2 _f32x2_mul_b_722 = make_float2(k2, k2);
                            float2 tw_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_38) : "l"(*(const unsigned long long*)&av_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_722));
                            float _exp2_723 = approx_exp2(te_38.x);
                            float _exp2_724 = approx_exp2(te_38.y);
                            float2 e_38 = make_float2(_exp2_723, _exp2_724);
                            float _exp2_725 = approx_exp2(tw_38.x);
                            float _exp2_726 = approx_exp2(tw_38.y);
                            float2 w_38 = make_float2(_exp2_725, _exp2_726);
                            float2 _f32x2_mul_b_727 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_38) : "l"(*(const unsigned long long*)&ax_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_727));
                            float _exp2_728 = approx_exp2_noftz(tp_38.x);
                            float _exp2_729 = approx_exp2_noftz(tp_38.y);
                            float2 p_38 = make_float2(_exp2_728, _exp2_729);
                            float2 d1_38 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_38);
                            float2 d2_38 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_38);
                            float2 dw_38 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_38);
                            float2 ng_38 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_38);
                            float2 nu_38 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_38);
                            float _rcp_730 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_38, d2_38), dw_38).x);
                            float _rcp_731 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_38, d2_38), dw_38).y);
                            float2 r_78 = make_float2(_rcp_730, _rcp_731);
                            float2 r_77 = r_78;
                            float2 _f32x2_mul_a_732 = mul_f32x2_noftz(ng_38, nu_38);
                            float2 _f32x2_mul_b_733 = mul_f32x2_noftz(r_77, make_float2(bl, bl));
                            float2 m_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_732), "l"(*(const unsigned long long*)&_f32x2_mul_b_733));
                            float2 ns_38 = make_float2(((x_39.x < 0.0f) ? p_38.x : 1.0f), ((x_39.y < 0.0f) ? p_38.y : 1.0f));
                            gate_packed_1[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_38, ns_38))) | sign_39;
                            float2 x_40 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[8]));
                            float2 v_40 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[8]));
                            unsigned int sign_40 = (gate_packed_1[8] ^ up_packed_1[8]) & 2147516416u;
                            float _abs_734;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_734) : "f"(x_40.x));
                            float _abs_735;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_735) : "f"(x_40.y));
                            float2 ax_39 = make_float2(_abs_734, _abs_735);
                            float _abs_736;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_736) : "f"(v_40.x));
                            float _abs_737;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_737) : "f"(v_40.y));
                            float2 av_39 = make_float2(_abs_736, _abs_737);
                            float2 _f32x2_mul_b_738 = make_float2(k1, k1);
                            float2 te_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_39) : "l"(*(const unsigned long long*)&ax_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_738));
                            float2 _f32x2_mul_b_739 = make_float2(k2, k2);
                            float2 tw_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_39) : "l"(*(const unsigned long long*)&av_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_739));
                            float _exp2_740 = approx_exp2(te_39.x);
                            float _exp2_741 = approx_exp2(te_39.y);
                            float2 e_39 = make_float2(_exp2_740, _exp2_741);
                            float _exp2_742 = approx_exp2(tw_39.x);
                            float _exp2_743 = approx_exp2(tw_39.y);
                            float2 w_39 = make_float2(_exp2_742, _exp2_743);
                            float2 _f32x2_mul_b_744 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_39) : "l"(*(const unsigned long long*)&ax_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_744));
                            float _exp2_745 = approx_exp2_noftz(tp_39.x);
                            float _exp2_746 = approx_exp2_noftz(tp_39.y);
                            float2 p_39 = make_float2(_exp2_745, _exp2_746);
                            float2 d1_39 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_39);
                            float2 d2_39 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_39);
                            float2 dw_39 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_39);
                            float2 ng_39 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_39);
                            float2 nu_39 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_39);
                            float _rcp_747 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_39, d2_39), dw_39).x);
                            float _rcp_748 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_39, d2_39), dw_39).y);
                            float2 r_80 = make_float2(_rcp_747, _rcp_748);
                            float2 r_79 = r_80;
                            float2 _f32x2_mul_a_749 = mul_f32x2_noftz(ng_39, nu_39);
                            float2 _f32x2_mul_b_750 = mul_f32x2_noftz(r_79, make_float2(bl, bl));
                            float2 m_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_749), "l"(*(const unsigned long long*)&_f32x2_mul_b_750));
                            float2 ns_39 = make_float2(((x_40.x < 0.0f) ? p_39.x : 1.0f), ((x_40.y < 0.0f) ? p_39.y : 1.0f));
                            gate_packed_1[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_39, ns_39))) | sign_40;
                            float2 x_41 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[9]));
                            float2 v_41 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[9]));
                            unsigned int sign_41 = (gate_packed_1[9] ^ up_packed_1[9]) & 2147516416u;
                            float _abs_751;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_751) : "f"(x_41.x));
                            float _abs_752;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_752) : "f"(x_41.y));
                            float2 ax_40 = make_float2(_abs_751, _abs_752);
                            float _abs_753;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_753) : "f"(v_41.x));
                            float _abs_754;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_754) : "f"(v_41.y));
                            float2 av_40 = make_float2(_abs_753, _abs_754);
                            float2 _f32x2_mul_b_755 = make_float2(k1, k1);
                            float2 te_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_40) : "l"(*(const unsigned long long*)&ax_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_755));
                            float2 _f32x2_mul_b_756 = make_float2(k2, k2);
                            float2 tw_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_40) : "l"(*(const unsigned long long*)&av_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_756));
                            float _exp2_757 = approx_exp2(te_40.x);
                            float _exp2_758 = approx_exp2(te_40.y);
                            float2 e_40 = make_float2(_exp2_757, _exp2_758);
                            float _exp2_759 = approx_exp2(tw_40.x);
                            float _exp2_760 = approx_exp2(tw_40.y);
                            float2 w_40 = make_float2(_exp2_759, _exp2_760);
                            float2 _f32x2_mul_b_761 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_40) : "l"(*(const unsigned long long*)&ax_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_761));
                            float _exp2_762 = approx_exp2_noftz(tp_40.x);
                            float _exp2_763 = approx_exp2_noftz(tp_40.y);
                            float2 p_40 = make_float2(_exp2_762, _exp2_763);
                            float2 d1_40 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_40);
                            float2 d2_40 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_40);
                            float2 dw_40 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_40);
                            float2 ng_40 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_40);
                            float2 nu_40 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_40);
                            float _rcp_764 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_40, d2_40), dw_40).x);
                            float _rcp_765 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_40, d2_40), dw_40).y);
                            float2 r_82 = make_float2(_rcp_764, _rcp_765);
                            float2 r_81 = r_82;
                            float2 _f32x2_mul_a_766 = mul_f32x2_noftz(ng_40, nu_40);
                            float2 _f32x2_mul_b_767 = mul_f32x2_noftz(r_81, make_float2(bl, bl));
                            float2 m_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_766), "l"(*(const unsigned long long*)&_f32x2_mul_b_767));
                            float2 ns_40 = make_float2(((x_41.x < 0.0f) ? p_40.x : 1.0f), ((x_41.y < 0.0f) ? p_40.y : 1.0f));
                            gate_packed_1[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_40, ns_40))) | sign_41;
                            float2 x_42 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[10]));
                            float2 v_42 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[10]));
                            unsigned int sign_42 = (gate_packed_1[10] ^ up_packed_1[10]) & 2147516416u;
                            float _abs_768;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_768) : "f"(x_42.x));
                            float _abs_769;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_769) : "f"(x_42.y));
                            float2 ax_41 = make_float2(_abs_768, _abs_769);
                            float _abs_770;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_770) : "f"(v_42.x));
                            float _abs_771;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_771) : "f"(v_42.y));
                            float2 av_41 = make_float2(_abs_770, _abs_771);
                            float2 _f32x2_mul_b_772 = make_float2(k1, k1);
                            float2 te_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_41) : "l"(*(const unsigned long long*)&ax_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_772));
                            float2 _f32x2_mul_b_773 = make_float2(k2, k2);
                            float2 tw_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_41) : "l"(*(const unsigned long long*)&av_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_773));
                            float _exp2_774 = approx_exp2(te_41.x);
                            float _exp2_775 = approx_exp2(te_41.y);
                            float2 e_41 = make_float2(_exp2_774, _exp2_775);
                            float _exp2_776 = approx_exp2(tw_41.x);
                            float _exp2_777 = approx_exp2(tw_41.y);
                            float2 w_41 = make_float2(_exp2_776, _exp2_777);
                            float2 _f32x2_mul_b_778 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_41) : "l"(*(const unsigned long long*)&ax_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_778));
                            float _exp2_779 = approx_exp2_noftz(tp_41.x);
                            float _exp2_780 = approx_exp2_noftz(tp_41.y);
                            float2 p_41 = make_float2(_exp2_779, _exp2_780);
                            float2 d1_41 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_41);
                            float2 d2_41 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_41);
                            float2 dw_41 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_41);
                            float2 ng_41 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_41);
                            float2 nu_41 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_41);
                            float _rcp_781 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_41, d2_41), dw_41).x);
                            float _rcp_782 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_41, d2_41), dw_41).y);
                            float2 r_84 = make_float2(_rcp_781, _rcp_782);
                            float2 r_83 = r_84;
                            float2 _f32x2_mul_a_783 = mul_f32x2_noftz(ng_41, nu_41);
                            float2 _f32x2_mul_b_784 = mul_f32x2_noftz(r_83, make_float2(bl, bl));
                            float2 m_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_783), "l"(*(const unsigned long long*)&_f32x2_mul_b_784));
                            float2 ns_41 = make_float2(((x_42.x < 0.0f) ? p_41.x : 1.0f), ((x_42.y < 0.0f) ? p_41.y : 1.0f));
                            gate_packed_1[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_41, ns_41))) | sign_42;
                            float2 x_43 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[11]));
                            float2 v_43 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[11]));
                            unsigned int sign_43 = (gate_packed_1[11] ^ up_packed_1[11]) & 2147516416u;
                            float _abs_785;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_785) : "f"(x_43.x));
                            float _abs_786;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_786) : "f"(x_43.y));
                            float2 ax_42 = make_float2(_abs_785, _abs_786);
                            float _abs_787;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_787) : "f"(v_43.x));
                            float _abs_788;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_788) : "f"(v_43.y));
                            float2 av_42 = make_float2(_abs_787, _abs_788);
                            float2 _f32x2_mul_b_789 = make_float2(k1, k1);
                            float2 te_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_42) : "l"(*(const unsigned long long*)&ax_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_789));
                            float2 _f32x2_mul_b_790 = make_float2(k2, k2);
                            float2 tw_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_42) : "l"(*(const unsigned long long*)&av_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_790));
                            float _exp2_791 = approx_exp2(te_42.x);
                            float _exp2_792 = approx_exp2(te_42.y);
                            float2 e_42 = make_float2(_exp2_791, _exp2_792);
                            float _exp2_793 = approx_exp2(tw_42.x);
                            float _exp2_794 = approx_exp2(tw_42.y);
                            float2 w_42 = make_float2(_exp2_793, _exp2_794);
                            float2 _f32x2_mul_b_795 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_42) : "l"(*(const unsigned long long*)&ax_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_795));
                            float _exp2_796 = approx_exp2_noftz(tp_42.x);
                            float _exp2_797 = approx_exp2_noftz(tp_42.y);
                            float2 p_42 = make_float2(_exp2_796, _exp2_797);
                            float2 d1_42 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_42);
                            float2 d2_42 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_42);
                            float2 dw_42 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_42);
                            float2 ng_42 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_42);
                            float2 nu_42 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_42);
                            float _rcp_798 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_42, d2_42), dw_42).x);
                            float _rcp_799 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_42, d2_42), dw_42).y);
                            float2 r_86 = make_float2(_rcp_798, _rcp_799);
                            float2 r_85 = r_86;
                            float2 _f32x2_mul_a_800 = mul_f32x2_noftz(ng_42, nu_42);
                            float2 _f32x2_mul_b_801 = mul_f32x2_noftz(r_85, make_float2(bl, bl));
                            float2 m_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_800), "l"(*(const unsigned long long*)&_f32x2_mul_b_801));
                            float2 ns_42 = make_float2(((x_43.x < 0.0f) ? p_42.x : 1.0f), ((x_43.y < 0.0f) ? p_42.y : 1.0f));
                            gate_packed_1[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_42, ns_42))) | sign_43;
                            float2 x_44 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[12]));
                            float2 v_44 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[12]));
                            unsigned int sign_44 = (gate_packed_1[12] ^ up_packed_1[12]) & 2147516416u;
                            float _abs_802;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_802) : "f"(x_44.x));
                            float _abs_803;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_803) : "f"(x_44.y));
                            float2 ax_43 = make_float2(_abs_802, _abs_803);
                            float _abs_804;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_804) : "f"(v_44.x));
                            float _abs_805;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_805) : "f"(v_44.y));
                            float2 av_43 = make_float2(_abs_804, _abs_805);
                            float2 _f32x2_mul_b_806 = make_float2(k1, k1);
                            float2 te_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_43) : "l"(*(const unsigned long long*)&ax_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_806));
                            float2 _f32x2_mul_b_807 = make_float2(k2, k2);
                            float2 tw_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_43) : "l"(*(const unsigned long long*)&av_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_807));
                            float _exp2_808 = approx_exp2(te_43.x);
                            float _exp2_809 = approx_exp2(te_43.y);
                            float2 e_43 = make_float2(_exp2_808, _exp2_809);
                            float _exp2_810 = approx_exp2(tw_43.x);
                            float _exp2_811 = approx_exp2(tw_43.y);
                            float2 w_43 = make_float2(_exp2_810, _exp2_811);
                            float2 _f32x2_mul_b_812 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_43) : "l"(*(const unsigned long long*)&ax_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_812));
                            float _exp2_813 = approx_exp2_noftz(tp_43.x);
                            float _exp2_814 = approx_exp2_noftz(tp_43.y);
                            float2 p_43 = make_float2(_exp2_813, _exp2_814);
                            float2 d1_43 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_43);
                            float2 d2_43 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_43);
                            float2 dw_43 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_43);
                            float2 ng_43 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_43);
                            float2 nu_43 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_43);
                            float _rcp_815 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_43, d2_43), dw_43).x);
                            float _rcp_816 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_43, d2_43), dw_43).y);
                            float2 r_88 = make_float2(_rcp_815, _rcp_816);
                            float2 r_87 = r_88;
                            float2 _f32x2_mul_a_817 = mul_f32x2_noftz(ng_43, nu_43);
                            float2 _f32x2_mul_b_818 = mul_f32x2_noftz(r_87, make_float2(bl, bl));
                            float2 m_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_817), "l"(*(const unsigned long long*)&_f32x2_mul_b_818));
                            float2 ns_43 = make_float2(((x_44.x < 0.0f) ? p_43.x : 1.0f), ((x_44.y < 0.0f) ? p_43.y : 1.0f));
                            gate_packed_1[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_43, ns_43))) | sign_44;
                            float2 x_45 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[13]));
                            float2 v_45 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[13]));
                            unsigned int sign_45 = (gate_packed_1[13] ^ up_packed_1[13]) & 2147516416u;
                            float _abs_819;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_819) : "f"(x_45.x));
                            float _abs_820;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_820) : "f"(x_45.y));
                            float2 ax_44 = make_float2(_abs_819, _abs_820);
                            float _abs_821;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_821) : "f"(v_45.x));
                            float _abs_822;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_822) : "f"(v_45.y));
                            float2 av_44 = make_float2(_abs_821, _abs_822);
                            float2 _f32x2_mul_b_823 = make_float2(k1, k1);
                            float2 te_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_44) : "l"(*(const unsigned long long*)&ax_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_823));
                            float2 _f32x2_mul_b_824 = make_float2(k2, k2);
                            float2 tw_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_44) : "l"(*(const unsigned long long*)&av_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_824));
                            float _exp2_825 = approx_exp2(te_44.x);
                            float _exp2_826 = approx_exp2(te_44.y);
                            float2 e_44 = make_float2(_exp2_825, _exp2_826);
                            float _exp2_827 = approx_exp2(tw_44.x);
                            float _exp2_828 = approx_exp2(tw_44.y);
                            float2 w_44 = make_float2(_exp2_827, _exp2_828);
                            float2 _f32x2_mul_b_829 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_44) : "l"(*(const unsigned long long*)&ax_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_829));
                            float _exp2_830 = approx_exp2_noftz(tp_44.x);
                            float _exp2_831 = approx_exp2_noftz(tp_44.y);
                            float2 p_44 = make_float2(_exp2_830, _exp2_831);
                            float2 d1_44 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_44);
                            float2 d2_44 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_44);
                            float2 dw_44 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_44);
                            float2 ng_44 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_44);
                            float2 nu_44 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_44);
                            float _rcp_832 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_44, d2_44), dw_44).x);
                            float _rcp_833 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_44, d2_44), dw_44).y);
                            float2 r_90 = make_float2(_rcp_832, _rcp_833);
                            float2 r_89 = r_90;
                            float2 _f32x2_mul_a_834 = mul_f32x2_noftz(ng_44, nu_44);
                            float2 _f32x2_mul_b_835 = mul_f32x2_noftz(r_89, make_float2(bl, bl));
                            float2 m_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_834), "l"(*(const unsigned long long*)&_f32x2_mul_b_835));
                            float2 ns_44 = make_float2(((x_45.x < 0.0f) ? p_44.x : 1.0f), ((x_45.y < 0.0f) ? p_44.y : 1.0f));
                            gate_packed_1[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_44, ns_44))) | sign_45;
                            float2 x_46 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[14]));
                            float2 v_46 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[14]));
                            unsigned int sign_46 = (gate_packed_1[14] ^ up_packed_1[14]) & 2147516416u;
                            float _abs_836;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_836) : "f"(x_46.x));
                            float _abs_837;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_837) : "f"(x_46.y));
                            float2 ax_45 = make_float2(_abs_836, _abs_837);
                            float _abs_838;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_838) : "f"(v_46.x));
                            float _abs_839;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_839) : "f"(v_46.y));
                            float2 av_45 = make_float2(_abs_838, _abs_839);
                            float2 _f32x2_mul_b_840 = make_float2(k1, k1);
                            float2 te_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_45) : "l"(*(const unsigned long long*)&ax_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_840));
                            float2 _f32x2_mul_b_841 = make_float2(k2, k2);
                            float2 tw_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_45) : "l"(*(const unsigned long long*)&av_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_841));
                            float _exp2_842 = approx_exp2(te_45.x);
                            float _exp2_843 = approx_exp2(te_45.y);
                            float2 e_45 = make_float2(_exp2_842, _exp2_843);
                            float _exp2_844 = approx_exp2(tw_45.x);
                            float _exp2_845 = approx_exp2(tw_45.y);
                            float2 w_45 = make_float2(_exp2_844, _exp2_845);
                            float2 _f32x2_mul_b_846 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_45) : "l"(*(const unsigned long long*)&ax_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_846));
                            float _exp2_847 = approx_exp2_noftz(tp_45.x);
                            float _exp2_848 = approx_exp2_noftz(tp_45.y);
                            float2 p_45 = make_float2(_exp2_847, _exp2_848);
                            float2 d1_45 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_45);
                            float2 d2_45 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_45);
                            float2 dw_45 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_45);
                            float2 ng_45 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_45);
                            float2 nu_45 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_45);
                            float _rcp_849 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_45, d2_45), dw_45).x);
                            float _rcp_850 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_45, d2_45), dw_45).y);
                            float2 r_92 = make_float2(_rcp_849, _rcp_850);
                            float2 r_91 = r_92;
                            float2 _f32x2_mul_a_851 = mul_f32x2_noftz(ng_45, nu_45);
                            float2 _f32x2_mul_b_852 = mul_f32x2_noftz(r_91, make_float2(bl, bl));
                            float2 m_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_851), "l"(*(const unsigned long long*)&_f32x2_mul_b_852));
                            float2 ns_45 = make_float2(((x_46.x < 0.0f) ? p_45.x : 1.0f), ((x_46.y < 0.0f) ? p_45.y : 1.0f));
                            gate_packed_1[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_45, ns_45))) | sign_46;
                            float2 x_47 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[15]));
                            float2 v_47 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[15]));
                            unsigned int sign_47 = (gate_packed_1[15] ^ up_packed_1[15]) & 2147516416u;
                            float _abs_853;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_853) : "f"(x_47.x));
                            float _abs_854;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_854) : "f"(x_47.y));
                            float2 ax_46 = make_float2(_abs_853, _abs_854);
                            float _abs_855;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_855) : "f"(v_47.x));
                            float _abs_856;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_856) : "f"(v_47.y));
                            float2 av_46 = make_float2(_abs_855, _abs_856);
                            float2 _f32x2_mul_b_857 = make_float2(k1, k1);
                            float2 te_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_46) : "l"(*(const unsigned long long*)&ax_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_857));
                            float2 _f32x2_mul_b_858 = make_float2(k2, k2);
                            float2 tw_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_46) : "l"(*(const unsigned long long*)&av_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_858));
                            float _exp2_859 = approx_exp2(te_46.x);
                            float _exp2_860 = approx_exp2(te_46.y);
                            float2 e_46 = make_float2(_exp2_859, _exp2_860);
                            float _exp2_861 = approx_exp2(tw_46.x);
                            float _exp2_862 = approx_exp2(tw_46.y);
                            float2 w_46 = make_float2(_exp2_861, _exp2_862);
                            float2 _f32x2_mul_b_863 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_46) : "l"(*(const unsigned long long*)&ax_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_863));
                            float _exp2_864 = approx_exp2_noftz(tp_46.x);
                            float _exp2_865 = approx_exp2_noftz(tp_46.y);
                            float2 p_46 = make_float2(_exp2_864, _exp2_865);
                            float2 d1_46 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_46);
                            float2 d2_46 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_46);
                            float2 dw_46 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_46);
                            float2 ng_46 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_46);
                            float2 nu_46 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_46);
                            float _rcp_866 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_46, d2_46), dw_46).x);
                            float _rcp_867 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_46, d2_46), dw_46).y);
                            float2 r_94 = make_float2(_rcp_866, _rcp_867);
                            float2 r_93 = r_94;
                            float2 _f32x2_mul_a_868 = mul_f32x2_noftz(ng_46, nu_46);
                            float2 _f32x2_mul_b_869 = mul_f32x2_noftz(r_93, make_float2(bl, bl));
                            float2 m_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_868), "l"(*(const unsigned long long*)&_f32x2_mul_b_869));
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
                                    int row_14 = warp_9 * 32 + half_10 * 16 + lane_9 % 16;
                                    int col_11 = col_tile_10 * 16 + lane_9 / 16 * 8;
                                    unsigned int address_10 = d_smem_addr + 8192 + (unsigned int)((row_14 * 32 + col_11) * 2);
                                    address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                    int offset_11 = half_10 * 8 + col_tile_10 * 4;
                                    uint32_t _stmatrix_addr_870 = static_cast<uint32_t>(address_10);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_870), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_11 + 3]))
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
                            float _abs_871;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_871) : "f"(x_48.x));
                            float _abs_872;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_872) : "f"(x_48.y));
                            float2 ax_47 = make_float2(_abs_871, _abs_872);
                            float _abs_873;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_873) : "f"(v_48.x));
                            float _abs_874;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_874) : "f"(v_48.y));
                            float2 av_47 = make_float2(_abs_873, _abs_874);
                            float2 _f32x2_mul_b_875 = make_float2(k1, k1);
                            float2 te_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_47) : "l"(*(const unsigned long long*)&ax_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_875));
                            float2 _f32x2_mul_b_876 = make_float2(k2, k2);
                            float2 tw_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_47) : "l"(*(const unsigned long long*)&av_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_876));
                            float _exp2_877 = approx_exp2(te_47.x);
                            float _exp2_878 = approx_exp2(te_47.y);
                            float2 e_47 = make_float2(_exp2_877, _exp2_878);
                            float _exp2_879 = approx_exp2(tw_47.x);
                            float _exp2_880 = approx_exp2(tw_47.y);
                            float2 w_47 = make_float2(_exp2_879, _exp2_880);
                            float2 _f32x2_mul_b_881 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_47) : "l"(*(const unsigned long long*)&ax_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_881));
                            float _exp2_882 = approx_exp2_noftz(tp_47.x);
                            float _exp2_883 = approx_exp2_noftz(tp_47.y);
                            float2 p_47 = make_float2(_exp2_882, _exp2_883);
                            float2 d1_47 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_47);
                            float2 d2_47 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_47);
                            float2 dw_47 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_47);
                            float2 ng_47 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_47);
                            float2 nu_47 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_47);
                            float _rcp_884 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_47, d2_47), dw_47).x);
                            float _rcp_885 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_47, d2_47), dw_47).y);
                            float2 r_96 = make_float2(_rcp_884, _rcp_885);
                            float2 r_95 = r_96;
                            float2 _f32x2_mul_a_886 = mul_f32x2_noftz(ng_47, nu_47);
                            float2 _f32x2_mul_b_887 = mul_f32x2_noftz(r_95, make_float2(bl, bl));
                            float2 m_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_886), "l"(*(const unsigned long long*)&_f32x2_mul_b_887));
                            float2 ns_47 = make_float2(((x_48.x < 0.0f) ? p_47.x : 1.0f), ((x_48.y < 0.0f) ? p_47.y : 1.0f));
                            gate_packed_1[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_47, ns_47))) | sign_48;
                            float2 x_49 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[17]));
                            float2 v_49 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[17]));
                            unsigned int sign_49 = (gate_packed_1[17] ^ up_packed_1[17]) & 2147516416u;
                            float _abs_888;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_888) : "f"(x_49.x));
                            float _abs_889;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_889) : "f"(x_49.y));
                            float2 ax_48 = make_float2(_abs_888, _abs_889);
                            float _abs_890;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_890) : "f"(v_49.x));
                            float _abs_891;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_891) : "f"(v_49.y));
                            float2 av_48 = make_float2(_abs_890, _abs_891);
                            float2 _f32x2_mul_b_892 = make_float2(k1, k1);
                            float2 te_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_48) : "l"(*(const unsigned long long*)&ax_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_892));
                            float2 _f32x2_mul_b_893 = make_float2(k2, k2);
                            float2 tw_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_48) : "l"(*(const unsigned long long*)&av_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_893));
                            float _exp2_894 = approx_exp2(te_48.x);
                            float _exp2_895 = approx_exp2(te_48.y);
                            float2 e_48 = make_float2(_exp2_894, _exp2_895);
                            float _exp2_896 = approx_exp2(tw_48.x);
                            float _exp2_897 = approx_exp2(tw_48.y);
                            float2 w_48 = make_float2(_exp2_896, _exp2_897);
                            float2 _f32x2_mul_b_898 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_48) : "l"(*(const unsigned long long*)&ax_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_898));
                            float _exp2_899 = approx_exp2_noftz(tp_48.x);
                            float _exp2_900 = approx_exp2_noftz(tp_48.y);
                            float2 p_48 = make_float2(_exp2_899, _exp2_900);
                            float2 d1_48 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_48);
                            float2 d2_48 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_48);
                            float2 dw_48 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_48);
                            float2 ng_48 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_48);
                            float2 nu_48 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_48);
                            float _rcp_901 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_48, d2_48), dw_48).x);
                            float _rcp_902 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_48, d2_48), dw_48).y);
                            float2 r_98 = make_float2(_rcp_901, _rcp_902);
                            float2 r_97 = r_98;
                            float2 _f32x2_mul_a_903 = mul_f32x2_noftz(ng_48, nu_48);
                            float2 _f32x2_mul_b_904 = mul_f32x2_noftz(r_97, make_float2(bl, bl));
                            float2 m_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_903), "l"(*(const unsigned long long*)&_f32x2_mul_b_904));
                            float2 ns_48 = make_float2(((x_49.x < 0.0f) ? p_48.x : 1.0f), ((x_49.y < 0.0f) ? p_48.y : 1.0f));
                            gate_packed_1[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_48, ns_48))) | sign_49;
                            float2 x_50 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[18]));
                            float2 v_50 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[18]));
                            unsigned int sign_50 = (gate_packed_1[18] ^ up_packed_1[18]) & 2147516416u;
                            float _abs_905;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_905) : "f"(x_50.x));
                            float _abs_906;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_906) : "f"(x_50.y));
                            float2 ax_49 = make_float2(_abs_905, _abs_906);
                            float _abs_907;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_907) : "f"(v_50.x));
                            float _abs_908;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_908) : "f"(v_50.y));
                            float2 av_49 = make_float2(_abs_907, _abs_908);
                            float2 _f32x2_mul_b_909 = make_float2(k1, k1);
                            float2 te_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_49) : "l"(*(const unsigned long long*)&ax_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_909));
                            float2 _f32x2_mul_b_910 = make_float2(k2, k2);
                            float2 tw_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_49) : "l"(*(const unsigned long long*)&av_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_910));
                            float _exp2_911 = approx_exp2(te_49.x);
                            float _exp2_912 = approx_exp2(te_49.y);
                            float2 e_49 = make_float2(_exp2_911, _exp2_912);
                            float _exp2_913 = approx_exp2(tw_49.x);
                            float _exp2_914 = approx_exp2(tw_49.y);
                            float2 w_49 = make_float2(_exp2_913, _exp2_914);
                            float2 _f32x2_mul_b_915 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_49) : "l"(*(const unsigned long long*)&ax_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_915));
                            float _exp2_916 = approx_exp2_noftz(tp_49.x);
                            float _exp2_917 = approx_exp2_noftz(tp_49.y);
                            float2 p_49 = make_float2(_exp2_916, _exp2_917);
                            float2 d1_49 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_49);
                            float2 d2_49 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_49);
                            float2 dw_49 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_49);
                            float2 ng_49 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_49);
                            float2 nu_49 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_49);
                            float _rcp_918 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_49, d2_49), dw_49).x);
                            float _rcp_919 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_49, d2_49), dw_49).y);
                            float2 r_100 = make_float2(_rcp_918, _rcp_919);
                            float2 r_99 = r_100;
                            float2 _f32x2_mul_a_920 = mul_f32x2_noftz(ng_49, nu_49);
                            float2 _f32x2_mul_b_921 = mul_f32x2_noftz(r_99, make_float2(bl, bl));
                            float2 m_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_920), "l"(*(const unsigned long long*)&_f32x2_mul_b_921));
                            float2 ns_49 = make_float2(((x_50.x < 0.0f) ? p_49.x : 1.0f), ((x_50.y < 0.0f) ? p_49.y : 1.0f));
                            gate_packed_1[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_49, ns_49))) | sign_50;
                            float2 x_51 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[19]));
                            float2 v_51 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[19]));
                            unsigned int sign_51 = (gate_packed_1[19] ^ up_packed_1[19]) & 2147516416u;
                            float _abs_922;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_922) : "f"(x_51.x));
                            float _abs_923;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_923) : "f"(x_51.y));
                            float2 ax_50 = make_float2(_abs_922, _abs_923);
                            float _abs_924;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_924) : "f"(v_51.x));
                            float _abs_925;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_925) : "f"(v_51.y));
                            float2 av_50 = make_float2(_abs_924, _abs_925);
                            float2 _f32x2_mul_b_926 = make_float2(k1, k1);
                            float2 te_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_50) : "l"(*(const unsigned long long*)&ax_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_926));
                            float2 _f32x2_mul_b_927 = make_float2(k2, k2);
                            float2 tw_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_50) : "l"(*(const unsigned long long*)&av_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_927));
                            float _exp2_928 = approx_exp2(te_50.x);
                            float _exp2_929 = approx_exp2(te_50.y);
                            float2 e_50 = make_float2(_exp2_928, _exp2_929);
                            float _exp2_930 = approx_exp2(tw_50.x);
                            float _exp2_931 = approx_exp2(tw_50.y);
                            float2 w_50 = make_float2(_exp2_930, _exp2_931);
                            float2 _f32x2_mul_b_932 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_50) : "l"(*(const unsigned long long*)&ax_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_932));
                            float _exp2_933 = approx_exp2_noftz(tp_50.x);
                            float _exp2_934 = approx_exp2_noftz(tp_50.y);
                            float2 p_50 = make_float2(_exp2_933, _exp2_934);
                            float2 d1_50 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_50);
                            float2 d2_50 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_50);
                            float2 dw_50 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_50);
                            float2 ng_50 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_50);
                            float2 nu_50 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_50);
                            float _rcp_935 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_50, d2_50), dw_50).x);
                            float _rcp_936 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_50, d2_50), dw_50).y);
                            float2 r_102 = make_float2(_rcp_935, _rcp_936);
                            float2 r_101 = r_102;
                            float2 _f32x2_mul_a_937 = mul_f32x2_noftz(ng_50, nu_50);
                            float2 _f32x2_mul_b_938 = mul_f32x2_noftz(r_101, make_float2(bl, bl));
                            float2 m_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_937), "l"(*(const unsigned long long*)&_f32x2_mul_b_938));
                            float2 ns_50 = make_float2(((x_51.x < 0.0f) ? p_50.x : 1.0f), ((x_51.y < 0.0f) ? p_50.y : 1.0f));
                            gate_packed_1[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_50, ns_50))) | sign_51;
                            float2 x_52 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[20]));
                            float2 v_52 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[20]));
                            unsigned int sign_52 = (gate_packed_1[20] ^ up_packed_1[20]) & 2147516416u;
                            float _abs_939;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_939) : "f"(x_52.x));
                            float _abs_940;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_940) : "f"(x_52.y));
                            float2 ax_51 = make_float2(_abs_939, _abs_940);
                            float _abs_941;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_941) : "f"(v_52.x));
                            float _abs_942;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_942) : "f"(v_52.y));
                            float2 av_51 = make_float2(_abs_941, _abs_942);
                            float2 _f32x2_mul_b_943 = make_float2(k1, k1);
                            float2 te_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_51) : "l"(*(const unsigned long long*)&ax_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_943));
                            float2 _f32x2_mul_b_944 = make_float2(k2, k2);
                            float2 tw_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_51) : "l"(*(const unsigned long long*)&av_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_944));
                            float _exp2_945 = approx_exp2(te_51.x);
                            float _exp2_946 = approx_exp2(te_51.y);
                            float2 e_51 = make_float2(_exp2_945, _exp2_946);
                            float _exp2_947 = approx_exp2(tw_51.x);
                            float _exp2_948 = approx_exp2(tw_51.y);
                            float2 w_51 = make_float2(_exp2_947, _exp2_948);
                            float2 _f32x2_mul_b_949 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_51) : "l"(*(const unsigned long long*)&ax_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_949));
                            float _exp2_950 = approx_exp2_noftz(tp_51.x);
                            float _exp2_951 = approx_exp2_noftz(tp_51.y);
                            float2 p_51 = make_float2(_exp2_950, _exp2_951);
                            float2 d1_51 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_51);
                            float2 d2_51 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_51);
                            float2 dw_51 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_51);
                            float2 ng_51 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_51);
                            float2 nu_51 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_51);
                            float _rcp_952 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_51, d2_51), dw_51).x);
                            float _rcp_953 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_51, d2_51), dw_51).y);
                            float2 r_104 = make_float2(_rcp_952, _rcp_953);
                            float2 r_103 = r_104;
                            float2 _f32x2_mul_a_954 = mul_f32x2_noftz(ng_51, nu_51);
                            float2 _f32x2_mul_b_955 = mul_f32x2_noftz(r_103, make_float2(bl, bl));
                            float2 m_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_954), "l"(*(const unsigned long long*)&_f32x2_mul_b_955));
                            float2 ns_51 = make_float2(((x_52.x < 0.0f) ? p_51.x : 1.0f), ((x_52.y < 0.0f) ? p_51.y : 1.0f));
                            gate_packed_1[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_51, ns_51))) | sign_52;
                            float2 x_53 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[21]));
                            float2 v_53 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[21]));
                            unsigned int sign_53 = (gate_packed_1[21] ^ up_packed_1[21]) & 2147516416u;
                            float _abs_956;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_956) : "f"(x_53.x));
                            float _abs_957;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_957) : "f"(x_53.y));
                            float2 ax_52 = make_float2(_abs_956, _abs_957);
                            float _abs_958;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_958) : "f"(v_53.x));
                            float _abs_959;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_959) : "f"(v_53.y));
                            float2 av_52 = make_float2(_abs_958, _abs_959);
                            float2 _f32x2_mul_b_960 = make_float2(k1, k1);
                            float2 te_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_52) : "l"(*(const unsigned long long*)&ax_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_960));
                            float2 _f32x2_mul_b_961 = make_float2(k2, k2);
                            float2 tw_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_52) : "l"(*(const unsigned long long*)&av_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_961));
                            float _exp2_962 = approx_exp2(te_52.x);
                            float _exp2_963 = approx_exp2(te_52.y);
                            float2 e_52 = make_float2(_exp2_962, _exp2_963);
                            float _exp2_964 = approx_exp2(tw_52.x);
                            float _exp2_965 = approx_exp2(tw_52.y);
                            float2 w_52 = make_float2(_exp2_964, _exp2_965);
                            float2 _f32x2_mul_b_966 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_52) : "l"(*(const unsigned long long*)&ax_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_966));
                            float _exp2_967 = approx_exp2_noftz(tp_52.x);
                            float _exp2_968 = approx_exp2_noftz(tp_52.y);
                            float2 p_52 = make_float2(_exp2_967, _exp2_968);
                            float2 d1_52 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_52);
                            float2 d2_52 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_52);
                            float2 dw_52 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_52);
                            float2 ng_52 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_52);
                            float2 nu_52 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_52);
                            float _rcp_969 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_52, d2_52), dw_52).x);
                            float _rcp_970 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_52, d2_52), dw_52).y);
                            float2 r_106 = make_float2(_rcp_969, _rcp_970);
                            float2 r_105 = r_106;
                            float2 _f32x2_mul_a_971 = mul_f32x2_noftz(ng_52, nu_52);
                            float2 _f32x2_mul_b_972 = mul_f32x2_noftz(r_105, make_float2(bl, bl));
                            float2 m_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_971), "l"(*(const unsigned long long*)&_f32x2_mul_b_972));
                            float2 ns_52 = make_float2(((x_53.x < 0.0f) ? p_52.x : 1.0f), ((x_53.y < 0.0f) ? p_52.y : 1.0f));
                            gate_packed_1[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_52, ns_52))) | sign_53;
                            float2 x_54 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[22]));
                            float2 v_54 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[22]));
                            unsigned int sign_54 = (gate_packed_1[22] ^ up_packed_1[22]) & 2147516416u;
                            float _abs_973;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_973) : "f"(x_54.x));
                            float _abs_974;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_974) : "f"(x_54.y));
                            float2 ax_53 = make_float2(_abs_973, _abs_974);
                            float _abs_975;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_975) : "f"(v_54.x));
                            float _abs_976;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_976) : "f"(v_54.y));
                            float2 av_53 = make_float2(_abs_975, _abs_976);
                            float2 _f32x2_mul_b_977 = make_float2(k1, k1);
                            float2 te_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_53) : "l"(*(const unsigned long long*)&ax_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_977));
                            float2 _f32x2_mul_b_978 = make_float2(k2, k2);
                            float2 tw_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_53) : "l"(*(const unsigned long long*)&av_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_978));
                            float _exp2_979 = approx_exp2(te_53.x);
                            float _exp2_980 = approx_exp2(te_53.y);
                            float2 e_53 = make_float2(_exp2_979, _exp2_980);
                            float _exp2_981 = approx_exp2(tw_53.x);
                            float _exp2_982 = approx_exp2(tw_53.y);
                            float2 w_53 = make_float2(_exp2_981, _exp2_982);
                            float2 _f32x2_mul_b_983 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_53) : "l"(*(const unsigned long long*)&ax_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_983));
                            float _exp2_984 = approx_exp2_noftz(tp_53.x);
                            float _exp2_985 = approx_exp2_noftz(tp_53.y);
                            float2 p_53 = make_float2(_exp2_984, _exp2_985);
                            float2 d1_53 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_53);
                            float2 d2_53 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_53);
                            float2 dw_53 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_53);
                            float2 ng_53 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_53);
                            float2 nu_53 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_53);
                            float _rcp_986 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_53, d2_53), dw_53).x);
                            float _rcp_987 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_53, d2_53), dw_53).y);
                            float2 r_108 = make_float2(_rcp_986, _rcp_987);
                            float2 r_107 = r_108;
                            float2 _f32x2_mul_a_988 = mul_f32x2_noftz(ng_53, nu_53);
                            float2 _f32x2_mul_b_989 = mul_f32x2_noftz(r_107, make_float2(bl, bl));
                            float2 m_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_988), "l"(*(const unsigned long long*)&_f32x2_mul_b_989));
                            float2 ns_53 = make_float2(((x_54.x < 0.0f) ? p_53.x : 1.0f), ((x_54.y < 0.0f) ? p_53.y : 1.0f));
                            gate_packed_1[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_53, ns_53))) | sign_54;
                            float2 x_55 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[23]));
                            float2 v_55 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[23]));
                            unsigned int sign_55 = (gate_packed_1[23] ^ up_packed_1[23]) & 2147516416u;
                            float _abs_990;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_990) : "f"(x_55.x));
                            float _abs_991;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_991) : "f"(x_55.y));
                            float2 ax_54 = make_float2(_abs_990, _abs_991);
                            float _abs_992;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_992) : "f"(v_55.x));
                            float _abs_993;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_993) : "f"(v_55.y));
                            float2 av_54 = make_float2(_abs_992, _abs_993);
                            float2 _f32x2_mul_b_994 = make_float2(k1, k1);
                            float2 te_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_54) : "l"(*(const unsigned long long*)&ax_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_994));
                            float2 _f32x2_mul_b_995 = make_float2(k2, k2);
                            float2 tw_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_54) : "l"(*(const unsigned long long*)&av_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_995));
                            float _exp2_996 = approx_exp2(te_54.x);
                            float _exp2_997 = approx_exp2(te_54.y);
                            float2 e_54 = make_float2(_exp2_996, _exp2_997);
                            float _exp2_998 = approx_exp2(tw_54.x);
                            float _exp2_999 = approx_exp2(tw_54.y);
                            float2 w_54 = make_float2(_exp2_998, _exp2_999);
                            float2 _f32x2_mul_b_1000 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_54) : "l"(*(const unsigned long long*)&ax_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_1000));
                            float _exp2_1001 = approx_exp2_noftz(tp_54.x);
                            float _exp2_1002 = approx_exp2_noftz(tp_54.y);
                            float2 p_54 = make_float2(_exp2_1001, _exp2_1002);
                            float2 d1_54 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_54);
                            float2 d2_54 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_54);
                            float2 dw_54 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_54);
                            float2 ng_54 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_54);
                            float2 nu_54 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_54);
                            float _rcp_1003 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_54, d2_54), dw_54).x);
                            float _rcp_1004 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_54, d2_54), dw_54).y);
                            float2 r_110 = make_float2(_rcp_1003, _rcp_1004);
                            float2 r_109 = r_110;
                            float2 _f32x2_mul_a_1005 = mul_f32x2_noftz(ng_54, nu_54);
                            float2 _f32x2_mul_b_1006 = mul_f32x2_noftz(r_109, make_float2(bl, bl));
                            float2 m_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1005), "l"(*(const unsigned long long*)&_f32x2_mul_b_1006));
                            float2 ns_54 = make_float2(((x_55.x < 0.0f) ? p_54.x : 1.0f), ((x_55.y < 0.0f) ? p_54.y : 1.0f));
                            gate_packed_1[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_54, ns_54))) | sign_55;
                            float2 x_56 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[24]));
                            float2 v_56 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[24]));
                            unsigned int sign_56 = (gate_packed_1[24] ^ up_packed_1[24]) & 2147516416u;
                            float _abs_1007;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1007) : "f"(x_56.x));
                            float _abs_1008;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1008) : "f"(x_56.y));
                            float2 ax_55 = make_float2(_abs_1007, _abs_1008);
                            float _abs_1009;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1009) : "f"(v_56.x));
                            float _abs_1010;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1010) : "f"(v_56.y));
                            float2 av_55 = make_float2(_abs_1009, _abs_1010);
                            float2 _f32x2_mul_b_1011 = make_float2(k1, k1);
                            float2 te_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_55) : "l"(*(const unsigned long long*)&ax_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_1011));
                            float2 _f32x2_mul_b_1012 = make_float2(k2, k2);
                            float2 tw_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_55) : "l"(*(const unsigned long long*)&av_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_1012));
                            float _exp2_1013 = approx_exp2(te_55.x);
                            float _exp2_1014 = approx_exp2(te_55.y);
                            float2 e_55 = make_float2(_exp2_1013, _exp2_1014);
                            float _exp2_1015 = approx_exp2(tw_55.x);
                            float _exp2_1016 = approx_exp2(tw_55.y);
                            float2 w_55 = make_float2(_exp2_1015, _exp2_1016);
                            float2 _f32x2_mul_b_1017 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_55) : "l"(*(const unsigned long long*)&ax_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_1017));
                            float _exp2_1018 = approx_exp2_noftz(tp_55.x);
                            float _exp2_1019 = approx_exp2_noftz(tp_55.y);
                            float2 p_55 = make_float2(_exp2_1018, _exp2_1019);
                            float2 d1_55 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_55);
                            float2 d2_55 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_55);
                            float2 dw_55 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_55);
                            float2 ng_55 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_55);
                            float2 nu_55 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_55);
                            float _rcp_1020 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_55, d2_55), dw_55).x);
                            float _rcp_1021 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_55, d2_55), dw_55).y);
                            float2 r_112 = make_float2(_rcp_1020, _rcp_1021);
                            float2 r_111 = r_112;
                            float2 _f32x2_mul_a_1022 = mul_f32x2_noftz(ng_55, nu_55);
                            float2 _f32x2_mul_b_1023 = mul_f32x2_noftz(r_111, make_float2(bl, bl));
                            float2 m_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1022), "l"(*(const unsigned long long*)&_f32x2_mul_b_1023));
                            float2 ns_55 = make_float2(((x_56.x < 0.0f) ? p_55.x : 1.0f), ((x_56.y < 0.0f) ? p_55.y : 1.0f));
                            gate_packed_1[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_55, ns_55))) | sign_56;
                            float2 x_57 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[25]));
                            float2 v_57 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[25]));
                            unsigned int sign_57 = (gate_packed_1[25] ^ up_packed_1[25]) & 2147516416u;
                            float _abs_1024;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1024) : "f"(x_57.x));
                            float _abs_1025;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1025) : "f"(x_57.y));
                            float2 ax_56 = make_float2(_abs_1024, _abs_1025);
                            float _abs_1026;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1026) : "f"(v_57.x));
                            float _abs_1027;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1027) : "f"(v_57.y));
                            float2 av_56 = make_float2(_abs_1026, _abs_1027);
                            float2 _f32x2_mul_b_1028 = make_float2(k1, k1);
                            float2 te_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_56) : "l"(*(const unsigned long long*)&ax_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_1028));
                            float2 _f32x2_mul_b_1029 = make_float2(k2, k2);
                            float2 tw_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_56) : "l"(*(const unsigned long long*)&av_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_1029));
                            float _exp2_1030 = approx_exp2(te_56.x);
                            float _exp2_1031 = approx_exp2(te_56.y);
                            float2 e_56 = make_float2(_exp2_1030, _exp2_1031);
                            float _exp2_1032 = approx_exp2(tw_56.x);
                            float _exp2_1033 = approx_exp2(tw_56.y);
                            float2 w_56 = make_float2(_exp2_1032, _exp2_1033);
                            float2 _f32x2_mul_b_1034 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_56) : "l"(*(const unsigned long long*)&ax_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_1034));
                            float _exp2_1035 = approx_exp2_noftz(tp_56.x);
                            float _exp2_1036 = approx_exp2_noftz(tp_56.y);
                            float2 p_56 = make_float2(_exp2_1035, _exp2_1036);
                            float2 d1_56 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_56);
                            float2 d2_56 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_56);
                            float2 dw_56 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_56);
                            float2 ng_56 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_56);
                            float2 nu_56 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_56);
                            float _rcp_1037 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_56, d2_56), dw_56).x);
                            float _rcp_1038 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_56, d2_56), dw_56).y);
                            float2 r_114 = make_float2(_rcp_1037, _rcp_1038);
                            float2 r_113 = r_114;
                            float2 _f32x2_mul_a_1039 = mul_f32x2_noftz(ng_56, nu_56);
                            float2 _f32x2_mul_b_1040 = mul_f32x2_noftz(r_113, make_float2(bl, bl));
                            float2 m_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1039), "l"(*(const unsigned long long*)&_f32x2_mul_b_1040));
                            float2 ns_56 = make_float2(((x_57.x < 0.0f) ? p_56.x : 1.0f), ((x_57.y < 0.0f) ? p_56.y : 1.0f));
                            gate_packed_1[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_56, ns_56))) | sign_57;
                            float2 x_58 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[26]));
                            float2 v_58 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[26]));
                            unsigned int sign_58 = (gate_packed_1[26] ^ up_packed_1[26]) & 2147516416u;
                            float _abs_1041;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1041) : "f"(x_58.x));
                            float _abs_1042;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1042) : "f"(x_58.y));
                            float2 ax_57 = make_float2(_abs_1041, _abs_1042);
                            float _abs_1043;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1043) : "f"(v_58.x));
                            float _abs_1044;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1044) : "f"(v_58.y));
                            float2 av_57 = make_float2(_abs_1043, _abs_1044);
                            float2 _f32x2_mul_b_1045 = make_float2(k1, k1);
                            float2 te_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_57) : "l"(*(const unsigned long long*)&ax_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_1045));
                            float2 _f32x2_mul_b_1046 = make_float2(k2, k2);
                            float2 tw_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_57) : "l"(*(const unsigned long long*)&av_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_1046));
                            float _exp2_1047 = approx_exp2(te_57.x);
                            float _exp2_1048 = approx_exp2(te_57.y);
                            float2 e_57 = make_float2(_exp2_1047, _exp2_1048);
                            float _exp2_1049 = approx_exp2(tw_57.x);
                            float _exp2_1050 = approx_exp2(tw_57.y);
                            float2 w_57 = make_float2(_exp2_1049, _exp2_1050);
                            float2 _f32x2_mul_b_1051 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_57) : "l"(*(const unsigned long long*)&ax_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_1051));
                            float _exp2_1052 = approx_exp2_noftz(tp_57.x);
                            float _exp2_1053 = approx_exp2_noftz(tp_57.y);
                            float2 p_57 = make_float2(_exp2_1052, _exp2_1053);
                            float2 d1_57 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_57);
                            float2 d2_57 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_57);
                            float2 dw_57 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_57);
                            float2 ng_57 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_57);
                            float2 nu_57 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_57);
                            float _rcp_1054 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_57, d2_57), dw_57).x);
                            float _rcp_1055 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_57, d2_57), dw_57).y);
                            float2 r_116 = make_float2(_rcp_1054, _rcp_1055);
                            float2 r_115 = r_116;
                            float2 _f32x2_mul_a_1056 = mul_f32x2_noftz(ng_57, nu_57);
                            float2 _f32x2_mul_b_1057 = mul_f32x2_noftz(r_115, make_float2(bl, bl));
                            float2 m_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1056), "l"(*(const unsigned long long*)&_f32x2_mul_b_1057));
                            float2 ns_57 = make_float2(((x_58.x < 0.0f) ? p_57.x : 1.0f), ((x_58.y < 0.0f) ? p_57.y : 1.0f));
                            gate_packed_1[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_57, ns_57))) | sign_58;
                            float2 x_59 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[27]));
                            float2 v_59 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[27]));
                            unsigned int sign_59 = (gate_packed_1[27] ^ up_packed_1[27]) & 2147516416u;
                            float _abs_1058;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1058) : "f"(x_59.x));
                            float _abs_1059;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1059) : "f"(x_59.y));
                            float2 ax_58 = make_float2(_abs_1058, _abs_1059);
                            float _abs_1060;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1060) : "f"(v_59.x));
                            float _abs_1061;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1061) : "f"(v_59.y));
                            float2 av_58 = make_float2(_abs_1060, _abs_1061);
                            float2 _f32x2_mul_b_1062 = make_float2(k1, k1);
                            float2 te_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_58) : "l"(*(const unsigned long long*)&ax_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_1062));
                            float2 _f32x2_mul_b_1063 = make_float2(k2, k2);
                            float2 tw_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_58) : "l"(*(const unsigned long long*)&av_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_1063));
                            float _exp2_1064 = approx_exp2(te_58.x);
                            float _exp2_1065 = approx_exp2(te_58.y);
                            float2 e_58 = make_float2(_exp2_1064, _exp2_1065);
                            float _exp2_1066 = approx_exp2(tw_58.x);
                            float _exp2_1067 = approx_exp2(tw_58.y);
                            float2 w_58 = make_float2(_exp2_1066, _exp2_1067);
                            float2 _f32x2_mul_b_1068 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_58) : "l"(*(const unsigned long long*)&ax_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_1068));
                            float _exp2_1069 = approx_exp2_noftz(tp_58.x);
                            float _exp2_1070 = approx_exp2_noftz(tp_58.y);
                            float2 p_58 = make_float2(_exp2_1069, _exp2_1070);
                            float2 d1_58 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_58);
                            float2 d2_58 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_58);
                            float2 dw_58 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_58);
                            float2 ng_58 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_58);
                            float2 nu_58 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_58);
                            float _rcp_1071 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_58, d2_58), dw_58).x);
                            float _rcp_1072 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_58, d2_58), dw_58).y);
                            float2 r_118 = make_float2(_rcp_1071, _rcp_1072);
                            float2 r_117 = r_118;
                            float2 _f32x2_mul_a_1073 = mul_f32x2_noftz(ng_58, nu_58);
                            float2 _f32x2_mul_b_1074 = mul_f32x2_noftz(r_117, make_float2(bl, bl));
                            float2 m_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1073), "l"(*(const unsigned long long*)&_f32x2_mul_b_1074));
                            float2 ns_58 = make_float2(((x_59.x < 0.0f) ? p_58.x : 1.0f), ((x_59.y < 0.0f) ? p_58.y : 1.0f));
                            gate_packed_1[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_58, ns_58))) | sign_59;
                            float2 x_60 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[28]));
                            float2 v_60 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[28]));
                            unsigned int sign_60 = (gate_packed_1[28] ^ up_packed_1[28]) & 2147516416u;
                            float _abs_1075;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1075) : "f"(x_60.x));
                            float _abs_1076;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1076) : "f"(x_60.y));
                            float2 ax_59 = make_float2(_abs_1075, _abs_1076);
                            float _abs_1077;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1077) : "f"(v_60.x));
                            float _abs_1078;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1078) : "f"(v_60.y));
                            float2 av_59 = make_float2(_abs_1077, _abs_1078);
                            float2 _f32x2_mul_b_1079 = make_float2(k1, k1);
                            float2 te_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_59) : "l"(*(const unsigned long long*)&ax_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_1079));
                            float2 _f32x2_mul_b_1080 = make_float2(k2, k2);
                            float2 tw_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_59) : "l"(*(const unsigned long long*)&av_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_1080));
                            float _exp2_1081 = approx_exp2(te_59.x);
                            float _exp2_1082 = approx_exp2(te_59.y);
                            float2 e_59 = make_float2(_exp2_1081, _exp2_1082);
                            float _exp2_1083 = approx_exp2(tw_59.x);
                            float _exp2_1084 = approx_exp2(tw_59.y);
                            float2 w_59 = make_float2(_exp2_1083, _exp2_1084);
                            float2 _f32x2_mul_b_1085 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_59) : "l"(*(const unsigned long long*)&ax_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_1085));
                            float _exp2_1086 = approx_exp2_noftz(tp_59.x);
                            float _exp2_1087 = approx_exp2_noftz(tp_59.y);
                            float2 p_59 = make_float2(_exp2_1086, _exp2_1087);
                            float2 d1_59 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_59);
                            float2 d2_59 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_59);
                            float2 dw_59 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_59);
                            float2 ng_59 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_59);
                            float2 nu_59 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_59);
                            float _rcp_1088 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_59, d2_59), dw_59).x);
                            float _rcp_1089 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_59, d2_59), dw_59).y);
                            float2 r_120 = make_float2(_rcp_1088, _rcp_1089);
                            float2 r_119 = r_120;
                            float2 _f32x2_mul_a_1090 = mul_f32x2_noftz(ng_59, nu_59);
                            float2 _f32x2_mul_b_1091 = mul_f32x2_noftz(r_119, make_float2(bl, bl));
                            float2 m_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1090), "l"(*(const unsigned long long*)&_f32x2_mul_b_1091));
                            float2 ns_59 = make_float2(((x_60.x < 0.0f) ? p_59.x : 1.0f), ((x_60.y < 0.0f) ? p_59.y : 1.0f));
                            gate_packed_1[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_59, ns_59))) | sign_60;
                            float2 x_61 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[29]));
                            float2 v_61 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[29]));
                            unsigned int sign_61 = (gate_packed_1[29] ^ up_packed_1[29]) & 2147516416u;
                            float _abs_1092;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1092) : "f"(x_61.x));
                            float _abs_1093;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1093) : "f"(x_61.y));
                            float2 ax_60 = make_float2(_abs_1092, _abs_1093);
                            float _abs_1094;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1094) : "f"(v_61.x));
                            float _abs_1095;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1095) : "f"(v_61.y));
                            float2 av_60 = make_float2(_abs_1094, _abs_1095);
                            float2 _f32x2_mul_b_1096 = make_float2(k1, k1);
                            float2 te_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_60) : "l"(*(const unsigned long long*)&ax_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_1096));
                            float2 _f32x2_mul_b_1097 = make_float2(k2, k2);
                            float2 tw_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_60) : "l"(*(const unsigned long long*)&av_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_1097));
                            float _exp2_1098 = approx_exp2(te_60.x);
                            float _exp2_1099 = approx_exp2(te_60.y);
                            float2 e_60 = make_float2(_exp2_1098, _exp2_1099);
                            float _exp2_1100 = approx_exp2(tw_60.x);
                            float _exp2_1101 = approx_exp2(tw_60.y);
                            float2 w_60 = make_float2(_exp2_1100, _exp2_1101);
                            float2 _f32x2_mul_b_1102 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_60) : "l"(*(const unsigned long long*)&ax_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_1102));
                            float _exp2_1103 = approx_exp2_noftz(tp_60.x);
                            float _exp2_1104 = approx_exp2_noftz(tp_60.y);
                            float2 p_60 = make_float2(_exp2_1103, _exp2_1104);
                            float2 d1_60 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_60);
                            float2 d2_60 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_60);
                            float2 dw_60 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_60);
                            float2 ng_60 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_60);
                            float2 nu_60 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_60);
                            float _rcp_1105 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_60, d2_60), dw_60).x);
                            float _rcp_1106 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_60, d2_60), dw_60).y);
                            float2 r_122 = make_float2(_rcp_1105, _rcp_1106);
                            float2 r_121 = r_122;
                            float2 _f32x2_mul_a_1107 = mul_f32x2_noftz(ng_60, nu_60);
                            float2 _f32x2_mul_b_1108 = mul_f32x2_noftz(r_121, make_float2(bl, bl));
                            float2 m_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1108));
                            float2 ns_60 = make_float2(((x_61.x < 0.0f) ? p_60.x : 1.0f), ((x_61.y < 0.0f) ? p_60.y : 1.0f));
                            gate_packed_1[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_60, ns_60))) | sign_61;
                            float2 x_62 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[30]));
                            float2 v_62 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[30]));
                            unsigned int sign_62 = (gate_packed_1[30] ^ up_packed_1[30]) & 2147516416u;
                            float _abs_1109;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1109) : "f"(x_62.x));
                            float _abs_1110;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1110) : "f"(x_62.y));
                            float2 ax_61 = make_float2(_abs_1109, _abs_1110);
                            float _abs_1111;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1111) : "f"(v_62.x));
                            float _abs_1112;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1112) : "f"(v_62.y));
                            float2 av_61 = make_float2(_abs_1111, _abs_1112);
                            float2 _f32x2_mul_b_1113 = make_float2(k1, k1);
                            float2 te_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_61) : "l"(*(const unsigned long long*)&ax_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_1113));
                            float2 _f32x2_mul_b_1114 = make_float2(k2, k2);
                            float2 tw_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_61) : "l"(*(const unsigned long long*)&av_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_1114));
                            float _exp2_1115 = approx_exp2(te_61.x);
                            float _exp2_1116 = approx_exp2(te_61.y);
                            float2 e_61 = make_float2(_exp2_1115, _exp2_1116);
                            float _exp2_1117 = approx_exp2(tw_61.x);
                            float _exp2_1118 = approx_exp2(tw_61.y);
                            float2 w_61 = make_float2(_exp2_1117, _exp2_1118);
                            float2 _f32x2_mul_b_1119 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_61) : "l"(*(const unsigned long long*)&ax_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_1119));
                            float _exp2_1120 = approx_exp2_noftz(tp_61.x);
                            float _exp2_1121 = approx_exp2_noftz(tp_61.y);
                            float2 p_61 = make_float2(_exp2_1120, _exp2_1121);
                            float2 d1_61 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_61);
                            float2 d2_61 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_61);
                            float2 dw_61 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_61);
                            float2 ng_61 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_61);
                            float2 nu_61 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_61);
                            float _rcp_1122 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_61, d2_61), dw_61).x);
                            float _rcp_1123 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_61, d2_61), dw_61).y);
                            float2 r_124 = make_float2(_rcp_1122, _rcp_1123);
                            float2 r_123 = r_124;
                            float2 _f32x2_mul_a_1124 = mul_f32x2_noftz(ng_61, nu_61);
                            float2 _f32x2_mul_b_1125 = mul_f32x2_noftz(r_123, make_float2(bl, bl));
                            float2 m_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1124), "l"(*(const unsigned long long*)&_f32x2_mul_b_1125));
                            float2 ns_61 = make_float2(((x_62.x < 0.0f) ? p_61.x : 1.0f), ((x_62.y < 0.0f) ? p_61.y : 1.0f));
                            gate_packed_1[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_61, ns_61))) | sign_62;
                            float2 x_63 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[31]));
                            float2 v_63 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[31]));
                            unsigned int sign_63 = (gate_packed_1[31] ^ up_packed_1[31]) & 2147516416u;
                            float _abs_1126;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1126) : "f"(x_63.x));
                            float _abs_1127;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1127) : "f"(x_63.y));
                            float2 ax_62 = make_float2(_abs_1126, _abs_1127);
                            float _abs_1128;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1128) : "f"(v_63.x));
                            float _abs_1129;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1129) : "f"(v_63.y));
                            float2 av_62 = make_float2(_abs_1128, _abs_1129);
                            float2 _f32x2_mul_b_1130 = make_float2(k1, k1);
                            float2 te_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_62) : "l"(*(const unsigned long long*)&ax_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_1130));
                            float2 _f32x2_mul_b_1131 = make_float2(k2, k2);
                            float2 tw_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_62) : "l"(*(const unsigned long long*)&av_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_1131));
                            float _exp2_1132 = approx_exp2(te_62.x);
                            float _exp2_1133 = approx_exp2(te_62.y);
                            float2 e_62 = make_float2(_exp2_1132, _exp2_1133);
                            float _exp2_1134 = approx_exp2(tw_62.x);
                            float _exp2_1135 = approx_exp2(tw_62.y);
                            float2 w_62 = make_float2(_exp2_1134, _exp2_1135);
                            float2 _f32x2_mul_b_1136 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_62) : "l"(*(const unsigned long long*)&ax_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_1136));
                            float _exp2_1137 = approx_exp2_noftz(tp_62.x);
                            float _exp2_1138 = approx_exp2_noftz(tp_62.y);
                            float2 p_62 = make_float2(_exp2_1137, _exp2_1138);
                            float2 d1_62 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_62);
                            float2 d2_62 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_62);
                            float2 dw_62 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_62);
                            float2 ng_62 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_62);
                            float2 nu_62 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_62);
                            float _rcp_1139 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_62, d2_62), dw_62).x);
                            float _rcp_1140 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_62, d2_62), dw_62).y);
                            float2 r_126 = make_float2(_rcp_1139, _rcp_1140);
                            float2 r_125 = r_126;
                            float2 _f32x2_mul_a_1141 = mul_f32x2_noftz(ng_62, nu_62);
                            float2 _f32x2_mul_b_1142 = mul_f32x2_noftz(r_125, make_float2(bl, bl));
                            float2 m_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1141), "l"(*(const unsigned long long*)&_f32x2_mul_b_1142));
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
                                    int row_15 = warp_10 * 32 + half_11 * 16 + lane_10 % 16;
                                    int col_12 = col_tile_11 * 16 + lane_10 / 16 * 8;
                                    unsigned int address_11 = d_smem_addr + 16384 + (unsigned int)((row_15 * 32 + col_12) * 2);
                                    address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                    int offset_12 = 16 + half_11 * 8 + col_tile_11 * 4;
                                    uint32_t _stmatrix_addr_1143 = static_cast<uint32_t>(address_11);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1143), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_12 + 3]))
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
                                    int row_16 = warp_11 * 32 + half_12 * 16 + lane_11 % 16;
                                    int col_13 = col_tile_12 * 16 + lane_11 / 16 * 8;
                                    unsigned int address_12 = d_smem_addr + (unsigned int)((row_16 * 32 + col_13) * 2);
                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                    int offset_13 = half_12 * 8 + col_tile_12 * 4;
                                    uint32_t _stmatrix_addr_1144 = static_cast<uint32_t>(address_12);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1144), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_13 + 3]))
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
                                    int row_17 = warp_12 * 32 + half_13 * 16 + lane_12 % 16;
                                    int col_14 = col_tile_13 * 16 + lane_12 / 16 * 8;
                                    unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)((row_17 * 32 + col_14) * 2);
                                    address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                    int offset_14 = half_13 * 8 + col_tile_13 * 4;
                                    uint32_t _stmatrix_addr_1145 = static_cast<uint32_t>(address_13);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1145), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_14 + 3]))
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
                                    int row_18 = warp_13 * 32 + half_14 * 16 + lane_13 % 16;
                                    int col_15 = col_tile_14 * 16 + lane_13 / 16 * 8;
                                    unsigned int address_14 = d_smem_addr + 16384 + (unsigned int)((row_18 * 32 + col_15) * 2);
                                    address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                    int offset_15 = 16 + half_14 * 8 + col_tile_14 * 4;
                                    uint32_t _stmatrix_addr_1146 = static_cast<uint32_t>(address_14);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1146), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 3]))
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
                                    int row_19 = warp_14 * 32 + half_15 * 16 + lane_14 % 16;
                                    int col_16 = col_tile_15 * 16 + lane_14 / 16 * 8;
                                    unsigned int address_15 = d_smem_addr + (unsigned int)((row_19 * 32 + col_16) * 2);
                                    address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                    int offset_16 = 16 + half_15 * 8 + col_tile_15 * 4;
                                    uint32_t _stmatrix_addr_1147 = static_cast<uint32_t>(address_15);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1147), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 3]))
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
                            float _abs_1148;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1148) : "f"(x_64.x));
                            float _abs_1149;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1149) : "f"(x_64.y));
                            float2 ax_63 = make_float2(_abs_1148, _abs_1149);
                            float _abs_1150;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1150) : "f"(v_64.x));
                            float _abs_1151;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1151) : "f"(v_64.y));
                            float2 av_63 = make_float2(_abs_1150, _abs_1151);
                            float2 _f32x2_mul_b_1152 = make_float2(k1, k1);
                            float2 te_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_63) : "l"(*(const unsigned long long*)&ax_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_1152));
                            float2 _f32x2_mul_b_1153 = make_float2(k2, k2);
                            float2 tw_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_63) : "l"(*(const unsigned long long*)&av_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_1153));
                            float _exp2_1154 = approx_exp2(te_63.x);
                            float _exp2_1155 = approx_exp2(te_63.y);
                            float2 e_63 = make_float2(_exp2_1154, _exp2_1155);
                            float _exp2_1156 = approx_exp2(tw_63.x);
                            float _exp2_1157 = approx_exp2(tw_63.y);
                            float2 w_63 = make_float2(_exp2_1156, _exp2_1157);
                            float2 _f32x2_mul_b_1158 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_63) : "l"(*(const unsigned long long*)&ax_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_1158));
                            float _exp2_1159 = approx_exp2_noftz(tp_63.x);
                            float _exp2_1160 = approx_exp2_noftz(tp_63.y);
                            float2 p_63 = make_float2(_exp2_1159, _exp2_1160);
                            float2 d1_63 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_63);
                            float2 d2_63 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_63);
                            float2 dw_63 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_63);
                            float2 ng_63 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_63);
                            float2 nu_63 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_63);
                            float _rcp_1161 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_63, d2_63), dw_63).x);
                            float _rcp_1162 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_63, d2_63), dw_63).y);
                            float2 r_128 = make_float2(_rcp_1161, _rcp_1162);
                            float2 r_127 = r_128;
                            float2 _f32x2_mul_a_1163 = mul_f32x2_noftz(ng_63, nu_63);
                            float2 _f32x2_mul_b_1164 = mul_f32x2_noftz(r_127, make_float2(bl, bl));
                            float2 m_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1163), "l"(*(const unsigned long long*)&_f32x2_mul_b_1164));
                            float2 ns_63 = make_float2(((x_64.x < 0.0f) ? p_63.x : 1.0f), ((x_64.y < 0.0f) ? p_63.y : 1.0f));
                            gate_packed_2[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_63, ns_63))) | sign_64;
                            float2 x_65 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[1]));
                            float2 v_65 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[1]));
                            unsigned int sign_65 = (gate_packed_2[1] ^ up_packed_2[1]) & 2147516416u;
                            float _abs_1165;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1165) : "f"(x_65.x));
                            float _abs_1166;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1166) : "f"(x_65.y));
                            float2 ax_64 = make_float2(_abs_1165, _abs_1166);
                            float _abs_1167;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1167) : "f"(v_65.x));
                            float _abs_1168;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1168) : "f"(v_65.y));
                            float2 av_64 = make_float2(_abs_1167, _abs_1168);
                            float2 _f32x2_mul_b_1169 = make_float2(k1, k1);
                            float2 te_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_64) : "l"(*(const unsigned long long*)&ax_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_1169));
                            float2 _f32x2_mul_b_1170 = make_float2(k2, k2);
                            float2 tw_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_64) : "l"(*(const unsigned long long*)&av_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_1170));
                            float _exp2_1171 = approx_exp2(te_64.x);
                            float _exp2_1172 = approx_exp2(te_64.y);
                            float2 e_64 = make_float2(_exp2_1171, _exp2_1172);
                            float _exp2_1173 = approx_exp2(tw_64.x);
                            float _exp2_1174 = approx_exp2(tw_64.y);
                            float2 w_64 = make_float2(_exp2_1173, _exp2_1174);
                            float2 _f32x2_mul_b_1175 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_64) : "l"(*(const unsigned long long*)&ax_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_1175));
                            float _exp2_1176 = approx_exp2_noftz(tp_64.x);
                            float _exp2_1177 = approx_exp2_noftz(tp_64.y);
                            float2 p_64 = make_float2(_exp2_1176, _exp2_1177);
                            float2 d1_64 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_64);
                            float2 d2_64 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_64);
                            float2 dw_64 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_64);
                            float2 ng_64 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_64);
                            float2 nu_64 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_64);
                            float _rcp_1178 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_64, d2_64), dw_64).x);
                            float _rcp_1179 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_64, d2_64), dw_64).y);
                            float2 r_130 = make_float2(_rcp_1178, _rcp_1179);
                            float2 r_129 = r_130;
                            float2 _f32x2_mul_a_1180 = mul_f32x2_noftz(ng_64, nu_64);
                            float2 _f32x2_mul_b_1181 = mul_f32x2_noftz(r_129, make_float2(bl, bl));
                            float2 m_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1180), "l"(*(const unsigned long long*)&_f32x2_mul_b_1181));
                            float2 ns_64 = make_float2(((x_65.x < 0.0f) ? p_64.x : 1.0f), ((x_65.y < 0.0f) ? p_64.y : 1.0f));
                            gate_packed_2[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_64, ns_64))) | sign_65;
                            float2 x_66 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[2]));
                            float2 v_66 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[2]));
                            unsigned int sign_66 = (gate_packed_2[2] ^ up_packed_2[2]) & 2147516416u;
                            float _abs_1182;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1182) : "f"(x_66.x));
                            float _abs_1183;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1183) : "f"(x_66.y));
                            float2 ax_65 = make_float2(_abs_1182, _abs_1183);
                            float _abs_1184;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1184) : "f"(v_66.x));
                            float _abs_1185;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1185) : "f"(v_66.y));
                            float2 av_65 = make_float2(_abs_1184, _abs_1185);
                            float2 _f32x2_mul_b_1186 = make_float2(k1, k1);
                            float2 te_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_65) : "l"(*(const unsigned long long*)&ax_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_1186));
                            float2 _f32x2_mul_b_1187 = make_float2(k2, k2);
                            float2 tw_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_65) : "l"(*(const unsigned long long*)&av_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_1187));
                            float _exp2_1188 = approx_exp2(te_65.x);
                            float _exp2_1189 = approx_exp2(te_65.y);
                            float2 e_65 = make_float2(_exp2_1188, _exp2_1189);
                            float _exp2_1190 = approx_exp2(tw_65.x);
                            float _exp2_1191 = approx_exp2(tw_65.y);
                            float2 w_65 = make_float2(_exp2_1190, _exp2_1191);
                            float2 _f32x2_mul_b_1192 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_65) : "l"(*(const unsigned long long*)&ax_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_1192));
                            float _exp2_1193 = approx_exp2_noftz(tp_65.x);
                            float _exp2_1194 = approx_exp2_noftz(tp_65.y);
                            float2 p_65 = make_float2(_exp2_1193, _exp2_1194);
                            float2 d1_65 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_65);
                            float2 d2_65 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_65);
                            float2 dw_65 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_65);
                            float2 ng_65 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_65);
                            float2 nu_65 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_65);
                            float _rcp_1195 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_65, d2_65), dw_65).x);
                            float _rcp_1196 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_65, d2_65), dw_65).y);
                            float2 r_132 = make_float2(_rcp_1195, _rcp_1196);
                            float2 r_131 = r_132;
                            float2 _f32x2_mul_a_1197 = mul_f32x2_noftz(ng_65, nu_65);
                            float2 _f32x2_mul_b_1198 = mul_f32x2_noftz(r_131, make_float2(bl, bl));
                            float2 m_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1197), "l"(*(const unsigned long long*)&_f32x2_mul_b_1198));
                            float2 ns_65 = make_float2(((x_66.x < 0.0f) ? p_65.x : 1.0f), ((x_66.y < 0.0f) ? p_65.y : 1.0f));
                            gate_packed_2[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_65, ns_65))) | sign_66;
                            float2 x_67 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[3]));
                            float2 v_67 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[3]));
                            unsigned int sign_67 = (gate_packed_2[3] ^ up_packed_2[3]) & 2147516416u;
                            float _abs_1199;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1199) : "f"(x_67.x));
                            float _abs_1200;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1200) : "f"(x_67.y));
                            float2 ax_66 = make_float2(_abs_1199, _abs_1200);
                            float _abs_1201;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1201) : "f"(v_67.x));
                            float _abs_1202;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1202) : "f"(v_67.y));
                            float2 av_66 = make_float2(_abs_1201, _abs_1202);
                            float2 _f32x2_mul_b_1203 = make_float2(k1, k1);
                            float2 te_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_66) : "l"(*(const unsigned long long*)&ax_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_1203));
                            float2 _f32x2_mul_b_1204 = make_float2(k2, k2);
                            float2 tw_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_66) : "l"(*(const unsigned long long*)&av_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_1204));
                            float _exp2_1205 = approx_exp2(te_66.x);
                            float _exp2_1206 = approx_exp2(te_66.y);
                            float2 e_66 = make_float2(_exp2_1205, _exp2_1206);
                            float _exp2_1207 = approx_exp2(tw_66.x);
                            float _exp2_1208 = approx_exp2(tw_66.y);
                            float2 w_66 = make_float2(_exp2_1207, _exp2_1208);
                            float2 _f32x2_mul_b_1209 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_66) : "l"(*(const unsigned long long*)&ax_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_1209));
                            float _exp2_1210 = approx_exp2_noftz(tp_66.x);
                            float _exp2_1211 = approx_exp2_noftz(tp_66.y);
                            float2 p_66 = make_float2(_exp2_1210, _exp2_1211);
                            float2 d1_66 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_66);
                            float2 d2_66 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_66);
                            float2 dw_66 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_66);
                            float2 ng_66 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_66);
                            float2 nu_66 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_66);
                            float _rcp_1212 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_66, d2_66), dw_66).x);
                            float _rcp_1213 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_66, d2_66), dw_66).y);
                            float2 r_134 = make_float2(_rcp_1212, _rcp_1213);
                            float2 r_133 = r_134;
                            float2 _f32x2_mul_a_1214 = mul_f32x2_noftz(ng_66, nu_66);
                            float2 _f32x2_mul_b_1215 = mul_f32x2_noftz(r_133, make_float2(bl, bl));
                            float2 m_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1214), "l"(*(const unsigned long long*)&_f32x2_mul_b_1215));
                            float2 ns_66 = make_float2(((x_67.x < 0.0f) ? p_66.x : 1.0f), ((x_67.y < 0.0f) ? p_66.y : 1.0f));
                            gate_packed_2[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_66, ns_66))) | sign_67;
                            float2 x_68 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[4]));
                            float2 v_68 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[4]));
                            unsigned int sign_68 = (gate_packed_2[4] ^ up_packed_2[4]) & 2147516416u;
                            float _abs_1216;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1216) : "f"(x_68.x));
                            float _abs_1217;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1217) : "f"(x_68.y));
                            float2 ax_67 = make_float2(_abs_1216, _abs_1217);
                            float _abs_1218;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1218) : "f"(v_68.x));
                            float _abs_1219;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1219) : "f"(v_68.y));
                            float2 av_67 = make_float2(_abs_1218, _abs_1219);
                            float2 _f32x2_mul_b_1220 = make_float2(k1, k1);
                            float2 te_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_67) : "l"(*(const unsigned long long*)&ax_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_1220));
                            float2 _f32x2_mul_b_1221 = make_float2(k2, k2);
                            float2 tw_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_67) : "l"(*(const unsigned long long*)&av_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_1221));
                            float _exp2_1222 = approx_exp2(te_67.x);
                            float _exp2_1223 = approx_exp2(te_67.y);
                            float2 e_67 = make_float2(_exp2_1222, _exp2_1223);
                            float _exp2_1224 = approx_exp2(tw_67.x);
                            float _exp2_1225 = approx_exp2(tw_67.y);
                            float2 w_67 = make_float2(_exp2_1224, _exp2_1225);
                            float2 _f32x2_mul_b_1226 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_67) : "l"(*(const unsigned long long*)&ax_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_1226));
                            float _exp2_1227 = approx_exp2_noftz(tp_67.x);
                            float _exp2_1228 = approx_exp2_noftz(tp_67.y);
                            float2 p_67 = make_float2(_exp2_1227, _exp2_1228);
                            float2 d1_67 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_67);
                            float2 d2_67 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_67);
                            float2 dw_67 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_67);
                            float2 ng_67 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_67);
                            float2 nu_67 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_67);
                            float _rcp_1229 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_67, d2_67), dw_67).x);
                            float _rcp_1230 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_67, d2_67), dw_67).y);
                            float2 r_136 = make_float2(_rcp_1229, _rcp_1230);
                            float2 r_135 = r_136;
                            float2 _f32x2_mul_a_1231 = mul_f32x2_noftz(ng_67, nu_67);
                            float2 _f32x2_mul_b_1232 = mul_f32x2_noftz(r_135, make_float2(bl, bl));
                            float2 m_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_67) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1231), "l"(*(const unsigned long long*)&_f32x2_mul_b_1232));
                            float2 ns_67 = make_float2(((x_68.x < 0.0f) ? p_67.x : 1.0f), ((x_68.y < 0.0f) ? p_67.y : 1.0f));
                            gate_packed_2[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_67, ns_67))) | sign_68;
                            float2 x_69 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[5]));
                            float2 v_69 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[5]));
                            unsigned int sign_69 = (gate_packed_2[5] ^ up_packed_2[5]) & 2147516416u;
                            float _abs_1233;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1233) : "f"(x_69.x));
                            float _abs_1234;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1234) : "f"(x_69.y));
                            float2 ax_68 = make_float2(_abs_1233, _abs_1234);
                            float _abs_1235;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1235) : "f"(v_69.x));
                            float _abs_1236;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1236) : "f"(v_69.y));
                            float2 av_68 = make_float2(_abs_1235, _abs_1236);
                            float2 _f32x2_mul_b_1237 = make_float2(k1, k1);
                            float2 te_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_68) : "l"(*(const unsigned long long*)&ax_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_1237));
                            float2 _f32x2_mul_b_1238 = make_float2(k2, k2);
                            float2 tw_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_68) : "l"(*(const unsigned long long*)&av_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_1238));
                            float _exp2_1239 = approx_exp2(te_68.x);
                            float _exp2_1240 = approx_exp2(te_68.y);
                            float2 e_68 = make_float2(_exp2_1239, _exp2_1240);
                            float _exp2_1241 = approx_exp2(tw_68.x);
                            float _exp2_1242 = approx_exp2(tw_68.y);
                            float2 w_68 = make_float2(_exp2_1241, _exp2_1242);
                            float2 _f32x2_mul_b_1243 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_68) : "l"(*(const unsigned long long*)&ax_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_1243));
                            float _exp2_1244 = approx_exp2_noftz(tp_68.x);
                            float _exp2_1245 = approx_exp2_noftz(tp_68.y);
                            float2 p_68 = make_float2(_exp2_1244, _exp2_1245);
                            float2 d1_68 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_68);
                            float2 d2_68 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_68);
                            float2 dw_68 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_68);
                            float2 ng_68 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_68);
                            float2 nu_68 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_68);
                            float _rcp_1246 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_68, d2_68), dw_68).x);
                            float _rcp_1247 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_68, d2_68), dw_68).y);
                            float2 r_138 = make_float2(_rcp_1246, _rcp_1247);
                            float2 r_137 = r_138;
                            float2 _f32x2_mul_a_1248 = mul_f32x2_noftz(ng_68, nu_68);
                            float2 _f32x2_mul_b_1249 = mul_f32x2_noftz(r_137, make_float2(bl, bl));
                            float2 m_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_68) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1248), "l"(*(const unsigned long long*)&_f32x2_mul_b_1249));
                            float2 ns_68 = make_float2(((x_69.x < 0.0f) ? p_68.x : 1.0f), ((x_69.y < 0.0f) ? p_68.y : 1.0f));
                            gate_packed_2[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_68, ns_68))) | sign_69;
                            float2 x_70 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[6]));
                            float2 v_70 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[6]));
                            unsigned int sign_70 = (gate_packed_2[6] ^ up_packed_2[6]) & 2147516416u;
                            float _abs_1250;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1250) : "f"(x_70.x));
                            float _abs_1251;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1251) : "f"(x_70.y));
                            float2 ax_69 = make_float2(_abs_1250, _abs_1251);
                            float _abs_1252;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1252) : "f"(v_70.x));
                            float _abs_1253;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1253) : "f"(v_70.y));
                            float2 av_69 = make_float2(_abs_1252, _abs_1253);
                            float2 _f32x2_mul_b_1254 = make_float2(k1, k1);
                            float2 te_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_69) : "l"(*(const unsigned long long*)&ax_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_1254));
                            float2 _f32x2_mul_b_1255 = make_float2(k2, k2);
                            float2 tw_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_69) : "l"(*(const unsigned long long*)&av_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_1255));
                            float _exp2_1256 = approx_exp2(te_69.x);
                            float _exp2_1257 = approx_exp2(te_69.y);
                            float2 e_69 = make_float2(_exp2_1256, _exp2_1257);
                            float _exp2_1258 = approx_exp2(tw_69.x);
                            float _exp2_1259 = approx_exp2(tw_69.y);
                            float2 w_69 = make_float2(_exp2_1258, _exp2_1259);
                            float2 _f32x2_mul_b_1260 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_69) : "l"(*(const unsigned long long*)&ax_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_1260));
                            float _exp2_1261 = approx_exp2_noftz(tp_69.x);
                            float _exp2_1262 = approx_exp2_noftz(tp_69.y);
                            float2 p_69 = make_float2(_exp2_1261, _exp2_1262);
                            float2 d1_69 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_69);
                            float2 d2_69 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_69);
                            float2 dw_69 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_69);
                            float2 ng_69 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_69);
                            float2 nu_69 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_69);
                            float _rcp_1263 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_69, d2_69), dw_69).x);
                            float _rcp_1264 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_69, d2_69), dw_69).y);
                            float2 r_140 = make_float2(_rcp_1263, _rcp_1264);
                            float2 r_139 = r_140;
                            float2 _f32x2_mul_a_1265 = mul_f32x2_noftz(ng_69, nu_69);
                            float2 _f32x2_mul_b_1266 = mul_f32x2_noftz(r_139, make_float2(bl, bl));
                            float2 m_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_69) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1265), "l"(*(const unsigned long long*)&_f32x2_mul_b_1266));
                            float2 ns_69 = make_float2(((x_70.x < 0.0f) ? p_69.x : 1.0f), ((x_70.y < 0.0f) ? p_69.y : 1.0f));
                            gate_packed_2[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_69, ns_69))) | sign_70;
                            float2 x_71 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[7]));
                            float2 v_71 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[7]));
                            unsigned int sign_71 = (gate_packed_2[7] ^ up_packed_2[7]) & 2147516416u;
                            float _abs_1267;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1267) : "f"(x_71.x));
                            float _abs_1268;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1268) : "f"(x_71.y));
                            float2 ax_70 = make_float2(_abs_1267, _abs_1268);
                            float _abs_1269;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1269) : "f"(v_71.x));
                            float _abs_1270;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1270) : "f"(v_71.y));
                            float2 av_70 = make_float2(_abs_1269, _abs_1270);
                            float2 _f32x2_mul_b_1271 = make_float2(k1, k1);
                            float2 te_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_70) : "l"(*(const unsigned long long*)&ax_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_1271));
                            float2 _f32x2_mul_b_1272 = make_float2(k2, k2);
                            float2 tw_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_70) : "l"(*(const unsigned long long*)&av_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_1272));
                            float _exp2_1273 = approx_exp2(te_70.x);
                            float _exp2_1274 = approx_exp2(te_70.y);
                            float2 e_70 = make_float2(_exp2_1273, _exp2_1274);
                            float _exp2_1275 = approx_exp2(tw_70.x);
                            float _exp2_1276 = approx_exp2(tw_70.y);
                            float2 w_70 = make_float2(_exp2_1275, _exp2_1276);
                            float2 _f32x2_mul_b_1277 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_70) : "l"(*(const unsigned long long*)&ax_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_1277));
                            float _exp2_1278 = approx_exp2_noftz(tp_70.x);
                            float _exp2_1279 = approx_exp2_noftz(tp_70.y);
                            float2 p_70 = make_float2(_exp2_1278, _exp2_1279);
                            float2 d1_70 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_70);
                            float2 d2_70 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_70);
                            float2 dw_70 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_70);
                            float2 ng_70 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_70);
                            float2 nu_70 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_70);
                            float _rcp_1280 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_70, d2_70), dw_70).x);
                            float _rcp_1281 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_70, d2_70), dw_70).y);
                            float2 r_142 = make_float2(_rcp_1280, _rcp_1281);
                            float2 r_141 = r_142;
                            float2 _f32x2_mul_a_1282 = mul_f32x2_noftz(ng_70, nu_70);
                            float2 _f32x2_mul_b_1283 = mul_f32x2_noftz(r_141, make_float2(bl, bl));
                            float2 m_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_70) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1282), "l"(*(const unsigned long long*)&_f32x2_mul_b_1283));
                            float2 ns_70 = make_float2(((x_71.x < 0.0f) ? p_70.x : 1.0f), ((x_71.y < 0.0f) ? p_70.y : 1.0f));
                            gate_packed_2[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_70, ns_70))) | sign_71;
                            float2 x_72 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[8]));
                            float2 v_72 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[8]));
                            unsigned int sign_72 = (gate_packed_2[8] ^ up_packed_2[8]) & 2147516416u;
                            float _abs_1284;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1284) : "f"(x_72.x));
                            float _abs_1285;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1285) : "f"(x_72.y));
                            float2 ax_71 = make_float2(_abs_1284, _abs_1285);
                            float _abs_1286;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1286) : "f"(v_72.x));
                            float _abs_1287;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1287) : "f"(v_72.y));
                            float2 av_71 = make_float2(_abs_1286, _abs_1287);
                            float2 _f32x2_mul_b_1288 = make_float2(k1, k1);
                            float2 te_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_71) : "l"(*(const unsigned long long*)&ax_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_1288));
                            float2 _f32x2_mul_b_1289 = make_float2(k2, k2);
                            float2 tw_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_71) : "l"(*(const unsigned long long*)&av_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_1289));
                            float _exp2_1290 = approx_exp2(te_71.x);
                            float _exp2_1291 = approx_exp2(te_71.y);
                            float2 e_71 = make_float2(_exp2_1290, _exp2_1291);
                            float _exp2_1292 = approx_exp2(tw_71.x);
                            float _exp2_1293 = approx_exp2(tw_71.y);
                            float2 w_71 = make_float2(_exp2_1292, _exp2_1293);
                            float2 _f32x2_mul_b_1294 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_71) : "l"(*(const unsigned long long*)&ax_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_1294));
                            float _exp2_1295 = approx_exp2_noftz(tp_71.x);
                            float _exp2_1296 = approx_exp2_noftz(tp_71.y);
                            float2 p_71 = make_float2(_exp2_1295, _exp2_1296);
                            float2 d1_71 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_71);
                            float2 d2_71 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_71);
                            float2 dw_71 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_71);
                            float2 ng_71 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_71);
                            float2 nu_71 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_71);
                            float _rcp_1297 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_71, d2_71), dw_71).x);
                            float _rcp_1298 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_71, d2_71), dw_71).y);
                            float2 r_144 = make_float2(_rcp_1297, _rcp_1298);
                            float2 r_143 = r_144;
                            float2 _f32x2_mul_a_1299 = mul_f32x2_noftz(ng_71, nu_71);
                            float2 _f32x2_mul_b_1300 = mul_f32x2_noftz(r_143, make_float2(bl, bl));
                            float2 m_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_71) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1299), "l"(*(const unsigned long long*)&_f32x2_mul_b_1300));
                            float2 ns_71 = make_float2(((x_72.x < 0.0f) ? p_71.x : 1.0f), ((x_72.y < 0.0f) ? p_71.y : 1.0f));
                            gate_packed_2[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_71, ns_71))) | sign_72;
                            float2 x_73 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[9]));
                            float2 v_73 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[9]));
                            unsigned int sign_73 = (gate_packed_2[9] ^ up_packed_2[9]) & 2147516416u;
                            float _abs_1301;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1301) : "f"(x_73.x));
                            float _abs_1302;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1302) : "f"(x_73.y));
                            float2 ax_72 = make_float2(_abs_1301, _abs_1302);
                            float _abs_1303;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1303) : "f"(v_73.x));
                            float _abs_1304;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1304) : "f"(v_73.y));
                            float2 av_72 = make_float2(_abs_1303, _abs_1304);
                            float2 _f32x2_mul_b_1305 = make_float2(k1, k1);
                            float2 te_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_72) : "l"(*(const unsigned long long*)&ax_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_1305));
                            float2 _f32x2_mul_b_1306 = make_float2(k2, k2);
                            float2 tw_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_72) : "l"(*(const unsigned long long*)&av_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_1306));
                            float _exp2_1307 = approx_exp2(te_72.x);
                            float _exp2_1308 = approx_exp2(te_72.y);
                            float2 e_72 = make_float2(_exp2_1307, _exp2_1308);
                            float _exp2_1309 = approx_exp2(tw_72.x);
                            float _exp2_1310 = approx_exp2(tw_72.y);
                            float2 w_72 = make_float2(_exp2_1309, _exp2_1310);
                            float2 _f32x2_mul_b_1311 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_72) : "l"(*(const unsigned long long*)&ax_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_1311));
                            float _exp2_1312 = approx_exp2_noftz(tp_72.x);
                            float _exp2_1313 = approx_exp2_noftz(tp_72.y);
                            float2 p_72 = make_float2(_exp2_1312, _exp2_1313);
                            float2 d1_72 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_72);
                            float2 d2_72 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_72);
                            float2 dw_72 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_72);
                            float2 ng_72 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_72);
                            float2 nu_72 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_72);
                            float _rcp_1314 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_72, d2_72), dw_72).x);
                            float _rcp_1315 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_72, d2_72), dw_72).y);
                            float2 r_146 = make_float2(_rcp_1314, _rcp_1315);
                            float2 r_145 = r_146;
                            float2 _f32x2_mul_a_1316 = mul_f32x2_noftz(ng_72, nu_72);
                            float2 _f32x2_mul_b_1317 = mul_f32x2_noftz(r_145, make_float2(bl, bl));
                            float2 m_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1316), "l"(*(const unsigned long long*)&_f32x2_mul_b_1317));
                            float2 ns_72 = make_float2(((x_73.x < 0.0f) ? p_72.x : 1.0f), ((x_73.y < 0.0f) ? p_72.y : 1.0f));
                            gate_packed_2[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_72, ns_72))) | sign_73;
                            float2 x_74 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[10]));
                            float2 v_74 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[10]));
                            unsigned int sign_74 = (gate_packed_2[10] ^ up_packed_2[10]) & 2147516416u;
                            float _abs_1318;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1318) : "f"(x_74.x));
                            float _abs_1319;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1319) : "f"(x_74.y));
                            float2 ax_73 = make_float2(_abs_1318, _abs_1319);
                            float _abs_1320;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1320) : "f"(v_74.x));
                            float _abs_1321;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1321) : "f"(v_74.y));
                            float2 av_73 = make_float2(_abs_1320, _abs_1321);
                            float2 _f32x2_mul_b_1322 = make_float2(k1, k1);
                            float2 te_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_73) : "l"(*(const unsigned long long*)&ax_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_1322));
                            float2 _f32x2_mul_b_1323 = make_float2(k2, k2);
                            float2 tw_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_73) : "l"(*(const unsigned long long*)&av_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_1323));
                            float _exp2_1324 = approx_exp2(te_73.x);
                            float _exp2_1325 = approx_exp2(te_73.y);
                            float2 e_73 = make_float2(_exp2_1324, _exp2_1325);
                            float _exp2_1326 = approx_exp2(tw_73.x);
                            float _exp2_1327 = approx_exp2(tw_73.y);
                            float2 w_73 = make_float2(_exp2_1326, _exp2_1327);
                            float2 _f32x2_mul_b_1328 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_73) : "l"(*(const unsigned long long*)&ax_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_1328));
                            float _exp2_1329 = approx_exp2_noftz(tp_73.x);
                            float _exp2_1330 = approx_exp2_noftz(tp_73.y);
                            float2 p_73 = make_float2(_exp2_1329, _exp2_1330);
                            float2 d1_73 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_73);
                            float2 d2_73 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_73);
                            float2 dw_73 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_73);
                            float2 ng_73 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_73);
                            float2 nu_73 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_73);
                            float _rcp_1331 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_73, d2_73), dw_73).x);
                            float _rcp_1332 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_73, d2_73), dw_73).y);
                            float2 r_148 = make_float2(_rcp_1331, _rcp_1332);
                            float2 r_147 = r_148;
                            float2 _f32x2_mul_a_1333 = mul_f32x2_noftz(ng_73, nu_73);
                            float2 _f32x2_mul_b_1334 = mul_f32x2_noftz(r_147, make_float2(bl, bl));
                            float2 m_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_73) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1333), "l"(*(const unsigned long long*)&_f32x2_mul_b_1334));
                            float2 ns_73 = make_float2(((x_74.x < 0.0f) ? p_73.x : 1.0f), ((x_74.y < 0.0f) ? p_73.y : 1.0f));
                            gate_packed_2[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_73, ns_73))) | sign_74;
                            float2 x_75 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[11]));
                            float2 v_75 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[11]));
                            unsigned int sign_75 = (gate_packed_2[11] ^ up_packed_2[11]) & 2147516416u;
                            float _abs_1335;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1335) : "f"(x_75.x));
                            float _abs_1336;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1336) : "f"(x_75.y));
                            float2 ax_74 = make_float2(_abs_1335, _abs_1336);
                            float _abs_1337;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1337) : "f"(v_75.x));
                            float _abs_1338;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1338) : "f"(v_75.y));
                            float2 av_74 = make_float2(_abs_1337, _abs_1338);
                            float2 _f32x2_mul_b_1339 = make_float2(k1, k1);
                            float2 te_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_74) : "l"(*(const unsigned long long*)&ax_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_1339));
                            float2 _f32x2_mul_b_1340 = make_float2(k2, k2);
                            float2 tw_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_74) : "l"(*(const unsigned long long*)&av_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_1340));
                            float _exp2_1341 = approx_exp2(te_74.x);
                            float _exp2_1342 = approx_exp2(te_74.y);
                            float2 e_74 = make_float2(_exp2_1341, _exp2_1342);
                            float _exp2_1343 = approx_exp2(tw_74.x);
                            float _exp2_1344 = approx_exp2(tw_74.y);
                            float2 w_74 = make_float2(_exp2_1343, _exp2_1344);
                            float2 _f32x2_mul_b_1345 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_74) : "l"(*(const unsigned long long*)&ax_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_1345));
                            float _exp2_1346 = approx_exp2_noftz(tp_74.x);
                            float _exp2_1347 = approx_exp2_noftz(tp_74.y);
                            float2 p_74 = make_float2(_exp2_1346, _exp2_1347);
                            float2 d1_74 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_74);
                            float2 d2_74 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_74);
                            float2 dw_74 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_74);
                            float2 ng_74 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_74);
                            float2 nu_74 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_74);
                            float _rcp_1348 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_74, d2_74), dw_74).x);
                            float _rcp_1349 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_74, d2_74), dw_74).y);
                            float2 r_150 = make_float2(_rcp_1348, _rcp_1349);
                            float2 r_149 = r_150;
                            float2 _f32x2_mul_a_1350 = mul_f32x2_noftz(ng_74, nu_74);
                            float2 _f32x2_mul_b_1351 = mul_f32x2_noftz(r_149, make_float2(bl, bl));
                            float2 m_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_74) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1350), "l"(*(const unsigned long long*)&_f32x2_mul_b_1351));
                            float2 ns_74 = make_float2(((x_75.x < 0.0f) ? p_74.x : 1.0f), ((x_75.y < 0.0f) ? p_74.y : 1.0f));
                            gate_packed_2[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_74, ns_74))) | sign_75;
                            float2 x_76 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[12]));
                            float2 v_76 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[12]));
                            unsigned int sign_76 = (gate_packed_2[12] ^ up_packed_2[12]) & 2147516416u;
                            float _abs_1352;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1352) : "f"(x_76.x));
                            float _abs_1353;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1353) : "f"(x_76.y));
                            float2 ax_75 = make_float2(_abs_1352, _abs_1353);
                            float _abs_1354;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1354) : "f"(v_76.x));
                            float _abs_1355;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1355) : "f"(v_76.y));
                            float2 av_75 = make_float2(_abs_1354, _abs_1355);
                            float2 _f32x2_mul_b_1356 = make_float2(k1, k1);
                            float2 te_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_75) : "l"(*(const unsigned long long*)&ax_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_1356));
                            float2 _f32x2_mul_b_1357 = make_float2(k2, k2);
                            float2 tw_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_75) : "l"(*(const unsigned long long*)&av_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_1357));
                            float _exp2_1358 = approx_exp2(te_75.x);
                            float _exp2_1359 = approx_exp2(te_75.y);
                            float2 e_75 = make_float2(_exp2_1358, _exp2_1359);
                            float _exp2_1360 = approx_exp2(tw_75.x);
                            float _exp2_1361 = approx_exp2(tw_75.y);
                            float2 w_75 = make_float2(_exp2_1360, _exp2_1361);
                            float2 _f32x2_mul_b_1362 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_75) : "l"(*(const unsigned long long*)&ax_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_1362));
                            float _exp2_1363 = approx_exp2_noftz(tp_75.x);
                            float _exp2_1364 = approx_exp2_noftz(tp_75.y);
                            float2 p_75 = make_float2(_exp2_1363, _exp2_1364);
                            float2 d1_75 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_75);
                            float2 d2_75 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_75);
                            float2 dw_75 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_75);
                            float2 ng_75 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_75);
                            float2 nu_75 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_75);
                            float _rcp_1365 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_75, d2_75), dw_75).x);
                            float _rcp_1366 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_75, d2_75), dw_75).y);
                            float2 r_152 = make_float2(_rcp_1365, _rcp_1366);
                            float2 r_151 = r_152;
                            float2 _f32x2_mul_a_1367 = mul_f32x2_noftz(ng_75, nu_75);
                            float2 _f32x2_mul_b_1368 = mul_f32x2_noftz(r_151, make_float2(bl, bl));
                            float2 m_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_75) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1367), "l"(*(const unsigned long long*)&_f32x2_mul_b_1368));
                            float2 ns_75 = make_float2(((x_76.x < 0.0f) ? p_75.x : 1.0f), ((x_76.y < 0.0f) ? p_75.y : 1.0f));
                            gate_packed_2[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_75, ns_75))) | sign_76;
                            float2 x_77 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[13]));
                            float2 v_77 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[13]));
                            unsigned int sign_77 = (gate_packed_2[13] ^ up_packed_2[13]) & 2147516416u;
                            float _abs_1369;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1369) : "f"(x_77.x));
                            float _abs_1370;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1370) : "f"(x_77.y));
                            float2 ax_76 = make_float2(_abs_1369, _abs_1370);
                            float _abs_1371;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1371) : "f"(v_77.x));
                            float _abs_1372;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1372) : "f"(v_77.y));
                            float2 av_76 = make_float2(_abs_1371, _abs_1372);
                            float2 _f32x2_mul_b_1373 = make_float2(k1, k1);
                            float2 te_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_76) : "l"(*(const unsigned long long*)&ax_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_1373));
                            float2 _f32x2_mul_b_1374 = make_float2(k2, k2);
                            float2 tw_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_76) : "l"(*(const unsigned long long*)&av_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_1374));
                            float _exp2_1375 = approx_exp2(te_76.x);
                            float _exp2_1376 = approx_exp2(te_76.y);
                            float2 e_76 = make_float2(_exp2_1375, _exp2_1376);
                            float _exp2_1377 = approx_exp2(tw_76.x);
                            float _exp2_1378 = approx_exp2(tw_76.y);
                            float2 w_76 = make_float2(_exp2_1377, _exp2_1378);
                            float2 _f32x2_mul_b_1379 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_76) : "l"(*(const unsigned long long*)&ax_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_1379));
                            float _exp2_1380 = approx_exp2_noftz(tp_76.x);
                            float _exp2_1381 = approx_exp2_noftz(tp_76.y);
                            float2 p_76 = make_float2(_exp2_1380, _exp2_1381);
                            float2 d1_76 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_76);
                            float2 d2_76 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_76);
                            float2 dw_76 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_76);
                            float2 ng_76 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_76);
                            float2 nu_76 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_76);
                            float _rcp_1382 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_76, d2_76), dw_76).x);
                            float _rcp_1383 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_76, d2_76), dw_76).y);
                            float2 r_154 = make_float2(_rcp_1382, _rcp_1383);
                            float2 r_153 = r_154;
                            float2 _f32x2_mul_a_1384 = mul_f32x2_noftz(ng_76, nu_76);
                            float2 _f32x2_mul_b_1385 = mul_f32x2_noftz(r_153, make_float2(bl, bl));
                            float2 m_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_76) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1384), "l"(*(const unsigned long long*)&_f32x2_mul_b_1385));
                            float2 ns_76 = make_float2(((x_77.x < 0.0f) ? p_76.x : 1.0f), ((x_77.y < 0.0f) ? p_76.y : 1.0f));
                            gate_packed_2[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_76, ns_76))) | sign_77;
                            float2 x_78 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[14]));
                            float2 v_78 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[14]));
                            unsigned int sign_78 = (gate_packed_2[14] ^ up_packed_2[14]) & 2147516416u;
                            float _abs_1386;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1386) : "f"(x_78.x));
                            float _abs_1387;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1387) : "f"(x_78.y));
                            float2 ax_77 = make_float2(_abs_1386, _abs_1387);
                            float _abs_1388;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1388) : "f"(v_78.x));
                            float _abs_1389;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1389) : "f"(v_78.y));
                            float2 av_77 = make_float2(_abs_1388, _abs_1389);
                            float2 _f32x2_mul_b_1390 = make_float2(k1, k1);
                            float2 te_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_77) : "l"(*(const unsigned long long*)&ax_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_1390));
                            float2 _f32x2_mul_b_1391 = make_float2(k2, k2);
                            float2 tw_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_77) : "l"(*(const unsigned long long*)&av_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_1391));
                            float _exp2_1392 = approx_exp2(te_77.x);
                            float _exp2_1393 = approx_exp2(te_77.y);
                            float2 e_77 = make_float2(_exp2_1392, _exp2_1393);
                            float _exp2_1394 = approx_exp2(tw_77.x);
                            float _exp2_1395 = approx_exp2(tw_77.y);
                            float2 w_77 = make_float2(_exp2_1394, _exp2_1395);
                            float2 _f32x2_mul_b_1396 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_77) : "l"(*(const unsigned long long*)&ax_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_1396));
                            float _exp2_1397 = approx_exp2_noftz(tp_77.x);
                            float _exp2_1398 = approx_exp2_noftz(tp_77.y);
                            float2 p_77 = make_float2(_exp2_1397, _exp2_1398);
                            float2 d1_77 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_77);
                            float2 d2_77 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_77);
                            float2 dw_77 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_77);
                            float2 ng_77 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_77);
                            float2 nu_77 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_77);
                            float _rcp_1399 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_77, d2_77), dw_77).x);
                            float _rcp_1400 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_77, d2_77), dw_77).y);
                            float2 r_156 = make_float2(_rcp_1399, _rcp_1400);
                            float2 r_155 = r_156;
                            float2 _f32x2_mul_a_1401 = mul_f32x2_noftz(ng_77, nu_77);
                            float2 _f32x2_mul_b_1402 = mul_f32x2_noftz(r_155, make_float2(bl, bl));
                            float2 m_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_77) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1401), "l"(*(const unsigned long long*)&_f32x2_mul_b_1402));
                            float2 ns_77 = make_float2(((x_78.x < 0.0f) ? p_77.x : 1.0f), ((x_78.y < 0.0f) ? p_77.y : 1.0f));
                            gate_packed_2[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_77, ns_77))) | sign_78;
                            float2 x_79 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[15]));
                            float2 v_79 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[15]));
                            unsigned int sign_79 = (gate_packed_2[15] ^ up_packed_2[15]) & 2147516416u;
                            float _abs_1403;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1403) : "f"(x_79.x));
                            float _abs_1404;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1404) : "f"(x_79.y));
                            float2 ax_78 = make_float2(_abs_1403, _abs_1404);
                            float _abs_1405;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1405) : "f"(v_79.x));
                            float _abs_1406;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1406) : "f"(v_79.y));
                            float2 av_78 = make_float2(_abs_1405, _abs_1406);
                            float2 _f32x2_mul_b_1407 = make_float2(k1, k1);
                            float2 te_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_78) : "l"(*(const unsigned long long*)&ax_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_1407));
                            float2 _f32x2_mul_b_1408 = make_float2(k2, k2);
                            float2 tw_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_78) : "l"(*(const unsigned long long*)&av_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_1408));
                            float _exp2_1409 = approx_exp2(te_78.x);
                            float _exp2_1410 = approx_exp2(te_78.y);
                            float2 e_78 = make_float2(_exp2_1409, _exp2_1410);
                            float _exp2_1411 = approx_exp2(tw_78.x);
                            float _exp2_1412 = approx_exp2(tw_78.y);
                            float2 w_78 = make_float2(_exp2_1411, _exp2_1412);
                            float2 _f32x2_mul_b_1413 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_78) : "l"(*(const unsigned long long*)&ax_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_1413));
                            float _exp2_1414 = approx_exp2_noftz(tp_78.x);
                            float _exp2_1415 = approx_exp2_noftz(tp_78.y);
                            float2 p_78 = make_float2(_exp2_1414, _exp2_1415);
                            float2 d1_78 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_78);
                            float2 d2_78 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_78);
                            float2 dw_78 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_78);
                            float2 ng_78 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_78);
                            float2 nu_78 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_78);
                            float _rcp_1416 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_78, d2_78), dw_78).x);
                            float _rcp_1417 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_78, d2_78), dw_78).y);
                            float2 r_158 = make_float2(_rcp_1416, _rcp_1417);
                            float2 r_157 = r_158;
                            float2 _f32x2_mul_a_1418 = mul_f32x2_noftz(ng_78, nu_78);
                            float2 _f32x2_mul_b_1419 = mul_f32x2_noftz(r_157, make_float2(bl, bl));
                            float2 m_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_78) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1418), "l"(*(const unsigned long long*)&_f32x2_mul_b_1419));
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
                                    int row_20 = warp_15 * 32 + half_16 * 16 + lane_15 % 16;
                                    int col_17 = col_tile_16 * 16 + lane_15 / 16 * 8;
                                    unsigned int address_16 = d_smem_addr + 8192 + (unsigned int)((row_20 * 32 + col_17) * 2);
                                    address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                    int offset_17 = half_16 * 8 + col_tile_16 * 4;
                                    uint32_t _stmatrix_addr_1420 = static_cast<uint32_t>(address_16);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1420), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_17 + 3]))
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
                            float _abs_1421;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1421) : "f"(x_80.x));
                            float _abs_1422;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1422) : "f"(x_80.y));
                            float2 ax_79 = make_float2(_abs_1421, _abs_1422);
                            float _abs_1423;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1423) : "f"(v_80.x));
                            float _abs_1424;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1424) : "f"(v_80.y));
                            float2 av_79 = make_float2(_abs_1423, _abs_1424);
                            float2 _f32x2_mul_b_1425 = make_float2(k1, k1);
                            float2 te_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_79) : "l"(*(const unsigned long long*)&ax_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_1425));
                            float2 _f32x2_mul_b_1426 = make_float2(k2, k2);
                            float2 tw_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_79) : "l"(*(const unsigned long long*)&av_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_1426));
                            float _exp2_1427 = approx_exp2(te_79.x);
                            float _exp2_1428 = approx_exp2(te_79.y);
                            float2 e_79 = make_float2(_exp2_1427, _exp2_1428);
                            float _exp2_1429 = approx_exp2(tw_79.x);
                            float _exp2_1430 = approx_exp2(tw_79.y);
                            float2 w_79 = make_float2(_exp2_1429, _exp2_1430);
                            float2 _f32x2_mul_b_1431 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_79) : "l"(*(const unsigned long long*)&ax_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_1431));
                            float _exp2_1432 = approx_exp2_noftz(tp_79.x);
                            float _exp2_1433 = approx_exp2_noftz(tp_79.y);
                            float2 p_79 = make_float2(_exp2_1432, _exp2_1433);
                            float2 d1_79 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_79);
                            float2 d2_79 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_79);
                            float2 dw_79 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_79);
                            float2 ng_79 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_79);
                            float2 nu_79 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_79);
                            float _rcp_1434 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_79, d2_79), dw_79).x);
                            float _rcp_1435 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_79, d2_79), dw_79).y);
                            float2 r_160 = make_float2(_rcp_1434, _rcp_1435);
                            float2 r_159 = r_160;
                            float2 _f32x2_mul_a_1436 = mul_f32x2_noftz(ng_79, nu_79);
                            float2 _f32x2_mul_b_1437 = mul_f32x2_noftz(r_159, make_float2(bl, bl));
                            float2 m_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_79) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1436), "l"(*(const unsigned long long*)&_f32x2_mul_b_1437));
                            float2 ns_79 = make_float2(((x_80.x < 0.0f) ? p_79.x : 1.0f), ((x_80.y < 0.0f) ? p_79.y : 1.0f));
                            gate_packed_2[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_79, ns_79))) | sign_80;
                            float2 x_81 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[17]));
                            float2 v_81 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[17]));
                            unsigned int sign_81 = (gate_packed_2[17] ^ up_packed_2[17]) & 2147516416u;
                            float _abs_1438;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1438) : "f"(x_81.x));
                            float _abs_1439;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1439) : "f"(x_81.y));
                            float2 ax_80 = make_float2(_abs_1438, _abs_1439);
                            float _abs_1440;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1440) : "f"(v_81.x));
                            float _abs_1441;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1441) : "f"(v_81.y));
                            float2 av_80 = make_float2(_abs_1440, _abs_1441);
                            float2 _f32x2_mul_b_1442 = make_float2(k1, k1);
                            float2 te_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_80) : "l"(*(const unsigned long long*)&ax_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_1442));
                            float2 _f32x2_mul_b_1443 = make_float2(k2, k2);
                            float2 tw_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_80) : "l"(*(const unsigned long long*)&av_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_1443));
                            float _exp2_1444 = approx_exp2(te_80.x);
                            float _exp2_1445 = approx_exp2(te_80.y);
                            float2 e_80 = make_float2(_exp2_1444, _exp2_1445);
                            float _exp2_1446 = approx_exp2(tw_80.x);
                            float _exp2_1447 = approx_exp2(tw_80.y);
                            float2 w_80 = make_float2(_exp2_1446, _exp2_1447);
                            float2 _f32x2_mul_b_1448 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_80) : "l"(*(const unsigned long long*)&ax_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_1448));
                            float _exp2_1449 = approx_exp2_noftz(tp_80.x);
                            float _exp2_1450 = approx_exp2_noftz(tp_80.y);
                            float2 p_80 = make_float2(_exp2_1449, _exp2_1450);
                            float2 d1_80 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_80);
                            float2 d2_80 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_80);
                            float2 dw_80 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_80);
                            float2 ng_80 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_80);
                            float2 nu_80 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_80);
                            float _rcp_1451 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_80, d2_80), dw_80).x);
                            float _rcp_1452 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_80, d2_80), dw_80).y);
                            float2 r_162 = make_float2(_rcp_1451, _rcp_1452);
                            float2 r_161 = r_162;
                            float2 _f32x2_mul_a_1453 = mul_f32x2_noftz(ng_80, nu_80);
                            float2 _f32x2_mul_b_1454 = mul_f32x2_noftz(r_161, make_float2(bl, bl));
                            float2 m_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_80) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1453), "l"(*(const unsigned long long*)&_f32x2_mul_b_1454));
                            float2 ns_80 = make_float2(((x_81.x < 0.0f) ? p_80.x : 1.0f), ((x_81.y < 0.0f) ? p_80.y : 1.0f));
                            gate_packed_2[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_80, ns_80))) | sign_81;
                            float2 x_82 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[18]));
                            float2 v_82 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[18]));
                            unsigned int sign_82 = (gate_packed_2[18] ^ up_packed_2[18]) & 2147516416u;
                            float _abs_1455;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1455) : "f"(x_82.x));
                            float _abs_1456;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1456) : "f"(x_82.y));
                            float2 ax_81 = make_float2(_abs_1455, _abs_1456);
                            float _abs_1457;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1457) : "f"(v_82.x));
                            float _abs_1458;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1458) : "f"(v_82.y));
                            float2 av_81 = make_float2(_abs_1457, _abs_1458);
                            float2 _f32x2_mul_b_1459 = make_float2(k1, k1);
                            float2 te_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_81) : "l"(*(const unsigned long long*)&ax_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_1459));
                            float2 _f32x2_mul_b_1460 = make_float2(k2, k2);
                            float2 tw_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_81) : "l"(*(const unsigned long long*)&av_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_1460));
                            float _exp2_1461 = approx_exp2(te_81.x);
                            float _exp2_1462 = approx_exp2(te_81.y);
                            float2 e_81 = make_float2(_exp2_1461, _exp2_1462);
                            float _exp2_1463 = approx_exp2(tw_81.x);
                            float _exp2_1464 = approx_exp2(tw_81.y);
                            float2 w_81 = make_float2(_exp2_1463, _exp2_1464);
                            float2 _f32x2_mul_b_1465 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_81) : "l"(*(const unsigned long long*)&ax_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_1465));
                            float _exp2_1466 = approx_exp2_noftz(tp_81.x);
                            float _exp2_1467 = approx_exp2_noftz(tp_81.y);
                            float2 p_81 = make_float2(_exp2_1466, _exp2_1467);
                            float2 d1_81 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_81);
                            float2 d2_81 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_81);
                            float2 dw_81 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_81);
                            float2 ng_81 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_81);
                            float2 nu_81 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_81);
                            float _rcp_1468 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_81, d2_81), dw_81).x);
                            float _rcp_1469 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_81, d2_81), dw_81).y);
                            float2 r_164 = make_float2(_rcp_1468, _rcp_1469);
                            float2 r_163 = r_164;
                            float2 _f32x2_mul_a_1470 = mul_f32x2_noftz(ng_81, nu_81);
                            float2 _f32x2_mul_b_1471 = mul_f32x2_noftz(r_163, make_float2(bl, bl));
                            float2 m_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_81) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1470), "l"(*(const unsigned long long*)&_f32x2_mul_b_1471));
                            float2 ns_81 = make_float2(((x_82.x < 0.0f) ? p_81.x : 1.0f), ((x_82.y < 0.0f) ? p_81.y : 1.0f));
                            gate_packed_2[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_81, ns_81))) | sign_82;
                            float2 x_83 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[19]));
                            float2 v_83 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[19]));
                            unsigned int sign_83 = (gate_packed_2[19] ^ up_packed_2[19]) & 2147516416u;
                            float _abs_1472;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1472) : "f"(x_83.x));
                            float _abs_1473;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1473) : "f"(x_83.y));
                            float2 ax_82 = make_float2(_abs_1472, _abs_1473);
                            float _abs_1474;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1474) : "f"(v_83.x));
                            float _abs_1475;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1475) : "f"(v_83.y));
                            float2 av_82 = make_float2(_abs_1474, _abs_1475);
                            float2 _f32x2_mul_b_1476 = make_float2(k1, k1);
                            float2 te_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_82) : "l"(*(const unsigned long long*)&ax_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_1476));
                            float2 _f32x2_mul_b_1477 = make_float2(k2, k2);
                            float2 tw_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_82) : "l"(*(const unsigned long long*)&av_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_1477));
                            float _exp2_1478 = approx_exp2(te_82.x);
                            float _exp2_1479 = approx_exp2(te_82.y);
                            float2 e_82 = make_float2(_exp2_1478, _exp2_1479);
                            float _exp2_1480 = approx_exp2(tw_82.x);
                            float _exp2_1481 = approx_exp2(tw_82.y);
                            float2 w_82 = make_float2(_exp2_1480, _exp2_1481);
                            float2 _f32x2_mul_b_1482 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_82) : "l"(*(const unsigned long long*)&ax_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_1482));
                            float _exp2_1483 = approx_exp2_noftz(tp_82.x);
                            float _exp2_1484 = approx_exp2_noftz(tp_82.y);
                            float2 p_82 = make_float2(_exp2_1483, _exp2_1484);
                            float2 d1_82 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_82);
                            float2 d2_82 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_82);
                            float2 dw_82 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_82);
                            float2 ng_82 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_82);
                            float2 nu_82 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_82);
                            float _rcp_1485 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_82, d2_82), dw_82).x);
                            float _rcp_1486 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_82, d2_82), dw_82).y);
                            float2 r_166 = make_float2(_rcp_1485, _rcp_1486);
                            float2 r_165 = r_166;
                            float2 _f32x2_mul_a_1487 = mul_f32x2_noftz(ng_82, nu_82);
                            float2 _f32x2_mul_b_1488 = mul_f32x2_noftz(r_165, make_float2(bl, bl));
                            float2 m_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_82) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1487), "l"(*(const unsigned long long*)&_f32x2_mul_b_1488));
                            float2 ns_82 = make_float2(((x_83.x < 0.0f) ? p_82.x : 1.0f), ((x_83.y < 0.0f) ? p_82.y : 1.0f));
                            gate_packed_2[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_82, ns_82))) | sign_83;
                            float2 x_84 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[20]));
                            float2 v_84 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[20]));
                            unsigned int sign_84 = (gate_packed_2[20] ^ up_packed_2[20]) & 2147516416u;
                            float _abs_1489;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1489) : "f"(x_84.x));
                            float _abs_1490;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1490) : "f"(x_84.y));
                            float2 ax_83 = make_float2(_abs_1489, _abs_1490);
                            float _abs_1491;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1491) : "f"(v_84.x));
                            float _abs_1492;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1492) : "f"(v_84.y));
                            float2 av_83 = make_float2(_abs_1491, _abs_1492);
                            float2 _f32x2_mul_b_1493 = make_float2(k1, k1);
                            float2 te_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_83) : "l"(*(const unsigned long long*)&ax_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_1493));
                            float2 _f32x2_mul_b_1494 = make_float2(k2, k2);
                            float2 tw_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_83) : "l"(*(const unsigned long long*)&av_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_1494));
                            float _exp2_1495 = approx_exp2(te_83.x);
                            float _exp2_1496 = approx_exp2(te_83.y);
                            float2 e_83 = make_float2(_exp2_1495, _exp2_1496);
                            float _exp2_1497 = approx_exp2(tw_83.x);
                            float _exp2_1498 = approx_exp2(tw_83.y);
                            float2 w_83 = make_float2(_exp2_1497, _exp2_1498);
                            float2 _f32x2_mul_b_1499 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_83) : "l"(*(const unsigned long long*)&ax_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_1499));
                            float _exp2_1500 = approx_exp2_noftz(tp_83.x);
                            float _exp2_1501 = approx_exp2_noftz(tp_83.y);
                            float2 p_83 = make_float2(_exp2_1500, _exp2_1501);
                            float2 d1_83 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_83);
                            float2 d2_83 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_83);
                            float2 dw_83 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_83);
                            float2 ng_83 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_83);
                            float2 nu_83 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_83);
                            float _rcp_1502 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_83, d2_83), dw_83).x);
                            float _rcp_1503 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_83, d2_83), dw_83).y);
                            float2 r_168 = make_float2(_rcp_1502, _rcp_1503);
                            float2 r_167 = r_168;
                            float2 _f32x2_mul_a_1504 = mul_f32x2_noftz(ng_83, nu_83);
                            float2 _f32x2_mul_b_1505 = mul_f32x2_noftz(r_167, make_float2(bl, bl));
                            float2 m_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_83) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1504), "l"(*(const unsigned long long*)&_f32x2_mul_b_1505));
                            float2 ns_83 = make_float2(((x_84.x < 0.0f) ? p_83.x : 1.0f), ((x_84.y < 0.0f) ? p_83.y : 1.0f));
                            gate_packed_2[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_83, ns_83))) | sign_84;
                            float2 x_85 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[21]));
                            float2 v_85 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[21]));
                            unsigned int sign_85 = (gate_packed_2[21] ^ up_packed_2[21]) & 2147516416u;
                            float _abs_1506;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1506) : "f"(x_85.x));
                            float _abs_1507;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1507) : "f"(x_85.y));
                            float2 ax_84 = make_float2(_abs_1506, _abs_1507);
                            float _abs_1508;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1508) : "f"(v_85.x));
                            float _abs_1509;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1509) : "f"(v_85.y));
                            float2 av_84 = make_float2(_abs_1508, _abs_1509);
                            float2 _f32x2_mul_b_1510 = make_float2(k1, k1);
                            float2 te_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_84) : "l"(*(const unsigned long long*)&ax_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_1510));
                            float2 _f32x2_mul_b_1511 = make_float2(k2, k2);
                            float2 tw_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_84) : "l"(*(const unsigned long long*)&av_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_1511));
                            float _exp2_1512 = approx_exp2(te_84.x);
                            float _exp2_1513 = approx_exp2(te_84.y);
                            float2 e_84 = make_float2(_exp2_1512, _exp2_1513);
                            float _exp2_1514 = approx_exp2(tw_84.x);
                            float _exp2_1515 = approx_exp2(tw_84.y);
                            float2 w_84 = make_float2(_exp2_1514, _exp2_1515);
                            float2 _f32x2_mul_b_1516 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_84) : "l"(*(const unsigned long long*)&ax_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_1516));
                            float _exp2_1517 = approx_exp2_noftz(tp_84.x);
                            float _exp2_1518 = approx_exp2_noftz(tp_84.y);
                            float2 p_84 = make_float2(_exp2_1517, _exp2_1518);
                            float2 d1_84 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_84);
                            float2 d2_84 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_84);
                            float2 dw_84 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_84);
                            float2 ng_84 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_84);
                            float2 nu_84 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_84);
                            float _rcp_1519 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_84, d2_84), dw_84).x);
                            float _rcp_1520 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_84, d2_84), dw_84).y);
                            float2 r_170 = make_float2(_rcp_1519, _rcp_1520);
                            float2 r_169 = r_170;
                            float2 _f32x2_mul_a_1521 = mul_f32x2_noftz(ng_84, nu_84);
                            float2 _f32x2_mul_b_1522 = mul_f32x2_noftz(r_169, make_float2(bl, bl));
                            float2 m_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1521), "l"(*(const unsigned long long*)&_f32x2_mul_b_1522));
                            float2 ns_84 = make_float2(((x_85.x < 0.0f) ? p_84.x : 1.0f), ((x_85.y < 0.0f) ? p_84.y : 1.0f));
                            gate_packed_2[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_84, ns_84))) | sign_85;
                            float2 x_86 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[22]));
                            float2 v_86 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[22]));
                            unsigned int sign_86 = (gate_packed_2[22] ^ up_packed_2[22]) & 2147516416u;
                            float _abs_1523;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1523) : "f"(x_86.x));
                            float _abs_1524;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1524) : "f"(x_86.y));
                            float2 ax_85 = make_float2(_abs_1523, _abs_1524);
                            float _abs_1525;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1525) : "f"(v_86.x));
                            float _abs_1526;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1526) : "f"(v_86.y));
                            float2 av_85 = make_float2(_abs_1525, _abs_1526);
                            float2 _f32x2_mul_b_1527 = make_float2(k1, k1);
                            float2 te_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_85) : "l"(*(const unsigned long long*)&ax_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_1527));
                            float2 _f32x2_mul_b_1528 = make_float2(k2, k2);
                            float2 tw_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_85) : "l"(*(const unsigned long long*)&av_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_1528));
                            float _exp2_1529 = approx_exp2(te_85.x);
                            float _exp2_1530 = approx_exp2(te_85.y);
                            float2 e_85 = make_float2(_exp2_1529, _exp2_1530);
                            float _exp2_1531 = approx_exp2(tw_85.x);
                            float _exp2_1532 = approx_exp2(tw_85.y);
                            float2 w_85 = make_float2(_exp2_1531, _exp2_1532);
                            float2 _f32x2_mul_b_1533 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_85) : "l"(*(const unsigned long long*)&ax_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_1533));
                            float _exp2_1534 = approx_exp2_noftz(tp_85.x);
                            float _exp2_1535 = approx_exp2_noftz(tp_85.y);
                            float2 p_85 = make_float2(_exp2_1534, _exp2_1535);
                            float2 d1_85 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_85);
                            float2 d2_85 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_85);
                            float2 dw_85 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_85);
                            float2 ng_85 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_85);
                            float2 nu_85 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_85);
                            float _rcp_1536 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_85, d2_85), dw_85).x);
                            float _rcp_1537 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_85, d2_85), dw_85).y);
                            float2 r_172 = make_float2(_rcp_1536, _rcp_1537);
                            float2 r_171 = r_172;
                            float2 _f32x2_mul_a_1538 = mul_f32x2_noftz(ng_85, nu_85);
                            float2 _f32x2_mul_b_1539 = mul_f32x2_noftz(r_171, make_float2(bl, bl));
                            float2 m_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_85) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1538), "l"(*(const unsigned long long*)&_f32x2_mul_b_1539));
                            float2 ns_85 = make_float2(((x_86.x < 0.0f) ? p_85.x : 1.0f), ((x_86.y < 0.0f) ? p_85.y : 1.0f));
                            gate_packed_2[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_85, ns_85))) | sign_86;
                            float2 x_87 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[23]));
                            float2 v_87 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[23]));
                            unsigned int sign_87 = (gate_packed_2[23] ^ up_packed_2[23]) & 2147516416u;
                            float _abs_1540;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1540) : "f"(x_87.x));
                            float _abs_1541;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1541) : "f"(x_87.y));
                            float2 ax_86 = make_float2(_abs_1540, _abs_1541);
                            float _abs_1542;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1542) : "f"(v_87.x));
                            float _abs_1543;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1543) : "f"(v_87.y));
                            float2 av_86 = make_float2(_abs_1542, _abs_1543);
                            float2 _f32x2_mul_b_1544 = make_float2(k1, k1);
                            float2 te_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_86) : "l"(*(const unsigned long long*)&ax_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_1544));
                            float2 _f32x2_mul_b_1545 = make_float2(k2, k2);
                            float2 tw_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_86) : "l"(*(const unsigned long long*)&av_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_1545));
                            float _exp2_1546 = approx_exp2(te_86.x);
                            float _exp2_1547 = approx_exp2(te_86.y);
                            float2 e_86 = make_float2(_exp2_1546, _exp2_1547);
                            float _exp2_1548 = approx_exp2(tw_86.x);
                            float _exp2_1549 = approx_exp2(tw_86.y);
                            float2 w_86 = make_float2(_exp2_1548, _exp2_1549);
                            float2 _f32x2_mul_b_1550 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_86) : "l"(*(const unsigned long long*)&ax_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_1550));
                            float _exp2_1551 = approx_exp2_noftz(tp_86.x);
                            float _exp2_1552 = approx_exp2_noftz(tp_86.y);
                            float2 p_86 = make_float2(_exp2_1551, _exp2_1552);
                            float2 d1_86 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_86);
                            float2 d2_86 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_86);
                            float2 dw_86 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_86);
                            float2 ng_86 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_86);
                            float2 nu_86 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_86);
                            float _rcp_1553 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_86, d2_86), dw_86).x);
                            float _rcp_1554 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_86, d2_86), dw_86).y);
                            float2 r_174 = make_float2(_rcp_1553, _rcp_1554);
                            float2 r_173 = r_174;
                            float2 _f32x2_mul_a_1555 = mul_f32x2_noftz(ng_86, nu_86);
                            float2 _f32x2_mul_b_1556 = mul_f32x2_noftz(r_173, make_float2(bl, bl));
                            float2 m_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1555), "l"(*(const unsigned long long*)&_f32x2_mul_b_1556));
                            float2 ns_86 = make_float2(((x_87.x < 0.0f) ? p_86.x : 1.0f), ((x_87.y < 0.0f) ? p_86.y : 1.0f));
                            gate_packed_2[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_86, ns_86))) | sign_87;
                            float2 x_88 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[24]));
                            float2 v_88 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[24]));
                            unsigned int sign_88 = (gate_packed_2[24] ^ up_packed_2[24]) & 2147516416u;
                            float _abs_1557;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1557) : "f"(x_88.x));
                            float _abs_1558;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1558) : "f"(x_88.y));
                            float2 ax_87 = make_float2(_abs_1557, _abs_1558);
                            float _abs_1559;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1559) : "f"(v_88.x));
                            float _abs_1560;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1560) : "f"(v_88.y));
                            float2 av_87 = make_float2(_abs_1559, _abs_1560);
                            float2 _f32x2_mul_b_1561 = make_float2(k1, k1);
                            float2 te_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_87) : "l"(*(const unsigned long long*)&ax_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_1561));
                            float2 _f32x2_mul_b_1562 = make_float2(k2, k2);
                            float2 tw_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_87) : "l"(*(const unsigned long long*)&av_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_1562));
                            float _exp2_1563 = approx_exp2(te_87.x);
                            float _exp2_1564 = approx_exp2(te_87.y);
                            float2 e_87 = make_float2(_exp2_1563, _exp2_1564);
                            float _exp2_1565 = approx_exp2(tw_87.x);
                            float _exp2_1566 = approx_exp2(tw_87.y);
                            float2 w_87 = make_float2(_exp2_1565, _exp2_1566);
                            float2 _f32x2_mul_b_1567 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_87) : "l"(*(const unsigned long long*)&ax_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_1567));
                            float _exp2_1568 = approx_exp2_noftz(tp_87.x);
                            float _exp2_1569 = approx_exp2_noftz(tp_87.y);
                            float2 p_87 = make_float2(_exp2_1568, _exp2_1569);
                            float2 d1_87 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_87);
                            float2 d2_87 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_87);
                            float2 dw_87 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_87);
                            float2 ng_87 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_87);
                            float2 nu_87 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_87);
                            float _rcp_1570 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_87, d2_87), dw_87).x);
                            float _rcp_1571 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_87, d2_87), dw_87).y);
                            float2 r_176 = make_float2(_rcp_1570, _rcp_1571);
                            float2 r_175 = r_176;
                            float2 _f32x2_mul_a_1572 = mul_f32x2_noftz(ng_87, nu_87);
                            float2 _f32x2_mul_b_1573 = mul_f32x2_noftz(r_175, make_float2(bl, bl));
                            float2 m_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_87) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1572), "l"(*(const unsigned long long*)&_f32x2_mul_b_1573));
                            float2 ns_87 = make_float2(((x_88.x < 0.0f) ? p_87.x : 1.0f), ((x_88.y < 0.0f) ? p_87.y : 1.0f));
                            gate_packed_2[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_87, ns_87))) | sign_88;
                            float2 x_89 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[25]));
                            float2 v_89 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[25]));
                            unsigned int sign_89 = (gate_packed_2[25] ^ up_packed_2[25]) & 2147516416u;
                            float _abs_1574;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1574) : "f"(x_89.x));
                            float _abs_1575;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1575) : "f"(x_89.y));
                            float2 ax_88 = make_float2(_abs_1574, _abs_1575);
                            float _abs_1576;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1576) : "f"(v_89.x));
                            float _abs_1577;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1577) : "f"(v_89.y));
                            float2 av_88 = make_float2(_abs_1576, _abs_1577);
                            float2 _f32x2_mul_b_1578 = make_float2(k1, k1);
                            float2 te_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_88) : "l"(*(const unsigned long long*)&ax_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_1578));
                            float2 _f32x2_mul_b_1579 = make_float2(k2, k2);
                            float2 tw_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_88) : "l"(*(const unsigned long long*)&av_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_1579));
                            float _exp2_1580 = approx_exp2(te_88.x);
                            float _exp2_1581 = approx_exp2(te_88.y);
                            float2 e_88 = make_float2(_exp2_1580, _exp2_1581);
                            float _exp2_1582 = approx_exp2(tw_88.x);
                            float _exp2_1583 = approx_exp2(tw_88.y);
                            float2 w_88 = make_float2(_exp2_1582, _exp2_1583);
                            float2 _f32x2_mul_b_1584 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_88) : "l"(*(const unsigned long long*)&ax_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_1584));
                            float _exp2_1585 = approx_exp2_noftz(tp_88.x);
                            float _exp2_1586 = approx_exp2_noftz(tp_88.y);
                            float2 p_88 = make_float2(_exp2_1585, _exp2_1586);
                            float2 d1_88 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_88);
                            float2 d2_88 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_88);
                            float2 dw_88 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_88);
                            float2 ng_88 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_88);
                            float2 nu_88 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_88);
                            float _rcp_1587 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_88, d2_88), dw_88).x);
                            float _rcp_1588 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_88, d2_88), dw_88).y);
                            float2 r_178 = make_float2(_rcp_1587, _rcp_1588);
                            float2 r_177 = r_178;
                            float2 _f32x2_mul_a_1589 = mul_f32x2_noftz(ng_88, nu_88);
                            float2 _f32x2_mul_b_1590 = mul_f32x2_noftz(r_177, make_float2(bl, bl));
                            float2 m_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1589), "l"(*(const unsigned long long*)&_f32x2_mul_b_1590));
                            float2 ns_88 = make_float2(((x_89.x < 0.0f) ? p_88.x : 1.0f), ((x_89.y < 0.0f) ? p_88.y : 1.0f));
                            gate_packed_2[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_88, ns_88))) | sign_89;
                            float2 x_90 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[26]));
                            float2 v_90 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[26]));
                            unsigned int sign_90 = (gate_packed_2[26] ^ up_packed_2[26]) & 2147516416u;
                            float _abs_1591;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1591) : "f"(x_90.x));
                            float _abs_1592;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1592) : "f"(x_90.y));
                            float2 ax_89 = make_float2(_abs_1591, _abs_1592);
                            float _abs_1593;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1593) : "f"(v_90.x));
                            float _abs_1594;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1594) : "f"(v_90.y));
                            float2 av_89 = make_float2(_abs_1593, _abs_1594);
                            float2 _f32x2_mul_b_1595 = make_float2(k1, k1);
                            float2 te_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_89) : "l"(*(const unsigned long long*)&ax_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_1595));
                            float2 _f32x2_mul_b_1596 = make_float2(k2, k2);
                            float2 tw_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_89) : "l"(*(const unsigned long long*)&av_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_1596));
                            float _exp2_1597 = approx_exp2(te_89.x);
                            float _exp2_1598 = approx_exp2(te_89.y);
                            float2 e_89 = make_float2(_exp2_1597, _exp2_1598);
                            float _exp2_1599 = approx_exp2(tw_89.x);
                            float _exp2_1600 = approx_exp2(tw_89.y);
                            float2 w_89 = make_float2(_exp2_1599, _exp2_1600);
                            float2 _f32x2_mul_b_1601 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_89) : "l"(*(const unsigned long long*)&ax_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_1601));
                            float _exp2_1602 = approx_exp2_noftz(tp_89.x);
                            float _exp2_1603 = approx_exp2_noftz(tp_89.y);
                            float2 p_89 = make_float2(_exp2_1602, _exp2_1603);
                            float2 d1_89 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_89);
                            float2 d2_89 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_89);
                            float2 dw_89 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_89);
                            float2 ng_89 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_89);
                            float2 nu_89 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_89);
                            float _rcp_1604 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_89, d2_89), dw_89).x);
                            float _rcp_1605 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_89, d2_89), dw_89).y);
                            float2 r_180 = make_float2(_rcp_1604, _rcp_1605);
                            float2 r_179 = r_180;
                            float2 _f32x2_mul_a_1606 = mul_f32x2_noftz(ng_89, nu_89);
                            float2 _f32x2_mul_b_1607 = mul_f32x2_noftz(r_179, make_float2(bl, bl));
                            float2 m_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1606), "l"(*(const unsigned long long*)&_f32x2_mul_b_1607));
                            float2 ns_89 = make_float2(((x_90.x < 0.0f) ? p_89.x : 1.0f), ((x_90.y < 0.0f) ? p_89.y : 1.0f));
                            gate_packed_2[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_89, ns_89))) | sign_90;
                            float2 x_91 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[27]));
                            float2 v_91 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[27]));
                            unsigned int sign_91 = (gate_packed_2[27] ^ up_packed_2[27]) & 2147516416u;
                            float _abs_1608;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1608) : "f"(x_91.x));
                            float _abs_1609;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1609) : "f"(x_91.y));
                            float2 ax_90 = make_float2(_abs_1608, _abs_1609);
                            float _abs_1610;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1610) : "f"(v_91.x));
                            float _abs_1611;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1611) : "f"(v_91.y));
                            float2 av_90 = make_float2(_abs_1610, _abs_1611);
                            float2 _f32x2_mul_b_1612 = make_float2(k1, k1);
                            float2 te_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_90) : "l"(*(const unsigned long long*)&ax_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_1612));
                            float2 _f32x2_mul_b_1613 = make_float2(k2, k2);
                            float2 tw_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_90) : "l"(*(const unsigned long long*)&av_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_1613));
                            float _exp2_1614 = approx_exp2(te_90.x);
                            float _exp2_1615 = approx_exp2(te_90.y);
                            float2 e_90 = make_float2(_exp2_1614, _exp2_1615);
                            float _exp2_1616 = approx_exp2(tw_90.x);
                            float _exp2_1617 = approx_exp2(tw_90.y);
                            float2 w_90 = make_float2(_exp2_1616, _exp2_1617);
                            float2 _f32x2_mul_b_1618 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_90) : "l"(*(const unsigned long long*)&ax_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_1618));
                            float _exp2_1619 = approx_exp2_noftz(tp_90.x);
                            float _exp2_1620 = approx_exp2_noftz(tp_90.y);
                            float2 p_90 = make_float2(_exp2_1619, _exp2_1620);
                            float2 d1_90 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_90);
                            float2 d2_90 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_90);
                            float2 dw_90 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_90);
                            float2 ng_90 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_90);
                            float2 nu_90 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_90);
                            float _rcp_1621 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_90, d2_90), dw_90).x);
                            float _rcp_1622 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_90, d2_90), dw_90).y);
                            float2 r_182 = make_float2(_rcp_1621, _rcp_1622);
                            float2 r_181 = r_182;
                            float2 _f32x2_mul_a_1623 = mul_f32x2_noftz(ng_90, nu_90);
                            float2 _f32x2_mul_b_1624 = mul_f32x2_noftz(r_181, make_float2(bl, bl));
                            float2 m_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1623), "l"(*(const unsigned long long*)&_f32x2_mul_b_1624));
                            float2 ns_90 = make_float2(((x_91.x < 0.0f) ? p_90.x : 1.0f), ((x_91.y < 0.0f) ? p_90.y : 1.0f));
                            gate_packed_2[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_90, ns_90))) | sign_91;
                            float2 x_92 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[28]));
                            float2 v_92 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[28]));
                            unsigned int sign_92 = (gate_packed_2[28] ^ up_packed_2[28]) & 2147516416u;
                            float _abs_1625;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1625) : "f"(x_92.x));
                            float _abs_1626;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1626) : "f"(x_92.y));
                            float2 ax_91 = make_float2(_abs_1625, _abs_1626);
                            float _abs_1627;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1627) : "f"(v_92.x));
                            float _abs_1628;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1628) : "f"(v_92.y));
                            float2 av_91 = make_float2(_abs_1627, _abs_1628);
                            float2 _f32x2_mul_b_1629 = make_float2(k1, k1);
                            float2 te_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_91) : "l"(*(const unsigned long long*)&ax_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_1629));
                            float2 _f32x2_mul_b_1630 = make_float2(k2, k2);
                            float2 tw_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_91) : "l"(*(const unsigned long long*)&av_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_1630));
                            float _exp2_1631 = approx_exp2(te_91.x);
                            float _exp2_1632 = approx_exp2(te_91.y);
                            float2 e_91 = make_float2(_exp2_1631, _exp2_1632);
                            float _exp2_1633 = approx_exp2(tw_91.x);
                            float _exp2_1634 = approx_exp2(tw_91.y);
                            float2 w_91 = make_float2(_exp2_1633, _exp2_1634);
                            float2 _f32x2_mul_b_1635 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_91) : "l"(*(const unsigned long long*)&ax_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_1635));
                            float _exp2_1636 = approx_exp2_noftz(tp_91.x);
                            float _exp2_1637 = approx_exp2_noftz(tp_91.y);
                            float2 p_91 = make_float2(_exp2_1636, _exp2_1637);
                            float2 d1_91 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_91);
                            float2 d2_91 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_91);
                            float2 dw_91 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_91);
                            float2 ng_91 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_91);
                            float2 nu_91 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_91);
                            float _rcp_1638 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_91, d2_91), dw_91).x);
                            float _rcp_1639 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_91, d2_91), dw_91).y);
                            float2 r_184 = make_float2(_rcp_1638, _rcp_1639);
                            float2 r_183 = r_184;
                            float2 _f32x2_mul_a_1640 = mul_f32x2_noftz(ng_91, nu_91);
                            float2 _f32x2_mul_b_1641 = mul_f32x2_noftz(r_183, make_float2(bl, bl));
                            float2 m_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_91) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1640), "l"(*(const unsigned long long*)&_f32x2_mul_b_1641));
                            float2 ns_91 = make_float2(((x_92.x < 0.0f) ? p_91.x : 1.0f), ((x_92.y < 0.0f) ? p_91.y : 1.0f));
                            gate_packed_2[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_91, ns_91))) | sign_92;
                            float2 x_93 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[29]));
                            float2 v_93 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[29]));
                            unsigned int sign_93 = (gate_packed_2[29] ^ up_packed_2[29]) & 2147516416u;
                            float _abs_1642;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1642) : "f"(x_93.x));
                            float _abs_1643;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1643) : "f"(x_93.y));
                            float2 ax_92 = make_float2(_abs_1642, _abs_1643);
                            float _abs_1644;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1644) : "f"(v_93.x));
                            float _abs_1645;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1645) : "f"(v_93.y));
                            float2 av_92 = make_float2(_abs_1644, _abs_1645);
                            float2 _f32x2_mul_b_1646 = make_float2(k1, k1);
                            float2 te_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_92) : "l"(*(const unsigned long long*)&ax_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_1646));
                            float2 _f32x2_mul_b_1647 = make_float2(k2, k2);
                            float2 tw_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_92) : "l"(*(const unsigned long long*)&av_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_1647));
                            float _exp2_1648 = approx_exp2(te_92.x);
                            float _exp2_1649 = approx_exp2(te_92.y);
                            float2 e_92 = make_float2(_exp2_1648, _exp2_1649);
                            float _exp2_1650 = approx_exp2(tw_92.x);
                            float _exp2_1651 = approx_exp2(tw_92.y);
                            float2 w_92 = make_float2(_exp2_1650, _exp2_1651);
                            float2 _f32x2_mul_b_1652 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_92) : "l"(*(const unsigned long long*)&ax_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_1652));
                            float _exp2_1653 = approx_exp2_noftz(tp_92.x);
                            float _exp2_1654 = approx_exp2_noftz(tp_92.y);
                            float2 p_92 = make_float2(_exp2_1653, _exp2_1654);
                            float2 d1_92 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_92);
                            float2 d2_92 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_92);
                            float2 dw_92 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_92);
                            float2 ng_92 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_92);
                            float2 nu_92 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_92);
                            float _rcp_1655 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_92, d2_92), dw_92).x);
                            float _rcp_1656 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_92, d2_92), dw_92).y);
                            float2 r_186 = make_float2(_rcp_1655, _rcp_1656);
                            float2 r_185 = r_186;
                            float2 _f32x2_mul_a_1657 = mul_f32x2_noftz(ng_92, nu_92);
                            float2 _f32x2_mul_b_1658 = mul_f32x2_noftz(r_185, make_float2(bl, bl));
                            float2 m_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1657), "l"(*(const unsigned long long*)&_f32x2_mul_b_1658));
                            float2 ns_92 = make_float2(((x_93.x < 0.0f) ? p_92.x : 1.0f), ((x_93.y < 0.0f) ? p_92.y : 1.0f));
                            gate_packed_2[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_92, ns_92))) | sign_93;
                            float2 x_94 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[30]));
                            float2 v_94 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[30]));
                            unsigned int sign_94 = (gate_packed_2[30] ^ up_packed_2[30]) & 2147516416u;
                            float _abs_1659;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1659) : "f"(x_94.x));
                            float _abs_1660;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1660) : "f"(x_94.y));
                            float2 ax_93 = make_float2(_abs_1659, _abs_1660);
                            float _abs_1661;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1661) : "f"(v_94.x));
                            float _abs_1662;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1662) : "f"(v_94.y));
                            float2 av_93 = make_float2(_abs_1661, _abs_1662);
                            float2 _f32x2_mul_b_1663 = make_float2(k1, k1);
                            float2 te_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_93) : "l"(*(const unsigned long long*)&ax_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_1663));
                            float2 _f32x2_mul_b_1664 = make_float2(k2, k2);
                            float2 tw_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_93) : "l"(*(const unsigned long long*)&av_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_1664));
                            float _exp2_1665 = approx_exp2(te_93.x);
                            float _exp2_1666 = approx_exp2(te_93.y);
                            float2 e_93 = make_float2(_exp2_1665, _exp2_1666);
                            float _exp2_1667 = approx_exp2(tw_93.x);
                            float _exp2_1668 = approx_exp2(tw_93.y);
                            float2 w_93 = make_float2(_exp2_1667, _exp2_1668);
                            float2 _f32x2_mul_b_1669 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_93) : "l"(*(const unsigned long long*)&ax_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_1669));
                            float _exp2_1670 = approx_exp2_noftz(tp_93.x);
                            float _exp2_1671 = approx_exp2_noftz(tp_93.y);
                            float2 p_93 = make_float2(_exp2_1670, _exp2_1671);
                            float2 d1_93 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_93);
                            float2 d2_93 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_93);
                            float2 dw_93 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_93);
                            float2 ng_93 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_93);
                            float2 nu_93 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_93);
                            float _rcp_1672 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_93, d2_93), dw_93).x);
                            float _rcp_1673 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_93, d2_93), dw_93).y);
                            float2 r_188 = make_float2(_rcp_1672, _rcp_1673);
                            float2 r_187 = r_188;
                            float2 _f32x2_mul_a_1674 = mul_f32x2_noftz(ng_93, nu_93);
                            float2 _f32x2_mul_b_1675 = mul_f32x2_noftz(r_187, make_float2(bl, bl));
                            float2 m_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_93) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1674), "l"(*(const unsigned long long*)&_f32x2_mul_b_1675));
                            float2 ns_93 = make_float2(((x_94.x < 0.0f) ? p_93.x : 1.0f), ((x_94.y < 0.0f) ? p_93.y : 1.0f));
                            gate_packed_2[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_93, ns_93))) | sign_94;
                            float2 x_95 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[31]));
                            float2 v_95 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[31]));
                            unsigned int sign_95 = (gate_packed_2[31] ^ up_packed_2[31]) & 2147516416u;
                            float _abs_1676;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1676) : "f"(x_95.x));
                            float _abs_1677;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1677) : "f"(x_95.y));
                            float2 ax_94 = make_float2(_abs_1676, _abs_1677);
                            float _abs_1678;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1678) : "f"(v_95.x));
                            float _abs_1679;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1679) : "f"(v_95.y));
                            float2 av_94 = make_float2(_abs_1678, _abs_1679);
                            float2 _f32x2_mul_b_1680 = make_float2(k1, k1);
                            float2 te_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_94) : "l"(*(const unsigned long long*)&ax_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_1680));
                            float2 _f32x2_mul_b_1681 = make_float2(k2, k2);
                            float2 tw_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_94) : "l"(*(const unsigned long long*)&av_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_1681));
                            float _exp2_1682 = approx_exp2(te_94.x);
                            float _exp2_1683 = approx_exp2(te_94.y);
                            float2 e_94 = make_float2(_exp2_1682, _exp2_1683);
                            float _exp2_1684 = approx_exp2(tw_94.x);
                            float _exp2_1685 = approx_exp2(tw_94.y);
                            float2 w_94 = make_float2(_exp2_1684, _exp2_1685);
                            float2 _f32x2_mul_b_1686 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_94) : "l"(*(const unsigned long long*)&ax_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_1686));
                            float _exp2_1687 = approx_exp2_noftz(tp_94.x);
                            float _exp2_1688 = approx_exp2_noftz(tp_94.y);
                            float2 p_94 = make_float2(_exp2_1687, _exp2_1688);
                            float2 d1_94 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_94);
                            float2 d2_94 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_94);
                            float2 dw_94 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_94);
                            float2 ng_94 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_94);
                            float2 nu_94 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_94);
                            float _rcp_1689 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_94, d2_94), dw_94).x);
                            float _rcp_1690 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_94, d2_94), dw_94).y);
                            float2 r_190 = make_float2(_rcp_1689, _rcp_1690);
                            float2 r_189 = r_190;
                            float2 _f32x2_mul_a_1691 = mul_f32x2_noftz(ng_94, nu_94);
                            float2 _f32x2_mul_b_1692 = mul_f32x2_noftz(r_189, make_float2(bl, bl));
                            float2 m_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1691), "l"(*(const unsigned long long*)&_f32x2_mul_b_1692));
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
                                    int row_21 = warp_16 * 32 + half_17 * 16 + lane_16 % 16;
                                    int col_18 = col_tile_17 * 16 + lane_16 / 16 * 8;
                                    unsigned int address_17 = d_smem_addr + 16384 + (unsigned int)((row_21 * 32 + col_18) * 2);
                                    address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                    int offset_18 = 16 + half_17 * 8 + col_tile_17 * 4;
                                    uint32_t _stmatrix_addr_1693 = static_cast<uint32_t>(address_17);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1693), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_18 + 3]))
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
                                    int row_22 = warp_17 * 32 + half_18 * 16 + lane_17 % 16;
                                    int col_19 = col_tile_18 * 16 + lane_17 / 16 * 8;
                                    unsigned int address_18 = d_smem_addr + (unsigned int)((row_22 * 32 + col_19) * 2);
                                    address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                    int offset_19 = half_18 * 8 + col_tile_18 * 4;
                                    uint32_t _stmatrix_addr_1694 = static_cast<uint32_t>(address_18);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1694), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_19 + 3]))
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
                                    int row_23 = warp_18 * 32 + half_19 * 16 + lane_18 % 16;
                                    int col_20 = col_tile_19 * 16 + lane_18 / 16 * 8;
                                    unsigned int address_19 = d_smem_addr + 8192 + (unsigned int)((row_23 * 32 + col_20) * 2);
                                    address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                    int offset_20 = half_19 * 8 + col_tile_19 * 4;
                                    uint32_t _stmatrix_addr_1695 = static_cast<uint32_t>(address_19);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1695), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_20 + 3]))
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
                                    int row_24 = warp_19 * 32 + half_20 * 16 + lane_19 % 16;
                                    int col_21 = col_tile_20 * 16 + lane_19 / 16 * 8;
                                    unsigned int address_20 = d_smem_addr + 16384 + (unsigned int)((row_24 * 32 + col_21) * 2);
                                    address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                    int offset_21 = 16 + half_20 * 8 + col_tile_20 * 4;
                                    uint32_t _stmatrix_addr_1696 = static_cast<uint32_t>(address_20);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1696), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 3]))
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
                                    int row_25 = warp_20 * 32 + half_21 * 16 + lane_20 % 16;
                                    int col_22 = col_tile_21 * 16 + lane_20 / 16 * 8;
                                    unsigned int address_21 = d_smem_addr + (unsigned int)((row_25 * 32 + col_22) * 2);
                                    address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                    int offset_22 = 16 + half_21 * 8 + col_tile_21 * 4;
                                    uint32_t _stmatrix_addr_1697 = static_cast<uint32_t>(address_21);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1697), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 3]))
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
                            float _abs_1698;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1698) : "f"(x_96.x));
                            float _abs_1699;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1699) : "f"(x_96.y));
                            float2 ax_95 = make_float2(_abs_1698, _abs_1699);
                            float _abs_1700;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1700) : "f"(v_96.x));
                            float _abs_1701;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1701) : "f"(v_96.y));
                            float2 av_95 = make_float2(_abs_1700, _abs_1701);
                            float2 _f32x2_mul_b_1702 = make_float2(k1, k1);
                            float2 te_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_95) : "l"(*(const unsigned long long*)&ax_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_1702));
                            float2 _f32x2_mul_b_1703 = make_float2(k2, k2);
                            float2 tw_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_95) : "l"(*(const unsigned long long*)&av_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_1703));
                            float _exp2_1704 = approx_exp2(te_95.x);
                            float _exp2_1705 = approx_exp2(te_95.y);
                            float2 e_95 = make_float2(_exp2_1704, _exp2_1705);
                            float _exp2_1706 = approx_exp2(tw_95.x);
                            float _exp2_1707 = approx_exp2(tw_95.y);
                            float2 w_95 = make_float2(_exp2_1706, _exp2_1707);
                            float2 _f32x2_mul_b_1708 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_95) : "l"(*(const unsigned long long*)&ax_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_1708));
                            float _exp2_1709 = approx_exp2_noftz(tp_95.x);
                            float _exp2_1710 = approx_exp2_noftz(tp_95.y);
                            float2 p_95 = make_float2(_exp2_1709, _exp2_1710);
                            float2 d1_95 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_95);
                            float2 d2_95 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_95);
                            float2 dw_95 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_95);
                            float2 ng_95 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_95);
                            float2 nu_95 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_95);
                            float _rcp_1711 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_95, d2_95), dw_95).x);
                            float _rcp_1712 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_95, d2_95), dw_95).y);
                            float2 r_192 = make_float2(_rcp_1711, _rcp_1712);
                            float2 r_191 = r_192;
                            float2 _f32x2_mul_a_1713 = mul_f32x2_noftz(ng_95, nu_95);
                            float2 _f32x2_mul_b_1714 = mul_f32x2_noftz(r_191, make_float2(bl, bl));
                            float2 m_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_95) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1713), "l"(*(const unsigned long long*)&_f32x2_mul_b_1714));
                            float2 ns_95 = make_float2(((x_96.x < 0.0f) ? p_95.x : 1.0f), ((x_96.y < 0.0f) ? p_95.y : 1.0f));
                            gate_packed_3[0] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_95, ns_95))) | sign_96;
                            float2 x_97 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[1]));
                            float2 v_97 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[1]));
                            unsigned int sign_97 = (gate_packed_3[1] ^ up_packed_3[1]) & 2147516416u;
                            float _abs_1715;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1715) : "f"(x_97.x));
                            float _abs_1716;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1716) : "f"(x_97.y));
                            float2 ax_96 = make_float2(_abs_1715, _abs_1716);
                            float _abs_1717;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1717) : "f"(v_97.x));
                            float _abs_1718;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1718) : "f"(v_97.y));
                            float2 av_96 = make_float2(_abs_1717, _abs_1718);
                            float2 _f32x2_mul_b_1719 = make_float2(k1, k1);
                            float2 te_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_96) : "l"(*(const unsigned long long*)&ax_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_1719));
                            float2 _f32x2_mul_b_1720 = make_float2(k2, k2);
                            float2 tw_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_96) : "l"(*(const unsigned long long*)&av_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_1720));
                            float _exp2_1721 = approx_exp2(te_96.x);
                            float _exp2_1722 = approx_exp2(te_96.y);
                            float2 e_96 = make_float2(_exp2_1721, _exp2_1722);
                            float _exp2_1723 = approx_exp2(tw_96.x);
                            float _exp2_1724 = approx_exp2(tw_96.y);
                            float2 w_96 = make_float2(_exp2_1723, _exp2_1724);
                            float2 _f32x2_mul_b_1725 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_96) : "l"(*(const unsigned long long*)&ax_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_1725));
                            float _exp2_1726 = approx_exp2_noftz(tp_96.x);
                            float _exp2_1727 = approx_exp2_noftz(tp_96.y);
                            float2 p_96 = make_float2(_exp2_1726, _exp2_1727);
                            float2 d1_96 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_96);
                            float2 d2_96 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_96);
                            float2 dw_96 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_96);
                            float2 ng_96 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_96);
                            float2 nu_96 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_96);
                            float _rcp_1728 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_96, d2_96), dw_96).x);
                            float _rcp_1729 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_96, d2_96), dw_96).y);
                            float2 r_194 = make_float2(_rcp_1728, _rcp_1729);
                            float2 r_193 = r_194;
                            float2 _f32x2_mul_a_1730 = mul_f32x2_noftz(ng_96, nu_96);
                            float2 _f32x2_mul_b_1731 = mul_f32x2_noftz(r_193, make_float2(bl, bl));
                            float2 m_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_96) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1730), "l"(*(const unsigned long long*)&_f32x2_mul_b_1731));
                            float2 ns_96 = make_float2(((x_97.x < 0.0f) ? p_96.x : 1.0f), ((x_97.y < 0.0f) ? p_96.y : 1.0f));
                            gate_packed_3[1] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_96, ns_96))) | sign_97;
                            float2 x_98 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[2]));
                            float2 v_98 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[2]));
                            unsigned int sign_98 = (gate_packed_3[2] ^ up_packed_3[2]) & 2147516416u;
                            float _abs_1732;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1732) : "f"(x_98.x));
                            float _abs_1733;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1733) : "f"(x_98.y));
                            float2 ax_97 = make_float2(_abs_1732, _abs_1733);
                            float _abs_1734;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1734) : "f"(v_98.x));
                            float _abs_1735;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1735) : "f"(v_98.y));
                            float2 av_97 = make_float2(_abs_1734, _abs_1735);
                            float2 _f32x2_mul_b_1736 = make_float2(k1, k1);
                            float2 te_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_97) : "l"(*(const unsigned long long*)&ax_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_1736));
                            float2 _f32x2_mul_b_1737 = make_float2(k2, k2);
                            float2 tw_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_97) : "l"(*(const unsigned long long*)&av_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_1737));
                            float _exp2_1738 = approx_exp2(te_97.x);
                            float _exp2_1739 = approx_exp2(te_97.y);
                            float2 e_97 = make_float2(_exp2_1738, _exp2_1739);
                            float _exp2_1740 = approx_exp2(tw_97.x);
                            float _exp2_1741 = approx_exp2(tw_97.y);
                            float2 w_97 = make_float2(_exp2_1740, _exp2_1741);
                            float2 _f32x2_mul_b_1742 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_97) : "l"(*(const unsigned long long*)&ax_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_1742));
                            float _exp2_1743 = approx_exp2_noftz(tp_97.x);
                            float _exp2_1744 = approx_exp2_noftz(tp_97.y);
                            float2 p_97 = make_float2(_exp2_1743, _exp2_1744);
                            float2 d1_97 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_97);
                            float2 d2_97 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_97);
                            float2 dw_97 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_97);
                            float2 ng_97 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_97);
                            float2 nu_97 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_97);
                            float _rcp_1745 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_97, d2_97), dw_97).x);
                            float _rcp_1746 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_97, d2_97), dw_97).y);
                            float2 r_196 = make_float2(_rcp_1745, _rcp_1746);
                            float2 r_195 = r_196;
                            float2 _f32x2_mul_a_1747 = mul_f32x2_noftz(ng_97, nu_97);
                            float2 _f32x2_mul_b_1748 = mul_f32x2_noftz(r_195, make_float2(bl, bl));
                            float2 m_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_97) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1747), "l"(*(const unsigned long long*)&_f32x2_mul_b_1748));
                            float2 ns_97 = make_float2(((x_98.x < 0.0f) ? p_97.x : 1.0f), ((x_98.y < 0.0f) ? p_97.y : 1.0f));
                            gate_packed_3[2] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_97, ns_97))) | sign_98;
                            float2 x_99 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[3]));
                            float2 v_99 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[3]));
                            unsigned int sign_99 = (gate_packed_3[3] ^ up_packed_3[3]) & 2147516416u;
                            float _abs_1749;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1749) : "f"(x_99.x));
                            float _abs_1750;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1750) : "f"(x_99.y));
                            float2 ax_98 = make_float2(_abs_1749, _abs_1750);
                            float _abs_1751;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1751) : "f"(v_99.x));
                            float _abs_1752;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1752) : "f"(v_99.y));
                            float2 av_98 = make_float2(_abs_1751, _abs_1752);
                            float2 _f32x2_mul_b_1753 = make_float2(k1, k1);
                            float2 te_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_98) : "l"(*(const unsigned long long*)&ax_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_1753));
                            float2 _f32x2_mul_b_1754 = make_float2(k2, k2);
                            float2 tw_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_98) : "l"(*(const unsigned long long*)&av_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_1754));
                            float _exp2_1755 = approx_exp2(te_98.x);
                            float _exp2_1756 = approx_exp2(te_98.y);
                            float2 e_98 = make_float2(_exp2_1755, _exp2_1756);
                            float _exp2_1757 = approx_exp2(tw_98.x);
                            float _exp2_1758 = approx_exp2(tw_98.y);
                            float2 w_98 = make_float2(_exp2_1757, _exp2_1758);
                            float2 _f32x2_mul_b_1759 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_98) : "l"(*(const unsigned long long*)&ax_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_1759));
                            float _exp2_1760 = approx_exp2_noftz(tp_98.x);
                            float _exp2_1761 = approx_exp2_noftz(tp_98.y);
                            float2 p_98 = make_float2(_exp2_1760, _exp2_1761);
                            float2 d1_98 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_98);
                            float2 d2_98 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_98);
                            float2 dw_98 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_98);
                            float2 ng_98 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_98);
                            float2 nu_98 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_98);
                            float _rcp_1762 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_98, d2_98), dw_98).x);
                            float _rcp_1763 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_98, d2_98), dw_98).y);
                            float2 r_198 = make_float2(_rcp_1762, _rcp_1763);
                            float2 r_197 = r_198;
                            float2 _f32x2_mul_a_1764 = mul_f32x2_noftz(ng_98, nu_98);
                            float2 _f32x2_mul_b_1765 = mul_f32x2_noftz(r_197, make_float2(bl, bl));
                            float2 m_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1764), "l"(*(const unsigned long long*)&_f32x2_mul_b_1765));
                            float2 ns_98 = make_float2(((x_99.x < 0.0f) ? p_98.x : 1.0f), ((x_99.y < 0.0f) ? p_98.y : 1.0f));
                            gate_packed_3[3] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_98, ns_98))) | sign_99;
                            float2 x_100 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[4]));
                            float2 v_100 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[4]));
                            unsigned int sign_100 = (gate_packed_3[4] ^ up_packed_3[4]) & 2147516416u;
                            float _abs_1766;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1766) : "f"(x_100.x));
                            float _abs_1767;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1767) : "f"(x_100.y));
                            float2 ax_99 = make_float2(_abs_1766, _abs_1767);
                            float _abs_1768;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1768) : "f"(v_100.x));
                            float _abs_1769;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1769) : "f"(v_100.y));
                            float2 av_99 = make_float2(_abs_1768, _abs_1769);
                            float2 _f32x2_mul_b_1770 = make_float2(k1, k1);
                            float2 te_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_99) : "l"(*(const unsigned long long*)&ax_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_1770));
                            float2 _f32x2_mul_b_1771 = make_float2(k2, k2);
                            float2 tw_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_99) : "l"(*(const unsigned long long*)&av_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_1771));
                            float _exp2_1772 = approx_exp2(te_99.x);
                            float _exp2_1773 = approx_exp2(te_99.y);
                            float2 e_99 = make_float2(_exp2_1772, _exp2_1773);
                            float _exp2_1774 = approx_exp2(tw_99.x);
                            float _exp2_1775 = approx_exp2(tw_99.y);
                            float2 w_99 = make_float2(_exp2_1774, _exp2_1775);
                            float2 _f32x2_mul_b_1776 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_99) : "l"(*(const unsigned long long*)&ax_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_1776));
                            float _exp2_1777 = approx_exp2_noftz(tp_99.x);
                            float _exp2_1778 = approx_exp2_noftz(tp_99.y);
                            float2 p_99 = make_float2(_exp2_1777, _exp2_1778);
                            float2 d1_99 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_99);
                            float2 d2_99 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_99);
                            float2 dw_99 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_99);
                            float2 ng_99 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_99);
                            float2 nu_99 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_99);
                            float _rcp_1779 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_99, d2_99), dw_99).x);
                            float _rcp_1780 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_99, d2_99), dw_99).y);
                            float2 r_200 = make_float2(_rcp_1779, _rcp_1780);
                            float2 r_199 = r_200;
                            float2 _f32x2_mul_a_1781 = mul_f32x2_noftz(ng_99, nu_99);
                            float2 _f32x2_mul_b_1782 = mul_f32x2_noftz(r_199, make_float2(bl, bl));
                            float2 m_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_99) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1781), "l"(*(const unsigned long long*)&_f32x2_mul_b_1782));
                            float2 ns_99 = make_float2(((x_100.x < 0.0f) ? p_99.x : 1.0f), ((x_100.y < 0.0f) ? p_99.y : 1.0f));
                            gate_packed_3[4] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_99, ns_99))) | sign_100;
                            float2 x_101 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[5]));
                            float2 v_101 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[5]));
                            unsigned int sign_101 = (gate_packed_3[5] ^ up_packed_3[5]) & 2147516416u;
                            float _abs_1783;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1783) : "f"(x_101.x));
                            float _abs_1784;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1784) : "f"(x_101.y));
                            float2 ax_100 = make_float2(_abs_1783, _abs_1784);
                            float _abs_1785;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1785) : "f"(v_101.x));
                            float _abs_1786;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1786) : "f"(v_101.y));
                            float2 av_100 = make_float2(_abs_1785, _abs_1786);
                            float2 _f32x2_mul_b_1787 = make_float2(k1, k1);
                            float2 te_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_100) : "l"(*(const unsigned long long*)&ax_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_1787));
                            float2 _f32x2_mul_b_1788 = make_float2(k2, k2);
                            float2 tw_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_100) : "l"(*(const unsigned long long*)&av_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_1788));
                            float _exp2_1789 = approx_exp2(te_100.x);
                            float _exp2_1790 = approx_exp2(te_100.y);
                            float2 e_100 = make_float2(_exp2_1789, _exp2_1790);
                            float _exp2_1791 = approx_exp2(tw_100.x);
                            float _exp2_1792 = approx_exp2(tw_100.y);
                            float2 w_100 = make_float2(_exp2_1791, _exp2_1792);
                            float2 _f32x2_mul_b_1793 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_100) : "l"(*(const unsigned long long*)&ax_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_1793));
                            float _exp2_1794 = approx_exp2_noftz(tp_100.x);
                            float _exp2_1795 = approx_exp2_noftz(tp_100.y);
                            float2 p_100 = make_float2(_exp2_1794, _exp2_1795);
                            float2 d1_100 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_100);
                            float2 d2_100 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_100);
                            float2 dw_100 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_100);
                            float2 ng_100 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_100);
                            float2 nu_100 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_100);
                            float _rcp_1796 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_100, d2_100), dw_100).x);
                            float _rcp_1797 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_100, d2_100), dw_100).y);
                            float2 r_202 = make_float2(_rcp_1796, _rcp_1797);
                            float2 r_201 = r_202;
                            float2 _f32x2_mul_a_1798 = mul_f32x2_noftz(ng_100, nu_100);
                            float2 _f32x2_mul_b_1799 = mul_f32x2_noftz(r_201, make_float2(bl, bl));
                            float2 m_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_100) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1798), "l"(*(const unsigned long long*)&_f32x2_mul_b_1799));
                            float2 ns_100 = make_float2(((x_101.x < 0.0f) ? p_100.x : 1.0f), ((x_101.y < 0.0f) ? p_100.y : 1.0f));
                            gate_packed_3[5] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_100, ns_100))) | sign_101;
                            float2 x_102 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[6]));
                            float2 v_102 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[6]));
                            unsigned int sign_102 = (gate_packed_3[6] ^ up_packed_3[6]) & 2147516416u;
                            float _abs_1800;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1800) : "f"(x_102.x));
                            float _abs_1801;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1801) : "f"(x_102.y));
                            float2 ax_101 = make_float2(_abs_1800, _abs_1801);
                            float _abs_1802;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1802) : "f"(v_102.x));
                            float _abs_1803;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1803) : "f"(v_102.y));
                            float2 av_101 = make_float2(_abs_1802, _abs_1803);
                            float2 _f32x2_mul_b_1804 = make_float2(k1, k1);
                            float2 te_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_101) : "l"(*(const unsigned long long*)&ax_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_1804));
                            float2 _f32x2_mul_b_1805 = make_float2(k2, k2);
                            float2 tw_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_101) : "l"(*(const unsigned long long*)&av_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_1805));
                            float _exp2_1806 = approx_exp2(te_101.x);
                            float _exp2_1807 = approx_exp2(te_101.y);
                            float2 e_101 = make_float2(_exp2_1806, _exp2_1807);
                            float _exp2_1808 = approx_exp2(tw_101.x);
                            float _exp2_1809 = approx_exp2(tw_101.y);
                            float2 w_101 = make_float2(_exp2_1808, _exp2_1809);
                            float2 _f32x2_mul_b_1810 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_101) : "l"(*(const unsigned long long*)&ax_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_1810));
                            float _exp2_1811 = approx_exp2_noftz(tp_101.x);
                            float _exp2_1812 = approx_exp2_noftz(tp_101.y);
                            float2 p_101 = make_float2(_exp2_1811, _exp2_1812);
                            float2 d1_101 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_101);
                            float2 d2_101 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_101);
                            float2 dw_101 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_101);
                            float2 ng_101 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_101);
                            float2 nu_101 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_101);
                            float _rcp_1813 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_101, d2_101), dw_101).x);
                            float _rcp_1814 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_101, d2_101), dw_101).y);
                            float2 r_204 = make_float2(_rcp_1813, _rcp_1814);
                            float2 r_203 = r_204;
                            float2 _f32x2_mul_a_1815 = mul_f32x2_noftz(ng_101, nu_101);
                            float2 _f32x2_mul_b_1816 = mul_f32x2_noftz(r_203, make_float2(bl, bl));
                            float2 m_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_101) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1815), "l"(*(const unsigned long long*)&_f32x2_mul_b_1816));
                            float2 ns_101 = make_float2(((x_102.x < 0.0f) ? p_101.x : 1.0f), ((x_102.y < 0.0f) ? p_101.y : 1.0f));
                            gate_packed_3[6] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_101, ns_101))) | sign_102;
                            float2 x_103 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[7]));
                            float2 v_103 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[7]));
                            unsigned int sign_103 = (gate_packed_3[7] ^ up_packed_3[7]) & 2147516416u;
                            float _abs_1817;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1817) : "f"(x_103.x));
                            float _abs_1818;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1818) : "f"(x_103.y));
                            float2 ax_102 = make_float2(_abs_1817, _abs_1818);
                            float _abs_1819;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1819) : "f"(v_103.x));
                            float _abs_1820;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1820) : "f"(v_103.y));
                            float2 av_102 = make_float2(_abs_1819, _abs_1820);
                            float2 _f32x2_mul_b_1821 = make_float2(k1, k1);
                            float2 te_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_102) : "l"(*(const unsigned long long*)&ax_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_1821));
                            float2 _f32x2_mul_b_1822 = make_float2(k2, k2);
                            float2 tw_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_102) : "l"(*(const unsigned long long*)&av_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_1822));
                            float _exp2_1823 = approx_exp2(te_102.x);
                            float _exp2_1824 = approx_exp2(te_102.y);
                            float2 e_102 = make_float2(_exp2_1823, _exp2_1824);
                            float _exp2_1825 = approx_exp2(tw_102.x);
                            float _exp2_1826 = approx_exp2(tw_102.y);
                            float2 w_102 = make_float2(_exp2_1825, _exp2_1826);
                            float2 _f32x2_mul_b_1827 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_102) : "l"(*(const unsigned long long*)&ax_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_1827));
                            float _exp2_1828 = approx_exp2_noftz(tp_102.x);
                            float _exp2_1829 = approx_exp2_noftz(tp_102.y);
                            float2 p_102 = make_float2(_exp2_1828, _exp2_1829);
                            float2 d1_102 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_102);
                            float2 d2_102 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_102);
                            float2 dw_102 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_102);
                            float2 ng_102 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_102);
                            float2 nu_102 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_102);
                            float _rcp_1830 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_102, d2_102), dw_102).x);
                            float _rcp_1831 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_102, d2_102), dw_102).y);
                            float2 r_206 = make_float2(_rcp_1830, _rcp_1831);
                            float2 r_205 = r_206;
                            float2 _f32x2_mul_a_1832 = mul_f32x2_noftz(ng_102, nu_102);
                            float2 _f32x2_mul_b_1833 = mul_f32x2_noftz(r_205, make_float2(bl, bl));
                            float2 m_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_102) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1832), "l"(*(const unsigned long long*)&_f32x2_mul_b_1833));
                            float2 ns_102 = make_float2(((x_103.x < 0.0f) ? p_102.x : 1.0f), ((x_103.y < 0.0f) ? p_102.y : 1.0f));
                            gate_packed_3[7] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_102, ns_102))) | sign_103;
                            float2 x_104 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[8]));
                            float2 v_104 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[8]));
                            unsigned int sign_104 = (gate_packed_3[8] ^ up_packed_3[8]) & 2147516416u;
                            float _abs_1834;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1834) : "f"(x_104.x));
                            float _abs_1835;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1835) : "f"(x_104.y));
                            float2 ax_103 = make_float2(_abs_1834, _abs_1835);
                            float _abs_1836;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1836) : "f"(v_104.x));
                            float _abs_1837;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1837) : "f"(v_104.y));
                            float2 av_103 = make_float2(_abs_1836, _abs_1837);
                            float2 _f32x2_mul_b_1838 = make_float2(k1, k1);
                            float2 te_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_103) : "l"(*(const unsigned long long*)&ax_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_1838));
                            float2 _f32x2_mul_b_1839 = make_float2(k2, k2);
                            float2 tw_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_103) : "l"(*(const unsigned long long*)&av_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_1839));
                            float _exp2_1840 = approx_exp2(te_103.x);
                            float _exp2_1841 = approx_exp2(te_103.y);
                            float2 e_103 = make_float2(_exp2_1840, _exp2_1841);
                            float _exp2_1842 = approx_exp2(tw_103.x);
                            float _exp2_1843 = approx_exp2(tw_103.y);
                            float2 w_103 = make_float2(_exp2_1842, _exp2_1843);
                            float2 _f32x2_mul_b_1844 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_103) : "l"(*(const unsigned long long*)&ax_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_1844));
                            float _exp2_1845 = approx_exp2_noftz(tp_103.x);
                            float _exp2_1846 = approx_exp2_noftz(tp_103.y);
                            float2 p_103 = make_float2(_exp2_1845, _exp2_1846);
                            float2 d1_103 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_103);
                            float2 d2_103 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_103);
                            float2 dw_103 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_103);
                            float2 ng_103 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_103);
                            float2 nu_103 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_103);
                            float _rcp_1847 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_103, d2_103), dw_103).x);
                            float _rcp_1848 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_103, d2_103), dw_103).y);
                            float2 r_208 = make_float2(_rcp_1847, _rcp_1848);
                            float2 r_207 = r_208;
                            float2 _f32x2_mul_a_1849 = mul_f32x2_noftz(ng_103, nu_103);
                            float2 _f32x2_mul_b_1850 = mul_f32x2_noftz(r_207, make_float2(bl, bl));
                            float2 m_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_103) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1849), "l"(*(const unsigned long long*)&_f32x2_mul_b_1850));
                            float2 ns_103 = make_float2(((x_104.x < 0.0f) ? p_103.x : 1.0f), ((x_104.y < 0.0f) ? p_103.y : 1.0f));
                            gate_packed_3[8] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_103, ns_103))) | sign_104;
                            float2 x_105 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[9]));
                            float2 v_105 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[9]));
                            unsigned int sign_105 = (gate_packed_3[9] ^ up_packed_3[9]) & 2147516416u;
                            float _abs_1851;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1851) : "f"(x_105.x));
                            float _abs_1852;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1852) : "f"(x_105.y));
                            float2 ax_104 = make_float2(_abs_1851, _abs_1852);
                            float _abs_1853;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1853) : "f"(v_105.x));
                            float _abs_1854;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1854) : "f"(v_105.y));
                            float2 av_104 = make_float2(_abs_1853, _abs_1854);
                            float2 _f32x2_mul_b_1855 = make_float2(k1, k1);
                            float2 te_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_104) : "l"(*(const unsigned long long*)&ax_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_1855));
                            float2 _f32x2_mul_b_1856 = make_float2(k2, k2);
                            float2 tw_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_104) : "l"(*(const unsigned long long*)&av_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_1856));
                            float _exp2_1857 = approx_exp2(te_104.x);
                            float _exp2_1858 = approx_exp2(te_104.y);
                            float2 e_104 = make_float2(_exp2_1857, _exp2_1858);
                            float _exp2_1859 = approx_exp2(tw_104.x);
                            float _exp2_1860 = approx_exp2(tw_104.y);
                            float2 w_104 = make_float2(_exp2_1859, _exp2_1860);
                            float2 _f32x2_mul_b_1861 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_104) : "l"(*(const unsigned long long*)&ax_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_1861));
                            float _exp2_1862 = approx_exp2_noftz(tp_104.x);
                            float _exp2_1863 = approx_exp2_noftz(tp_104.y);
                            float2 p_104 = make_float2(_exp2_1862, _exp2_1863);
                            float2 d1_104 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_104);
                            float2 d2_104 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_104);
                            float2 dw_104 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_104);
                            float2 ng_104 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_104);
                            float2 nu_104 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_104);
                            float _rcp_1864 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_104, d2_104), dw_104).x);
                            float _rcp_1865 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_104, d2_104), dw_104).y);
                            float2 r_210 = make_float2(_rcp_1864, _rcp_1865);
                            float2 r_209 = r_210;
                            float2 _f32x2_mul_a_1866 = mul_f32x2_noftz(ng_104, nu_104);
                            float2 _f32x2_mul_b_1867 = mul_f32x2_noftz(r_209, make_float2(bl, bl));
                            float2 m_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_104) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1866), "l"(*(const unsigned long long*)&_f32x2_mul_b_1867));
                            float2 ns_104 = make_float2(((x_105.x < 0.0f) ? p_104.x : 1.0f), ((x_105.y < 0.0f) ? p_104.y : 1.0f));
                            gate_packed_3[9] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_104, ns_104))) | sign_105;
                            float2 x_106 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[10]));
                            float2 v_106 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[10]));
                            unsigned int sign_106 = (gate_packed_3[10] ^ up_packed_3[10]) & 2147516416u;
                            float _abs_1868;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1868) : "f"(x_106.x));
                            float _abs_1869;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1869) : "f"(x_106.y));
                            float2 ax_105 = make_float2(_abs_1868, _abs_1869);
                            float _abs_1870;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1870) : "f"(v_106.x));
                            float _abs_1871;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1871) : "f"(v_106.y));
                            float2 av_105 = make_float2(_abs_1870, _abs_1871);
                            float2 _f32x2_mul_b_1872 = make_float2(k1, k1);
                            float2 te_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_105) : "l"(*(const unsigned long long*)&ax_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_1872));
                            float2 _f32x2_mul_b_1873 = make_float2(k2, k2);
                            float2 tw_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_105) : "l"(*(const unsigned long long*)&av_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_1873));
                            float _exp2_1874 = approx_exp2(te_105.x);
                            float _exp2_1875 = approx_exp2(te_105.y);
                            float2 e_105 = make_float2(_exp2_1874, _exp2_1875);
                            float _exp2_1876 = approx_exp2(tw_105.x);
                            float _exp2_1877 = approx_exp2(tw_105.y);
                            float2 w_105 = make_float2(_exp2_1876, _exp2_1877);
                            float2 _f32x2_mul_b_1878 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_105) : "l"(*(const unsigned long long*)&ax_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_1878));
                            float _exp2_1879 = approx_exp2_noftz(tp_105.x);
                            float _exp2_1880 = approx_exp2_noftz(tp_105.y);
                            float2 p_105 = make_float2(_exp2_1879, _exp2_1880);
                            float2 d1_105 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_105);
                            float2 d2_105 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_105);
                            float2 dw_105 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_105);
                            float2 ng_105 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_105);
                            float2 nu_105 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_105);
                            float _rcp_1881 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_105, d2_105), dw_105).x);
                            float _rcp_1882 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_105, d2_105), dw_105).y);
                            float2 r_212 = make_float2(_rcp_1881, _rcp_1882);
                            float2 r_211 = r_212;
                            float2 _f32x2_mul_a_1883 = mul_f32x2_noftz(ng_105, nu_105);
                            float2 _f32x2_mul_b_1884 = mul_f32x2_noftz(r_211, make_float2(bl, bl));
                            float2 m_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1883), "l"(*(const unsigned long long*)&_f32x2_mul_b_1884));
                            float2 ns_105 = make_float2(((x_106.x < 0.0f) ? p_105.x : 1.0f), ((x_106.y < 0.0f) ? p_105.y : 1.0f));
                            gate_packed_3[10] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_105, ns_105))) | sign_106;
                            float2 x_107 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[11]));
                            float2 v_107 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[11]));
                            unsigned int sign_107 = (gate_packed_3[11] ^ up_packed_3[11]) & 2147516416u;
                            float _abs_1885;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1885) : "f"(x_107.x));
                            float _abs_1886;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1886) : "f"(x_107.y));
                            float2 ax_106 = make_float2(_abs_1885, _abs_1886);
                            float _abs_1887;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1887) : "f"(v_107.x));
                            float _abs_1888;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1888) : "f"(v_107.y));
                            float2 av_106 = make_float2(_abs_1887, _abs_1888);
                            float2 _f32x2_mul_b_1889 = make_float2(k1, k1);
                            float2 te_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_106) : "l"(*(const unsigned long long*)&ax_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_1889));
                            float2 _f32x2_mul_b_1890 = make_float2(k2, k2);
                            float2 tw_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_106) : "l"(*(const unsigned long long*)&av_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_1890));
                            float _exp2_1891 = approx_exp2(te_106.x);
                            float _exp2_1892 = approx_exp2(te_106.y);
                            float2 e_106 = make_float2(_exp2_1891, _exp2_1892);
                            float _exp2_1893 = approx_exp2(tw_106.x);
                            float _exp2_1894 = approx_exp2(tw_106.y);
                            float2 w_106 = make_float2(_exp2_1893, _exp2_1894);
                            float2 _f32x2_mul_b_1895 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_106) : "l"(*(const unsigned long long*)&ax_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_1895));
                            float _exp2_1896 = approx_exp2_noftz(tp_106.x);
                            float _exp2_1897 = approx_exp2_noftz(tp_106.y);
                            float2 p_106 = make_float2(_exp2_1896, _exp2_1897);
                            float2 d1_106 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_106);
                            float2 d2_106 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_106);
                            float2 dw_106 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_106);
                            float2 ng_106 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_106);
                            float2 nu_106 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_106);
                            float _rcp_1898 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_106, d2_106), dw_106).x);
                            float _rcp_1899 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_106, d2_106), dw_106).y);
                            float2 r_214 = make_float2(_rcp_1898, _rcp_1899);
                            float2 r_213 = r_214;
                            float2 _f32x2_mul_a_1900 = mul_f32x2_noftz(ng_106, nu_106);
                            float2 _f32x2_mul_b_1901 = mul_f32x2_noftz(r_213, make_float2(bl, bl));
                            float2 m_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_106) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1900), "l"(*(const unsigned long long*)&_f32x2_mul_b_1901));
                            float2 ns_106 = make_float2(((x_107.x < 0.0f) ? p_106.x : 1.0f), ((x_107.y < 0.0f) ? p_106.y : 1.0f));
                            gate_packed_3[11] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_106, ns_106))) | sign_107;
                            float2 x_108 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[12]));
                            float2 v_108 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[12]));
                            unsigned int sign_108 = (gate_packed_3[12] ^ up_packed_3[12]) & 2147516416u;
                            float _abs_1902;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1902) : "f"(x_108.x));
                            float _abs_1903;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1903) : "f"(x_108.y));
                            float2 ax_107 = make_float2(_abs_1902, _abs_1903);
                            float _abs_1904;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1904) : "f"(v_108.x));
                            float _abs_1905;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1905) : "f"(v_108.y));
                            float2 av_107 = make_float2(_abs_1904, _abs_1905);
                            float2 _f32x2_mul_b_1906 = make_float2(k1, k1);
                            float2 te_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_107) : "l"(*(const unsigned long long*)&ax_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1906));
                            float2 _f32x2_mul_b_1907 = make_float2(k2, k2);
                            float2 tw_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_107) : "l"(*(const unsigned long long*)&av_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1907));
                            float _exp2_1908 = approx_exp2(te_107.x);
                            float _exp2_1909 = approx_exp2(te_107.y);
                            float2 e_107 = make_float2(_exp2_1908, _exp2_1909);
                            float _exp2_1910 = approx_exp2(tw_107.x);
                            float _exp2_1911 = approx_exp2(tw_107.y);
                            float2 w_107 = make_float2(_exp2_1910, _exp2_1911);
                            float2 _f32x2_mul_b_1912 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_107) : "l"(*(const unsigned long long*)&ax_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_1912));
                            float _exp2_1913 = approx_exp2_noftz(tp_107.x);
                            float _exp2_1914 = approx_exp2_noftz(tp_107.y);
                            float2 p_107 = make_float2(_exp2_1913, _exp2_1914);
                            float2 d1_107 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_107);
                            float2 d2_107 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_107);
                            float2 dw_107 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_107);
                            float2 ng_107 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_107);
                            float2 nu_107 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_107);
                            float _rcp_1915 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_107, d2_107), dw_107).x);
                            float _rcp_1916 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_107, d2_107), dw_107).y);
                            float2 r_216 = make_float2(_rcp_1915, _rcp_1916);
                            float2 r_215 = r_216;
                            float2 _f32x2_mul_a_1917 = mul_f32x2_noftz(ng_107, nu_107);
                            float2 _f32x2_mul_b_1918 = mul_f32x2_noftz(r_215, make_float2(bl, bl));
                            float2 m_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_107) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1917), "l"(*(const unsigned long long*)&_f32x2_mul_b_1918));
                            float2 ns_107 = make_float2(((x_108.x < 0.0f) ? p_107.x : 1.0f), ((x_108.y < 0.0f) ? p_107.y : 1.0f));
                            gate_packed_3[12] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_107, ns_107))) | sign_108;
                            float2 x_109 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[13]));
                            float2 v_109 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[13]));
                            unsigned int sign_109 = (gate_packed_3[13] ^ up_packed_3[13]) & 2147516416u;
                            float _abs_1919;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1919) : "f"(x_109.x));
                            float _abs_1920;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1920) : "f"(x_109.y));
                            float2 ax_108 = make_float2(_abs_1919, _abs_1920);
                            float _abs_1921;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1921) : "f"(v_109.x));
                            float _abs_1922;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1922) : "f"(v_109.y));
                            float2 av_108 = make_float2(_abs_1921, _abs_1922);
                            float2 _f32x2_mul_b_1923 = make_float2(k1, k1);
                            float2 te_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_108) : "l"(*(const unsigned long long*)&ax_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_1923));
                            float2 _f32x2_mul_b_1924 = make_float2(k2, k2);
                            float2 tw_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_108) : "l"(*(const unsigned long long*)&av_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_1924));
                            float _exp2_1925 = approx_exp2(te_108.x);
                            float _exp2_1926 = approx_exp2(te_108.y);
                            float2 e_108 = make_float2(_exp2_1925, _exp2_1926);
                            float _exp2_1927 = approx_exp2(tw_108.x);
                            float _exp2_1928 = approx_exp2(tw_108.y);
                            float2 w_108 = make_float2(_exp2_1927, _exp2_1928);
                            float2 _f32x2_mul_b_1929 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_108) : "l"(*(const unsigned long long*)&ax_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_1929));
                            float _exp2_1930 = approx_exp2_noftz(tp_108.x);
                            float _exp2_1931 = approx_exp2_noftz(tp_108.y);
                            float2 p_108 = make_float2(_exp2_1930, _exp2_1931);
                            float2 d1_108 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_108);
                            float2 d2_108 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_108);
                            float2 dw_108 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_108);
                            float2 ng_108 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_108);
                            float2 nu_108 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_108);
                            float _rcp_1932 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_108, d2_108), dw_108).x);
                            float _rcp_1933 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_108, d2_108), dw_108).y);
                            float2 r_218 = make_float2(_rcp_1932, _rcp_1933);
                            float2 r_217 = r_218;
                            float2 _f32x2_mul_a_1934 = mul_f32x2_noftz(ng_108, nu_108);
                            float2 _f32x2_mul_b_1935 = mul_f32x2_noftz(r_217, make_float2(bl, bl));
                            float2 m_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_108) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1934), "l"(*(const unsigned long long*)&_f32x2_mul_b_1935));
                            float2 ns_108 = make_float2(((x_109.x < 0.0f) ? p_108.x : 1.0f), ((x_109.y < 0.0f) ? p_108.y : 1.0f));
                            gate_packed_3[13] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_108, ns_108))) | sign_109;
                            float2 x_110 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[14]));
                            float2 v_110 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[14]));
                            unsigned int sign_110 = (gate_packed_3[14] ^ up_packed_3[14]) & 2147516416u;
                            float _abs_1936;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1936) : "f"(x_110.x));
                            float _abs_1937;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1937) : "f"(x_110.y));
                            float2 ax_109 = make_float2(_abs_1936, _abs_1937);
                            float _abs_1938;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1938) : "f"(v_110.x));
                            float _abs_1939;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1939) : "f"(v_110.y));
                            float2 av_109 = make_float2(_abs_1938, _abs_1939);
                            float2 _f32x2_mul_b_1940 = make_float2(k1, k1);
                            float2 te_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_109) : "l"(*(const unsigned long long*)&ax_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_1940));
                            float2 _f32x2_mul_b_1941 = make_float2(k2, k2);
                            float2 tw_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_109) : "l"(*(const unsigned long long*)&av_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_1941));
                            float _exp2_1942 = approx_exp2(te_109.x);
                            float _exp2_1943 = approx_exp2(te_109.y);
                            float2 e_109 = make_float2(_exp2_1942, _exp2_1943);
                            float _exp2_1944 = approx_exp2(tw_109.x);
                            float _exp2_1945 = approx_exp2(tw_109.y);
                            float2 w_109 = make_float2(_exp2_1944, _exp2_1945);
                            float2 _f32x2_mul_b_1946 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_109) : "l"(*(const unsigned long long*)&ax_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_1946));
                            float _exp2_1947 = approx_exp2_noftz(tp_109.x);
                            float _exp2_1948 = approx_exp2_noftz(tp_109.y);
                            float2 p_109 = make_float2(_exp2_1947, _exp2_1948);
                            float2 d1_109 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_109);
                            float2 d2_109 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_109);
                            float2 dw_109 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_109);
                            float2 ng_109 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_109);
                            float2 nu_109 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_109);
                            float _rcp_1949 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_109, d2_109), dw_109).x);
                            float _rcp_1950 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_109, d2_109), dw_109).y);
                            float2 r_220 = make_float2(_rcp_1949, _rcp_1950);
                            float2 r_219 = r_220;
                            float2 _f32x2_mul_a_1951 = mul_f32x2_noftz(ng_109, nu_109);
                            float2 _f32x2_mul_b_1952 = mul_f32x2_noftz(r_219, make_float2(bl, bl));
                            float2 m_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_109) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1951), "l"(*(const unsigned long long*)&_f32x2_mul_b_1952));
                            float2 ns_109 = make_float2(((x_110.x < 0.0f) ? p_109.x : 1.0f), ((x_110.y < 0.0f) ? p_109.y : 1.0f));
                            gate_packed_3[14] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_109, ns_109))) | sign_110;
                            float2 x_111 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[15]));
                            float2 v_111 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[15]));
                            unsigned int sign_111 = (gate_packed_3[15] ^ up_packed_3[15]) & 2147516416u;
                            float _abs_1953;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1953) : "f"(x_111.x));
                            float _abs_1954;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1954) : "f"(x_111.y));
                            float2 ax_110 = make_float2(_abs_1953, _abs_1954);
                            float _abs_1955;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1955) : "f"(v_111.x));
                            float _abs_1956;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1956) : "f"(v_111.y));
                            float2 av_110 = make_float2(_abs_1955, _abs_1956);
                            float2 _f32x2_mul_b_1957 = make_float2(k1, k1);
                            float2 te_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_110) : "l"(*(const unsigned long long*)&ax_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_1957));
                            float2 _f32x2_mul_b_1958 = make_float2(k2, k2);
                            float2 tw_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_110) : "l"(*(const unsigned long long*)&av_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_1958));
                            float _exp2_1959 = approx_exp2(te_110.x);
                            float _exp2_1960 = approx_exp2(te_110.y);
                            float2 e_110 = make_float2(_exp2_1959, _exp2_1960);
                            float _exp2_1961 = approx_exp2(tw_110.x);
                            float _exp2_1962 = approx_exp2(tw_110.y);
                            float2 w_110 = make_float2(_exp2_1961, _exp2_1962);
                            float2 _f32x2_mul_b_1963 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_110) : "l"(*(const unsigned long long*)&ax_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_1963));
                            float _exp2_1964 = approx_exp2_noftz(tp_110.x);
                            float _exp2_1965 = approx_exp2_noftz(tp_110.y);
                            float2 p_110 = make_float2(_exp2_1964, _exp2_1965);
                            float2 d1_110 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_110);
                            float2 d2_110 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_110);
                            float2 dw_110 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_110);
                            float2 ng_110 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_110);
                            float2 nu_110 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_110);
                            float _rcp_1966 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_110, d2_110), dw_110).x);
                            float _rcp_1967 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_110, d2_110), dw_110).y);
                            float2 r_222 = make_float2(_rcp_1966, _rcp_1967);
                            float2 r_221 = r_222;
                            float2 _f32x2_mul_a_1968 = mul_f32x2_noftz(ng_110, nu_110);
                            float2 _f32x2_mul_b_1969 = mul_f32x2_noftz(r_221, make_float2(bl, bl));
                            float2 m_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1968), "l"(*(const unsigned long long*)&_f32x2_mul_b_1969));
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
                                    int row_26 = warp_21 * 32 + half_22 * 16 + lane_21 % 16;
                                    int col_23 = col_tile_22 * 16 + lane_21 / 16 * 8;
                                    unsigned int address_22 = d_smem_addr + 8192 + (unsigned int)((row_26 * 32 + col_23) * 2);
                                    address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                    int offset_23 = half_22 * 8 + col_tile_22 * 4;
                                    uint32_t _stmatrix_addr_1970 = static_cast<uint32_t>(address_22);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1970), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_23 + 3]))
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
                            float _abs_1971;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1971) : "f"(x_112.x));
                            float _abs_1972;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1972) : "f"(x_112.y));
                            float2 ax_111 = make_float2(_abs_1971, _abs_1972);
                            float _abs_1973;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1973) : "f"(v_112.x));
                            float _abs_1974;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1974) : "f"(v_112.y));
                            float2 av_111 = make_float2(_abs_1973, _abs_1974);
                            float2 _f32x2_mul_b_1975 = make_float2(k1, k1);
                            float2 te_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_111) : "l"(*(const unsigned long long*)&ax_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_1975));
                            float2 _f32x2_mul_b_1976 = make_float2(k2, k2);
                            float2 tw_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_111) : "l"(*(const unsigned long long*)&av_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_1976));
                            float _exp2_1977 = approx_exp2(te_111.x);
                            float _exp2_1978 = approx_exp2(te_111.y);
                            float2 e_111 = make_float2(_exp2_1977, _exp2_1978);
                            float _exp2_1979 = approx_exp2(tw_111.x);
                            float _exp2_1980 = approx_exp2(tw_111.y);
                            float2 w_111 = make_float2(_exp2_1979, _exp2_1980);
                            float2 _f32x2_mul_b_1981 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_111) : "l"(*(const unsigned long long*)&ax_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_1981));
                            float _exp2_1982 = approx_exp2_noftz(tp_111.x);
                            float _exp2_1983 = approx_exp2_noftz(tp_111.y);
                            float2 p_111 = make_float2(_exp2_1982, _exp2_1983);
                            float2 d1_111 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_111);
                            float2 d2_111 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_111);
                            float2 dw_111 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_111);
                            float2 ng_111 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_111);
                            float2 nu_111 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_111);
                            float _rcp_1984 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_111, d2_111), dw_111).x);
                            float _rcp_1985 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_111, d2_111), dw_111).y);
                            float2 r_224 = make_float2(_rcp_1984, _rcp_1985);
                            float2 r_223 = r_224;
                            float2 _f32x2_mul_a_1986 = mul_f32x2_noftz(ng_111, nu_111);
                            float2 _f32x2_mul_b_1987 = mul_f32x2_noftz(r_223, make_float2(bl, bl));
                            float2 m_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_111) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1986), "l"(*(const unsigned long long*)&_f32x2_mul_b_1987));
                            float2 ns_111 = make_float2(((x_112.x < 0.0f) ? p_111.x : 1.0f), ((x_112.y < 0.0f) ? p_111.y : 1.0f));
                            gate_packed_3[16] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_111, ns_111))) | sign_112;
                            float2 x_113 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[17]));
                            float2 v_113 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[17]));
                            unsigned int sign_113 = (gate_packed_3[17] ^ up_packed_3[17]) & 2147516416u;
                            float _abs_1988;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1988) : "f"(x_113.x));
                            float _abs_1989;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1989) : "f"(x_113.y));
                            float2 ax_112 = make_float2(_abs_1988, _abs_1989);
                            float _abs_1990;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1990) : "f"(v_113.x));
                            float _abs_1991;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_1991) : "f"(v_113.y));
                            float2 av_112 = make_float2(_abs_1990, _abs_1991);
                            float2 _f32x2_mul_b_1992 = make_float2(k1, k1);
                            float2 te_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_112) : "l"(*(const unsigned long long*)&ax_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_1992));
                            float2 _f32x2_mul_b_1993 = make_float2(k2, k2);
                            float2 tw_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_112) : "l"(*(const unsigned long long*)&av_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_1993));
                            float _exp2_1994 = approx_exp2(te_112.x);
                            float _exp2_1995 = approx_exp2(te_112.y);
                            float2 e_112 = make_float2(_exp2_1994, _exp2_1995);
                            float _exp2_1996 = approx_exp2(tw_112.x);
                            float _exp2_1997 = approx_exp2(tw_112.y);
                            float2 w_112 = make_float2(_exp2_1996, _exp2_1997);
                            float2 _f32x2_mul_b_1998 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_112) : "l"(*(const unsigned long long*)&ax_112), "l"(*(const unsigned long long*)&_f32x2_mul_b_1998));
                            float _exp2_1999 = approx_exp2_noftz(tp_112.x);
                            float _exp2_2000 = approx_exp2_noftz(tp_112.y);
                            float2 p_112 = make_float2(_exp2_1999, _exp2_2000);
                            float2 d1_112 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_112);
                            float2 d2_112 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_112);
                            float2 dw_112 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_112);
                            float2 ng_112 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_112);
                            float2 nu_112 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_112);
                            float _rcp_2001 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_112, d2_112), dw_112).x);
                            float _rcp_2002 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_112, d2_112), dw_112).y);
                            float2 r_226 = make_float2(_rcp_2001, _rcp_2002);
                            float2 r_225 = r_226;
                            float2 _f32x2_mul_a_2003 = mul_f32x2_noftz(ng_112, nu_112);
                            float2 _f32x2_mul_b_2004 = mul_f32x2_noftz(r_225, make_float2(bl, bl));
                            float2 m_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_112) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2003), "l"(*(const unsigned long long*)&_f32x2_mul_b_2004));
                            float2 ns_112 = make_float2(((x_113.x < 0.0f) ? p_112.x : 1.0f), ((x_113.y < 0.0f) ? p_112.y : 1.0f));
                            gate_packed_3[17] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_112, ns_112))) | sign_113;
                            float2 x_114 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[18]));
                            float2 v_114 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[18]));
                            unsigned int sign_114 = (gate_packed_3[18] ^ up_packed_3[18]) & 2147516416u;
                            float _abs_2005;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2005) : "f"(x_114.x));
                            float _abs_2006;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2006) : "f"(x_114.y));
                            float2 ax_113 = make_float2(_abs_2005, _abs_2006);
                            float _abs_2007;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2007) : "f"(v_114.x));
                            float _abs_2008;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2008) : "f"(v_114.y));
                            float2 av_113 = make_float2(_abs_2007, _abs_2008);
                            float2 _f32x2_mul_b_2009 = make_float2(k1, k1);
                            float2 te_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_113) : "l"(*(const unsigned long long*)&ax_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_2009));
                            float2 _f32x2_mul_b_2010 = make_float2(k2, k2);
                            float2 tw_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_113) : "l"(*(const unsigned long long*)&av_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_2010));
                            float _exp2_2011 = approx_exp2(te_113.x);
                            float _exp2_2012 = approx_exp2(te_113.y);
                            float2 e_113 = make_float2(_exp2_2011, _exp2_2012);
                            float _exp2_2013 = approx_exp2(tw_113.x);
                            float _exp2_2014 = approx_exp2(tw_113.y);
                            float2 w_113 = make_float2(_exp2_2013, _exp2_2014);
                            float2 _f32x2_mul_b_2015 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_113) : "l"(*(const unsigned long long*)&ax_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_2015));
                            float _exp2_2016 = approx_exp2_noftz(tp_113.x);
                            float _exp2_2017 = approx_exp2_noftz(tp_113.y);
                            float2 p_113 = make_float2(_exp2_2016, _exp2_2017);
                            float2 d1_113 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_113);
                            float2 d2_113 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_113);
                            float2 dw_113 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_113);
                            float2 ng_113 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_113);
                            float2 nu_113 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_113);
                            float _rcp_2018 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_113, d2_113), dw_113).x);
                            float _rcp_2019 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_113, d2_113), dw_113).y);
                            float2 r_228 = make_float2(_rcp_2018, _rcp_2019);
                            float2 r_227 = r_228;
                            float2 _f32x2_mul_a_2020 = mul_f32x2_noftz(ng_113, nu_113);
                            float2 _f32x2_mul_b_2021 = mul_f32x2_noftz(r_227, make_float2(bl, bl));
                            float2 m_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_113) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2020), "l"(*(const unsigned long long*)&_f32x2_mul_b_2021));
                            float2 ns_113 = make_float2(((x_114.x < 0.0f) ? p_113.x : 1.0f), ((x_114.y < 0.0f) ? p_113.y : 1.0f));
                            gate_packed_3[18] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_113, ns_113))) | sign_114;
                            float2 x_115 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[19]));
                            float2 v_115 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[19]));
                            unsigned int sign_115 = (gate_packed_3[19] ^ up_packed_3[19]) & 2147516416u;
                            float _abs_2022;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2022) : "f"(x_115.x));
                            float _abs_2023;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2023) : "f"(x_115.y));
                            float2 ax_114 = make_float2(_abs_2022, _abs_2023);
                            float _abs_2024;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2024) : "f"(v_115.x));
                            float _abs_2025;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2025) : "f"(v_115.y));
                            float2 av_114 = make_float2(_abs_2024, _abs_2025);
                            float2 _f32x2_mul_b_2026 = make_float2(k1, k1);
                            float2 te_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_114) : "l"(*(const unsigned long long*)&ax_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_2026));
                            float2 _f32x2_mul_b_2027 = make_float2(k2, k2);
                            float2 tw_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_114) : "l"(*(const unsigned long long*)&av_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_2027));
                            float _exp2_2028 = approx_exp2(te_114.x);
                            float _exp2_2029 = approx_exp2(te_114.y);
                            float2 e_114 = make_float2(_exp2_2028, _exp2_2029);
                            float _exp2_2030 = approx_exp2(tw_114.x);
                            float _exp2_2031 = approx_exp2(tw_114.y);
                            float2 w_114 = make_float2(_exp2_2030, _exp2_2031);
                            float2 _f32x2_mul_b_2032 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_114) : "l"(*(const unsigned long long*)&ax_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_2032));
                            float _exp2_2033 = approx_exp2_noftz(tp_114.x);
                            float _exp2_2034 = approx_exp2_noftz(tp_114.y);
                            float2 p_114 = make_float2(_exp2_2033, _exp2_2034);
                            float2 d1_114 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_114);
                            float2 d2_114 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_114);
                            float2 dw_114 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_114);
                            float2 ng_114 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_114);
                            float2 nu_114 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_114);
                            float _rcp_2035 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_114, d2_114), dw_114).x);
                            float _rcp_2036 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_114, d2_114), dw_114).y);
                            float2 r_230 = make_float2(_rcp_2035, _rcp_2036);
                            float2 r_229 = r_230;
                            float2 _f32x2_mul_a_2037 = mul_f32x2_noftz(ng_114, nu_114);
                            float2 _f32x2_mul_b_2038 = mul_f32x2_noftz(r_229, make_float2(bl, bl));
                            float2 m_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_114) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2037), "l"(*(const unsigned long long*)&_f32x2_mul_b_2038));
                            float2 ns_114 = make_float2(((x_115.x < 0.0f) ? p_114.x : 1.0f), ((x_115.y < 0.0f) ? p_114.y : 1.0f));
                            gate_packed_3[19] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_114, ns_114))) | sign_115;
                            float2 x_116 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[20]));
                            float2 v_116 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[20]));
                            unsigned int sign_116 = (gate_packed_3[20] ^ up_packed_3[20]) & 2147516416u;
                            float _abs_2039;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2039) : "f"(x_116.x));
                            float _abs_2040;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2040) : "f"(x_116.y));
                            float2 ax_115 = make_float2(_abs_2039, _abs_2040);
                            float _abs_2041;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2041) : "f"(v_116.x));
                            float _abs_2042;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2042) : "f"(v_116.y));
                            float2 av_115 = make_float2(_abs_2041, _abs_2042);
                            float2 _f32x2_mul_b_2043 = make_float2(k1, k1);
                            float2 te_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_115) : "l"(*(const unsigned long long*)&ax_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_2043));
                            float2 _f32x2_mul_b_2044 = make_float2(k2, k2);
                            float2 tw_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_115) : "l"(*(const unsigned long long*)&av_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_2044));
                            float _exp2_2045 = approx_exp2(te_115.x);
                            float _exp2_2046 = approx_exp2(te_115.y);
                            float2 e_115 = make_float2(_exp2_2045, _exp2_2046);
                            float _exp2_2047 = approx_exp2(tw_115.x);
                            float _exp2_2048 = approx_exp2(tw_115.y);
                            float2 w_115 = make_float2(_exp2_2047, _exp2_2048);
                            float2 _f32x2_mul_b_2049 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_115) : "l"(*(const unsigned long long*)&ax_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_2049));
                            float _exp2_2050 = approx_exp2_noftz(tp_115.x);
                            float _exp2_2051 = approx_exp2_noftz(tp_115.y);
                            float2 p_115 = make_float2(_exp2_2050, _exp2_2051);
                            float2 d1_115 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_115);
                            float2 d2_115 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_115);
                            float2 dw_115 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_115);
                            float2 ng_115 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_115);
                            float2 nu_115 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_115);
                            float _rcp_2052 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_115, d2_115), dw_115).x);
                            float _rcp_2053 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_115, d2_115), dw_115).y);
                            float2 r_232 = make_float2(_rcp_2052, _rcp_2053);
                            float2 r_231 = r_232;
                            float2 _f32x2_mul_a_2054 = mul_f32x2_noftz(ng_115, nu_115);
                            float2 _f32x2_mul_b_2055 = mul_f32x2_noftz(r_231, make_float2(bl, bl));
                            float2 m_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_115) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2054), "l"(*(const unsigned long long*)&_f32x2_mul_b_2055));
                            float2 ns_115 = make_float2(((x_116.x < 0.0f) ? p_115.x : 1.0f), ((x_116.y < 0.0f) ? p_115.y : 1.0f));
                            gate_packed_3[20] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_115, ns_115))) | sign_116;
                            float2 x_117 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[21]));
                            float2 v_117 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[21]));
                            unsigned int sign_117 = (gate_packed_3[21] ^ up_packed_3[21]) & 2147516416u;
                            float _abs_2056;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2056) : "f"(x_117.x));
                            float _abs_2057;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2057) : "f"(x_117.y));
                            float2 ax_116 = make_float2(_abs_2056, _abs_2057);
                            float _abs_2058;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2058) : "f"(v_117.x));
                            float _abs_2059;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2059) : "f"(v_117.y));
                            float2 av_116 = make_float2(_abs_2058, _abs_2059);
                            float2 _f32x2_mul_b_2060 = make_float2(k1, k1);
                            float2 te_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_116) : "l"(*(const unsigned long long*)&ax_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_2060));
                            float2 _f32x2_mul_b_2061 = make_float2(k2, k2);
                            float2 tw_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_116) : "l"(*(const unsigned long long*)&av_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_2061));
                            float _exp2_2062 = approx_exp2(te_116.x);
                            float _exp2_2063 = approx_exp2(te_116.y);
                            float2 e_116 = make_float2(_exp2_2062, _exp2_2063);
                            float _exp2_2064 = approx_exp2(tw_116.x);
                            float _exp2_2065 = approx_exp2(tw_116.y);
                            float2 w_116 = make_float2(_exp2_2064, _exp2_2065);
                            float2 _f32x2_mul_b_2066 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_116) : "l"(*(const unsigned long long*)&ax_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_2066));
                            float _exp2_2067 = approx_exp2_noftz(tp_116.x);
                            float _exp2_2068 = approx_exp2_noftz(tp_116.y);
                            float2 p_116 = make_float2(_exp2_2067, _exp2_2068);
                            float2 d1_116 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_116);
                            float2 d2_116 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_116);
                            float2 dw_116 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_116);
                            float2 ng_116 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_116);
                            float2 nu_116 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_116);
                            float _rcp_2069 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_116, d2_116), dw_116).x);
                            float _rcp_2070 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_116, d2_116), dw_116).y);
                            float2 r_234 = make_float2(_rcp_2069, _rcp_2070);
                            float2 r_233 = r_234;
                            float2 _f32x2_mul_a_2071 = mul_f32x2_noftz(ng_116, nu_116);
                            float2 _f32x2_mul_b_2072 = mul_f32x2_noftz(r_233, make_float2(bl, bl));
                            float2 m_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2071), "l"(*(const unsigned long long*)&_f32x2_mul_b_2072));
                            float2 ns_116 = make_float2(((x_117.x < 0.0f) ? p_116.x : 1.0f), ((x_117.y < 0.0f) ? p_116.y : 1.0f));
                            gate_packed_3[21] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_116, ns_116))) | sign_117;
                            float2 x_118 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[22]));
                            float2 v_118 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[22]));
                            unsigned int sign_118 = (gate_packed_3[22] ^ up_packed_3[22]) & 2147516416u;
                            float _abs_2073;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2073) : "f"(x_118.x));
                            float _abs_2074;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2074) : "f"(x_118.y));
                            float2 ax_117 = make_float2(_abs_2073, _abs_2074);
                            float _abs_2075;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2075) : "f"(v_118.x));
                            float _abs_2076;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2076) : "f"(v_118.y));
                            float2 av_117 = make_float2(_abs_2075, _abs_2076);
                            float2 _f32x2_mul_b_2077 = make_float2(k1, k1);
                            float2 te_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_117) : "l"(*(const unsigned long long*)&ax_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_2077));
                            float2 _f32x2_mul_b_2078 = make_float2(k2, k2);
                            float2 tw_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_117) : "l"(*(const unsigned long long*)&av_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_2078));
                            float _exp2_2079 = approx_exp2(te_117.x);
                            float _exp2_2080 = approx_exp2(te_117.y);
                            float2 e_117 = make_float2(_exp2_2079, _exp2_2080);
                            float _exp2_2081 = approx_exp2(tw_117.x);
                            float _exp2_2082 = approx_exp2(tw_117.y);
                            float2 w_117 = make_float2(_exp2_2081, _exp2_2082);
                            float2 _f32x2_mul_b_2083 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_117) : "l"(*(const unsigned long long*)&ax_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_2083));
                            float _exp2_2084 = approx_exp2_noftz(tp_117.x);
                            float _exp2_2085 = approx_exp2_noftz(tp_117.y);
                            float2 p_117 = make_float2(_exp2_2084, _exp2_2085);
                            float2 d1_117 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_117);
                            float2 d2_117 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_117);
                            float2 dw_117 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_117);
                            float2 ng_117 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_117);
                            float2 nu_117 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_117);
                            float _rcp_2086 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_117, d2_117), dw_117).x);
                            float _rcp_2087 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_117, d2_117), dw_117).y);
                            float2 r_236 = make_float2(_rcp_2086, _rcp_2087);
                            float2 r_235 = r_236;
                            float2 _f32x2_mul_a_2088 = mul_f32x2_noftz(ng_117, nu_117);
                            float2 _f32x2_mul_b_2089 = mul_f32x2_noftz(r_235, make_float2(bl, bl));
                            float2 m_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_117) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2088), "l"(*(const unsigned long long*)&_f32x2_mul_b_2089));
                            float2 ns_117 = make_float2(((x_118.x < 0.0f) ? p_117.x : 1.0f), ((x_118.y < 0.0f) ? p_117.y : 1.0f));
                            gate_packed_3[22] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_117, ns_117))) | sign_118;
                            float2 x_119 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[23]));
                            float2 v_119 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[23]));
                            unsigned int sign_119 = (gate_packed_3[23] ^ up_packed_3[23]) & 2147516416u;
                            float _abs_2090;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2090) : "f"(x_119.x));
                            float _abs_2091;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2091) : "f"(x_119.y));
                            float2 ax_118 = make_float2(_abs_2090, _abs_2091);
                            float _abs_2092;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2092) : "f"(v_119.x));
                            float _abs_2093;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2093) : "f"(v_119.y));
                            float2 av_118 = make_float2(_abs_2092, _abs_2093);
                            float2 _f32x2_mul_b_2094 = make_float2(k1, k1);
                            float2 te_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_118) : "l"(*(const unsigned long long*)&ax_118), "l"(*(const unsigned long long*)&_f32x2_mul_b_2094));
                            float2 _f32x2_mul_b_2095 = make_float2(k2, k2);
                            float2 tw_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_118) : "l"(*(const unsigned long long*)&av_118), "l"(*(const unsigned long long*)&_f32x2_mul_b_2095));
                            float _exp2_2096 = approx_exp2(te_118.x);
                            float _exp2_2097 = approx_exp2(te_118.y);
                            float2 e_118 = make_float2(_exp2_2096, _exp2_2097);
                            float _exp2_2098 = approx_exp2(tw_118.x);
                            float _exp2_2099 = approx_exp2(tw_118.y);
                            float2 w_118 = make_float2(_exp2_2098, _exp2_2099);
                            float2 _f32x2_mul_b_2100 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_118) : "l"(*(const unsigned long long*)&ax_118), "l"(*(const unsigned long long*)&_f32x2_mul_b_2100));
                            float _exp2_2101 = approx_exp2_noftz(tp_118.x);
                            float _exp2_2102 = approx_exp2_noftz(tp_118.y);
                            float2 p_118 = make_float2(_exp2_2101, _exp2_2102);
                            float2 d1_118 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_118);
                            float2 d2_118 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_118);
                            float2 dw_118 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_118);
                            float2 ng_118 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_118);
                            float2 nu_118 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_118);
                            float _rcp_2103 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_118, d2_118), dw_118).x);
                            float _rcp_2104 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_118, d2_118), dw_118).y);
                            float2 r_238 = make_float2(_rcp_2103, _rcp_2104);
                            float2 r_237 = r_238;
                            float2 _f32x2_mul_a_2105 = mul_f32x2_noftz(ng_118, nu_118);
                            float2 _f32x2_mul_b_2106 = mul_f32x2_noftz(r_237, make_float2(bl, bl));
                            float2 m_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_118) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2105), "l"(*(const unsigned long long*)&_f32x2_mul_b_2106));
                            float2 ns_118 = make_float2(((x_119.x < 0.0f) ? p_118.x : 1.0f), ((x_119.y < 0.0f) ? p_118.y : 1.0f));
                            gate_packed_3[23] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_118, ns_118))) | sign_119;
                            float2 x_120 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[24]));
                            float2 v_120 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[24]));
                            unsigned int sign_120 = (gate_packed_3[24] ^ up_packed_3[24]) & 2147516416u;
                            float _abs_2107;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2107) : "f"(x_120.x));
                            float _abs_2108;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2108) : "f"(x_120.y));
                            float2 ax_119 = make_float2(_abs_2107, _abs_2108);
                            float _abs_2109;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2109) : "f"(v_120.x));
                            float _abs_2110;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2110) : "f"(v_120.y));
                            float2 av_119 = make_float2(_abs_2109, _abs_2110);
                            float2 _f32x2_mul_b_2111 = make_float2(k1, k1);
                            float2 te_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_119) : "l"(*(const unsigned long long*)&ax_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_2111));
                            float2 _f32x2_mul_b_2112 = make_float2(k2, k2);
                            float2 tw_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_119) : "l"(*(const unsigned long long*)&av_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_2112));
                            float _exp2_2113 = approx_exp2(te_119.x);
                            float _exp2_2114 = approx_exp2(te_119.y);
                            float2 e_119 = make_float2(_exp2_2113, _exp2_2114);
                            float _exp2_2115 = approx_exp2(tw_119.x);
                            float _exp2_2116 = approx_exp2(tw_119.y);
                            float2 w_119 = make_float2(_exp2_2115, _exp2_2116);
                            float2 _f32x2_mul_b_2117 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_119) : "l"(*(const unsigned long long*)&ax_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_2117));
                            float _exp2_2118 = approx_exp2_noftz(tp_119.x);
                            float _exp2_2119 = approx_exp2_noftz(tp_119.y);
                            float2 p_119 = make_float2(_exp2_2118, _exp2_2119);
                            float2 d1_119 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_119);
                            float2 d2_119 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_119);
                            float2 dw_119 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_119);
                            float2 ng_119 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_119);
                            float2 nu_119 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_119);
                            float _rcp_2120 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_119, d2_119), dw_119).x);
                            float _rcp_2121 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_119, d2_119), dw_119).y);
                            float2 r_240 = make_float2(_rcp_2120, _rcp_2121);
                            float2 r_239 = r_240;
                            float2 _f32x2_mul_a_2122 = mul_f32x2_noftz(ng_119, nu_119);
                            float2 _f32x2_mul_b_2123 = mul_f32x2_noftz(r_239, make_float2(bl, bl));
                            float2 m_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_119) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2123));
                            float2 ns_119 = make_float2(((x_120.x < 0.0f) ? p_119.x : 1.0f), ((x_120.y < 0.0f) ? p_119.y : 1.0f));
                            gate_packed_3[24] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_119, ns_119))) | sign_120;
                            float2 x_121 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[25]));
                            float2 v_121 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[25]));
                            unsigned int sign_121 = (gate_packed_3[25] ^ up_packed_3[25]) & 2147516416u;
                            float _abs_2124;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2124) : "f"(x_121.x));
                            float _abs_2125;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2125) : "f"(x_121.y));
                            float2 ax_120 = make_float2(_abs_2124, _abs_2125);
                            float _abs_2126;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2126) : "f"(v_121.x));
                            float _abs_2127;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2127) : "f"(v_121.y));
                            float2 av_120 = make_float2(_abs_2126, _abs_2127);
                            float2 _f32x2_mul_b_2128 = make_float2(k1, k1);
                            float2 te_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_120) : "l"(*(const unsigned long long*)&ax_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_2128));
                            float2 _f32x2_mul_b_2129 = make_float2(k2, k2);
                            float2 tw_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_120) : "l"(*(const unsigned long long*)&av_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_2129));
                            float _exp2_2130 = approx_exp2(te_120.x);
                            float _exp2_2131 = approx_exp2(te_120.y);
                            float2 e_120 = make_float2(_exp2_2130, _exp2_2131);
                            float _exp2_2132 = approx_exp2(tw_120.x);
                            float _exp2_2133 = approx_exp2(tw_120.y);
                            float2 w_120 = make_float2(_exp2_2132, _exp2_2133);
                            float2 _f32x2_mul_b_2134 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_120) : "l"(*(const unsigned long long*)&ax_120), "l"(*(const unsigned long long*)&_f32x2_mul_b_2134));
                            float _exp2_2135 = approx_exp2_noftz(tp_120.x);
                            float _exp2_2136 = approx_exp2_noftz(tp_120.y);
                            float2 p_120 = make_float2(_exp2_2135, _exp2_2136);
                            float2 d1_120 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_120);
                            float2 d2_120 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_120);
                            float2 dw_120 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_120);
                            float2 ng_120 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_120);
                            float2 nu_120 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_120);
                            float _rcp_2137 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_120, d2_120), dw_120).x);
                            float _rcp_2138 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_120, d2_120), dw_120).y);
                            float2 r_242 = make_float2(_rcp_2137, _rcp_2138);
                            float2 r_241 = r_242;
                            float2 _f32x2_mul_a_2139 = mul_f32x2_noftz(ng_120, nu_120);
                            float2 _f32x2_mul_b_2140 = mul_f32x2_noftz(r_241, make_float2(bl, bl));
                            float2 m_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_120) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2139), "l"(*(const unsigned long long*)&_f32x2_mul_b_2140));
                            float2 ns_120 = make_float2(((x_121.x < 0.0f) ? p_120.x : 1.0f), ((x_121.y < 0.0f) ? p_120.y : 1.0f));
                            gate_packed_3[25] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_120, ns_120))) | sign_121;
                            float2 x_122 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[26]));
                            float2 v_122 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[26]));
                            unsigned int sign_122 = (gate_packed_3[26] ^ up_packed_3[26]) & 2147516416u;
                            float _abs_2141;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2141) : "f"(x_122.x));
                            float _abs_2142;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2142) : "f"(x_122.y));
                            float2 ax_121 = make_float2(_abs_2141, _abs_2142);
                            float _abs_2143;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2143) : "f"(v_122.x));
                            float _abs_2144;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2144) : "f"(v_122.y));
                            float2 av_121 = make_float2(_abs_2143, _abs_2144);
                            float2 _f32x2_mul_b_2145 = make_float2(k1, k1);
                            float2 te_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_121) : "l"(*(const unsigned long long*)&ax_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_2145));
                            float2 _f32x2_mul_b_2146 = make_float2(k2, k2);
                            float2 tw_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_121) : "l"(*(const unsigned long long*)&av_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_2146));
                            float _exp2_2147 = approx_exp2(te_121.x);
                            float _exp2_2148 = approx_exp2(te_121.y);
                            float2 e_121 = make_float2(_exp2_2147, _exp2_2148);
                            float _exp2_2149 = approx_exp2(tw_121.x);
                            float _exp2_2150 = approx_exp2(tw_121.y);
                            float2 w_121 = make_float2(_exp2_2149, _exp2_2150);
                            float2 _f32x2_mul_b_2151 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_121) : "l"(*(const unsigned long long*)&ax_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_2151));
                            float _exp2_2152 = approx_exp2_noftz(tp_121.x);
                            float _exp2_2153 = approx_exp2_noftz(tp_121.y);
                            float2 p_121 = make_float2(_exp2_2152, _exp2_2153);
                            float2 d1_121 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_121);
                            float2 d2_121 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_121);
                            float2 dw_121 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_121);
                            float2 ng_121 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_121);
                            float2 nu_121 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_121);
                            float _rcp_2154 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_121, d2_121), dw_121).x);
                            float _rcp_2155 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_121, d2_121), dw_121).y);
                            float2 r_244 = make_float2(_rcp_2154, _rcp_2155);
                            float2 r_243 = r_244;
                            float2 _f32x2_mul_a_2156 = mul_f32x2_noftz(ng_121, nu_121);
                            float2 _f32x2_mul_b_2157 = mul_f32x2_noftz(r_243, make_float2(bl, bl));
                            float2 m_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_121) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2156), "l"(*(const unsigned long long*)&_f32x2_mul_b_2157));
                            float2 ns_121 = make_float2(((x_122.x < 0.0f) ? p_121.x : 1.0f), ((x_122.y < 0.0f) ? p_121.y : 1.0f));
                            gate_packed_3[26] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_121, ns_121))) | sign_122;
                            float2 x_123 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[27]));
                            float2 v_123 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[27]));
                            unsigned int sign_123 = (gate_packed_3[27] ^ up_packed_3[27]) & 2147516416u;
                            float _abs_2158;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2158) : "f"(x_123.x));
                            float _abs_2159;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2159) : "f"(x_123.y));
                            float2 ax_122 = make_float2(_abs_2158, _abs_2159);
                            float _abs_2160;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2160) : "f"(v_123.x));
                            float _abs_2161;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2161) : "f"(v_123.y));
                            float2 av_122 = make_float2(_abs_2160, _abs_2161);
                            float2 _f32x2_mul_b_2162 = make_float2(k1, k1);
                            float2 te_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_122) : "l"(*(const unsigned long long*)&ax_122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2162));
                            float2 _f32x2_mul_b_2163 = make_float2(k2, k2);
                            float2 tw_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_122) : "l"(*(const unsigned long long*)&av_122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2163));
                            float _exp2_2164 = approx_exp2(te_122.x);
                            float _exp2_2165 = approx_exp2(te_122.y);
                            float2 e_122 = make_float2(_exp2_2164, _exp2_2165);
                            float _exp2_2166 = approx_exp2(tw_122.x);
                            float _exp2_2167 = approx_exp2(tw_122.y);
                            float2 w_122 = make_float2(_exp2_2166, _exp2_2167);
                            float2 _f32x2_mul_b_2168 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_122) : "l"(*(const unsigned long long*)&ax_122), "l"(*(const unsigned long long*)&_f32x2_mul_b_2168));
                            float _exp2_2169 = approx_exp2_noftz(tp_122.x);
                            float _exp2_2170 = approx_exp2_noftz(tp_122.y);
                            float2 p_122 = make_float2(_exp2_2169, _exp2_2170);
                            float2 d1_122 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_122);
                            float2 d2_122 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_122);
                            float2 dw_122 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_122);
                            float2 ng_122 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_122);
                            float2 nu_122 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_122);
                            float _rcp_2171 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_122, d2_122), dw_122).x);
                            float _rcp_2172 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_122, d2_122), dw_122).y);
                            float2 r_246 = make_float2(_rcp_2171, _rcp_2172);
                            float2 r_245 = r_246;
                            float2 _f32x2_mul_a_2173 = mul_f32x2_noftz(ng_122, nu_122);
                            float2 _f32x2_mul_b_2174 = mul_f32x2_noftz(r_245, make_float2(bl, bl));
                            float2 m_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_122) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2173), "l"(*(const unsigned long long*)&_f32x2_mul_b_2174));
                            float2 ns_122 = make_float2(((x_123.x < 0.0f) ? p_122.x : 1.0f), ((x_123.y < 0.0f) ? p_122.y : 1.0f));
                            gate_packed_3[27] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_122, ns_122))) | sign_123;
                            float2 x_124 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[28]));
                            float2 v_124 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[28]));
                            unsigned int sign_124 = (gate_packed_3[28] ^ up_packed_3[28]) & 2147516416u;
                            float _abs_2175;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2175) : "f"(x_124.x));
                            float _abs_2176;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2176) : "f"(x_124.y));
                            float2 ax_123 = make_float2(_abs_2175, _abs_2176);
                            float _abs_2177;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2177) : "f"(v_124.x));
                            float _abs_2178;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2178) : "f"(v_124.y));
                            float2 av_123 = make_float2(_abs_2177, _abs_2178);
                            float2 _f32x2_mul_b_2179 = make_float2(k1, k1);
                            float2 te_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_123) : "l"(*(const unsigned long long*)&ax_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_2179));
                            float2 _f32x2_mul_b_2180 = make_float2(k2, k2);
                            float2 tw_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_123) : "l"(*(const unsigned long long*)&av_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_2180));
                            float _exp2_2181 = approx_exp2(te_123.x);
                            float _exp2_2182 = approx_exp2(te_123.y);
                            float2 e_123 = make_float2(_exp2_2181, _exp2_2182);
                            float _exp2_2183 = approx_exp2(tw_123.x);
                            float _exp2_2184 = approx_exp2(tw_123.y);
                            float2 w_123 = make_float2(_exp2_2183, _exp2_2184);
                            float2 _f32x2_mul_b_2185 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_123) : "l"(*(const unsigned long long*)&ax_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_2185));
                            float _exp2_2186 = approx_exp2_noftz(tp_123.x);
                            float _exp2_2187 = approx_exp2_noftz(tp_123.y);
                            float2 p_123 = make_float2(_exp2_2186, _exp2_2187);
                            float2 d1_123 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_123);
                            float2 d2_123 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_123);
                            float2 dw_123 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_123);
                            float2 ng_123 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_123);
                            float2 nu_123 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_123);
                            float _rcp_2188 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_123, d2_123), dw_123).x);
                            float _rcp_2189 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_123, d2_123), dw_123).y);
                            float2 r_248 = make_float2(_rcp_2188, _rcp_2189);
                            float2 r_247 = r_248;
                            float2 _f32x2_mul_a_2190 = mul_f32x2_noftz(ng_123, nu_123);
                            float2 _f32x2_mul_b_2191 = mul_f32x2_noftz(r_247, make_float2(bl, bl));
                            float2 m_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_123) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2190), "l"(*(const unsigned long long*)&_f32x2_mul_b_2191));
                            float2 ns_123 = make_float2(((x_124.x < 0.0f) ? p_123.x : 1.0f), ((x_124.y < 0.0f) ? p_123.y : 1.0f));
                            gate_packed_3[28] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_123, ns_123))) | sign_124;
                            float2 x_125 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[29]));
                            float2 v_125 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[29]));
                            unsigned int sign_125 = (gate_packed_3[29] ^ up_packed_3[29]) & 2147516416u;
                            float _abs_2192;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2192) : "f"(x_125.x));
                            float _abs_2193;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2193) : "f"(x_125.y));
                            float2 ax_124 = make_float2(_abs_2192, _abs_2193);
                            float _abs_2194;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2194) : "f"(v_125.x));
                            float _abs_2195;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2195) : "f"(v_125.y));
                            float2 av_124 = make_float2(_abs_2194, _abs_2195);
                            float2 _f32x2_mul_b_2196 = make_float2(k1, k1);
                            float2 te_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_124) : "l"(*(const unsigned long long*)&ax_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_2196));
                            float2 _f32x2_mul_b_2197 = make_float2(k2, k2);
                            float2 tw_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_124) : "l"(*(const unsigned long long*)&av_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_2197));
                            float _exp2_2198 = approx_exp2(te_124.x);
                            float _exp2_2199 = approx_exp2(te_124.y);
                            float2 e_124 = make_float2(_exp2_2198, _exp2_2199);
                            float _exp2_2200 = approx_exp2(tw_124.x);
                            float _exp2_2201 = approx_exp2(tw_124.y);
                            float2 w_124 = make_float2(_exp2_2200, _exp2_2201);
                            float2 _f32x2_mul_b_2202 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_124) : "l"(*(const unsigned long long*)&ax_124), "l"(*(const unsigned long long*)&_f32x2_mul_b_2202));
                            float _exp2_2203 = approx_exp2_noftz(tp_124.x);
                            float _exp2_2204 = approx_exp2_noftz(tp_124.y);
                            float2 p_124 = make_float2(_exp2_2203, _exp2_2204);
                            float2 d1_124 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_124);
                            float2 d2_124 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_124);
                            float2 dw_124 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_124);
                            float2 ng_124 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_124);
                            float2 nu_124 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_124);
                            float _rcp_2205 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_124, d2_124), dw_124).x);
                            float _rcp_2206 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_124, d2_124), dw_124).y);
                            float2 r_250 = make_float2(_rcp_2205, _rcp_2206);
                            float2 r_249 = r_250;
                            float2 _f32x2_mul_a_2207 = mul_f32x2_noftz(ng_124, nu_124);
                            float2 _f32x2_mul_b_2208 = mul_f32x2_noftz(r_249, make_float2(bl, bl));
                            float2 m_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_124) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2207), "l"(*(const unsigned long long*)&_f32x2_mul_b_2208));
                            float2 ns_124 = make_float2(((x_125.x < 0.0f) ? p_124.x : 1.0f), ((x_125.y < 0.0f) ? p_124.y : 1.0f));
                            gate_packed_3[29] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_124, ns_124))) | sign_125;
                            float2 x_126 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[30]));
                            float2 v_126 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[30]));
                            unsigned int sign_126 = (gate_packed_3[30] ^ up_packed_3[30]) & 2147516416u;
                            float _abs_2209;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2209) : "f"(x_126.x));
                            float _abs_2210;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2210) : "f"(x_126.y));
                            float2 ax_125 = make_float2(_abs_2209, _abs_2210);
                            float _abs_2211;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2211) : "f"(v_126.x));
                            float _abs_2212;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2212) : "f"(v_126.y));
                            float2 av_125 = make_float2(_abs_2211, _abs_2212);
                            float2 _f32x2_mul_b_2213 = make_float2(k1, k1);
                            float2 te_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_125) : "l"(*(const unsigned long long*)&ax_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_2213));
                            float2 _f32x2_mul_b_2214 = make_float2(k2, k2);
                            float2 tw_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_125) : "l"(*(const unsigned long long*)&av_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_2214));
                            float _exp2_2215 = approx_exp2(te_125.x);
                            float _exp2_2216 = approx_exp2(te_125.y);
                            float2 e_125 = make_float2(_exp2_2215, _exp2_2216);
                            float _exp2_2217 = approx_exp2(tw_125.x);
                            float _exp2_2218 = approx_exp2(tw_125.y);
                            float2 w_125 = make_float2(_exp2_2217, _exp2_2218);
                            float2 _f32x2_mul_b_2219 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_125) : "l"(*(const unsigned long long*)&ax_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_2219));
                            float _exp2_2220 = approx_exp2_noftz(tp_125.x);
                            float _exp2_2221 = approx_exp2_noftz(tp_125.y);
                            float2 p_125 = make_float2(_exp2_2220, _exp2_2221);
                            float2 d1_125 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_125);
                            float2 d2_125 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_125);
                            float2 dw_125 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_125);
                            float2 ng_125 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_125);
                            float2 nu_125 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_125);
                            float _rcp_2222 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_125, d2_125), dw_125).x);
                            float _rcp_2223 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_125, d2_125), dw_125).y);
                            float2 r_252 = make_float2(_rcp_2222, _rcp_2223);
                            float2 r_251 = r_252;
                            float2 _f32x2_mul_a_2224 = mul_f32x2_noftz(ng_125, nu_125);
                            float2 _f32x2_mul_b_2225 = mul_f32x2_noftz(r_251, make_float2(bl, bl));
                            float2 m_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_125) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2224), "l"(*(const unsigned long long*)&_f32x2_mul_b_2225));
                            float2 ns_125 = make_float2(((x_126.x < 0.0f) ? p_125.x : 1.0f), ((x_126.y < 0.0f) ? p_125.y : 1.0f));
                            gate_packed_3[30] = __as_u32(__float22bfloat162_rn(mul_f32x2_noftz(m_125, ns_125))) | sign_126;
                            float2 x_127 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[31]));
                            float2 v_127 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[31]));
                            unsigned int sign_127 = (gate_packed_3[31] ^ up_packed_3[31]) & 2147516416u;
                            float _abs_2226;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2226) : "f"(x_127.x));
                            float _abs_2227;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2227) : "f"(x_127.y));
                            float2 ax_126 = make_float2(_abs_2226, _abs_2227);
                            float _abs_2228;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2228) : "f"(v_127.x));
                            float _abs_2229;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_2229) : "f"(v_127.y));
                            float2 av_126 = make_float2(_abs_2228, _abs_2229);
                            float2 _f32x2_mul_b_2230 = make_float2(k1, k1);
                            float2 te_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&te_126) : "l"(*(const unsigned long long*)&ax_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_2230));
                            float2 _f32x2_mul_b_2231 = make_float2(k2, k2);
                            float2 tw_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tw_126) : "l"(*(const unsigned long long*)&av_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_2231));
                            float _exp2_2232 = approx_exp2(te_126.x);
                            float _exp2_2233 = approx_exp2(te_126.y);
                            float2 e_126 = make_float2(_exp2_2232, _exp2_2233);
                            float _exp2_2234 = approx_exp2(tw_126.x);
                            float _exp2_2235 = approx_exp2(tw_126.y);
                            float2 w_126 = make_float2(_exp2_2234, _exp2_2235);
                            float2 _f32x2_mul_b_2236 = make_float2(-1.442695f, -1.442695f);
                            float2 tp_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tp_126) : "l"(*(const unsigned long long*)&ax_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_2236));
                            float _exp2_2237 = approx_exp2_noftz(tp_126.x);
                            float _exp2_2238 = approx_exp2_noftz(tp_126.y);
                            float2 p_126 = make_float2(_exp2_2237, _exp2_2238);
                            float2 d1_126 = add_f32x2_noftz(make_float2(1.0f, 1.0f), e_126);
                            float2 d2_126 = add_f32x2_noftz(make_float2(1.0f, 1.0f), p_126);
                            float2 dw_126 = add_f32x2_noftz(make_float2(1.0f, 1.0f), w_126);
                            float2 ng_126 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), e_126);
                            float2 nu_126 = sub_f32x2_noftz(make_float2(1.0f, 1.0f), w_126);
                            float _rcp_2239 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_126, d2_126), dw_126).x);
                            float _rcp_2240 = approx_rcp(mul_f32x2_noftz(mul_f32x2_noftz(d1_126, d2_126), dw_126).y);
                            float2 r_254 = make_float2(_rcp_2239, _rcp_2240);
                            float2 r_253 = r_254;
                            float2 _f32x2_mul_a_2241 = mul_f32x2_noftz(ng_126, nu_126);
                            float2 _f32x2_mul_b_2242 = mul_f32x2_noftz(r_253, make_float2(bl, bl));
                            float2 m_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&m_126) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2241), "l"(*(const unsigned long long*)&_f32x2_mul_b_2242));
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
                                    int row_27 = warp_22 * 32 + half_23 * 16 + lane_22 % 16;
                                    int col_24 = col_tile_23 * 16 + lane_22 / 16 * 8;
                                    unsigned int address_23 = d_smem_addr + 16384 + (unsigned int)((row_27 * 32 + col_24) * 2);
                                    address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                    int offset_24 = 16 + half_23 * 8 + col_tile_23 * 4;
                                    uint32_t _stmatrix_addr_2243 = static_cast<uint32_t>(address_23);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_2243), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_24 + 3]))
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
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + macro_1 * (macro_size / 256) + x))), "r"(static_cast<unsigned int>(2)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    gemm_wide += iterations * (int)(expert >= 0);
                    gemm_out0 += (int)(expert >= 0);
                    gemm_out1 += (int)(expert >= 0);
                    gemm_acc_tiles += 2 * (int)(expert >= 0);
                } else {
                    int col_blocks_2 = (hidden + 512 - 1) / 512;
                    int x_128 = -1;
                    int y_1 = -1;
                    int expert_1 = -1;
                    int k_start_1 = 0;
                    int k_end_1 = 0;
                    int first_1 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                    int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int block_6 = first_block_1;
                    uint32_t fast_div_q_1 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(task_1 - mini_fused)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(task_1 - mini_fused));
                    uint32_t fast_div_r_1 = (uint32_t)((unsigned int)(task_1 - mini_fused)) - fast_div_q_1 * (uint32_t)(down_col_blocks_div.divisor);
                    block_6 = first_block_1 + (int)fast_div_q_1;
                    if (block_6 < end_block_1) {
                        int index_1 = counts[3 * experts + block_6];
                        int offset_25 = counts[experts + index_1] / 256;
                        int first_row_7 = ((first_block_1) > (offset_25) ? (first_block_1) : (offset_25));
                        int _min_2244 = ((end_block_1) < (offset_25 + counts[index_1] / 256) ? (end_block_1) : (offset_25 + counts[index_1] / 256));
                        int rows_6 = _min_2244 - first_row_7;
                        int supergroup_1 = (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_2) / (rows_6 * 8);
                        int full_cols_1 = col_blocks_2 / 8 * 8;
                        int row_28 = 0;
                        int col_25 = 0;
                        if (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_2 < rows_6 * full_cols_1) {
                            row_28 = (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_2) % (rows_6 * 8) / 8;
                            col_25 = supergroup_1 * 8 + (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_2) % 8;
                        } else {
                            row_28 = (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_2 - rows_6 * full_cols_1) / (col_blocks_2 - full_cols_1);
                            col_25 = full_cols_1 + (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_2 - rows_6 * full_cols_1) % (col_blocks_2 - full_cols_1);
                        }
                        if ((supergroup_1 & 1) != 0) {
                            row_28 = rows_6 - row_28 - 1;
                        }
                        x_128 = first_row_7 + row_28 - macro_1 * (macro_size / 256);
                        y_1 = col_25;
                        expert_1 = index_1;
                    }
                    int has_hi_1 = 0;
                    has_hi_1 = (int)((y_1 * 2 + 1) * 256 < hidden);
                    int global_mini_1 = macro_1 * (macro_size / mini_size) + mini_3;
                    int macro_rows_2 = macro_1 * (macro_size / 256);
                    int iterations_1 = intermediate / 64;
                    if (expert_1 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    bool enabled_value_4 = 1;
                                    if (enabled_value_4 != 0) {
                                        int poll_5_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_first) : "l"(hidden_ready + (shared_rows + macro_rows_2 + x_128)) : "memory");
                                        int poll_5 = poll_5_first;
                                        while (poll_5 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_5_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_next) : "l"(hidden_ready + (shared_rows + macro_rows_2 + x_128)) : "memory");
                                            poll_5 = poll_5_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_2245 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                    int mini_rows_5 = ((0) > (_min_2245) ? (0) : (_min_2245));
                                    int required_3 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
                                }
                                #pragma unroll 1
                                for (int idx_6 = 0; idx_6 < iterations_1; idx_6++) {
                                    int _stage_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_6) % 4);
                                    int _parity_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_6) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_1) * 8, _parity_gemm_finished_1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&hidden_routed_in)), "r"(0), "r"(x_128 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&wd_routed)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_1), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&wd_routed)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_1), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int head_1 = ((3) < (iterations_1) ? (3) : (iterations_1));
                                    int tail_2 = 0;
                                    int _min_2246 = ((3) < (iterations_1 - head_1) ? (3) : (iterations_1 - head_1));
                                    tail_2 = _min_2246;
                                    int _stage_output_finished_2 = (int)((unsigned int)gemm_acc_tiles % 2);
                                    int _parity_output_finished_2 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_2) * 8, _parity_output_finished_2);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_7 = 0; idx_7 < head_1; idx_7++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_7) % 4);
                                        int _parity_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_7) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_3) * 8, _parity_gemm_arrived_3);
                                        int _smem_desc_lo_2247 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_7) % 4) * 1024;
                                        int _smem_desc_lo_2248 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_7) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_2247), "r"(_smem_desc_lo_2248), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)) : "memory");
                                    }
                                    int _stage_output_finished_3 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                    int _parity_output_finished_3 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_3) * 8, _parity_output_finished_3);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_8 = 0; idx_8 < head_1; idx_8++) {
                                        int _smem_desc_lo_2249 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_8) % 4) * 1024;
                                        int _smem_desc_lo_2250 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_8) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_2249), "r"(_smem_desc_lo_2250), "r"((tmem_accumulator + (256))), "r"(((idx_8 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_8) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_9 = head_1; idx_9 < iterations_1 - tail_2; idx_9++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) % 4);
                                        int _parity_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_4) * 8, _parity_gemm_arrived_4);
                                        int _smem_desc_lo_2251 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                        int _smem_desc_lo_2252 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_2251), "r"(_smem_desc_lo_2252), "r"(tmem_accumulator), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                        int _smem_desc_lo_2253 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                        int _smem_desc_lo_2254 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_2253), "r"(_smem_desc_lo_2254), "r"((tmem_accumulator + (256))), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_10 = iterations_1 - tail_2; idx_10 < iterations_1; idx_10++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_10) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_10) % 4);
                                        int _parity_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_10) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_5) * 8, _parity_gemm_arrived_5);
                                        int _smem_desc_lo_2255 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_10) % 4) * 1024;
                                        int _smem_desc_lo_2256 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_10) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_2255), "r"(_smem_desc_lo_2256), "r"(tmem_accumulator), "r"(((idx_10 == 0) ? 0 : 1)) : "memory");
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    #pragma unroll 1
                                    for (int idx_11 = iterations_1 - tail_2; idx_11 < iterations_1; idx_11++) {
                                        int _smem_desc_lo_2257 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_11) % 4) * 1024;
                                        int _smem_desc_lo_2258 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_11) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_2257), "r"(_smem_desc_lo_2258), "r"((tmem_accumulator + (256))), "r"(((idx_11 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_11) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            int _parity_output_arrived_2 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_2);
                            unsigned int packed[128];
                            #pragma unroll
                            for (int chunk = 0; chunk < 8; chunk++) {
                                #pragma unroll
                                for (int sub = 0; sub < 2; sub++) {
                                    unsigned int address_24 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk * 32);
                                    float _tmem_load_32[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[15]))
                                        : "r"(address_24));
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; pair++) {
                                        packed[chunk * 16 + sub * 8 + pair] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32[pair * 2], _tmem_load_32[pair * 2 + 1])));
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int last = 1;
                            if (last != 0) {
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            if (tid == 0) {
                                int previous_offset_1 = (macro_1 + 1) * macro_size;
                                int output_row = x_128 * 256 + cta_rank_0 * 128;
                                int _min_2259 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                                if (output_row < _min_2259) {
                                    bool enabled_value_5 = 1;
                                    if (enabled_value_5 != 0) {
                                        int poll_6_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_first) : "l"(y_done + ((previous_offset_1 + output_row) / 128)) : "memory");
                                        int poll_6 = poll_6_first;
                                        while (poll_6 < 8 * ((hidden + 1023) / 1024)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_6_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_next) : "l"(y_done + ((previous_offset_1 + output_row) / 128)) : "memory");
                                            poll_6 = poll_6_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                            }
                            #pragma unroll
                            for (int chunk_1 = 0; chunk_1 < 8; chunk_1++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_23 = tid / 32;
                                int lane_23 = tid % 32;
                                #pragma unroll
                                for (int half_24 = 0; half_24 < 2; half_24++) {
                                    #pragma unroll
                                    for (int col_tile_24 = 0; col_tile_24 < 2; col_tile_24++) {
                                        int row_29 = warp_23 * 32 + half_24 * 16 + lane_23 % 16;
                                        int col_26 = col_tile_24 * 16 + lane_23 / 16 * 8;
                                        unsigned int address_25 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_29 * 32 + col_26) * 2);
                                        address_25 = address_25 ^ (address_25 & 511) >> 7 << 4;
                                        int offset_26 = chunk_1 * 16 + half_24 * 8 + col_tile_24 * 4;
                                        uint32_t _stmatrix_addr_2260 = static_cast<uint32_t>(address_25);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_2260), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&y_routed), 0, x_128 * 256 + cta_rank_0 * 128, y_1 * 2 * 8 + chunk_1, 0, 0, d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            int _parity_output_arrived_3 = (int)((unsigned int)gemm_out1 % 2);
                            mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_3);
                            if (has_hi_1 != 0) {
                                unsigned int packed_0[128];
                                #pragma unroll
                                for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                                    #pragma unroll
                                    for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                        unsigned int address_26 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_2 * 32);
                                        float _tmem_load_32_16[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[15]))
                                            : "r"(address_26));
                                        #pragma unroll
                                        for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                            packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16[pair_1 * 2], _tmem_load_32_16[pair_1 * 2 + 1])));
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_0 = 1;
                                if (last_0 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                #pragma unroll
                                for (int chunk_3 = 0; chunk_3 < 8; chunk_3++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_24 = tid / 32;
                                    int lane_24 = tid % 32;
                                    #pragma unroll
                                    for (int half_25 = 0; half_25 < 2; half_25++) {
                                        #pragma unroll
                                        for (int col_tile_25 = 0; col_tile_25 < 2; col_tile_25++) {
                                            int row_30 = warp_24 * 32 + half_25 * 16 + lane_24 % 16;
                                            int col_27 = col_tile_25 * 16 + lane_24 / 16 * 8;
                                            unsigned int address_27 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_30 * 32 + col_27) * 2);
                                            address_27 = address_27 ^ (address_27 & 511) >> 7 << 4;
                                            int offset_27 = chunk_3 * 16 + half_25 * 8 + col_tile_25 * 4;
                                            uint32_t _stmatrix_addr_2261 = static_cast<uint32_t>(address_27);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_2261), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&y_routed), 0, x_128 * 256 + cta_rank_0 * 128, (y_1 * 2 + 1) * 8 + chunk_3, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            if (has_hi_1 == 0) {
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (tid / 32 == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_6 = 1;
                                        if (enabled_value_6 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                        if (has_hi_1 != 0) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_0 = 1;
                                            if (enabled_value_0 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_wide += iterations_1 * (int)(expert_1 >= 0);
                    gemm_out0 += (int)(expert_1 >= 0);
                    gemm_out1 += (int)(expert_1 >= 0);
                    gemm_acc_tiles += 2 * (int)(expert_1 >= 0);
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
